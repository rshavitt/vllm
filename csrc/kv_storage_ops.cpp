// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// kv_storage_ops: Async-friendly disk I/O for KV-cache block offloading.
//
// Each KV block is stored as a single file containing exactly block_size bytes.
//
// submit_store_job / submit_load_job:
//   Non-blocking: acquires buffer pointers (GIL held), registers a JobState,
//   and enqueues one task per block into the thread pool, then returns.
//   FD creation and directory setup happen inside the worker task (not upfront)
//   to avoid holding large numbers of open file descriptors for queued jobs.
//
// get_finished_jobs:
//   Returns a list of (job_id, success) pairs for all jobs whose last task
//   has completed.  Removes those jobs from the registry.
//
// Thread pool:
//   Two queues (read, write) and two sets of threads:
//     - Read-priority threads: drain the read queue first, then the write queue.
//     - Write-priority threads: drain the write queue first, then the read queue.
//   This ensures load (read) tasks are always preferred by at least half the
//   threads, reducing latency for cache-miss restores.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace py = pybind11;
namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// JobState — tracks completion of all per-file tasks belonging to one job
// ---------------------------------------------------------------------------

struct JobState {
  const int total_tasks;
  std::atomic<int>  completed{0};
  std::atomic<bool> all_success{true};

  explicit JobState(int n) : total_tasks(n) {}

  void task_done(bool ok) {
    if (!ok)
      all_success.store(false, std::memory_order_relaxed);
    // Release ensures prior writes (e.g. all_success) are visible to the
    // acquire in is_done().
    completed.fetch_add(1, std::memory_order_release);
  }

  bool is_done() const {
    return completed.load(std::memory_order_acquire) == total_tasks;
  }
};

// ---------------------------------------------------------------------------
// DualQueueThreadPool — two queues, two priority classes of threads
// ---------------------------------------------------------------------------

class DualQueueThreadPool {
 public:
  // n_read_prio:  threads that try the read queue first, write queue second.
  // n_write_prio: threads that try the write queue first, read queue second.
  DualQueueThreadPool(size_t n_read_prio, size_t n_write_prio) : stop_(false) {
    for (size_t i = 0; i < n_read_prio; ++i)
      workers_.emplace_back([this] { worker_loop(/*read_priority=*/true); });
    for (size_t i = 0; i < n_write_prio; ++i)
      workers_.emplace_back([this] { worker_loop(/*read_priority=*/false); });
  }

  ~DualQueueThreadPool() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& w : workers_) w.join();
  }

  void enqueue_read(std::function<void()> f) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      read_queue_.push(std::move(f));
    }
    cv_.notify_one();
  }

  void enqueue_write(std::function<void()> f) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      write_queue_.push(std::move(f));
    }
    cv_.notify_one();
  }

 private:
  void worker_loop(bool read_priority) {
    for (;;) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [this] {
          return stop_ || !read_queue_.empty() || !write_queue_.empty();
        });
        if (stop_ && read_queue_.empty() && write_queue_.empty()) return;

        auto& primary   = read_priority ? read_queue_  : write_queue_;
        auto& secondary = read_priority ? write_queue_ : read_queue_;

        if (!primary.empty()) {
          task = std::move(primary.front());
          primary.pop();
        } else {
          task = std::move(secondary.front());
          secondary.pop();
        }
      }
      task();
    }
  }

  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> read_queue_;
  std::queue<std::function<void()>> write_queue_;
  std::mutex mu_;
  std::condition_variable cv_;
  bool stop_;
};

// ---------------------------------------------------------------------------
// Global pool (lazy-initialised)
// ---------------------------------------------------------------------------

static std::mutex g_pool_mu;
static std::unique_ptr<DualQueueThreadPool> g_pool;
static size_t g_n_read  = 0;  // 0 = use default
static size_t g_n_write = 0;

static std::pair<size_t, size_t> default_thread_counts() {
  return {32, 16};
}

static DualQueueThreadPool& pool() {
  std::lock_guard<std::mutex> lk(g_pool_mu);
  if (!g_pool) {
    size_t n_r = g_n_read, n_w = g_n_write;
    if (n_r == 0 && n_w == 0) {
      auto [dr, dw] = default_thread_counts();
      n_r = dr; n_w = dw;
    }
    g_pool = std::make_unique<DualQueueThreadPool>(n_r, n_w);
  }
  return *g_pool;
}

// ---------------------------------------------------------------------------
// Global job registry
// ---------------------------------------------------------------------------

static std::unordered_map<int64_t, std::shared_ptr<JobState>> g_jobs;
static std::mutex g_jobs_mu;

// ---------------------------------------------------------------------------
// Error-path helpers
// ---------------------------------------------------------------------------

// For write errors where the FD is still open (pwrite / close failure).
static bool fail_write(int fd, const std::string& tmp_path,
                        const char* op, int err) {
  std::cerr << "[kv_storage_ops] " << op << " failed for " << tmp_path
            << ": " << std::strerror(err) << "\n";
  if (fd >= 0) close(fd);
  unlink(tmp_path.c_str());
  return false;
}

// For read errors where the FD is still open.
static bool fail_read(int fd, const std::string& src_path,
                       const char* op, int err) {
  std::cerr << "[kv_storage_ops] " << op << " failed for " << src_path
            << ": " << std::strerror(err) << "\n";
  if (fd >= 0) close(fd);
  return false;
}

// ---------------------------------------------------------------------------
// Per-block I/O — called from worker threads (no GIL)
// ---------------------------------------------------------------------------

// Write one block from the source buffer into a single file.
// Handles directory creation, temp-file open, pwrite, and atomic rename.
static bool write_block(const uint8_t* src,
                        int64_t block_idx, int64_t block_size,
                        const std::string& dest_path) {
  // Create parent directory (idempotent; safe to call concurrently).
  fs::path parent = fs::path(dest_path).parent_path();
  if (!parent.empty()) {
    std::error_code ec;
    fs::create_directories(parent, ec);
    if (ec) {
      std::cerr << "[kv_storage_ops] create_directories failed for "
                << parent.string() << ": " << ec.message() << "\n";
      return false;
    }
  }

  // Each thread uses a unique random suffix so concurrent workers never
  // collide on the same temporary filename.
  thread_local std::string tmp_suffix =
      "_" + std::to_string(std::random_device{}()) + ".tmp";

  const std::string tmp_path = dest_path + tmp_suffix;
  int fd = open(tmp_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0644);
  if (fd < 0)
    return fail_write(-1, tmp_path, "open", errno);

  const uint8_t* ptr = src + block_idx * block_size;
  size_t written = 0;
  while (written < static_cast<size_t>(block_size)) {
    ssize_t n = pwrite(fd, ptr + written, block_size - written,
                       static_cast<int64_t>(written));
    if (n < 0) {
      if (errno == EINTR) continue;
      return fail_write(fd, tmp_path, "pwrite", errno);
    }
    if (n == 0)
      return fail_write(fd, tmp_path, "pwrite returned 0 unexpectedly", 0);
    written += n;
  }

  if (close(fd) != 0)
    return fail_write(-1, tmp_path, "close", errno);

  if (std::rename(tmp_path.c_str(), dest_path.c_str()) != 0)
    return fail_write(-1, tmp_path, "rename", errno);

  return true;
}

// Read one block from a file into the destination buffer.
static bool read_block(uint8_t* dst,
                       int64_t block_idx, int64_t block_size,
                       const std::string& src_path) {
  int fd = open(src_path.c_str(), O_RDONLY | O_DIRECT);
  if (fd < 0)
    return fail_read(-1, src_path, "open", errno);

  uint8_t* ptr = dst + block_idx * block_size;
  size_t total = 0;
  while (total < static_cast<size_t>(block_size)) {
    ssize_t n = pread(fd, ptr + total, block_size - total,
                      static_cast<int64_t>(total));
    if (n < 0) {
      if (errno == EINTR) continue;
      return fail_read(fd, src_path, "pread", errno);
    }
    if (n == 0) {
      std::cerr << "[kv_storage_ops] unexpected EOF in " << src_path
                << " after " << total << "/" << block_size << " bytes\n";
      close(fd);
      return false;
    }
    total += n;
  }

  close(fd);
  return true;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// Enqueue one write task per block.  Non-blocking: returns as soon as all
// tasks are enqueued.
//
// Args:
//   job_id        : unique identifier returned by get_finished_jobs().
//   buffer_list   : list of buffer-protocol objects (e.g. memoryviews) holding
//                   the CPU KV cache data.  The caller must keep them alive
//                   until get_finished_jobs() reports this job as done.
//   block_size    : bytes per block (== spec.block_stride_bytes).
//   block_indices : index of each block within the buffers.
//   dest_files    : destination file path for each block (one per block).
void submit_store_job(int64_t job_id,
                      py::buffer buf,
                      int64_t block_size,
                      std::vector<int64_t> block_indices,
                      std::vector<std::string> dest_files) {
  if (block_indices.size() != dest_files.size())
    throw std::invalid_argument(
        "block_indices and dest_files must have the same length");

  size_t n = block_indices.size();
  if (n == 0)
    throw std::invalid_argument("submit_store_job: no blocks to write");

  // Acquire buffer pointer and validate bounds while holding the GIL.
  auto info = buf.request(/*writable=*/false);
  const uint8_t* src_ptr = static_cast<const uint8_t*>(info.ptr);
  const int64_t buf_bytes = static_cast<int64_t>(info.size) *
                            static_cast<int64_t>(info.itemsize);

  for (size_t i = 0; i < n; ++i) {
    const int64_t start = block_indices[i] * block_size;
    const int64_t end   = start + block_size;
    if (start < 0 || end > buf_bytes)
      throw std::out_of_range(
          "submit_store_job: block_indices[" + std::to_string(i) +
          "] out of buffer range");
  }

  // Register the job state before enqueuing any tasks so that
  // get_finished_jobs() cannot observe the job before all tasks are in.
  auto state = std::make_shared<JobState>(static_cast<int>(n));
  {
    std::lock_guard<std::mutex> lk(g_jobs_mu);
    g_jobs[job_id] = state;
  }

  // Enqueue one write task per block. src_ptr is kept alive by the Python
  // caller (via _ActiveJob.buffer) for the lifetime of the job.
  auto& p = pool();
  for (size_t i = 0; i < n; ++i) {
    p.enqueue_write([src_ptr, block_idx = block_indices[i], block_size,
                     dest = dest_files[i], state]() {
      bool ok = write_block(src_ptr, block_idx, block_size, dest);
      state->task_done(ok);
    });
  }
}

// Enqueue one read task per block.  Non-blocking.
//
// Args:
//   job_id        : unique identifier returned by get_finished_jobs().
//   buffer_list   : writable buffer-protocol objects to read into.
//                   The caller must keep them alive until the job is done.
//   block_size    : bytes per block.
//   block_indices : index of each block within the buffers.
//   source_files  : source file path for each block.
void submit_load_job(int64_t job_id,
                     py::buffer buf,
                     int64_t block_size,
                     std::vector<int64_t> block_indices,
                     std::vector<std::string> source_files) {
  if (block_indices.size() != source_files.size())
    throw std::invalid_argument(
        "block_indices and source_files must have the same length");

  size_t n = block_indices.size();
  if (n == 0)
    throw std::invalid_argument("submit_load_job: no blocks to read");

  // Acquire buffer pointer and validate bounds while holding the GIL.
  auto info = buf.request(/*writable=*/true);
  uint8_t* dst_ptr = static_cast<uint8_t*>(info.ptr);
  const int64_t buf_bytes = static_cast<int64_t>(info.size) *
                            static_cast<int64_t>(info.itemsize);

  for (size_t i = 0; i < n; ++i) {
    const int64_t start = block_indices[i] * block_size;
    const int64_t end   = start + block_size;
    if (start < 0 || end > buf_bytes)
      throw std::out_of_range(
          "submit_load_job: block_indices[" + std::to_string(i) +
          "] out of buffer range");
  }

  auto state = std::make_shared<JobState>(static_cast<int>(n));
  {
    std::lock_guard<std::mutex> lk(g_jobs_mu);
    g_jobs[job_id] = state;
  }

  // dst_ptr is kept alive by the Python caller (via _ActiveJob.buffer).
  auto& p = pool();
  for (size_t i = 0; i < n; ++i) {
    p.enqueue_read([dst_ptr, block_idx = block_indices[i], block_size,
                    src = source_files[i], state]() {
      bool ok = read_block(dst_ptr, block_idx, block_size, src);
      state->task_done(ok);
    });
  }
}

// Poll for completed jobs.
//
// Returns a list of (job_id, success) tuples for every job whose last task
// has finished since the previous call.  Completed jobs are removed from the
// internal registry.  Always call from the Python scheduler thread.
py::list get_finished_jobs() {
  py::list out;
  std::lock_guard<std::mutex> lk(g_jobs_mu);
  for (auto it = g_jobs.begin(); it != g_jobs.end(); ) {
    if (it->second->is_done()) {
      out.append(py::make_tuple(
          it->first,
          it->second->all_success.load(std::memory_order_relaxed)));
      it = g_jobs.erase(it);
    } else {
      ++it;
    }
  }
  return out;
}

// Set the number of I/O worker threads by replacing the global pool.
//
// WARNING: must only be called before any submit_store_job / submit_load_job
// calls are in flight.  Replacing the pool while tasks from the old pool are
// outstanding will join (and block on) all outstanding workers in the
// destructor, and any JobState shared_ptrs held by those workers will keep
// the state alive until all tasks finish.  Safe usage: call once at startup.
void set_thread_count(size_t n_read, size_t n_write) {
  if (n_read == 0 && n_write == 0)
    throw std::invalid_argument(
        "set_thread_count: at least one thread required");
  std::lock_guard<std::mutex> lk(g_pool_mu);
  g_n_read  = n_read;
  g_n_write = n_write;
  g_pool = std::make_unique<DualQueueThreadPool>(n_read, n_write);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "KV-cache block storage I/O (pread/pwrite, dual-queue thread pool)";

  m.def("submit_store_job", &submit_store_job,
        "Enqueue write tasks for a KV store job (one task per block file).\n"
        "Non-blocking. Call get_finished_jobs() to poll for completion.\n"
        "The caller must keep buf alive until the job is reported done.",
        py::arg("job_id"), py::arg("buf"), py::arg("block_size"),
        py::arg("block_indices"), py::arg("dest_files"));

  m.def("submit_load_job", &submit_load_job,
        "Enqueue read tasks for a KV load job (one task per block file).\n"
        "Non-blocking. Call get_finished_jobs() to poll for completion.\n"
        "The caller must keep buf alive until the job is reported done.",
        py::arg("job_id"), py::arg("buf"), py::arg("block_size"),
        py::arg("block_indices"), py::arg("source_files"));

  m.def("get_finished_jobs", &get_finished_jobs,
        "Return list of (job_id, success) for all completed jobs since the\n"
        "last call.  Removes completed jobs from the internal registry.");

  m.def("set_thread_count", &set_thread_count,
        "Set the number of read-priority and write-priority I/O threads.\n"
        "Recreates the global pool — call once at startup only.",
        py::arg("n_read"), py::arg("n_write"));
}
