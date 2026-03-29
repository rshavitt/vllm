# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
FileSystem tier benchmarks.

Measures the performance of CPU→FileSystem KV cache offloading versus a
CPU-only baseline.  Each benchmark function in this file targets a specific
access pattern; the results are printed as aligned tables for easy comparison.

Usage:
    python benchmarks/kv_offload/benchmark_filesystem_tier.py \\
        --model meta-llama/Llama-3.2-1B-Instruct \\
        --num-prompts-per-len 20 \\
        --output-len 16 \\
        --cpu-bytes $((512 * 1024 * 1024)) \\
        --fs-path /tmp/vllm_kv_bench \\
        --scenarios all

Benchmarks
----------
bench_cache_miss
    All prompts are unique (no shared prefix).  Every request is a cache
    miss, stressing the write path of the FS tier.
"""

import argparse
import random
import time

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

try:
    from vllm.tokenizers import get_tokenizer
except ImportError:
    from transformers import AutoTokenizer

    def get_tokenizer(model, **kwargs):
        return AutoTokenizer.from_pretrained(model, **kwargs)


# Input lengths swept by each benchmark.
INPUT_LENGTHS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

SCENARIOS = ["baseline", "cpu+fs"]


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------

def make_prompts(
    input_len: int,
    num: int,
    tokenizer,
    seed: int,
) -> list[str]:
    """Return `num` unique prompts, each exactly `input_len` tokens long.

    Tokens are sampled randomly from the non-special vocabulary, so prompts
    share no prefix and produce guaranteed cache misses.
    """
    rng = random.Random(seed)
    vocab = tokenizer.get_vocab()
    all_special_ids = set(tokenizer.all_special_ids)
    valid_ids = [v for v in vocab.values() if v not in all_special_ids]

    prompts = []
    for _ in range(num):
        token_ids = rng.choices(valid_ids, k=input_len)
        prompts.append(tokenizer.decode(token_ids))
    return prompts


# ---------------------------------------------------------------------------
# Connector config helpers
# ---------------------------------------------------------------------------

def build_kv_config(
    scenario: str,
    cpu_bytes: int,
    fs_path: str,
    max_fs_blocks: int,
) -> KVTransferConfig:
    if scenario == "baseline":
        extra: dict = {"cpu_bytes_to_use": cpu_bytes}
    elif scenario == "cpu+fs":
        extra = {
            "cpu_bytes_to_use": cpu_bytes,
            "spec_name": "TiersOffloadingSpec",
            "secondary_tiers": [
                {
                    "type": "storage",
                    "tier_name": "FS",
                    "base_path": fs_path,
                    "max_blocks": max_fs_blocks,
                }
            ],
        }
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config=extra,
    )


# ---------------------------------------------------------------------------
# Single-batch runner
# ---------------------------------------------------------------------------

def run_batch(
    llm: LLM,
    prompts: list[str],
    output_len: int,
) -> tuple[float, int]:
    """Run llm.generate() and return (wall_time_seconds, total_output_tokens)."""
    sampling_params = SamplingParams(temperature=0, max_tokens=output_len)

    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    elapsed = time.perf_counter() - start

    total_output_tokens = sum(
        len(o.token_ids) for out in outputs for o in out.outputs
    )
    return elapsed, total_output_tokens


# ---------------------------------------------------------------------------
# Benchmark: cache miss
# ---------------------------------------------------------------------------

def bench_cache_miss(args: argparse.Namespace, tokenizer) -> None:
    """
    All prompts are unique — every request is a cache miss.
    Sweeps INPUT_LENGTHS and compares scenarios side by side.
    """
    print("\n=== Cache Miss Benchmark ===")
    print(f"Prompts per length: {args.num_prompts_per_len}  |  "
          f"Output tokens: {args.output_len}  |  "
          f"Scenarios: {args.scenarios}\n")

    scenarios = SCENARIOS if "all" in args.scenarios else args.scenarios

    # results[input_len][scenario] = (time_s, tok_s)
    results: dict[int, dict[str, tuple[float, float]]] = {}

    for input_len in INPUT_LENGTHS:
        prompts = make_prompts(
            input_len, args.num_prompts_per_len, tokenizer, args.seed
        )
        results[input_len] = {}

        for scenario in scenarios:
            kv_cfg = build_kv_config(
                scenario, args.cpu_bytes, args.fs_path, args.max_fs_blocks
            )
            llm = LLM(
                model=args.model,
                gpu_memory_utilization=args.gpu_memory_utilization,
                kv_transfer_config=kv_cfg,
                enable_prefix_caching=True,
            )

            elapsed, out_toks = run_batch(llm, prompts, args.output_len)
            tok_per_s = out_toks / elapsed if elapsed > 0 else 0.0

            print(
                f"  input_len={input_len:4d}  scenario={scenario:<10s}"
                f"  time={elapsed:6.2f}s  tok/s={tok_per_s:8.1f}"
            )
            results[input_len][scenario] = (elapsed, tok_per_s)

            del llm  # release GPU memory before the next scenario

    # Print summary table
    col_w = 18
    header_parts = [f"{'Input len':>9}"]
    for s in scenarios:
        header_parts.append(f"{s + ' time(s)':>{col_w}}")
        header_parts.append(f"{s + ' tok/s':>{col_w}}")
    header = " | ".join(header_parts)
    separator = "-" * len(header)

    print(f"\n{separator}")
    print(header)
    print(separator)
    for input_len in INPUT_LENGTHS:
        row_parts = [f"{input_len:>9}"]
        for s in scenarios:
            if s in results[input_len]:
                t, tok_s = results[input_len][s]
                row_parts.append(f"{t:>{col_w}.2f}")
                row_parts.append(f"{tok_s:>{col_w}.1f}")
            else:
                row_parts.append(f"{'N/A':>{col_w}}")
                row_parts.append(f"{'N/A':>{col_w}}")
        print(" | ".join(row_parts))
    print(separator)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FileSystem tier benchmarks for vLLM KV cache offloading."
    )
    parser.add_argument("--model", required=True,
                        help="HuggingFace model name or path.")
    parser.add_argument("--num-prompts-per-len", type=int, default=20,
                        help="Number of prompts generated for each input length.")
    parser.add_argument("--output-len", type=int, default=16,
                        help="Number of output tokens per request.")
    parser.add_argument("--cpu-bytes", type=int, default=512 << 20,
                        help="CPU memory to allocate for the primary tier (bytes).")
    parser.add_argument("--fs-path", default="/tmp/vllm_kv_bench",
                        help="Base directory for the FileSystem tier.")
    parser.add_argument("--max-fs-blocks", type=int, default=10000,
                        help="Maximum number of blocks in the FileSystem tier.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=SCENARIOS + ["all"],
        default=["all"],
        help="Scenarios to run. Use 'all' for every scenario.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = get_tokenizer(args.model, trust_remote_code=True)
    bench_cache_miss(args, tokenizer)


if __name__ == "__main__":
    main()
