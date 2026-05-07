# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
JIT-compiles the _kv_file_system_ops C++ extension and installs it into the
vllm package directory so that `import vllm._kv_file_system_ops` works.

Run this after `VLLM_USE_PRECOMPILED=1 pip install -e .` to add the one
extension that has no precompiled binary.
"""
import os
import shutil

from torch.utils.cpp_extension import load

ext = load(
    name="_kv_file_system_ops",
    sources=[os.path.join("vllm","csrc", "kv_file_system_ops.cpp")],
    extra_cflags=["-O2", "-std=c++17"],
    verbose=True,
)

import vllm  # noqa: E402 — must import after build
dst = os.path.join(os.path.dirname(vllm.__file__), os.path.basename(ext.__file__))
shutil.copy2(ext.__file__, dst)
print(f"Installed: {dst}")