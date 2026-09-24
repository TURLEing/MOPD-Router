#!/usr/bin/env bash
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Build the MOPD-Router data pools end-to-end.
#
# Produces:
#   ${ROOT}/numina_tir/train.parquet          (natural math x code composite)
#   ${ROOT}/rlvr/math.parquet                 (intermediate: plain math, feeds step 3)
#   ${ROOT}/rlvr/ifeval.parquet               (intermediate: IF constraints, feeds step 3)
#   ${ROOT}/math_if_composite/train.parquet   (synthetic math x IF composite)
#   ${ROOT}/train_pool/train.parquet          (final unlabeled COMPOSITE-ONLY training
#                                              mix: numina_tir + math_if_composite)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(python3 -c 'import os,sys; print(os.path.expanduser("~/data/mopd_router"))')}"
PYTHON="${PYTHON:-python3}"
MAX_CHARS="${MAX_CHARS:-6000}"
# Per-source cap. Composite-only training pool: numina_tir (math x code) +
# math_if_composite (math x IF). numina_tir (~65k after filtering) is capped to
# ~50k so the pool lands at ~65k rows (50k + 15k). Single-domain slices
# (rlvr/math.parquet, rlvr/ifeval.parquet) are still produced as intermediate
# inputs for math_if_composite synthesis but are NOT merged into train_pool:
# on single-domain prompts the token-router has nothing to switch between,
# and including them would duplicate the math questions already wrapped in
# math_if_composite.
CAP="${CAP:-50000}"

echo "ROOT=${ROOT}"

# 1. NuminaMath-TIR -> composite math x code pool.
"${PYTHON}" "${SCRIPT_DIR}/numina_tir.py" \
    --local_save_dir "${ROOT}/numina_tir" \
    --max_problem_chars "${MAX_CHARS}"

# 2. RLVR-GSM-MATH-IF -> single-domain math + ifeval subsets.
"${PYTHON}" "${SCRIPT_DIR}/rlvr_math_if.py" \
    --local_save_dir "${ROOT}/rlvr" \
    --max_prompt_chars "${MAX_CHARS}"

# 3. Synthesize math x IF composite from the math subset + sampled IF constraints.
"${PYTHON}" "${SCRIPT_DIR}/build_math_if_composite.py" \
    --math_parquet "${ROOT}/rlvr/math.parquet" \
    --ifeval_parquet "${ROOT}/rlvr/ifeval.parquet" \
    --local_save_dir "${ROOT}/math_if_composite" \
    --max_prompt_chars "${MAX_CHARS}"

# 4. Final unlabeled COMPOSITE-ONLY training pool: merge numina_tir
#    (math x code) + math_if_composite (math x IF) and shuffle. No domain
#    labels are written -- the token-level router owns teacher selection at
#    training time (domain-unclear OPD setting).
"${PYTHON}" "${SCRIPT_DIR}/merge_pool.py" \
    --inputs "${ROOT}/numina_tir/train.parquet" "${ROOT}/math_if_composite/train.parquet" \
    --cap_per_source "${CAP}" \
    --local_save_dir "${ROOT}/train_pool" \
    --shuffle

echo "Done. Manifest:"
cat "${ROOT}/train_pool/manifest.json"
