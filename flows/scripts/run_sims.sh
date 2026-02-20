#!/usr/bin/env bash
set -euo pipefail

# Build and run a small set of `@testbench` designs via Verilator.

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
pyc_find_pycc

PYTHONPATH_VAL="$(pyc_pythonpath)"
OUT_BASE="$(pyc_out_root)/sim"
mkdir -p "${OUT_BASE}"

pyc_log "using pycc: ${PYCC}"

run_case() {
  local name="$1"
  local src="$2"
  local out_dir="${OUT_BASE}/${name}"
  rm -rf "${out_dir}" >/dev/null 2>&1 || true
  mkdir -p "${out_dir}"
  pyc_log "sim ${name}: ${src}"
  PYTHONPATH="${PYTHONPATH_VAL}" PYTHONDONTWRITEBYTECODE=1 PYCC="${PYCC}" \
    python3 -m pycircuit.cli build \
      "${src}" \
      --out-dir "${out_dir}" \
      --target verilator \
      --jobs "${PYC_SIM_JOBS:-4}" \
      --logic-depth "${PYC_SIM_LOGIC_DEPTH:-256}" \
      --run-verilator
}

run_case counter_tb "${PYC_ROOT_DIR}/designs/examples/counter_tb.py"
run_case issq "${PYC_ROOT_DIR}/designs/IssueQueue/tb_issq.py"
run_case regfile "${PYC_ROOT_DIR}/designs/RegisterFile/tb_regfile.py"
run_case bypass_unit "${PYC_ROOT_DIR}/designs/BypassUnit/tb_bypass_unit.py"

pyc_log "all sims passed"

