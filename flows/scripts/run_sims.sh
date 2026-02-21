#!/usr/bin/env bash
set -euo pipefail

# Strict simulation gate:
# - all normal-tier folderized examples via both backends (C++ + Verilator)
# - existing non-example testbench designs via Verilator

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
pyc_find_pycc

PYTHONPATH_VAL="$(pyc_pythonpath)"
OUT_BASE="$(pyc_out_root)/sim"
DISCOVER="${PYC_ROOT_DIR}/flows/tools/discover_examples.py"
mkdir -p "${OUT_BASE}"

pyc_log "using pycc: ${PYCC}"

run_case_examples() {
  local name="$1"
  local tb_src="$2"
  local out_dir="${OUT_BASE}/${name}"
  rm -rf "${out_dir}" >/dev/null 2>&1 || true
  mkdir -p "${out_dir}"
  pyc_log "sim(example) ${name}: ${tb_src}"
  PYTHONPATH="${PYTHONPATH_VAL}" PYTHONDONTWRITEBYTECODE=1 PYCC="${PYCC}" \
    python3 -m pycircuit.cli build \
      "${tb_src}" \
      --out-dir "${out_dir}" \
      --target both \
      --jobs "${PYC_SIM_JOBS:-4}" \
      --logic-depth "${PYC_SIM_LOGIC_DEPTH:-256}" \
      --run-verilator

  local cpp_bin
  cpp_bin="$(
    python3 - "${out_dir}/project_manifest.json" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    manifest = json.load(f)
print(manifest.get("cpp_executable", ""))
PY
  )"
  if [[ -z "${cpp_bin}" || ! -x "${cpp_bin}" ]]; then
    pyc_die "missing or non-executable cpp_executable for ${name}: ${cpp_bin}"
  fi
  pyc_log "run(cpp) ${name}: ${cpp_bin}"
  (cd "${out_dir}" && "${cpp_bin}")
}

while IFS=$'\t' read -r name _design tb _cfg _tier; do
  [[ -n "${name}" ]] || continue
  run_case_examples "example_${name}" "${tb}"
done < <(python3 "${DISCOVER}" --root "${PYC_ROOT_DIR}/designs/examples" --tier normal --format tsv)

run_case_nonexample() {
  local name="$1"
  local src="$2"
  local out_dir="${OUT_BASE}/${name}"
  rm -rf "${out_dir}" >/dev/null 2>&1 || true
  mkdir -p "${out_dir}"
  pyc_log "sim(non-example) ${name}: ${src}"
  PYTHONPATH="${PYTHONPATH_VAL}" PYTHONDONTWRITEBYTECODE=1 PYCC="${PYCC}" \
    python3 -m pycircuit.cli build \
      "${src}" \
      --out-dir "${out_dir}" \
      --target verilator \
      --jobs "${PYC_SIM_JOBS:-4}" \
      --logic-depth "${PYC_SIM_LOGIC_DEPTH:-256}" \
      --run-verilator
}

run_case_nonexample issq "${PYC_ROOT_DIR}/designs/IssueQueue/tb_issq.py"
run_case_nonexample regfile "${PYC_ROOT_DIR}/designs/RegisterFile/tb_regfile.py"
run_case_nonexample bypass_unit "${PYC_ROOT_DIR}/designs/BypassUnit/tb_bypass_unit.py"

pyc_log "all sims passed"
