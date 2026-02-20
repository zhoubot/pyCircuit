#!/usr/bin/env bash
set -euo pipefail

# Run all repo examples through emit + pycc (cpp) to sanity-check the
# compiler/codegen pipeline.

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
pyc_find_pycc

PYTHONPATH_VAL="$(pyc_pythonpath)"
EX_DIR="${PYC_ROOT_DIR}/designs/examples"

if [[ ! -d "${EX_DIR}" ]]; then
  pyc_die "examples dir not found: ${EX_DIR}"
fi

pyc_log "using pycc: ${PYCC}"

pyc_log "running strict API hygiene gate"
python3 "${PYC_ROOT_DIR}/flows/tools/check_api_hygiene.py" \
  compiler/frontend/pycircuit \
  designs/examples \
  docs \
  README.md

fail=0
count=0

for ex in "${EX_DIR}"/*.py; do
  [[ -f "${ex}" ]] || continue
  bn="$(basename "${ex}" .py)"

  # Skip package init modules; examples should be runnable design entrypoints.
  if [[ "${bn}" == "__init__" ]]; then
    continue
  fi

  count=$((count+1))
  out_root="$(pyc_out_root)/example-smoke/${bn}"
  rm -rf "${out_root}" >/dev/null 2>&1 || true
  mkdir -p "${out_root}"
  pyc_file="${out_root}/${bn}.pyc"
  cpp_dir="${out_root}/cpp"

  pyc_log "[${count}] emit ${bn}"
  if ! PYTHONPATH="${PYTHONPATH_VAL}" python3 -m pycircuit.cli emit "${ex}" -o "${pyc_file}"; then
    pyc_warn "emit failed: ${bn}"
    fail=1
    continue
  fi

  pyc_log "[${count}] compile(cpp) ${bn}"
  if ! "${PYCC}" "${pyc_file}" \
      --emit=cpp \
      --cpp-split=module \
      --out-dir="${cpp_dir}" \
      --logic-depth=64; then
    pyc_warn "pycc failed: ${bn}"
    fail=1
    continue
  fi

  # Basic artifact check.
  if [[ ! -f "${cpp_dir}/cpp_compile_manifest.json" ]]; then
    pyc_warn "missing cpp_compile_manifest.json: ${bn}"
    fail=1
  fi

done

# Project build flow smoke checks (multi-.pyc + parallel pycc + CMake/Ninja).
for bex in counter_tb huge_hierarchy_stress; do
  ex="${EX_DIR}/${bex}.py"
  [[ -f "${ex}" ]] || continue
  count=$((count+1))
  out_root="$(pyc_out_root)/example-build-smoke/${bex}"
  rm -rf "${out_root}" >/dev/null 2>&1 || true
  mkdir -p "${out_root}"
  pyc_log "[${count}] build ${bex}"
  if ! PYTHONPATH="${PYTHONPATH_VAL}" python3 -m pycircuit.cli build \
      "${ex}" \
      --out-dir "${out_root}" \
      --target cpp \
      --jobs "${PYC_EXAMPLE_JOBS:-4}" \
      --logic-depth 64; then
    pyc_warn "build failed: ${bex}"
    fail=1
    continue
  fi
  if [[ ! -f "${out_root}/project_manifest.json" ]]; then
    pyc_warn "missing project_manifest.json: ${bex}"
    fail=1
  fi
done

if [[ "${fail}" -ne 0 ]]; then
  pyc_die "one or more examples failed"
fi

pyc_log "all examples passed"
