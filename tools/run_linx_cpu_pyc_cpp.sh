#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=../scripts/lib.sh
source "${ROOT_DIR}/scripts/lib.sh"
pyc_find_pyc_compile

MEMH=""
ELF=""
EXPECTED=""
ELF_TEXT_BASE="0x10000"
ELF_DATA_BASE="0x20000"
ELF_PAGE_ALIGN="0x1000"
TB_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --memh)
      MEMH="${2:?missing value for --memh}"
      shift 2
      ;;
    --elf)
      ELF="${2:?missing value for --elf}"
      shift 2
      ;;
    --expected)
      EXPECTED="${2:?missing value for --expected}"
      shift 2
      ;;
    --base)
      ELF_TEXT_BASE="${2:?missing value for --base}"
      shift 2
      ;;
    --text-base)
      ELF_TEXT_BASE="${2:?missing value for --text-base}"
      shift 2
      ;;
    --data-base)
      ELF_DATA_BASE="${2:?missing value for --data-base}"
      shift 2
      ;;
    --page-align)
      ELF_PAGE_ALIGN="${2:?missing value for --page-align}"
      shift 2
      ;;
    -p)
      TB_ARGS+=("-p" "${2:?missing value for -p}")
      shift 2
      ;;
    -p1)
      TB_ARGS+=("-p1")
      shift 1
      ;;
    --pipeview|--pipfile|--swimlane|--swimfile|--boot-pc)
      TB_ARGS+=("$1" "${2:?missing value for $1}")
      shift 2
      ;;
    -h|--help)
      cat <<EOF
Usage:
  $0                     # run built-in regression memh tests
  $0 --memh <file> [--expected <hex>]   # run one memh program
  $0 --elf  <file> [--expected <hex>]   # convert ELF -> memh (apply relocs, load .data/.bss) and run

ELF options:
  --base <addr>       Alias for --text-base (default: 0x10000)
  --text-base <addr>  (default: 0x10000)
  --data-base <addr>  (default: 0x20000)
  --page-align <addr> (default: 0x1000)

Trace export options (forwarded to the C++ TB):
  -p1                  Enable Konata pipeview (O3PipeView)
  --pipeview <0|1>      Enable/disable pipeview
  --pipfile <path>      Pipeview output path (default: pipeview.log)
  --swimlane <0|1>      Enable/disable Perfetto swimlane trace
  --swimfile <path>     Swimlane output path (default: swimlane.trace.json)
  --boot-pc <addr>      Override boot PC for the TB (also supports PYC_BOOT_PC env)
EOF
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

WORK_DIR="$(mktemp -d -t linx_cpu_pyc.XXXXXX)"
trap 'rm -rf "${WORK_DIR}"' EXIT

cd "${ROOT_DIR}"

if [[ -n "${ELF}" ]]; then
  MEMH="${WORK_DIR}/program.memh"
  START_PC="$(PYTHONDONTWRITEBYTECODE=1 python3 tools/linxisa/elf_to_memh.py "${ELF}" --text-base "${ELF_TEXT_BASE}" --data-base "${ELF_DATA_BASE}" --page-align "${ELF_PAGE_ALIGN}" -o "${MEMH}" --print-start)"
  if [[ -z "${PYC_BOOT_PC:-}" ]]; then
    export PYC_BOOT_PC="${START_PC}"
  fi
fi

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$(pyc_pythonpath)" \
  python3 -m pycircuit.cli emit examples/linx_cpu_pyc/linx_cpu_pyc.py -o "${WORK_DIR}/linx_cpu_pyc.pyc"

"${PYC_COMPILE}" "${WORK_DIR}/linx_cpu_pyc.pyc" --emit=cpp -o "${WORK_DIR}/linx_cpu_pyc_gen.hpp"

CXX="${CXX:-clang++}"
if ! command -v "${CXX}" >/dev/null 2>&1; then
  if command -v g++ >/dev/null 2>&1; then
    CXX="g++"
  else
    echo "error: missing C++ compiler (clang++ or g++)" >&2
    exit 1
  fi
fi

"${CXX}" -std=c++17 -O2 \
  -I "${ROOT_DIR}/include" \
  -I "${WORK_DIR}" \
  -o "${WORK_DIR}/tb_linx_cpu_pyc" \
  "${ROOT_DIR}/examples/linx_cpu_pyc/tb_linx_cpu_pyc.cpp"

if [[ -n "${MEMH}" ]]; then
  if [[ -n "${EXPECTED}" ]]; then
    "${WORK_DIR}/tb_linx_cpu_pyc" "${TB_ARGS[@]}" "${MEMH}" "${EXPECTED}"
  else
    "${WORK_DIR}/tb_linx_cpu_pyc" "${TB_ARGS[@]}" "${MEMH}"
  fi
else
  "${WORK_DIR}/tb_linx_cpu_pyc" "${TB_ARGS[@]}"
fi
