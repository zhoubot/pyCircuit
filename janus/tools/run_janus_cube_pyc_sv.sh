#!/usr/bin/env bash
# Run Verilog simulation for janus_cube_pyc using Icarus Verilog
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../../scripts/lib.sh
source "${ROOT_DIR}/scripts/lib.sh"
pyc_find_pyc_compile

# Check for Icarus Verilog
IVERILOG="${IVERILOG:-$(command -v iverilog || true)}"
VVP="${VVP:-$(command -v vvp || true)}"
if [[ -z "${IVERILOG}" ]]; then
  echo "error: missing iverilog (install with: brew install icarus-verilog)" >&2
  exit 1
fi
if [[ -z "${VVP}" ]]; then
  echo "error: missing vvp (install with: brew install icarus-verilog)" >&2
  exit 1
fi

# Paths
GEN_DIR="${ROOT_DIR}/janus/generated/janus_cube_pyc"
VLOG="${GEN_DIR}/janus_cube_pyc.v"
TB_SV="${ROOT_DIR}/janus/tb/tb_janus_cube_pyc.sv"

# Regenerate if needed
if [[ ! -f "${VLOG}" ]]; then
  echo "[cube-sv] Generating Verilog..."
  bash "${ROOT_DIR}/janus/update_generated.sh"
fi

# Create work directory
WORK_DIR="$(mktemp -d -t janus_cube_pyc_sv.XXXXXX)"
trap 'rm -rf "${WORK_DIR}"' EXIT

echo "[cube-sv] Compiling with iverilog..."
"${IVERILOG}" -g2012 \
  -I "${ROOT_DIR}/include/pyc/verilog" \
  -o "${WORK_DIR}/tb_janus_cube_pyc.vvp" \
  "${TB_SV}" \
  "${VLOG}"

echo "[cube-sv] Running simulation..."
"${VVP}" "${WORK_DIR}/tb_janus_cube_pyc.vvp" "$@"
