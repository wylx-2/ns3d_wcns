#!/usr/bin/env bash
set -euo pipefail

# Run the solver sequentially with different Riemann_Solver values.
# The script edits solver.in in place, runs the executable, stores outputs per case,
# and restores solver.in when finished.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOLVER_IN="${ROOT_DIR}/solver.in"
BACKUP_IN="${ROOT_DIR}/solver.in.bak.riemann_batch"

# Allow users to override run command, e.g.:
RUN_CMD="mpirun -np 8 ./build/ns3d_wcns" #./tools/run_riemann_batch.sh
if [[ -n "${RUN_CMD:-}" ]]; then
  CMD="${RUN_CMD}"
elif [[ -x "${ROOT_DIR}/build/ns3d_wcns" ]]; then
  CMD="./build/ns3d_wcns"
elif [[ -x "${ROOT_DIR}/ns3d_wcns" ]]; then
  CMD="./ns3d_wcns"
else
  echo "Error: cannot find executable. Set RUN_CMD or build the project first." >&2
  exit 1
fi

SOLVERS=(Roe HLL HLLC HLLC_p Rusanov)

restore_solver_in() {
  if [[ -f "${BACKUP_IN}" ]]; then
    mv -f "${BACKUP_IN}" "${SOLVER_IN}"
  fi
}

trap restore_solver_in EXIT

if [[ ! -f "${SOLVER_IN}" ]]; then
  echo "Error: ${SOLVER_IN} not found." >&2
  exit 1
fi

cp "${SOLVER_IN}" "${BACKUP_IN}"

cd "${ROOT_DIR}"

for solver in "${SOLVERS[@]}"; do
  echo "============================================================"
  echo "Running case with Riemann_Solver=${solver}"

  sed -i -E "s|^([[:space:]]*Riemann_Solver[[:space:]]*=[[:space:]]*).*$|\\1${solver}|" "${SOLVER_IN}"

  # Keep each case's logs and outputs isolated.
  rm -rf output
  mkdir -p output

  set +e
  bash -lc "${CMD}" > "run_${solver}.log" 2>&1
  status=$?
  set -e

  if [[ ${status} -ne 0 ]]; then
    echo "Case ${solver} failed. See run_${solver}.log" >&2
    exit ${status}
  fi

  case_out_dir="output_${solver}"
  rm -rf "${case_out_dir}"
  mv output "${case_out_dir}"
  echo "Finished ${solver}, output saved to ${case_out_dir}, log: run_${solver}.log"
done

echo "============================================================"
echo "All Riemann_Solver cases finished successfully."
