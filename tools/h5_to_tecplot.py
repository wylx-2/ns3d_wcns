#!/usr/bin/env python3
"""
Convert NS3D HDF5 field output to Tecplot ASCII (.dat).

Expected HDF5 datasets (3D, shape = [K, J, I]):
X, Y, Z, rho, u, v, w, E, p, T
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert HDF5 field file to Tecplot ASCII format."
    )
    parser.add_argument("input_h5", help="Input HDF5 file path, e.g. output/time_0/field.h5")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output Tecplot .dat path (default: same name as input with .dat suffix)",
    )
    parser.add_argument(
        "--vars",
        nargs="+",
        default=["X", "Y", "Z", "rho", "u", "v", "w", "E", "p", "T"],
        help="Dataset names to export in order",
    )
    parser.add_argument(
        "--zone-title",
        default="field",
        help="Tecplot zone title",
    )
    parser.add_argument(
        "--title",
        default="NS3D Field",
        help="Tecplot file title",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=8,
        help="Scientific notation precision (default: 8)",
    )
    return parser.parse_args()


def _validate_vars(h5f: h5py.File, var_names: List[str]) -> None:
    missing = [name for name in var_names if name not in h5f]
    if missing:
        raise KeyError(f"Missing datasets in HDF5 file: {missing}")



def _load_and_validate_arrays(h5f: h5py.File, var_names: List[str]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    ref_shape = None

    for name in var_names:
        arr = np.asarray(h5f[name])
        if arr.ndim != 3:
            raise ValueError(f"Dataset '{name}' must be 3D, got shape {arr.shape}")
        if ref_shape is None:
            ref_shape = arr.shape
        elif arr.shape != ref_shape:
            raise ValueError(
                f"Dataset '{name}' shape {arr.shape} does not match reference shape {ref_shape}"
            )
        arrays[name] = arr

    return arrays



def write_tecplot_dat(
    input_h5: Path,
    output_dat: Path,
    var_names: List[str],
    title: str,
    zone_title: str,
    precision: int,
) -> None:
    with h5py.File(input_h5, "r") as h5f:
        _validate_vars(h5f, var_names)
        arrays = _load_and_validate_arrays(h5f, var_names)

    nz, ny, nx = arrays[var_names[0]].shape
    fmt = f"{{:.{precision}e}}"

    output_dat.parent.mkdir(parents=True, exist_ok=True)
    with output_dat.open("w", encoding="utf-8") as f:
        f.write(f'TITLE = "{title}"\n')
        vars_text = " ".join([f'\"{v}\"' for v in var_names])
        f.write(f"VARIABLES = {vars_text}\n")
        f.write(
            f'ZONE T="{zone_title}" I={nx} J={ny} K={nz} DATAPACKING=POINT\n'
        )

        # Tecplot POINT order: i fast, then j, then k.
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    values = [arrays[name][k, j, i] for name in var_names]
                    line = " ".join(fmt.format(float(v)) for v in values)
                    f.write(line + "\n")


def main() -> None:
    args = parse_args()

    input_h5 = Path(args.input_h5)
    if not input_h5.is_file():
        raise FileNotFoundError(f"Input HDF5 file not found: {input_h5}")

    if args.output is None:
        output_dat = input_h5.with_suffix(".dat")
    else:
        output_dat = Path(args.output)

    write_tecplot_dat(
        input_h5=input_h5,
        output_dat=output_dat,
        var_names=args.vars,
        title=args.title,
        zone_title=args.zone_title,
        precision=args.precision,
    )

    print(f"Wrote Tecplot file: {output_dat}")


if __name__ == "__main__":
    main()
