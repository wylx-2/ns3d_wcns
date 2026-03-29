#!/usr/bin/env python3
"""将结构网格 HDF5 转换为 Tecplot 可读 ASCII 数据文件。

输入要求：
- HDF5 中包含三维坐标数据集 x/y/z（或 X/Y/Z）
- 三个数据集 shape 一致，布局为 [nz, ny, nx]

输出：
- Tecplot ASCII .dat，DATAPACKING=POINT，变量为 X Y Z
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert structured grid HDF5 to Tecplot ASCII .dat"
    )
    parser.add_argument("input_h5", help="Input structured grid HDF5 path")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output Tecplot .dat path (default: same name as input)",
    )
    parser.add_argument("--title", default="Structured Grid", help="Tecplot file title")
    parser.add_argument("--zone-title", default="grid", help="Tecplot zone title")
    parser.add_argument(
        "--precision",
        type=int,
        default=10,
        help="Scientific notation precision for output values",
    )
    return parser.parse_args()


def _pick_coord_names(h5f: h5py.File) -> Tuple[str, str, str]:
    lower = ("x", "y", "z")
    upper = ("X", "Y", "Z")

    if all(name in h5f for name in lower):
        return lower
    if all(name in h5f for name in upper):
        return upper

    keys = sorted(list(h5f.keys()))
    raise KeyError(
        "Cannot find coordinate datasets x/y/z or X/Y/Z in HDF5 file. "
        f"Available datasets: {keys}"
    )


def _load_coords(h5f: h5py.File) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx_name, ny_name, nz_name = _pick_coord_names(h5f)
    x = np.asarray(h5f[nx_name], dtype=np.float64)
    y = np.asarray(h5f[ny_name], dtype=np.float64)
    z = np.asarray(h5f[nz_name], dtype=np.float64)

    if x.ndim != 3 or y.ndim != 3 or z.ndim != 3:
        raise ValueError(
            f"Coordinate datasets must be 3D, got shapes x={x.shape}, y={y.shape}, z={z.shape}"
        )
    if x.shape != y.shape or x.shape != z.shape:
        raise ValueError(
            f"Coordinate datasets shape mismatch: x={x.shape}, y={y.shape}, z={z.shape}"
        )

    return x, y, z


def write_tecplot_grid(
    input_h5: Path,
    output_dat: Path,
    title: str,
    zone_title: str,
    precision: int,
) -> None:
    with h5py.File(input_h5, "r") as h5f:
        x, y, z = _load_coords(h5f)

    nz, ny, nx = x.shape
    fmt = f"{{:.{precision}e}}"

    output_dat.parent.mkdir(parents=True, exist_ok=True)
    with output_dat.open("w", encoding="utf-8") as f:
        f.write(f'TITLE = "{title}"\n')
        f.write('VARIABLES = "X" "Y" "Z"\n')
        f.write(f'ZONE T="{zone_title}" I={nx} J={ny} K={nz} DATAPACKING=POINT\n')

        # Tecplot POINT 格式默认 i 最快，然后 j，然后 k
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    f.write(
                        f"{fmt.format(float(x[k, j, i]))} "
                        f"{fmt.format(float(y[k, j, i]))} "
                        f"{fmt.format(float(z[k, j, i]))}\n"
                    )


def main() -> None:
    args = parse_args()

    input_h5 = Path(args.input_h5)
    if not input_h5.is_file():
        raise FileNotFoundError(f"Input file not found: {input_h5}")

    if args.output is None:
        output_dat = input_h5.with_suffix(".dat")
    else:
        output_dat = Path(args.output)

    write_tecplot_grid(
        input_h5=input_h5,
        output_dat=output_dat,
        title=args.title,
        zone_title=args.zone_title,
        precision=args.precision,
    )

    print(f"Wrote Tecplot grid file: {output_dat}")


if __name__ == "__main__":
    main()
