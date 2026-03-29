#!/usr/bin/env python3
"""生成结构网格 HDF5 文件（x/y/z 三个坐标数据集）。

数据布局：
- x, y, z 数据集均为三维 shape = (nz, ny, nx)
- 索引顺序为 [k, j, i]，对应 z, y, x 方向

函数映射：
- 在参数坐标 (xi, eta, zeta) in [0, 1]^3 上定义映射
- x = fx(xi, eta, zeta)
- y = fy(xi, eta, zeta)
- z = fz(xi, eta, zeta)

默认使用线性映射到 [xmin,xmax]x[ymin,ymax]x[zmin,zmax]。
"""

import argparse
import math
from typing import Dict, Any

import h5py
import numpy as np


SAFE_NAMES: Dict[str, Any] = {
    "np": np,
    "math": math,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "exp": np.exp,
    "sqrt": np.sqrt,
    "pi": np.pi,
    "tanh": np.tanh,
}


def eval_expression(expr: str, xi: np.ndarray, eta: np.ndarray, zeta: np.ndarray) -> np.ndarray:
    """在受限命名空间中计算用户给定表达式。"""
    local_names = {
        "xi": xi,
        "eta": eta,
        "zeta": zeta,
    }
    value = eval(expr, {"__builtins__": {}}, {**SAFE_NAMES, **local_names})
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != xi.shape:
        raise ValueError(
            f"表达式结果形状不匹配: got {arr.shape}, expected {xi.shape}."
        )
    return arr


def build_grid(
    nx: int,
    ny: int,
    nz: int,
    fx: str,
    fy: str,
    fz: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构建三维结构网格坐标。"""
    i = np.linspace(0.0, 1.0, nx, dtype=np.float64)
    j = np.linspace(0.0, 1.0, ny, dtype=np.float64)
    k = np.linspace(0.0, 1.0, nz, dtype=np.float64)

    # 索引顺序与求解器一致: [k, j, i] -> [z, y, x]
    zeta, eta, xi = np.meshgrid(k, j, i, indexing="ij")

    x = eval_expression(fx, xi, eta, zeta)
    y = eval_expression(fy, xi, eta, zeta)
    z = eval_expression(fz, xi, eta, zeta)
    return x, y, z


def write_hdf5(path: str, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    """写入 HDF5 文件。"""
    with h5py.File(path, "w") as f:
        dset_x = f.create_dataset("x", data=x, dtype="f8")
        dset_y = f.create_dataset("y", data=y, dtype="f8")
        dset_z = f.create_dataset("z", data=z, dtype="f8")

        # 基础元数据
        f.attrs["nx"] = int(x.shape[2])
        f.attrs["ny"] = int(x.shape[1])
        f.attrs["nz"] = int(x.shape[0])
        f.attrs["layout"] = "[nz, ny, nx]"

        dset_x.attrs["description"] = "x coordinate"
        dset_y.attrs["description"] = "y coordinate"
        dset_z.attrs["description"] = "z coordinate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成结构网格 HDF5 文件")
    parser.add_argument("--nx", type=int, required=True, help="x 方向网格点数")
    parser.add_argument("--ny", type=int, required=True, help="y 方向网格点数")
    parser.add_argument("--nz", type=int, required=True, help="z 方向网格点数")
    parser.add_argument("--output", type=str, default="grid.h5", help="输出 HDF5 文件名")

    parser.add_argument("--xmin", type=float, default=0.0, help="默认线性映射的 xmin")
    parser.add_argument("--xmax", type=float, default=1.0, help="默认线性映射的 xmax")
    parser.add_argument("--ymin", type=float, default=0.0, help="默认线性映射的 ymin")
    parser.add_argument("--ymax", type=float, default=1.0, help="默认线性映射的 ymax")
    parser.add_argument("--zmin", type=float, default=0.0, help="默认线性映射的 zmin")
    parser.add_argument("--zmax", type=float, default=1.0, help="默认线性映射的 zmax")

    parser.add_argument(
        "--fx",
        type=str,
        default="xmin + (xmax - xmin) * xi",
        help="x 坐标表达式，支持变量 xi, eta, zeta 及 np/math 函数",
    )
    parser.add_argument(
        "--fy",
        type=str,
        default="ymin + (ymax - ymin) * eta",
        help="y 坐标表达式，支持变量 xi, eta, zeta 及 np/math 函数",
    )
    parser.add_argument(
        "--fz",
        type=str,
        default="zmin + (zmax - zmin) * zeta",
        help="z 坐标表达式，支持变量 xi, eta, zeta 及 np/math 函数",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.nx <= 0 or args.ny <= 0 or args.nz <= 0:
        raise ValueError("nx, ny, nz 必须为正整数")

    # 把边界变量注入表达式环境（例如 fx 默认式里用到 xmin/xmax）
    fx = args.fx
    fy = args.fy
    fz = args.fz

    for name, value in {
        "xmin": args.xmin,
        "xmax": args.xmax,
        "ymin": args.ymin,
        "ymax": args.ymax,
        "zmin": args.zmin,
        "zmax": args.zmax,
    }.items():
        SAFE_NAMES[name] = float(value)

    x, y, z = build_grid(args.nx, args.ny, args.nz, fx, fy, fz)
    write_hdf5(args.output, x, y, z)

    print(f"Grid file generated: {args.output}")
    print(f"shape = {x.shape}  (layout: [nz, ny, nx])")
    print(
        "x range = [{:.6g}, {:.6g}], y range = [{:.6g}, {:.6g}], z range = [{:.6g}, {:.6g}]".format(
            float(np.min(x)),
            float(np.max(x)),
            float(np.min(y)),
            float(np.max(y)),
            float(np.min(z)),
            float(np.max(z)),
        )
    )


if __name__ == "__main__":
    main()
