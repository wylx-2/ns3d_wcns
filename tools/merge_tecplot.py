#!/usr/bin/env python3
"""
Merge per-rank Tecplot ASCII files (written by write_tecplot_field) into a single Tecplot
ASCII file containing one structured ZONE with global I, J, K.

Usage:
    python3 merge_tecplot.py prefix output_filename

Example:
    python3 ./tools/merge_tecplot.py ./output/initial_field intitial_field_merged.dat

Notes:
- Assumes per-rank files are named <prefix>_rank<id>.dat and contain the VARIABLES line
  with the first three variables X Y Z followed by data variables.
- Assumes each rank wrote only physical cells (no ghost overlap).
- This script reconstructs the global grid by collecting unique X/Y/Z coordinates and
  placing each point into the global array using coordinate matching (with tolerance).
"""
import sys
import glob
import numpy as np
from collections import OrderedDict


def read_tecplot_file(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()

    # find VARIABLES line
    varline = None
    zoneline_idx = None
    for i, L in enumerate(lines):
        if L.strip().upper().startswith('VARIABLES'):
            varline = L.strip()
        if L.strip().upper().startswith('ZONE'):
            zoneline_idx = i
            break

    if varline is None:
        raise RuntimeError(f"No VARIABLES line in {fname}")

    # parse variable names (basic parsing of quoted names)
    import re
    names = re.findall(r'"([^"]+)"', varline)
    # data starts after zoneline_idx (if present)
    data_start = zoneline_idx + 1 if zoneline_idx is not None else 0

    # load numeric data from remaining lines
    data = np.loadtxt(lines[data_start:])
    # data columns: first three are X Y Z
    return names, data


def main(prefix, outname):
    files = sorted(glob.glob(f"{prefix}_rank*.dat"))
    if not files:
        print(f"No files found matching {prefix}_rank*.dat")
        return

    all_names = None
    points = []  # list of arrays Nx x M

    for fname in files:
        names, data = read_tecplot_file(fname)
        if all_names is None:
            all_names = names
        else:
            if names != all_names:
                raise RuntimeError(f"Variable mismatch in {fname}")
        points.append(data)
        print(f"Read {data.shape[0]} points from {fname}")

    data_all = np.vstack(points)
    X = data_all[:,0]
    Y = data_all[:,1]
    Z = data_all[:,2]
    vars_all = data_all[:,3:]

    # build unique sorted coordinate arrays with tolerance
    def unique_sorted(arr):
        vals = np.unique(np.round(arr, 10))
        return np.sort(vals)

    xs = unique_sorted(X)
    ys = unique_sorted(Y)
    zs = unique_sorted(Z)

    nx = xs.size; ny = ys.size; nz = zs.size
    print(f"Global grid inferred: nx={nx}, ny={ny}, nz={nz}")

    # build maps from coordinate to index
    # use dictionary with rounded key to avoid floating point issues
    def build_map(vals):
        m = {}
        for idx, v in enumerate(vals):
            key = round(float(v), 10)
            m[key] = idx
        return m

    mx = build_map(xs); my = build_map(ys); mz = build_map(zs)

    nvars = vars_all.shape[1]
    # allocate data in i,j,k order (i fastest) to match DATAPACKING=POINT used in writer
    data_grid = np.zeros((nx, ny, nz, nvars), dtype=float)

    # fill grid
    for p in range(data_all.shape[0]):
        xi = round(float(X[p]), 10)
        yi = round(float(Y[p]), 10)
        zi = round(float(Z[p]), 10)
        ix = mx.get(xi, None)
        iy = my.get(yi, None)
        iz = mz.get(zi, None)
        if ix is None or iy is None or iz is None:
            raise RuntimeError(f"Coordinate {X[p]}, {Y[p]}, {Z[p]} not found in maps")
        data_grid[ix,iy,iz,:] = vars_all[p,:]

    # write merged file
    with open(outname, 'w') as fo:
        fo.write(f"TITLE = \"Merged Tecplot: {outname}\"\n")
        # write variable names: keep X Y Z and then the rest
        fo.write('VARIABLES = ')
        for nm in all_names:
            fo.write(f'"{nm}" ')
        fo.write('\n')
        fo.write(f"ZONE T=\"merged\" I={nx} J={ny} K={nz} DATAPACKING=POINT\n")
        fo.write('\n')
        # output in k,j,i loops where i varies fastest
        for kz in range(nz):
            for jy in range(ny):
                for ix in range(nx):
                    x = xs[ix]; y = ys[jy]; z = zs[kz]
                    fo.write(f"{x:.8e} {y:.8e} {z:.8e}")
                    vals = data_grid[ix,jy,kz,:]
                    for v in vals:
                        fo.write(f" {v:.8e}")
                    fo.write('\n')
    print(f"Merged file written to {outname}")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: merge_tecplot.py prefix output_filename')
        sys.exit(1)
    prefix = sys.argv[1]
    outname = sys.argv[2]
    main(prefix, outname)
