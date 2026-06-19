# -*- coding: utf-8 -*-
"""
stage_subset.py  --  arma una carpeta LIGERA y enviable a partir de los
sweeps pesados (Lorenz-96 y SWE).

Idea
----
Los snapshots .npz son lo que pesa (2.4 GB). Los CSV pesan poco. Asi que:

  * copiamos TODOS los CSV de ambos sweeps (master_*.csv, sweep_config.json,
    y los CSV de cada celda),
  * pero de los .npz copiamos SOLO las celdas que elijas (las que se van a
    dibujar como esferas / radar),
  * todo va a una carpeta NUEVA (--out), lista para comprimir y mandar.

Las celdas se nombran como en tus sweeps:
  Lorenz-96 : N{N}_r{r}_s{s}     (r = radio de localizacion, s = spacing)
  SWE       : N{N}_r{r}          (r = radio; coverage fijo 70%)

Uso tipico (desde /mnt/data/pyteda):

  python stage_subset.py \
      --lorenz sweep_lorenz96_obswoodbury_NrS_20260526_025703 \
      --swe    sweep_swe_obswoodbury_20260526_025703 \
      --out    subset_para_figuras \
      --lorenz-cells N40_r3_s1 N40_r3_s2 N40_r3_s3 N20_r3_s1 N80_r3_s1 \
      --swe-cells    N40_r3 N20_r3 N40_r1 N40_r5

Si NO pasas --lorenz-cells / --swe-cells, usa un set por defecto razonable
(ver DEFAULT_* abajo) y te avisa. Con --list solo lista las celdas
disponibles y sale (util para decidir cuales pedir).

El script tambien intenta copiar el cache de verdad de SWE (swe_cache/truth.nc
y la malla) si existe, porque las esferas necesitan el campo "truth" para la
columna de comparacion. Si no lo encuentra, avisa y sigue.
"""

from __future__ import annotations

import os
import sys
import json
import shutil
import argparse
import fnmatch


# Celdas SWE por defecto para snapshots (esferas). Lorenz NO lleva snapshots:
# se copian todos sus CSV y ningun .npz.
DEFAULT_SWE_CELLS = ["N40_r3", "N20_r3", "N40_r1", "N40_r5"]

# Que copiar de cada celda. Los CSV siempre; el .npz solo si la celda fue
# seleccionada para esferas.
CELL_CSVS = ["summary.csv", "summary_aggregated.csv",
             "diagnostics_summary.csv", "error_curves.csv"]
CELL_NPZ = "snapshots.npz"

# CSV / config a nivel de sweep (siempre se copian, pesan poco).
SWEEP_LEVEL_FILES = ["master_summary.csv", "master_percell.csv",
                     "master_error_curves.csv", "sweep_config.json"]


def human(nbytes):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024.0:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024.0
    return f"{nbytes:.1f} PB"


def list_cells(sweep_dir):
    """Devuelve la lista de nombres de celda (carpetas dentro de results/)."""
    results = os.path.join(sweep_dir, "results")
    if not os.path.isdir(results):
        return []
    cells = sorted(d for d in os.listdir(results)
                   if os.path.isdir(os.path.join(results, d)))
    return cells


def resolve_cells(requested, available, label):
    """Expande patrones glob y valida contra las celdas disponibles."""
    if not requested:
        return []
    chosen = []
    for pat in requested:
        matches = fnmatch.filter(available, pat)
        if not matches:
            print(f"  [aviso] {label}: patron/celda '{pat}' no coincide con "
                  f"ninguna celda disponible — se ignora.")
        chosen.extend(matches)
    # dedup preservando orden
    seen, out = set(), []
    for c in chosen:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def copy_file(src, dst, stats):
    if not os.path.exists(src):
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    sz = os.path.getsize(src)
    stats["files"] += 1
    stats["bytes"] += sz
    return True


def stage_sweep(sweep_dir, out_dir, npz_cells, stats, label):
    """Copia un sweep completo (CSV de todas las celdas) + .npz de npz_cells."""
    if not os.path.isdir(sweep_dir):
        print(f"  [SKIP] {label}: no existe la carpeta {sweep_dir}")
        return

    base = os.path.basename(os.path.normpath(sweep_dir))
    dst_sweep = os.path.join(out_dir, base)
    print(f"\n  {label}: {sweep_dir}")
    print(f"     -> {dst_sweep}")

    # 1) Archivos a nivel de sweep.
    for fname in SWEEP_LEVEL_FILES:
        if copy_file(os.path.join(sweep_dir, fname),
                     os.path.join(dst_sweep, fname), stats):
            print(f"     + {fname}")

    # 2) CSV de TODAS las celdas (peso bajo) + .npz solo de las elegidas.
    all_cells = list_cells(sweep_dir)
    npz_set = set(npz_cells)
    n_csv_cells, n_npz = 0, 0
    npz_bytes = 0
    for cell in all_cells:
        src_cell = os.path.join(sweep_dir, "results", cell)
        dst_cell = os.path.join(dst_sweep, "results", cell)
        any_csv = False
        for csv in CELL_CSVS:
            if copy_file(os.path.join(src_cell, csv),
                         os.path.join(dst_cell, csv), stats):
                any_csv = True
        if any_csv:
            n_csv_cells += 1
        if cell in npz_set:
            src_npz = os.path.join(src_cell, CELL_NPZ)
            if os.path.exists(src_npz):
                sz = os.path.getsize(src_npz)
                copy_file(src_npz, os.path.join(dst_cell, CELL_NPZ), stats)
                npz_bytes += sz
                n_npz += 1
                print(f"     + results/{cell}/{CELL_NPZ}  ({human(sz)})")
            else:
                print(f"     [aviso] {cell}: se pidio .npz pero no existe "
                      f"{src_npz}")

    print(f"     resumen: CSV de {n_csv_cells} celdas, "
          f"{n_npz} .npz copiados ({human(npz_bytes)} en snapshots)")

    # 3) Manifiesto de celdas con .npz para que make_all_figures sepa cuales hay.
    manifest = {
        "sweep": base,
        "all_cells": all_cells,
        "npz_cells": [c for c in npz_cells if c in all_cells],
    }
    with open(os.path.join(dst_sweep, "subset_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def stage_swe_cache(swe_cache_dir, out_dir, stats):
    """Copia el truth (y malla) de SWE — necesario para la columna 'truth' de
    las esferas. Solo los .nc pequenos de truth/x0/xb; NO el pool de ensemble."""
    if not swe_cache_dir or not os.path.isdir(swe_cache_dir):
        print(f"\n  [aviso] swe_cache no encontrado ({swe_cache_dir}). "
              f"Las esferas no tendran columna 'truth' a menos que lo incluyas.")
        return
    dst = os.path.join(out_dir, os.path.basename(os.path.normpath(swe_cache_dir)))
    print(f"\n  SWE cache: {swe_cache_dir} -> {dst}")
    # Copiamos truth y los estados de referencia; evitamos el pool de ensemble
    # (initial_ensemble_N*.nc) que puede pesar.
    wanted = ["truth.nc", "x0_ref.nc", "xb.nc"]
    for fname in wanted:
        if copy_file(os.path.join(swe_cache_dir, fname),
                     os.path.join(dst, fname), stats):
            sz = os.path.getsize(os.path.join(swe_cache_dir, fname))
            print(f"     + {fname}  ({human(sz)})")
    # Avisar de lo que se omite deliberadamente.
    for fname in os.listdir(swe_cache_dir):
        if fname.startswith("initial_ensemble_"):
            print(f"     (omitido a proposito: {fname} — no hace falta para figuras)")


def main():
    ap = argparse.ArgumentParser(
        description="Arma una carpeta ligera (CSV + .npz selectos) para enviar.")
    ap.add_argument("--lorenz", default=None, help="carpeta del sweep Lorenz-96")
    ap.add_argument("--swe", default=None, help="carpeta del sweep SWE")
    ap.add_argument("--swe-cache", default="swe_cache",
                    help="carpeta de cache de SWE (truth.nc, etc.)")
    ap.add_argument("--out", default="subset_para_figuras",
                    help="carpeta de salida (se crea)")
    ap.add_argument("--lorenz-cells", nargs="*", default=None,
                    help="celdas Lorenz para .npz (por defecto NINGUNA: "
                         "Lorenz lleva todos los CSV pero cero snapshots)")
    ap.add_argument("--swe-cells", nargs="*", default=None,
                    help="celdas SWE para .npz, p.ej. N40_r3 (acepta glob)")
    ap.add_argument("--list", action="store_true",
                    help="solo listar celdas disponibles y salir")
    args = ap.parse_args()

    if not args.lorenz and not args.swe:
        ap.error("Debes pasar al menos --lorenz o --swe.")

    # Modo lista: muestra que celdas existen para que elijas.
    if args.list:
        for label, d in (("Lorenz-96", args.lorenz), ("SWE", args.swe)):
            if d:
                cells = list_cells(d)
                print(f"\n{label}  ({d}) — {len(cells)} celdas:")
                print("  " + "  ".join(cells))
        return

    # Resolver celdas pedidas (o defaults).
    lor_avail = list_cells(args.lorenz) if args.lorenz else []
    swe_avail = list_cells(args.swe) if args.swe else []

    lor_req = args.lorenz_cells
    if args.lorenz and lor_req is None:
        lor_req = []  # Lorenz: todos los CSV, CERO snapshots (decision pedida).
        print("[info] Lorenz-96: copio TODOS los CSV y NINGUN .npz "
              "(usa --lorenz-cells si quieres algun snapshot).")
    swe_req = args.swe_cells
    if args.swe and swe_req is None:
        swe_req = DEFAULT_SWE_CELLS
        print(f"[info] --swe-cells no dado; uso defaults: {swe_req}")

    lor_cells = resolve_cells(lor_req, lor_avail, "Lorenz-96")
    swe_cells = resolve_cells(swe_req, swe_avail, "SWE")

    os.makedirs(args.out, exist_ok=True)
    stats = {"files": 0, "bytes": 0}

    print("=" * 72)
    print(f" STAGING -> {os.path.abspath(args.out)}")
    print("=" * 72)
    print(f"  Lorenz .npz de celdas: {lor_cells or '(ninguna)'}")
    print(f"  SWE    .npz de celdas: {swe_cells or '(ninguna)'}")

    if args.lorenz:
        stage_sweep(args.lorenz, args.out, lor_cells, stats, "Lorenz-96")
    if args.swe:
        stage_sweep(args.swe, args.out, swe_cells, stats, "SWE")
        stage_swe_cache(args.swe_cache, args.out, stats)

    # Manifiesto global del subset.
    top = {
        "lorenz_sweep": os.path.basename(os.path.normpath(args.lorenz)) if args.lorenz else None,
        "swe_sweep": os.path.basename(os.path.normpath(args.swe)) if args.swe else None,
        "lorenz_npz_cells": lor_cells,
        "swe_npz_cells": swe_cells,
    }
    with open(os.path.join(args.out, "SUBSET_INFO.json"), "w") as f:
        json.dump(top, f, indent=2)

    print("\n" + "=" * 72)
    print(f" LISTO. {stats['files']} archivos, {human(stats['bytes'])} en total.")
    print(f" Carpeta: {os.path.abspath(args.out)}")
    print("\n Para comprimir y mandar:")
    print(f"   tar -czvf {os.path.basename(args.out)}.tar.gz {args.out}")
    print("=" * 72)


if __name__ == "__main__":
    main()