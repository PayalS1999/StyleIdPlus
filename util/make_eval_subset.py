#!/usr/bin/env python
import os, shutil, itertools, pathlib, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cnt', required=True)    # master content dir
parser.add_argument('--sty', required=True)    # master style dir
parser.add_argument('--out_cnt', required=True)
parser.add_argument('--out_sty', required=True)
parser.add_argument('--cnt_ids', nargs='+', required=True)  # e.g. 00 01 02 03 04
parser.add_argument('--sty_ids', nargs='+', required=True)  # e.g. 00 … 39
args = parser.parse_args()

Path = pathlib.Path
Path(args.out_cnt).mkdir(parents=True, exist_ok=True)
Path(args.out_sty).mkdir(parents=True, exist_ok=True)

for c in args.cnt_ids:
    c_file = Path(args.cnt) / f"{c}.png"
    for s in args.sty_ids:
        # duplicate content: keep name pattern identical to stylised pair finder
        shutil.copy2(c_file, Path(args.out_cnt) / f"{c}_{s}.png")

for s in args.sty_ids:
    s_file = Path(args.sty) / f"{s}.png"
    for c in args.cnt_ids:
        shutil.copy2(s_file, Path(args.out_sty) / f"{c}_{s}.png")

print("Done – cnt_eval_5 and sty_eval_5 ready.")
