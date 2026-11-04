#!/usr/bin/env python3
"""Compare work/results/*.json with the numbers reported in the paper (expected/paper_numbers.json).

Usage: python scripts/check_against_paper.py --work work
Each expected value is compared after rounding to the precision written in the paper.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FILES = {"internal": "internal.json", "external_original": "external_original.json", "external_jpeg512": "external_jpeg512.json"}


def get(d, key):
    """Dotted path lookup that tolerates dots inside keys such as '1.0' or 'NMS, IoU 0.5'."""
    parts = key.split(".")
    i = 0
    while i < len(parts):
        for j in range(len(parts), i, -1):
            k = ".".join(parts[i:j])
            if isinstance(d, dict) and k in d:
                d, i = d[k], j
                break
            if isinstance(d, list) and k.isdigit() and int(k) < len(d):
                d, i = d[int(k)], j
                break
        else:
            raise KeyError(key)
    return d


def decimals(x):
    s = repr(float(x))
    return 0 if float(x).is_integer() else len(s.split(".")[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work")
    a = ap.parse_args()
    exp = json.loads((REPO / "expected/paper_numbers.json").read_text())
    n_ok = n_bad = n_skip = 0
    for sec, fname in FILES.items():
        p = Path(a.work) / "results" / fname
        if not p.exists():
            print(f"[skip] {p} not found")
            n_skip += len(exp[sec])
            continue
        R = json.loads(p.read_text())
        for key, want in exp[sec].items():
            got = get(R, key)
            want_l, got_l = (want, got) if isinstance(want, list) else ([want], [got])
            ok = all(round(float(g), decimals(w)) == round(float(w), decimals(w))
                     for w, g in zip(want_l, got_l))
            n_ok += ok
            n_bad += not ok
            if not ok:
                print(f"[DIFF] {sec}: {key}: paper {want}, here {got}")
    print(f"{n_ok} match, {n_bad} differ, {n_skip} not checked")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
