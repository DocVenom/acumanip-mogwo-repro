# -*- coding: utf-8 -*-
"""Standalone re-derivation of every split in this repository.

Deliberately does NOT import the authors' model/search code: the splits must be
reproducible from the public raw data alone.

usage:
    python regenerate_splits.py --ucihar "path/to/UCI HAR Dataset" \
                                --wisdm "path/to/WISDM_ar_v1.1_raw.txt"

Exit status 0 and a final "ALL SPLITS MATCH" means the JSON files in data/splits/
are exactly reproduced. Mismatches are reported per item and give a non-zero exit.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(HERE, "..", "data", "splits")

UCI_ACTIVITIES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
                  "SITTING", "STANDING", "LAYING"]
WISDM_ACTIVITIES = ["Walking", "Jogging", "Upstairs", "Downstairs", "Sitting", "Standing"]
WISDM_SEQ_LEN, WISDM_STEP, WISDM_SPLIT_SUBJECT = 128, 64, 28

SEARCH_POOL, SEARCH_VAL = 3000, 0.2
FULL_VAL = 0.15
SEED = 42

failures = []


def check(name, got, want):
    ok = got == want
    print(("  OK   " if ok else "  FAIL ") + name + ("" if ok else f"  (got {got}, want {want})"))
    if not ok:
        failures.append(name)


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def uci_subject_label(root):
    """官方 train / test 的被试与标签，保持文件原始顺序。"""
    out = []
    for part in ("train", "test"):
        s = np.loadtxt(os.path.join(root, part, f"subject_{part}.txt")).astype(int)
        y = np.loadtxt(os.path.join(root, part, f"y_{part}.txt")).astype(int) - 1
        out.append((part, s, y))
    return out


def regen_ucihar(root):
    print("UCI HAR")
    parts = uci_subject_label(root)
    (_, s_tr, y_tr), (_, s_te, y_te) = parts

    # ---- index table
    want_rows = []
    with open(os.path.join(SPLITS, "ucihar_index.csv"), encoding="utf-8") as f:
        header = f.readline().strip()
        for line in f:
            p, r, s, lab, act = line.strip().split(",")
            want_rows.append((p, int(r), int(s), int(lab), act))
    got_rows = []
    for part, s, y in parts:
        for i in range(len(y)):
            got_rows.append((part, i, int(s[i]), int(y[i]), UCI_ACTIVITIES[int(y[i])]))
    check("ucihar_index.csv rows", len(got_rows), len(want_rows))
    check("ucihar_index.csv content", got_rows, want_rows)

    # ---- search split: 3000 from official train, then 8:2
    sel, _ = train_test_split(np.arange(len(y_tr)), train_size=SEARCH_POOL,
                              stratify=y_tr, random_state=SEED)
    pos_tr, pos_va = train_test_split(np.arange(len(sel)), test_size=SEARCH_VAL,
                                      stratify=y_tr[sel], random_state=SEED)
    sjson = load_json(os.path.join(SPLITS, "ucihar_search_split.json"))
    check("search train indices", sorted(sel[pos_tr].tolist()), sorted(sjson["search_train_index"]))
    check("search val indices", sorted(sel[pos_va].tolist()), sorted(sjson["val_index"]))
    check("search train size", len(pos_tr), sjson["n_search_train"])
    check("search val size", len(pos_va), sjson["n_val"])

    # ---- full split: official train 85/15
    p_tr, p_va = train_test_split(np.arange(len(y_tr)), test_size=FULL_VAL,
                                  stratify=y_tr, random_state=SEED)
    fjson = load_json(os.path.join(SPLITS, "ucihar_full_split.json"))
    check("full train indices", sorted(p_tr.tolist()), sorted(fjson["train_index"]))
    check("full val indices", sorted(p_va.tolist()), sorted(fjson["val_index"]))

    # ---- LOSO folds
    all_s = np.concatenate([s_tr, s_te])
    ljson = load_json(os.path.join(SPLITS, "ucihar_loso_folds.json"))
    got = [{"subject": int(u), "n_test": int((all_s == u).sum()), "n_train_pool": int((all_s != u).sum())}
           for u in sorted(set(all_s.tolist()))]
    check("LOSO fold count", len(got), ljson["n_folds"])
    check("LOSO fold table", got, ljson["folds"])
    check("LOSO total held-out samples", sum(g["n_test"] for g in got), len(all_s))


def wisdm_parse(path):
    users, acts, xs, ys, zs = [], [], [], [], []
    act2id = {a: i for i, a in enumerate(WISDM_ACTIVITIES)}
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            s = line.strip().rstrip(";").strip()
            if not s:
                continue
            p = s.split(",")
            if len(p) != 6:
                continue
            try:
                u, a = int(p[0]), act2id[p[1]]
                x, y, z = float(p[3]), float(p[4]), float(p[5])
            except (ValueError, KeyError):
                continue
            users.append(u); acts.append(a); xs.append(x); ys.append(y); zs.append(z)
    return np.column_stack([users, acts, xs, ys, zs]).astype(np.float32)


def wisdm_window(arr):
    """切窗并返回每条窗口的 (user, activity)，用于核对数量与分布。"""
    n = len(arr)
    segs, start, cur = [], 0, (arr[0, 0], arr[0, 1])
    for i in range(1, n):
        key = (arr[i, 0], arr[i, 1])
        if key != cur:
            segs.append((start, i, cur))
            start, cur = i, key
    segs.append((start, n, cur))

    users, acts = [], []
    for s, e, (u, a) in segs:
        L = e - s
        if L < WISDM_SEQ_LEN:
            continue
        for _ in range(0, L - WISDM_SEQ_LEN + 1, WISDM_STEP):
            users.append(int(u)); acts.append(int(a))
    return np.asarray(users), np.asarray(acts)


def regen_wisdm(path):
    print("WISDM")
    arr = wisdm_parse(path)
    users, acts = wisdm_window(arr)
    tr = users < WISDM_SPLIT_SUBJECT
    j = load_json(os.path.join(SPLITS, "wisdm_split.json"))
    check("WISDM train windows", int(tr.sum()), j["n_train"])
    check("WISDM test windows", int((~tr).sum()), j["n_test"])
    check("WISDM train subjects", sorted(set(users[tr].tolist())), j["train_subjects"])
    check("WISDM test subjects", sorted(set(users[~tr].tolist())), j["test_subjects"])

    card = load_json(os.path.join(HERE, "..", "data", "cards", "wisdm.json"))
    got = {"train": np.bincount(acts[tr], minlength=6).tolist(),
           "test": np.bincount(acts[~tr], minlength=6).tolist()}
    check("WISDM label distribution", got, card["label_distribution"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ucihar", required=True, help='path to "UCI HAR Dataset" folder')
    ap.add_argument("--wisdm", required=True, help="path to WISDM_ar_v1.1_raw.txt")
    a = ap.parse_args()

    regen_ucihar(a.ucihar)
    regen_wisdm(a.wisdm)

    print()
    if failures:
        print("%d MISMATCH(ES): %s" % (len(failures), ", ".join(failures)))
        return 1
    print("ALL SPLITS MATCH")
    return 0


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.exit(main())
