# -*- coding: utf-8 -*-
import os

IN_TRAIN = r"train.txt"   # 你现在的格式：segid: i1 i2 ...
IN_TEST  = r"test.txt"    # 你现在的格式：segid: i_test

OUT_TRAIN = r"train_edges.txt"  # LightGCN常用：u i 1
OUT_TEST  = r"test_edges.txt"

def parse_train_line(line: str):
    # "123: 7 8 9"
    line = line.strip()
    if not line:
        return None
    if ":" not in line:
        return None
    left, right = line.split(":", 1)
    u = left.strip()
    items = right.strip().split()
    return u, items

def parse_test_line(line: str):
    # "123: 10"
    line = line.strip()
    if not line:
        return None
    if ":" not in line:
        return None
    left, right = line.split(":", 1)
    u = left.strip()
    i = right.strip()
    if not i:
        return None
    return u, i

def main():
    print("[PATH]", IN_TRAIN, os.path.exists(IN_TRAIN))
    print("[PATH]", IN_TEST, os.path.exists(IN_TEST))

    train_edges = 0
    with open(IN_TRAIN, "r", encoding="utf-8") as fin, open(OUT_TRAIN, "w", encoding="utf-8", newline="\n") as fout:
        for ln in fin:
            parsed = parse_train_line(ln)
            if not parsed:
                continue
            u, items = parsed
            for it in items:
                # u item 1
                fout.write(f"{u} {it} 1\n")
                train_edges += 1

    test_edges = 0
    with open(IN_TEST, "r", encoding="utf-8") as fin, open(OUT_TEST, "w", encoding="utf-8", newline="\n") as fout:
        for ln in fin:
            parsed = parse_test_line(ln)
            if not parsed:
                continue
            u, it = parsed
            fout.write(f"{u} {it} 1\n")
            test_edges += 1

    print("[DONE]")
    print("OUT_TRAIN:", OUT_TRAIN, "edges=", train_edges)
    print("OUT_TEST :", OUT_TEST,  "edges=", test_edges)

if __name__ == "__main__":
    main()
