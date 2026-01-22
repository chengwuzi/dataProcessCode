# -*- coding: utf-8 -*-
import json
from datetime import datetime

INPUT_PATH = r"Electronics.jsonl"
OUTPUT_PATH = r"Electronics_2019_2021.jsonl"

TARGET_YEARS = {2019, 2020, 2021}


def get_year(ts):
    if ts is None:
        return None
    ts = int(ts)
    if ts > 10_000_000_000:  # treat as ms
        ts //= 1000
    return datetime.utcfromtimestamp(ts).year


def main():
    total = 0
    kept = 0
    bad_lines = 0
    missing_ts = 0

    with open(INPUT_PATH, "r", encoding="utf-8") as fin, open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue

            total += 1
            try:
                obj = json.loads(line)
            except Exception:
                bad_lines += 1
                continue

            year = get_year(obj.get("timestamp"))
            if year is None:
                missing_ts += 1
                continue

            if year in TARGET_YEARS:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                kept += 1

            if idx % 1_000_000 == 0:
                print(f"processed {idx:,} lines, kept {kept:,}...")

    print("\n[DONE]")
    print(f"input lines processed: {total:,}")
    print(f"kept (2019-2021): {kept:,}")
    print(f"bad json lines: {bad_lines:,}")
    print(f"missing timestamp: {missing_ts:,}")
    print(f"output file: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
