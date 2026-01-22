# -*- coding: utf-8 -*-
import json
import os
from collections import defaultdict, Counter

# ============ 你只改这里 ============
SEG_JSONL = r"segments_all_with_aug.jsonl"
TRAIN_TXT = r"train.txt"
TEST_TXT  = r"test.txt"
ITEM2INT  = r"item2int.json"
STATS_JSON = r"stats.json"
# ==================================

def parse_train_or_test_line(line: str):
    """
    解析 "segid: i i i" 或 "segid: i"
    返回 (segid:int, items:list[int])
    """
    line = line.strip()
    if not line:
        return None
    if ":" not in line:
        raise ValueError(f"Bad line (no colon): {line[:80]}")
    left, right = line.split(":", 1)
    segid = int(left.strip())
    right = right.strip()
    if not right:
        items = []
    else:
        items = list(map(int, right.split()))
    return segid, items

def main():
    # --------- basic exists ----------
    for p in [SEG_JSONL, TRAIN_TXT, TEST_TXT, ITEM2INT, STATS_JSON]:
        print(f"[PATH] {p} exists? {os.path.exists(p)}")
    print()

    # --------- load stats, item2int ----------
    with open(STATS_JSON, "r", encoding="utf-8") as f:
        stats = json.load(f)

    with open(ITEM2INT, "r", encoding="utf-8") as f:
        item2int = json.load(f)

    int_set = set(item2int.values())
    max_int = max(int_set) if int_set else 0
    print(f"[ITEM2INT] unique items = {len(item2int):,}, max_int = {max_int:,}")
    # quick sanity: ids should be 1..N ideally
    missing_ints = []
    if max_int <= 500000:  # 防止太大时 O(N) 过慢；你这个 13万没问题
        for x in range(1, max_int + 1):
            if x not in int_set:
                missing_ints.append(x)
                if len(missing_ints) >= 10:
                    break
    if missing_ints:
        print(f"[WARN] item2int ids not contiguous, first missing: {missing_ints[:10]}")
    else:
        print("[OK] item2int ids look contiguous (1..max)")
    print()

    # --------- scan segments jsonl ----------
    seg_ids = set()
    seg_type_cnt = Counter()
    native_aug_declared = {}      # native_id -> aug_generated_count(声明值)
    aug_children_cnt = defaultdict(int)  # parent_id -> number of aug children
    aug_parent_type_bad = 0
    parent_missing = 0
    id_dup = 0
    id_gap = 0
    id_not_start0 = False

    # 为了检查 parent 是否 native，需要存 segment_type
    seg_type_by_id = {}

    min_id = None
    max_id_seen = None
    lines = 0

    with open(SEG_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lines += 1
            obj = json.loads(line)

            sid = obj.get("segment_id")
            stype = obj.get("segment_type")
            if sid is None or stype is None:
                raise ValueError(f"Missing segment_id/segment_type at line {lines}")

            if min_id is None:
                min_id = sid
            max_id_seen = sid

            # dup check
            if sid in seg_ids:
                id_dup += 1
            seg_ids.add(sid)
            seg_type_cnt[stype] += 1
            seg_type_by_id[sid] = stype

            if stype == "native":
                if "aug_generated_count" in obj:
                    native_aug_declared[sid] = int(obj["aug_generated_count"])
                else:
                    native_aug_declared[sid] = 0
            else:
                # aug segment
                pid = obj.get("parent_segment_id")
                if pid is None:
                    raise ValueError(f"Aug segment missing parent_segment_id at seg_id={sid}")
                aug_children_cnt[pid] += 1

    print(f"[SEGMENTS] jsonl lines = {lines:,}")
    print(f"[SEGMENTS] unique segment_id = {len(seg_ids):,}, dup_id = {id_dup}")
    print(f"[SEGMENTS] min_id={min_id}, max_id={max_id_seen}")
    if min_id != 0:
        id_not_start0 = True
        print(f"[WARN] segment_id does NOT start from 0 (min_id={min_id})")

    # continuous id check (0..max)
    if max_id_seen is not None:
        expected = max_id_seen - min_id + 1
        if expected != len(seg_ids):
            # 找几个缺口
            missing = []
            for x in range(min_id, max_id_seen + 1):
                if x not in seg_ids:
                    missing.append(x)
                    if len(missing) >= 10:
                        break
            id_gap = (expected - len(seg_ids))
            print(f"[WARN] segment_id not continuous, missing_count≈{id_gap}, first_missing={missing[:10]}")
        else:
            print("[OK] segment_id continuous")
    print()

    print("[SEGMENTS] type distribution:")
    for k, v in seg_type_cnt.most_common():
        print(f"  - {k}: {v:,}")
    print()

    # --------- check parent existence & parent type ----------
    for pid, c in aug_children_cnt.items():
        if pid not in seg_ids:
            parent_missing += 1
        else:
            if seg_type_by_id.get(pid) != "native":
                aug_parent_type_bad += 1

    if parent_missing == 0:
        print("[OK] all aug parent_segment_id exist")
    else:
        print(f"[BAD] missing parents for aug segments: {parent_missing:,}")

    if aug_parent_type_bad == 0:
        print("[OK] all aug parent_segment_id point to native")
    else:
        print(f"[WARN] aug parent points to NON-native: {aug_parent_type_bad:,}")
    print()

    # --------- check native aug_generated_count correctness ----------
    mismatch = 0
    mismatch_examples = []
    for nid, declared in native_aug_declared.items():
        real = aug_children_cnt.get(nid, 0)
        if declared != real:
            mismatch += 1
            if len(mismatch_examples) < 10:
                mismatch_examples.append((nid, declared, real))

    if mismatch == 0:
        print("[OK] native aug_generated_count matches real child count")
    else:
        print(f"[WARN] native aug_generated_count mismatch: {mismatch:,}")
        print("  examples (native_id, declared, real):")
        for e in mismatch_examples:
            print("  ", e)
    print()

    # --------- scan train/test consistency ----------
    def scan_train_test(path, name):
        line_cnt = 0
        bad_lines = 0
        missing_seg = 0
        item_out_of_range = 0
        empty_items = 0
        wrong_order = 0
        prev_sid = None

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                line_cnt += 1
                try:
                    sid, items = parse_train_or_test_line(line)
                except Exception:
                    bad_lines += 1
                    continue

                if prev_sid is not None and sid < prev_sid:
                    wrong_order += 1
                prev_sid = sid

                if sid not in seg_ids:
                    missing_seg += 1

                if len(items) == 0:
                    empty_items += 1

                # check item id range
                for it in items:
                    if it <= 0 or it > max_int:
                        item_out_of_range += 1
                        if item_out_of_range <= 5:
                            print(f"[{name} WARN] item int out of range: seg={sid}, it={it}")

        print(f"[{name}] lines={line_cnt:,}, bad_lines={bad_lines:,}, seg_not_in_segments={missing_seg:,}, "
              f"empty_items={empty_items:,}, wrong_order={wrong_order:,}, item_out_of_range={item_out_of_range:,}")
        return line_cnt, bad_lines, missing_seg, empty_items, wrong_order, item_out_of_range

    train_cnt, *_ = scan_train_test(TRAIN_TXT, "TRAIN")
    test_cnt,  *_ = scan_train_test(TEST_TXT, "TEST")
    print()

    # In your generator: one line per segment in train and test
    if train_cnt == len(seg_ids) and test_cnt == len(seg_ids):
        print("[OK] train/test line count == total segments")
    else:
        print(f"[WARN] train/test line count mismatch: segments={len(seg_ids):,}, train={train_cnt:,}, test={test_cnt:,}")
    print()

    # --------- compare with stats.json ----------
    # stats.json 里 segments_total 应该等于实际写出的段数（segment_id计数）
    # 但如果你 stats 记录的是 segment_id(最终) ，也就是 len(seg_ids)
    print("[STATS.json] reported:")
    for k in ["users_seen", "bursts_total", "native_segments_total", "aug_segments_total", "segments_total", "items_unique_total"]:
        if k in stats:
            print(f"  - {k}: {stats[k]}")
    print()

    # 对比几个关键的
    ok1 = (stats.get("segments_total") == len(seg_ids))
    ok2 = (stats.get("native_segments_total") == seg_type_cnt.get("native", 0))
    ok3 = (stats.get("aug_segments_total") == (len(seg_ids) - seg_type_cnt.get("native", 0)))
    ok4 = (stats.get("items_unique_total") == len(item2int))

    print("[COMPARE] segments_total:", "OK" if ok1 else f"BAD (stats={stats.get('segments_total')}, real={len(seg_ids)})")
    print("[COMPARE] native_segments_total:", "OK" if ok2 else f"BAD (stats={stats.get('native_segments_total')}, real={seg_type_cnt.get('native',0)})")
    print("[COMPARE] aug_segments_total:", "OK" if ok3 else f"BAD (stats={stats.get('aug_segments_total')}, real={len(seg_ids)-seg_type_cnt.get('native',0)})")
    print("[COMPARE] items_unique_total:", "OK" if ok4 else f"BAD (stats={stats.get('items_unique_total')}, real={len(item2int)})")

    print("\n==== DONE ====")

if __name__ == "__main__":
    main()
