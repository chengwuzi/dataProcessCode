# -*- coding: utf-8 -*-
import json
import os
import csv
from collections import Counter


# ================== 你只改这里 ==================
DATA_PATH = r"../dataProcess/Electronics_2019_2021_sorted_min15_canon.jsonl"

OUT_SEG_JSONL = r"segments_7d_native_plus_aug.jsonl"       # 每个片段一行 JSON
OUT_SEG_ITEM_CSV = r"segment_item_7d_native_plus_aug.csv"  # 训练用 (segment_id,item_id)

# 如果你还想顺便输出统计信息
OUT_STATS_JSON = r"segments_7d_native_plus_aug_stats.json"
# ===============================================

# ============ Segment 划分参数 ============
COMPRESS_HOURS = 1           # 1小时桶内去重（保序）
CUT_HOURS = 168              # gap>=7天 切 burst
SEG_MIN = 2                  # 最小 item 数
SEG_MAX = 30                 # 最大 item 数（native + aug 都不超过它）
# =======================================

# ============ Augmentation 参数 ============
ENABLE_AUGMENT = True
AUG_MIN_ITEMS = 12           # 仅对 >=12 的大片段做派生
AUG_SEG_MIN = 4              # 派生子段至少多少 item
AUG_WINDOW_W = 8             # 滑窗长度（前窗+后窗）
# 派生策略：时间二分 2段 + 滑窗 2段（尽量前窗/后窗）
# =========================================


def ts_to_seconds(ts):
    ts = int(ts)
    if ts > 10_000_000_000:  # ms
        return ts / 1000.0
    return float(ts)


def hour_bucket(ts_sec):
    return int(ts_sec // (COMPRESS_HOURS * 3600))


def compress_user_events(events):
    """
    1小时桶内去重 item（保序去重：同一桶内只保留第一次出现的 item，保持原始顺序）
    events: list of (ts_sec, item_id) sorted by time
    return: list of (approx_ts, item_id) sorted by approx_ts
    """
    out = []
    if not events:
        return out

    cur_b = None
    cur_seen = set()
    cur_list = []  # 保序

    def flush_bucket(bucket_id, items_in_order):
        approx_ts = bucket_id * COMPRESS_HOURS * 3600.0
        for it in items_in_order:
            out.append((approx_ts, it))

    for ts_sec, item in events:
        b = hour_bucket(ts_sec)
        if cur_b is None:
            cur_b = b

        if b != cur_b:
            flush_bucket(cur_b, cur_list)
            cur_b = b
            cur_seen = set()
            cur_list = []

        if item not in cur_seen:
            cur_seen.add(item)
            cur_list.append(item)

    flush_bucket(cur_b, cur_list)
    # out 本身已按 bucket 时间递增，无需再 sort
    return out


def split_bursts(comp_events):
    """
    gap>=CUT_HOURS 切 burst
    burst 内再做一次“全 burst 维度的 item 去重（保留第一次出现）”，得到 pairs（用于二分/滑窗）
    return bursts: list of dict {start, end, pairs, items_set}
      pairs: [(ts, item)] 按时间序列（这里 ts 是小时桶近似时间）
    """
    bursts = []
    if not comp_events:
        return bursts

    cut_sec = CUT_HOURS * 3600.0

    cur_start = comp_events[0][0]
    cur_last = comp_events[0][0]

    cur_pairs = []
    cur_items_set = set()
    cur_seen_in_burst = set()

    ts0, it0 = comp_events[0]
    cur_seen_in_burst.add(it0)
    cur_pairs.append((ts0, it0))
    cur_items_set.add(it0)

    for ts_sec, item in comp_events[1:]:
        if (ts_sec - cur_last) >= cut_sec:
            bursts.append({
                "start": cur_start,
                "end": cur_last,
                "pairs": cur_pairs,
                "items_set": cur_items_set,
            })
            # reset burst
            cur_start = ts_sec
            cur_pairs = []
            cur_items_set = set()
            cur_seen_in_burst = set()

        if item not in cur_seen_in_burst:
            cur_seen_in_burst.add(item)
            cur_pairs.append((ts_sec, item))
            cur_items_set.add(item)

        cur_last = ts_sec

    bursts.append({
        "start": cur_start,
        "end": cur_last,
        "pairs": cur_pairs,
        "items_set": cur_items_set,
    })
    return bursts


def _pairs_to_items(pairs):
    return [it for _, it in pairs]


def _dedup_keep_order(items):
    return list(dict.fromkeys(items))


def gen_aug_segments_from_pairs(pairs, parent_segment_id):
    """
    输入：native burst 的 pairs（时间序列、item 在 burst 内唯一）
    输出：最多 4 个派生段：split2_left, split2_right, window_0, window_1
    同一 parent 内 items 完全一致的派生段会去重丢弃
    """
    n = len(pairs)
    if n < AUG_MIN_ITEMS:
        return []

    out = []
    seen_keys = set()

    def push(seg_type, sub_pairs):
        items = _dedup_keep_order(_pairs_to_items(sub_pairs))
        if len(items) < max(SEG_MIN, AUG_SEG_MIN):
            return
        if len(items) > SEG_MAX:
            return
        key = tuple(items)
        if key in seen_keys:
            return
        seen_keys.add(key)
        out.append({
            "segment_type": seg_type,
            "parent_segment_id": parent_segment_id,
            "start_sec": sub_pairs[0][0],
            "end_sec": sub_pairs[-1][0],
            "items": items,
        })

    # 1) 时间二分 -> 两段
    mid = n // 2
    left = pairs[:mid]
    right = pairs[mid:]
    if left and right:
        push("aug_split2_left", left)
        push("aug_split2_right", right)

    # 2) 滑窗 -> 两段（前窗 + 后窗）
    W = min(AUG_WINDOW_W, n)
    if W >= max(SEG_MIN, AUG_SEG_MIN):
        start0 = 0
        start1 = max(0, n - W)
        w0 = pairs[start0:start0 + W]
        w1 = pairs[start1:start1 + W]
        if w0:
            push("aug_window_0", w0)
        if w1:
            push("aug_window_1", w1)

    return out


def main():
    print("cwd:", os.getcwd())
    print("DATA exists?", os.path.exists(DATA_PATH), DATA_PATH)
    print("ENABLE_AUGMENT:", ENABLE_AUGMENT)

    total_users = 0
    total_lines = 0

    native_kept = 0
    native_candidates = 0
    aug_total = 0
    large_native_count = 0
    aug_count_dist = Counter()  # key=派生段数量

    seg_id = 0  # 全局 segment_id（native + aug 连续递增）

    cur_uid = None
    cur_events = []  # (ts_sec, item)

    # 打开输出文件（流式写，避免内存爆）
    with open(OUT_SEG_JSONL, "w", encoding="utf-8") as fseg, \
         open(OUT_SEG_ITEM_CSV, "w", encoding="utf-8", newline="") as fc:

        writer = csv.writer(fc)
        writer.writerow(["segment_id", "item_id"])

        def write_one_segment(record, items):
            """写 JSONL + 写 CSV (segment_id,item_id)"""
            fseg.write(json.dumps(record, ensure_ascii=False) + "\n")
            for it in items:
                writer.writerow([record["segment_id"], it])

        def flush_user(uid, events):
            nonlocal total_users, native_kept, native_candidates
            nonlocal aug_total, large_native_count, seg_id

            if uid is None or not events:
                return

            total_users += 1

            # Step0: 1小时桶内保序去重
            comp = compress_user_events(events)

            # Step1: 7天 gap 切 burst
            bursts = split_bursts(comp)

            # Step2: burst -> native segment（并可派生 aug）
            for b in bursts:
                pairs = b["pairs"]
                n = len(pairs)

                # native 候选计数（可用于 sanity）
                if n >= SEG_MIN:
                    native_candidates += 1

                if not (SEG_MIN <= n <= SEG_MAX):
                    continue

                items = _dedup_keep_order(_pairs_to_items(pairs))
                if not (SEG_MIN <= len(items) <= SEG_MAX):
                    continue

                # 先拿到 native 的 id
                native_id = seg_id
                seg_id += 1

                # augmentation：基于真实 pairs 派生
                derived = []
                if ENABLE_AUGMENT and n >= AUG_MIN_ITEMS:
                    large_native_count += 1
                    derived = gen_aug_segments_from_pairs(pairs, parent_segment_id=native_id)

                aug_generated_count = len(derived)
                aug_total += aug_generated_count
                aug_count_dist[aug_generated_count] += 1

                # 写 native（原始大片段不动，只记录派生数量）
                native_record = {
                    "segment_id": native_id,
                    "segment_type": "native",
                    "user_id": uid,
                    "start_sec": b["start"],
                    "end_sec": b["end"],
                    "unique_item_count": len(set(items)),
                    "items": items,
                    "aug_generated_count": aug_generated_count,
                }
                write_one_segment(native_record, items)
                native_kept += 1

                # 写 aug（追加）
                for a in derived:
                    a_items = a["items"]
                    a_id = seg_id
                    seg_id += 1
                    a_record = {
                        "segment_id": a_id,
                        "segment_type": a["segment_type"],
                        "parent_segment_id": a["parent_segment_id"],
                        "user_id": uid,
                        "start_sec": a["start_sec"],
                        "end_sec": a["end_sec"],
                        "unique_item_count": len(set(a_items)),
                        "items": a_items,
                    }
                    write_one_segment(a_record, a_items)

        # 读数据（要求按 user_id 排序）
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total_lines += 1

                obj = json.loads(line)
                uid = obj.get("user_id")
                ts = obj.get("timestamp")
                if not uid or ts is None:
                    continue
                item = obj.get("item_id")
                if not item:
                    continue
                ts_sec = ts_to_seconds(ts)

                if cur_uid is None:
                    cur_uid = uid
                if uid != cur_uid:
                    flush_user(cur_uid, cur_events)
                    cur_uid = uid
                    cur_events = []
                cur_events.append((ts_sec, item))

            flush_user(cur_uid, cur_events)

    # 写统计
    stats = {
        "config": {
            "DATA_PATH": DATA_PATH,
            "COMPRESS_HOURS": COMPRESS_HOURS,
            "CUT_HOURS": CUT_HOURS,
            "SEG_MIN": SEG_MIN,
            "SEG_MAX": SEG_MAX,
            "ENABLE_AUGMENT": ENABLE_AUGMENT,
            "AUG_MIN_ITEMS": AUG_MIN_ITEMS,
            "AUG_SEG_MIN": AUG_SEG_MIN,
            "AUG_WINDOW_W": AUG_WINDOW_W,
        },
        "total_users": total_users,
        "total_lines_read": total_lines,
        "native_candidates_ge_segmin": native_candidates,
        "native_kept": native_kept,
        "large_native_count_in_kept": large_native_count,
        "aug_total": aug_total,
        "output_total_segments": native_kept + aug_total,
        "aug_count_distribution_over_native": dict(aug_count_dist),
        "outputs": {
            "segments_jsonl": OUT_SEG_JSONL,
            "segment_item_csv": OUT_SEG_ITEM_CSV,
        }
    }

    with open(OUT_STATS_JSON, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print("users:", f"{total_users:,}")
    print("lines:", f"{total_lines:,}")
    print("native_kept:", f"{native_kept:,}")
    print("aug_total:", f"{aug_total:,}")
    print("total_segments:", f"{native_kept + aug_total:,}")
    print("aug_count_dist (over native):", dict(aug_count_dist))
    print("written:", OUT_SEG_JSONL)
    print("written:", OUT_SEG_ITEM_CSV)
    print("written:", OUT_STATS_JSON)


if __name__ == "__main__":
    main()
