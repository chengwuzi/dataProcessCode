# -*- coding: utf-8 -*-
import json
import os
import random
from collections import Counter, defaultdict

# ================== 你只改这里 ==================
DATA_PATH = r"../dataProcess/Electronics_2019_2021_sorted_min15.jsonl"
META_PATH = r"../dataProcess/meta_Electronics.jsonl"  # 不想统计类目就填 None
OUTPUT_SAMPLE_PATH = r"segment_sample_7d_native_300_with_aug.jsonl"
OUTPUT_AUG_STATS_PATH = r"segment_sample_7d_native_300_with_aug_stats.json"  # 统计信息单独写一份
# ===============================================

CATEGORY_MODE = "L3"   # 可选: "L2", "L3", "LEAF", "MAIN"

SEED = 2026
SAMPLE_NATIVE = 300

COMPRESS_HOURS = 1           # 1小时桶内去重 item
CUT_HOURS = 168              # gap>=7天 切 burst
SEG_MIN = 2
SEG_MAX = 30

# ===== augmentation（只对“大段”做：保留原段不动，再派生小段）=====
AUG_MIN_ITEMS = 12           # 触发阈值：>=12 的大片段
AUG_SEG_MIN = 4              # 派生子段最小 item 数（建议 >=4 稳一点）
AUG_WINDOW_W = 8             # 滑窗窗口长度（实际用 min(AUG_WINDOW_W, n)）
# 生成策略：时间二分 2 段 + 滑窗 2 段（尽量前窗 + 后窗）
# ================================================================


def ts_to_seconds(ts):
    ts = int(ts)
    if ts > 10_000_000_000:  # ms
        return ts / 1000.0
    return float(ts)


def hour_bucket(ts_sec):
    return int(ts_sec // (COMPRESS_HOURS * 3600))


def reservoir_push(reservoir, item, k, seen_count, rng):
    """标准 reservoir sampling"""
    if len(reservoir) < k:
        reservoir.append(item)
    else:
        j = rng.randint(0, seen_count - 1)
        if j < k:
            reservoir[j] = item


def compress_user_events(events):
    """
    1小时桶内去重 item
    注意：同一小时桶里 items 用 sorted 保证稳定顺序（否则 set 会乱序）
    events: list of (ts_sec, item_id) sorted by time
    return: list of (ts_sec_approx, item_id) sorted by time
    """
    out = []
    if not events:
        return out

    cur_b = None
    cur_items = set()
    for ts_sec, item in events:
        b = hour_bucket(ts_sec)
        if cur_b is None:
            cur_b = b
            cur_items.add(item)
            continue

        if b == cur_b:
            cur_items.add(item)
        else:
            approx_ts = cur_b * COMPRESS_HOURS * 3600.0
            for it in sorted(cur_items):
                out.append((approx_ts, it))
            cur_b = b
            cur_items = {item}

    approx_ts = cur_b * COMPRESS_HOURS * 3600.0
    for it in sorted(cur_items):
        out.append((approx_ts, it))

    out.sort(key=lambda x: x[0])
    return out


def split_bursts(comp_events):
    """
    comp_events: list of (ts_sec, item_id) sorted
    return bursts: list of dict
      {
        "start": float,
        "end": float,
        "pairs": list[(ts, item)]  # burst 内按时间，且同一 item 只保留第一次出现（用于可切分）
        "items_set": set(item)     # 便于统计
      }
    """
    bursts = []
    if not comp_events:
        return bursts

    cut_sec = CUT_HOURS * 3600.0

    cur_pairs = []
    cur_items_set = set()
    cur_seen_in_burst = set()

    cur_start = comp_events[0][0]
    cur_last = comp_events[0][0]

    # init first
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
            # reset
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


def load_meta_categories(meta_path, needed_items):
    """只为 needed_items 建映射：item_id -> category(str or None)"""
    if not meta_path:
        return {}

    needed = set(needed_items)
    mapping = {}
    if not needed:
        return mapping

    def extract_cat(obj, mode="L2"):
        cats = obj.get("categories")

        def normalize_list(lst):
            if not isinstance(lst, list) or len(lst) == 0:
                return None
            if isinstance(lst[0], list):
                lst = lst[0]
                if not isinstance(lst, list) or len(lst) == 0:
                    return None
            return lst

        lst = normalize_list(cats)

        def pick(idx):
            if not lst:
                return None
            if idx == "LEAF":
                v = lst[-1] if len(lst) > 0 else None
            else:
                if len(lst) <= idx:
                    return None
                v = lst[idx]
            return v.strip() if isinstance(v, str) and v.strip() else None

        if mode == "L2":
            v = pick(1)
            if v: return v
        elif mode == "L3":
            v = pick(2)
            if v: return v
        elif mode == "LEAF":
            v = pick("LEAF")
            if v: return v

        mc = obj.get("main_category")
        if isinstance(mc, str) and mc.strip():
            mc = mc.strip()
            if mc.lower().startswith("all "):
                mc = mc[4:].strip()
            return mc
        return None

    found = 0
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            item = obj.get("parent_asin") or obj.get("asin")
            if not item or item not in needed:
                continue

            mapping[item] = extract_cat(obj, mode=CATEGORY_MODE)

            found += 1
            if found >= len(needed):
                break

    return mapping


def describe_dist(values, name):
    if not values:
        return f"{name}: empty"
    v = sorted(values)
    n = len(v)

    def pct(p):
        idx = int(round((p / 100) * (n - 1)))
        idx = max(0, min(n - 1, idx))
        return v[idx]

    return (f"{name}: n={n}, mean={sum(v)/n:.3f}, "
            f"P50={pct(50):.3f}, P90={pct(90):.3f}, P95={pct(95):.3f}, MAX={v[-1]:.3f}")


def _pairs_to_items(pairs):
    return [it for _, it in pairs]


def _dedup_keep_order(items):
    return list(dict.fromkeys(items))


def gen_aug_segments_from_pairs(pairs, parent_sample_id):
    """
    输入：burst 的 pairs（时间序列，item 唯一）
    输出：最多 4 个派生段：split2_left, split2_right, window_0, window_1
    去重：同一 parent 内如果生成出 items 完全一致的段则丢掉
    """
    n = len(pairs)
    if n < AUG_MIN_ITEMS:
        return []

    out = []
    seen_keys = set()

    def push(seg_type, sub_pairs):
        items = _dedup_keep_order(_pairs_to_items(sub_pairs))
        if len(items) < AUG_SEG_MIN:
            return
        key = tuple(items)
        if key in seen_keys:
            return
        seen_keys.add(key)
        out.append({
            "segment_type": seg_type,
            "parent_sample_id": parent_sample_id,
            "start_sec": sub_pairs[0][0],
            "end_sec": sub_pairs[-1][0],
            "items": items,
        })

    # 1) 时间二分
    mid = n // 2
    left = pairs[:mid]
    right = pairs[mid:]
    if left and right:
        push("aug_split2_left", left)
        push("aug_split2_right", right)

    # 2) 滑窗两段：尽量前窗 + 后窗
    W = min(AUG_WINDOW_W, n)
    if W >= AUG_SEG_MIN:
        start0 = 0
        start1 = max(0, n - W)
        # 若 n==W，则 start0==start1，只能生成 1 段（这种情况 n<=8，不会触发 AUG_MIN_ITEMS=12）
        if start1 == start0:
            start1 = min(n - W, W // 2)  # 理论上不会走到
        w0 = pairs[start0:start0 + W]
        w1 = pairs[start1:start1 + W]
        if w0:
            push("aug_window_0", w0)
        if w1:
            push("aug_window_1", w1)

    return out


def main():
    rng = random.Random(SEED)

    print("当前工作目录：", os.getcwd())
    print("数据文件存在？", os.path.exists(DATA_PATH), DATA_PATH)
    if META_PATH:
        print("meta 文件存在？", os.path.exists(META_PATH), META_PATH)
    else:
        print("meta 未提供：将跳过类目统计")

    native_res = []
    seen_native = 0

    cur_uid = None
    cur_events = []  # (ts_sec, item)

    def flush_user(uid, events):
        nonlocal seen_native
        if uid is None or not events:
            return

        comp = compress_user_events(events)
        bursts = split_bursts(comp)

        for b in bursts:
            pairs = b["pairs"]
            n = len(pairs)  # burst 内 unique item 数（按时间序列）
            if SEG_MIN <= n <= SEG_MAX:
                seen_native += 1
                seg = {
                    "segment_type": "native",
                    "user_id": uid,
                    "start": b["start"],
                    "end": b["end"],
                    "pairs": pairs,  # 存一下，后面派生用
                    "items": _pairs_to_items(pairs),
                }
                reservoir_push(native_res, seg, SAMPLE_NATIVE, seen_native, rng)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            uid = obj.get("user_id")
            ts = obj.get("timestamp")
            if not uid or ts is None:
                continue
            item = obj.get("parent_asin") or obj.get("asin")
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

    rng.shuffle(native_res)

    print("\n采样结果：")
    print(f"  native 段：采到 {len(native_res)} / 目标 {SAMPLE_NATIVE}；总候选数约 {seen_native:,}")

    # ===== 对采样到的 native 段做 augmentation（只对大片段，且“原始大片段不动”）=====
    aug_segments = []
    aug_count_per_parent = {}
    large_parent_count = 0

    for parent_id, s in enumerate(native_res):
        pairs = s["pairs"]
        n = len(pairs)
        if n >= AUG_MIN_ITEMS:
            large_parent_count += 1
            derived = gen_aug_segments_from_pairs(pairs, parent_sample_id=parent_id)
            aug_segments.extend(derived)
            aug_count_per_parent[parent_id] = len(derived)
        else:
            aug_count_per_parent[parent_id] = 0

    print("\nAugmentation 统计：")
    print(f"  触发阈值 AUG_MIN_ITEMS={AUG_MIN_ITEMS}")
    print(f"  采样的 300 段中，大片段（>= {AUG_MIN_ITEMS}）数量：{large_parent_count}")
    print(f"  从大片段派生出来的总段数：{len(aug_segments)}")
    if large_parent_count > 0:
        dist = list(aug_count_per_parent.values())
        print("  每个 native 段派生段数量分布：")
        print("   ", describe_dist(dist, "派生段数"))

    # ===== meta 映射（统计类目用）=====
    all_items = set()
    for s in native_res:
        for it in s["items"]:
            all_items.add(it)
    for a in aug_segments:
        for it in a["items"]:
            all_items.add(it)

    item2cat = load_meta_categories(META_PATH, all_items) if META_PATH else {}

    # ===== 写输出：先写 300 原生段，再写派生段（总行数 = 300 + 派生数）=====
    all_records = []

    # 1) native records
    for idx, s in enumerate(native_res):
        items = _dedup_keep_order(s["items"])
        record = {
            "sample_id": idx,
            "segment_type": "native",
            "user_id": s["user_id"],
            "start_sec": s["start"],
            "end_sec": s["end"],
            "unique_item_count": len(set(items)),
            "items": items[:50],
            "aug_generated_count": aug_count_per_parent.get(idx, 0),  # 你要的“派生数量记录”
        }
        if item2cat:
            cats = [item2cat.get(it) for it in items]
            record["categories"] = cats[:50]
            cc = Counter([c for c in cats if c])
            record["category_counter"] = dict(cc)
            record["category_count"] = len(cc)
        all_records.append(record)

    # 2) augmented records (append)
    base_id = len(all_records)
    for j, a in enumerate(aug_segments):
        items = _dedup_keep_order(a["items"])
        record = {
            "sample_id": base_id + j,
            "segment_type": a["segment_type"],
            "parent_sample_id": a["parent_sample_id"],
            "user_id": native_res[a["parent_sample_id"]]["user_id"],
            "start_sec": a["start_sec"],
            "end_sec": a["end_sec"],
            "unique_item_count": len(set(items)),
            "items": items[:50],
        }
        if item2cat:
            cats = [item2cat.get(it) for it in items]
            record["categories"] = cats[:50]
            cc = Counter([c for c in cats if c])
            record["category_counter"] = dict(cc)
            record["category_count"] = len(cc)
        all_records.append(record)

    with open(OUTPUT_SAMPLE_PATH, "w", encoding="utf-8") as out:
        for r in all_records:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ===== 额外写一份统计 JSON，方便你留档/画图 =====
    stats_obj = {
        "config": {
            "DATA_PATH": DATA_PATH,
            "META_PATH": META_PATH,
            "SAMPLE_NATIVE": SAMPLE_NATIVE,
            "COMPRESS_HOURS": COMPRESS_HOURS,
            "CUT_HOURS": CUT_HOURS,
            "SEG_MIN": SEG_MIN,
            "SEG_MAX": SEG_MAX,
            "AUG_MIN_ITEMS": AUG_MIN_ITEMS,
            "AUG_SEG_MIN": AUG_SEG_MIN,
            "AUG_WINDOW_W": AUG_WINDOW_W,
            "CATEGORY_MODE": CATEGORY_MODE,
            "SEED": SEED,
        },
        "native_sample_count": len(native_res),
        "native_candidate_total": seen_native,
        "large_native_count_in_sample": large_parent_count,
        "aug_total": len(aug_segments),
        "aug_count_per_native_sample": aug_count_per_parent,  # key=parent_sample_id
        "output_total_records": len(all_records),
    }
    with open(OUTPUT_AUG_STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats_obj, f, ensure_ascii=False, indent=2)

    print("\n写出完成：")
    print(f"  输出样本（含派生段）：{OUTPUT_SAMPLE_PATH}")
    print(f"  统计信息：{OUTPUT_AUG_STATS_PATH}")
    print(f"  总行数：{len(all_records)} = {len(native_res)}(native) + {len(aug_segments)}(aug)")


if __name__ == "__main__":
    main()
