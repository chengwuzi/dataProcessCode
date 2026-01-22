# -*- coding: utf-8 -*-
import json
import os
import random
from collections import Counter

# ================== 你只改这里 ==================
# DATA_PATH = r"Sports_and_Outdoors_2019_2021_sorted_min10.jsonl"
DATA_PATH = r"../dataProcess/Electronics_2019_2021_sorted_min15.jsonl"

# META_PATH = r"meta_Sports_and_Outdoors.jsonl"
META_PATH = r"../dataProcess/meta_Electronics.jsonl"
# META_PATH = None

OUTPUT_SAMPLE_PATH = r"segment_sample_7d_native_300.jsonl"
# ===============================================

CATEGORY_MODE = "L3"   # 可选: "L2", "L3", "LEAF", "MAIN"

SEED = 2026
SAMPLE_NATIVE = 300

COMPRESS_HOURS = 1           # 1小时压缩去重
CUT_HOURS = 168              # 7天切 burst
SEG_MIN = 2
SEG_MAX = 30

PRINT_FIRST_N = 0            # 这份脚本不需要打印详情，留0即可


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
    """1小时桶内去重 item"""
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
            for it in cur_items:
                out.append((approx_ts, it))
            cur_b = b
            cur_items = {item}

    approx_ts = cur_b * COMPRESS_HOURS * 3600.0
    for it in cur_items:
        out.append((approx_ts, it))

    out.sort(key=lambda x: x[0])
    return out


def split_bursts(comp_events):
    """gap>=CUT_HOURS 切 burst"""
    bursts = []
    if not comp_events:
        return bursts

    cut_sec = CUT_HOURS * 3600.0

    cur_start = comp_events[0][0]
    cur_last = comp_events[0][0]
    cur_items = set([comp_events[0][1]])

    for ts_sec, item in comp_events[1:]:
        if (ts_sec - cur_last) >= cut_sec:
            bursts.append({"start": cur_start, "end": cur_last, "items": cur_items})
            cur_start = ts_sec
            cur_items = set()
        cur_items.add(item)
        cur_last = ts_sec

    bursts.append({"start": cur_start, "end": cur_last, "items": cur_items})
    return bursts


def load_meta_categories(meta_path, needed_items):
    """
    只为 needed_items 建映射：item_id -> category(str or None)
    """
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
            # categories 可能是 list[list[str]]
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
            items = b["items"]
            n = len(items)

            if SEG_MIN <= n <= SEG_MAX:
                seen_native += 1
                seg = {
                    "segment_type": "native",
                    "user_id": uid,
                    "start": b["start"],
                    "end": b["end"],
                    "items": list(items),
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
    print(f"  写入：{OUTPUT_SAMPLE_PATH}")

    # meta 映射
    all_items = set()
    for s in native_res:
        for it in s["items"]:
            all_items.add(it)

    item2cat = load_meta_categories(META_PATH, all_items) if META_PATH else {}

    # 统计分布
    size_list = []
    cat_cnt_vals = []

    for s in native_res:
        items = list(dict.fromkeys(s["items"]))
        size_list.append(len(set(items)))

        if item2cat:
            cats = [item2cat.get(it) for it in items]
            cats = [c for c in cats if c]
            cat_cnt_vals.append(len(set(cats)))

    print("\n【Sample 段大小分布】")
    print(describe_dist(size_list, "unique item 数"))

    if item2cat:
        print(f"\n【Sample 类目数分布（CATEGORY_MODE={CATEGORY_MODE}）】")
        print(describe_dist(cat_cnt_vals, "类目数"))
        for t in [1, 2, 3, 4, 5]:
            c = sum(1 for x in cat_cnt_vals if x <= t)
            print(f"  - 类目数 <= {t}：{c}/{len(cat_cnt_vals)} ({c/len(cat_cnt_vals)*100:.2f}%)")
    else:
        print("\n未提供 meta：跳过类目统计。")

    # 写出 sample 文件
    with open(OUTPUT_SAMPLE_PATH, "w", encoding="utf-8") as out:
        for idx, s in enumerate(native_res):
            items = list(dict.fromkeys(s["items"]))
            record = {
                "sample_id": idx,
                "segment_type": "native",
                "user_id": s["user_id"],
                "start_sec": s["start"],
                "end_sec": s["end"],
                "unique_item_count": len(set(items)),
                "items": items[:50],
            }
            if item2cat:
                cats = [item2cat.get(it) for it in items]
                record["categories"] = cats[:50]
                cc = Counter([c for c in cats if c])
                record["category_counter"] = dict(cc)
                record["category_count"] = len(cc)

            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\n完成。")


if __name__ == "__main__":
    main()
