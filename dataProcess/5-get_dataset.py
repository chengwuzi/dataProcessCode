# -*- coding: utf-8 -*-
import json
import os
from collections import defaultdict, Counter

# ================== 你只改这里 ==================
DATA_PATH = r"Sports_2019_2021_sorted_min10_canon.jsonl"

OUT_SEGMENTS_JSONL = r"segments_all_with_aug.jsonl"
OUT_TRAIN_TXT      = r"train.txt"
OUT_TEST_TXT       = r"test.txt"
OUT_ITEM2INT_JSON  = r"item2int.json"
OUT_STATS_JSON     = r"stats.json"
# ===============================================

# ===== 切片逻辑参数 =====
COMPRESS_HOURS = 1           # 1小时桶内去重 item（保序）
CUT_HOURS = 168              # gap>=7天 切 burst
SEG_MIN = 2
SEG_MAX = 30

# ===== augmentation（只对“大段”做：保留原段不动，再派生小段）=====
AUG_MIN_ITEMS = 12           # native 段 item数 >=12 触发
AUG_SEG_MIN = 4              # 派生子段最小 item 数
AUG_WINDOW_W = 8             # 滑窗窗口长度（前窗 + 后窗）
# ================================================================

# ===== test 构造方式（只当仪表盘，不从 train 扣除！）=====
TEST_PICK_MODE = "last"   # "last" or "random"
SEED = 2026
# ========================

# ===== 你要的门槛：过滤后 items 至少 3 个 =====
MIN_ITEMS_FOR_SEG = 3

# ===== 过滤超热门 item =====
FILTER_HOT_ITEMS = True
# “超热门”的定义：出现在多少个 segment 以上就删（按 segment 频次，不是交互频次）
# 建议先用 0.005~0.02 试；越小删得越狠。比如 0.01 表示出现在 1% segment 以上就删。
HOT_ITEM_SEG_RATIO = 0.01
# 或者你更想用固定阈值：把下面设成一个整数（例如 5000），则优先用它
HOT_ITEM_SEG_ABS = None
# ========================


def ts_to_seconds(ts):
    ts = int(ts)
    if ts > 10_000_000_000:  # ms
        return ts / 1000.0
    return float(ts)


def hour_bucket(ts_sec):
    return int(ts_sec // (COMPRESS_HOURS * 3600))


def compress_user_events(events):
    """
    1小时桶内去重 item（保序：同一桶内只保留第一次出现）
    events: list of (ts_sec, item_id) sorted by time
    return: list of (approx_ts, item_id) (按桶时间递增)
    """
    out = []
    if not events:
        return out

    cur_b = None
    cur_seen = set()
    cur_list = []

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
    return out


def split_bursts(comp_events):
    """
    gap>=CUT_HOURS 切 burst
    burst 内 pairs：按时间顺序、每个 item 只保留第一次出现（用于二分/滑窗）
    """
    bursts = []
    if not comp_events:
        return bursts

    cut_sec = CUT_HOURS * 3600.0

    cur_start = comp_events[0][0]
    cur_last = comp_events[0][0]
    cur_pairs = []
    seen = set()

    ts0, it0 = comp_events[0]
    cur_pairs.append((ts0, it0))
    seen.add(it0)

    for ts_sec, item in comp_events[1:]:
        if (ts_sec - cur_last) >= cut_sec:
            bursts.append({"start": cur_start, "end": cur_last, "pairs": cur_pairs})
            cur_start = ts_sec
            cur_pairs = []
            seen = set()

        if item not in seen:
            cur_pairs.append((ts_sec, item))
            seen.add(item)

        cur_last = ts_sec

    bursts.append({"start": cur_start, "end": cur_last, "pairs": cur_pairs})
    return bursts


def dedup_keep_order(items):
    return list(dict.fromkeys(items))


def pairs_to_items(pairs):
    return [it for _, it in pairs]


def gen_aug_segments_from_pairs(pairs, parent_segment_id):
    """
    从 pairs 派生：
      - 时间二分：left/right
      - 滑窗：前窗/后窗
    同一 parent 内 items 完全一致的派生段会去重
    """
    n = len(pairs)
    if n < AUG_MIN_ITEMS:
        return []

    out = []
    seen_keys = set()

    def push(seg_type, sub_pairs):
        items = dedup_keep_order(pairs_to_items(sub_pairs))
        if len(items) < AUG_SEG_MIN:
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

    mid = n // 2
    left = pairs[:mid]
    right = pairs[mid:]
    if left and right:
        push("aug_split2_left", left)
        push("aug_split2_right", right)

    W = min(AUG_WINDOW_W, n)
    if W >= AUG_SEG_MIN:
        w0 = pairs[0:W]
        w1 = pairs[n - W:n]
        if w0:
            push("aug_window_0", w0)
        if w1:
            push("aug_window_1", w1)

    return out


class ItemIndexer:
    def __init__(self):
        self.item2int = {}
        self.next_id = 1  # 从1开始，0留给padding（很多实现习惯）

    def get(self, item_id: str) -> int:
        v = self.item2int.get(item_id)
        if v is None:
            v = self.next_id
            self.item2int[item_id] = v
            self.next_id += 1
        return v


def pick_test_item(items, rng):
    """只挑 test，不从 train 扣除"""
    assert len(items) >= 1
    if TEST_PICK_MODE == "last":
        return items[-1]
    else:
        return items[rng.randrange(len(items))]


def iter_segments_from_file(data_path):
    """
    第一遍：只生成 segment（native + aug），用于统计每个 item 出现在多少个 segment（segment-frequency）
    产出元素：dict {user_id, seg_type, start_sec, end_sec, items, parent_segment_id?, aug_generated_count?}
    注意：这里 segment_id 先不分配，等第二遍真正写出时再分配。
    """
    cur_uid = None
    cur_events = []

    def flush_user(uid, events):
        if uid is None or not events:
            return
        comp = compress_user_events(events)
        bursts = split_bursts(comp)
        for b in bursts:
            pairs = b["pairs"]
            items = pairs_to_items(pairs)
            n = len(items)
            if not (SEG_MIN <= n <= SEG_MAX):
                continue
            items = dedup_keep_order(items)
            if not (SEG_MIN <= len(items) <= SEG_MAX):
                continue

            # native 先产出（parent_segment_id 暂时用 None；第二遍会用真实 native_id）
            yield {
                "segment_type": "native",
                "user_id": uid,
                "start_sec": b["start"],
                "end_sec": b["end"],
                "items": items,
                "pairs": pairs,  # 留给第二遍生成 aug 用
            }

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
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
                yield from flush_user(cur_uid, cur_events) or ()
                cur_uid = uid
                cur_events = []
            cur_events.append((ts_sec, item))

        yield from flush_user(cur_uid, cur_events) or ()


def main():
    print("cwd:", os.getcwd())
    print("DATA exists?", os.path.exists(DATA_PATH), DATA_PATH)

    assert AUG_MIN_ITEMS >= MIN_ITEMS_FOR_SEG, \
        f"AUG_MIN_ITEMS({AUG_MIN_ITEMS}) must be >= MIN_ITEMS_FOR_SEG({MIN_ITEMS_FOR_SEG})"

    # ====== 第一遍：统计 item 的 segment-frequency，找超热门 ======
    hot_items = set()
    seg_freq = Counter()
    native_cnt_firstpass = 0

    if FILTER_HOT_ITEMS:
        print("[PASS1] scanning segments to compute segment-frequency for hot-item filtering...")
        for seg in iter_segments_from_file(DATA_PATH):
            native_cnt_firstpass += 1
            # 先按 native 的 items 统计频次（更稳定，aug 会放大热门的影响）
            for it in set(seg["items"]):
                seg_freq[it] += 1

        total_native = native_cnt_firstpass
        if HOT_ITEM_SEG_ABS is not None:
            thr = int(HOT_ITEM_SEG_ABS)
        else:
            thr = max(1, int(total_native * HOT_ITEM_SEG_RATIO))

        hot_items = {it for it, c in seg_freq.items() if c >= thr}
        print(f"[PASS1] native segments scanned: {total_native:,}")
        print(f"[PASS1] HOT threshold: seg_freq >= {thr} "
              f"({'ABS' if HOT_ITEM_SEG_ABS is not None else f'ratio={HOT_ITEM_SEG_RATIO}'})")
        print(f"[PASS1] hot items count: {len(hot_items):,}")
    else:
        print("[PASS1] skipped (FILTER_HOT_ITEMS=False)")

    # ====== 第二遍：正式写出 segments/train/test/item2int/stats ======
    stats = {
        "config": {
            "DATA_PATH": DATA_PATH,
            "COMPRESS_HOURS": COMPRESS_HOURS,
            "CUT_HOURS": CUT_HOURS,
            "SEG_MIN": SEG_MIN,
            "SEG_MAX": SEG_MAX,
            "AUG_MIN_ITEMS": AUG_MIN_ITEMS,
            "AUG_SEG_MIN": AUG_SEG_MIN,
            "AUG_WINDOW_W": AUG_WINDOW_W,
            "MIN_ITEMS_FOR_SEG": MIN_ITEMS_FOR_SEG,
            "TEST_PICK_MODE": TEST_PICK_MODE,
            "SEED": SEED,
            "FILTER_HOT_ITEMS": FILTER_HOT_ITEMS,
            "HOT_ITEM_SEG_RATIO": HOT_ITEM_SEG_RATIO,
            "HOT_ITEM_SEG_ABS": HOT_ITEM_SEG_ABS,
        },
        "users_seen": 0,
        "bursts_total": 0,
        "native_segments_kept": 0,
        "aug_segments_kept": 0,
        "segments_total": 0,
        "items_unique_total": 0,
        "native_large_triggered": 0,
        "aug_count_hist": defaultdict(int),

        "emit_ok_total": 0,
        "emit_skip_after_hot_filter_too_short": 0,

        "hot_items_count": len(hot_items),
    }

    indexer = ItemIndexer()

    rng = None
    if TEST_PICK_MODE == "random":
        import random
        rng = random.Random(SEED)

    seg_out = open(OUT_SEGMENTS_JSONL, "w", encoding="utf-8")
    train_out = open(OUT_TRAIN_TXT, "w", encoding="utf-8")
    test_out = open(OUT_TEST_TXT, "w", encoding="utf-8")

    segment_id = 0

    cur_uid = None
    cur_events = []

    def emit_one(segment_type, user_id, start_sec, end_sec, items,
                 parent_segment_id=None, aug_generated_count=None):
        nonlocal segment_id

        # 去重保序
        items = dedup_keep_order(items)

        # 过滤热门
        if FILTER_HOT_ITEMS and hot_items:
            items = [it for it in items if it not in hot_items]

        # 过滤后门槛
        if len(items) < MIN_ITEMS_FOR_SEG:
            stats["emit_skip_after_hot_filter_too_short"] += 1
            return False, None

        # train 喂满：train = items 全部（映射成 int）
        train_ints = [indexer.get(it) for it in items]

        # test 只当仪表盘：挑一个 item（映射成 int），允许与 train 重叠
        test_item = pick_test_item(items, rng=rng) if TEST_PICK_MODE == "random" else items[-1]
        test_int = indexer.get(test_item)

        rec = {
            "segment_id": segment_id,
            "user_id": user_id,
            "segment_type": segment_type,
            "start_sec": float(start_sec),
            "end_sec": float(end_sec),
            "items": items,
        }
        if parent_segment_id is not None:
            rec["parent_segment_id"] = parent_segment_id
        if aug_generated_count is not None:
            rec["aug_generated_count"] = aug_generated_count

        seg_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        train_out.write(f"{segment_id}: " + " ".join(map(str, train_ints)) + "\n")
        test_out.write(f"{segment_id}: {test_int}\n")

        cur_id = segment_id
        segment_id += 1
        stats["emit_ok_total"] += 1
        return True, cur_id

    def flush_user(uid, events):
        if uid is None or not events:
            return

        stats["users_seen"] += 1

        comp = compress_user_events(events)
        bursts = split_bursts(comp)
        stats["bursts_total"] += len(bursts)

        for b in bursts:
            pairs = b["pairs"]
            items = pairs_to_items(pairs)
            n = len(items)
            if not (SEG_MIN <= n <= SEG_MAX):
                continue

            # 先“尝试”写 native，拿到真实 native_id（鲁棒：只有写成功才有 native_id）
            # 先根据 native 的 pairs 生成 aug_list（需要 parent_segment_id = native_id，所以得先拿到 native_id）
            # 做法：先不生成 aug；先 emit native；成功后再生成 aug
            # 但又要把 aug_generated_count 写进 native：所以先“预计算”派生数量（基于原 pairs），写到 native 里即可
            aug_preview = []
            if n >= AUG_MIN_ITEMS:
                stats["native_large_triggered"] += 1
                # 这里先用占位 parent=-1 预览数量，真正写 aug 时会替换成真实 native_id
                aug_preview = gen_aug_segments_from_pairs(pairs, parent_segment_id=-1)

            ok_native, native_id = emit_one(
                segment_type="native",
                user_id=uid,
                start_sec=b["start"],
                end_sec=b["end"],
                items=items,
                parent_segment_id=None,
                aug_generated_count=len(aug_preview),
            )
            if not ok_native:
                continue
            stats["native_segments_kept"] += 1

            # native 写成功后，再生成/写 aug（parent_segment_id 用真实 native_id）
            if n >= AUG_MIN_ITEMS:
                aug_list = gen_aug_segments_from_pairs(pairs, parent_segment_id=native_id)
                if aug_list:
                    stats["aug_count_hist"][len(aug_list)] += 1
                    for a in aug_list:
                        ok_aug, _ = emit_one(
                            segment_type=a["segment_type"],
                            user_id=uid,
                            start_sec=a["start_sec"],
                            end_sec=a["end_sec"],
                            items=a["items"],
                            parent_segment_id=a["parent_segment_id"],
                            aug_generated_count=None,
                        )
                        if ok_aug:
                            stats["aug_segments_kept"] += 1

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

    seg_out.close()
    train_out.close()
    test_out.close()

    with open(OUT_ITEM2INT_JSON, "w", encoding="utf-8") as f:
        json.dump(indexer.item2int, f, ensure_ascii=False)

    stats["segments_total"] = segment_id
    stats["items_unique_total"] = len(indexer.item2int)
    stats["aug_count_hist"] = dict(stats["aug_count_hist"])

    with open(OUT_STATS_JSON, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print("segments:", OUT_SEGMENTS_JSONL)
    print("train   :", OUT_TRAIN_TXT)
    print("test    :", OUT_TEST_TXT)
    print("item2int:", OUT_ITEM2INT_JSON)
    print("stats   :", OUT_STATS_JSON)
    print(f"Total segments: {stats['segments_total']}")
    print(f"Native kept   : {stats['native_segments_kept']}, Aug kept: {stats['aug_segments_kept']}")
    print(f"Unique items  : {stats['items_unique_total']}")
    if FILTER_HOT_ITEMS:
        print(f"Hot items filtered: {stats['hot_items_count']}")


if __name__ == "__main__":
    main()
