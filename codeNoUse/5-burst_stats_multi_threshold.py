# -*- coding: utf-8 -*-
import json
import math
from collections import defaultdict

INPUT_PATH = r"../dataProcess/Electronics_2019_2021_sorted_min15.jsonl"

# 多个 gap 阈值（小时）用来对比
THRESHOLDS_H = [24, 72, 168, 336]  # 1天/3天/7天/14天

# 压缩粒度：1小时
COMPRESS_HOURS = 1

# segment 目标大小区间（你后续要训练用）
SEG_MIN = 2
SEG_MAX = 30


def ts_to_seconds(ts):
    ts = int(ts)
    if ts > 10_000_000_000:  # ms
        return ts / 1000.0
    return float(ts)


def percentile(arr_sorted, p):
    if not arr_sorted:
        return 0
    n = len(arr_sorted)
    idx = int(round((p / 100) * (n - 1)))
    idx = max(0, min(n - 1, idx))
    return arr_sorted[idx]


def summarize_dist(name, values):
    if not values:
        return f"{name}: (empty)"
    v = sorted(values)
    return (f"{name}: n={len(v):,}, "
            f"P50={percentile(v,50):.3f}, P90={percentile(v,90):.3f}, "
            f"P95={percentile(v,95):.3f}, P99={percentile(v,99):.3f}, MAX={v[-1]:.3f}")


def main():
    # 先把每个用户的交互读成 (ts_sec, item_id) 列表（按文件顺序已排序）
    # 注意：你当前文件规模 54万行，按用户聚合完全没问题
    user_events = defaultdict(list)

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
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
            user_events[uid].append((ts_sec, item))

    users = list(user_events.keys())
    print(f"用户数：{len(users):,}")
    print(f"总交互行数（有效uid+ts+item）：{sum(len(v) for v in user_events.values()):,}\n")

    # 针对每个阈值统计 burst 特性
    for T in THRESHOLDS_H:
        T_sec = T * 3600.0

        burst_count_per_user = []
        burst_unique_items = []
        burst_duration_hours = []
        burst_raw_len = []
        burst_compressed_len = []

        # 用于判断 burst 是否适合直接当 segment
        burst_in_range = 0
        burst_too_small = 0
        burst_too_large = 0
        total_bursts = 0

        for uid, evs in user_events.items():
            # evs 已按时间升序（因为输入文件按用户时间排了）
            # Step A: 1小时压缩：同一小时内去重 item
            compressed = []
            last_bucket = None
            bucket_items = set()

            for ts_sec, item in evs:
                bucket = int(ts_sec // (COMPRESS_HOURS * 3600))
                if last_bucket is None:
                    last_bucket = bucket
                    bucket_items = {item}
                    continue
                if bucket == last_bucket:
                    bucket_items.add(item)
                else:
                    # flush 前一个桶
                    # 这里把桶里 items 以任意顺序展开（只关心集合统计）
                    for it in bucket_items:
                        compressed.append((last_bucket, it))
                    last_bucket = bucket
                    bucket_items = {item}

            if last_bucket is not None:
                for it in bucket_items:
                    compressed.append((last_bucket, it))

            # 把压缩后的 bucket 还原成近似时间（桶编号 * 小时）
            # 只为了计算 gap / duration
            comp_events = [(b * COMPRESS_HOURS * 3600.0, it) for (b, it) in compressed]
            if not comp_events:
                continue

            # Step B: 以阈值 T 切 burst
            bursts = []
            cur_items = []
            cur_start = comp_events[0][0]
            cur_last = comp_events[0][0]

            for ts_sec, item in comp_events:
                if (ts_sec - cur_last) >= T_sec and cur_items:
                    bursts.append((cur_start, cur_last, cur_items))
                    cur_items = []
                    cur_start = ts_sec
                cur_items.append((ts_sec, item))
                cur_last = ts_sec

            if cur_items:
                bursts.append((cur_start, cur_last, cur_items))

            burst_count_per_user.append(len(bursts))

            # 统计每个 burst
            for bstart, blast, items in bursts:
                total_bursts += 1

                raw_len = len(items)  # 压缩后“事件数”（已经去掉同小时重复）
                uniq = len(set(it for _, it in items))
                dur_h = (blast - bstart) / 3600.0

                burst_raw_len.append(raw_len)
                burst_unique_items.append(uniq)
                burst_duration_hours.append(dur_h)

                # 估计压缩效果：用原始用户交互数作为对比（粗略，不严格对齐burst）
                # 这里不强求严格映射，只看整体分布即可
                # burst_compressed_len 用 uniq 更直观
                burst_compressed_len.append(uniq)

                if uniq < SEG_MIN:
                    burst_too_small += 1
                elif uniq > SEG_MAX:
                    burst_too_large += 1
                else:
                    burst_in_range += 1

        print("======================================================")
        print(f"切分阈值：gap >= {T} 小时 作为新 burst（压缩粒度：{COMPRESS_HOURS} 小时）")
        print(summarize_dist("每用户 burst 数", burst_count_per_user))
        print(summarize_dist("burst unique item 数", burst_unique_items))
        print(summarize_dist("burst 持续时间(小时)", burst_duration_hours))
        print(summarize_dist("burst 压缩后事件数(近似)", burst_raw_len))

        if total_bursts > 0:
            print("\nburst 作为 segment 的可用性（按 unique item 数判断）")
            print(f"  - 可直接用（{SEG_MIN}~{SEG_MAX}）：{burst_in_range:,} 个，占比 {burst_in_range/total_bursts*100:.2f}%")
            print(f"  - 太小（<{SEG_MIN}）：{burst_too_small:,} 个，占比 {burst_too_small/total_bursts*100:.2f}%")
            print(f"  - 太大（>{SEG_MAX}）：{burst_too_large:,} 个，占比 {burst_too_large/total_bursts*100:.2f}%")
        print("======================================================\n")


if __name__ == "__main__":
    main()
