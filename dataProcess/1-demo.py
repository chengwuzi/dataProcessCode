# -*- coding: utf-8 -*-
import json
from datetime import datetime

INPUT_PATH = r"Electronics_2019_2021.jsonl"
TARGET_YEARS = {2019, 2020, 2021}
K_LIST = [5, 10, 15, 20, 30, 40]

SECONDS_PER_DAY = 24 * 3600
DAYS_PER_MONTH = 30.0


def get_year(ts):
    if ts is None:
        return None
    ts = int(ts)
    # ms -> s
    if ts > 10_000_000_000:
        ts //= 1000
    return datetime.utcfromtimestamp(ts).year


def normalize_ts_seconds(ts):
    if ts is None:
        return None
    ts = int(ts)
    if ts > 10_000_000_000:
        ts //= 1000
    return ts


def quantiles(values, ps=(0, 50, 75, 90, 95, 99, 100)):
    """values: list of float; return dict p->value at percentile p (nearest-rank-ish)"""
    if not values:
        return {p: 0 for p in ps}
    arr = sorted(values)
    n = len(arr)
    out = {}
    for p in ps:
        if p <= 0:
            out[p] = arr[0]
        elif p >= 100:
            out[p] = arr[-1]
        else:
            # linear position
            idx = int(round((p / 100) * (n - 1)))
            if idx < 0:
                idx = 0
            if idx >= n:
                idx = n - 1
            out[p] = arr[idx]
    return out


def bucketize(values, buckets):
    """
    buckets: list of tuples (label, lo, hi, lo_inclusive, hi_inclusive)
    hi can be None meaning +inf
    """
    counts = {b[0]: 0 for b in buckets}
    for v in values:
        for label, lo, hi, lo_inc, hi_inc in buckets:
            if hi is None:
                ok_hi = True
            else:
                ok_hi = (v <= hi) if hi_inc else (v < hi)

            ok_lo = (v >= lo) if lo_inc else (v > lo)

            if ok_lo and ok_hi:
                counts[label] += 1
                break
    return counts


def print_bucket_report(title, counts, total):
    print(f"\n分桶分布：")
    for label, c in counts.items():
        ratio = (c / total * 100) if total else 0
        print(f"  - {label}：{c:,} 人，占比 {ratio:.2f}%")
    print("（注：单位均为“人”，占比按总用户数计算）")


def main():
    # user_id -> [cnt, min_ts, max_ts]
    user_stat = {}

    bad_lines = 0
    missing_ts = 0
    missing_uid = 0
    out_of_year = 0
    total_lines = 0
    kept_lines = 0

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            total_lines += 1
            try:
                obj = json.loads(line)
            except Exception:
                bad_lines += 1
                continue

            uid = obj.get("user_id")
            if not uid:
                missing_uid += 1
                continue

            ts = normalize_ts_seconds(obj.get("timestamp"))
            if ts is None:
                missing_ts += 1
                continue

            y = get_year(ts)
            if y not in TARGET_YEARS:
                out_of_year += 1
                continue

            kept_lines += 1

            if uid not in user_stat:
                user_stat[uid] = [1, ts, ts]
            else:
                st = user_stat[uid]
                st[0] += 1
                if ts < st[1]:
                    st[1] = ts
                if ts > st[2]:
                    st[2] = ts

            if idx % 1_000_000 == 0:
                print(f"已处理 {idx:,} 行；保留(2019-2021) {kept_lines:,} 行；当前用户数 {len(user_stat):,}...")

    total_users = len(user_stat)
    if total_users == 0:
        print("没有统计到任何用户（请检查文件路径/内容/年份过滤）。")
        return

    # ===== 指标1：>=k 用户数与占比 =====
    cnt_list = [st[0] for st in user_stat.values()]

    # ===== 指标2：首末跨度（天）分布 + 分位数 =====
    span_days_list = []
    # ===== 指标3：30天折算平均每月交互次数 分布 + 分位数 =====
    avg_per_month_list = []

    for cnt, tmin, tmax in user_stat.values():
        span_days = (tmax - tmin) / SECONDS_PER_DAY
        span_days_list.append(span_days)

        active_months = max(1.0, span_days / DAYS_PER_MONTH)  # 30天折算，最少按1个月
        avg_per_month = cnt / active_months
        avg_per_month_list.append(avg_per_month)

    # buckets
    span_buckets = [
        ("0 天（仅一次或同日）", 0, 0, True, True),
        ("(0, 7] 天", 0, 7, False, True),
        ("(7, 30] 天", 7, 30, False, True),
        ("(30, 90] 天", 30, 90, False, True),
        ("(90, 180] 天", 90, 180, False, True),
        ("(180, 365] 天", 180, 365, False, True),
        ("(365, 730] 天", 365, 730, False, True),
        ("> 730 天", 730, None, False, True),
    ]

    avg_buckets = [
        ("[0, 0.5) 次/月", 0, 0.5, True, False),
        ("[0.5, 1) 次/月", 0.5, 1, True, False),
        ("[1, 2) 次/月", 1, 2, True, False),
        ("[2, 5) 次/月", 2, 5, True, False),
        ("[5, 10) 次/月", 5, 10, True, False),
        ("[10, 20) 次/月", 10, 20, True, False),
        ("[20, 50) 次/月", 20, 50, True, False),
        ("[50, +∞) 次/月", 50, None, True, True),
    ]

    span_counts = bucketize(span_days_list, span_buckets)
    avg_counts = bucketize(avg_per_month_list, avg_buckets)

    span_q = quantiles(span_days_list)
    avg_q = quantiles(avg_per_month_list)
    cnt_q = quantiles([float(x) for x in cnt_list])

    # ===== 输出（中文说明）=====
    print("\n================= 2019-2021 细节统计报告 =================")
    print(f"输入文件：{INPUT_PATH}")
    print(f"总行数（非空行）：{total_lines:,} 行")
    print(f"纳入统计（2019-2021 且 user_id/timestamp 有效）：{kept_lines:,} 行（交互）")
    print(f"总用户数（去重）：{total_users:,} 人")

    print("\n【指标1：用户交互次数 >= k 的人数与占比】")
    for k in K_LIST:
        num_k = sum(1 for c in cnt_list if c >= k)
        ratio_k = num_k / total_users * 100
        print(f"  - 交互次数 >= {k}：{num_k:,} 人，占比 {ratio_k:.2f}%")

    print("\n【补充：用户交互次数分位数（单位：次）】")
    print(f"  - P0={cnt_q[0]:.0f}, P50={cnt_q[50]:.0f}, P75={cnt_q[75]:.0f}, "
          f"P90={cnt_q[90]:.0f}, P95={cnt_q[95]:.0f}, P99={cnt_q[99]:.0f}, P100={cnt_q[100]:.0f}")

    print("\n【指标2：用户首末次交互跨度 span_days（单位：天）分位数】")
    print(f"  - P0={span_q[0]:.2f}, P50={span_q[50]:.2f}, P75={span_q[75]:.2f}, "
          f"P90={span_q[90]:.2f}, P95={span_q[95]:.2f}, P99={span_q[99]:.2f}, P100={span_q[100]:.2f}")
    print_bucket_report("指标2：span_days（天）", span_counts, total_users)

    print("\n【指标3：平均每月交互次数 avg_per_month（30天折算，span=0 按1个月）分位数】")
    print(f"  - P0={avg_q[0]:.4f}, P50={avg_q[50]:.4f}, P75={avg_q[75]:.4f}, "
          f"P90={avg_q[90]:.4f}, P95={avg_q[95]:.4f}, P99={avg_q[99]:.4f}, P100={avg_q[100]:.4f}")
    print_bucket_report("指标3：avg_per_month（次/月）", avg_counts, total_users)

    print("\n【数据质量信息】")
    print(f"  - JSON 解析失败行：{bad_lines:,} 行")
    print(f"  - 缺失 user_id 行：{missing_uid:,} 行")
    print(f"  - 缺失 timestamp 行：{missing_ts:,} 行")
    print(f"  - timestamp 不在 2019-2021 的行（被过滤）：{out_of_year:,} 行")
    print("==========================================================\n")


if __name__ == "__main__":
    main()
