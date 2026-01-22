# -*- coding: utf-8 -*-
import json
import os

# ====== 改这里 ======
INPUT_PATH = r"../dataProcess/Electronics_2019_2021_sorted_min15.jsonl"  # 你在dataProcess目录运行就行

# 你想看的累计阈值（小时）
K_HOURS = [1, 6, 12, 24, 48, 72, 168, 720]

# gap 分桶（小时）——你觉得不合适可改
BUCKETS = [
    ("< 1小时", 0, 1),
    ("[1, 6)小时", 1, 6),
    ("[6, 12)小时", 6, 12),
    ("[12, 24)小时", 12, 24),
    ("[1, 2)天", 24, 48),
    ("[2, 3)天", 48, 72),
    ("[3, 7)天", 72, 168),
    ("[7, 30)天", 168, 720),
    ("[30, 90)天", 720, 2160),
    ("[90, 180)天", 2160, 4320),
    ("[180天, +∞)", 4320, None),
]

# 如果你文件同用户是新->旧，就设 True
TIME_DESC = False


def ts_to_seconds(ts):
    ts = int(ts)
    if ts > 10_000_000_000:  # ms
        return ts / 1000.0
    return float(ts)


def quantiles(sorted_vals, ps=(50, 75, 90, 95, 99, 100)):
    if not sorted_vals:
        return {p: 0 for p in ps}
    n = len(sorted_vals)
    out = {}
    for p in ps:
        if p <= 0:
            out[p] = sorted_vals[0]
        elif p >= 100:
            out[p] = sorted_vals[-1]
        else:
            idx = int(round((p / 100) * (n - 1)))
            idx = max(0, min(n - 1, idx))
            out[p] = sorted_vals[idx]
    return out


def main():
    print("当前工作目录：", os.getcwd())
    print("输入文件存在？", os.path.exists(INPUT_PATH), INPUT_PATH)

    total_lines = 0
    bad_lines = 0
    missing_uid = 0
    missing_ts = 0

    # gap级别统计
    gap_hours = []              # 存所有gap（40万级，完全没问题）
    gap_sum = 0.0
    k_counts = {k: 0 for k in K_HOURS}
    bucket_counts = {name: 0 for (name, _, _) in BUCKETS}

    # 排序检查
    negative_gap = 0

    prev_uid = None
    prev_ts = None

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
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
            ts = obj.get("timestamp")
            if not uid:
                missing_uid += 1
                continue
            if ts is None:
                missing_ts += 1
                continue

            ts_sec = ts_to_seconds(ts)

            # 新用户
            if uid != prev_uid:
                prev_uid = uid
                prev_ts = ts_sec
                continue

            # 同用户：算相邻gap
            diff_sec = ts_sec - prev_ts

            # 让gap为正；并统计“时间乱序”次数
            if TIME_DESC:
                # 新->旧：diff应该<=0
                if diff_sec > 0:
                    negative_gap += 1
                    diff_sec = -diff_sec
                else:
                    diff_sec = -diff_sec
            else:
                # 旧->新：diff应该>=0
                if diff_sec < 0:
                    negative_gap += 1
                    diff_sec = -diff_sec

            h = diff_sec / 3600.0
            gap_hours.append(h)
            gap_sum += h

            # >=k累计
            for k in K_HOURS:
                if h >= k:
                    k_counts[k] += 1

            # 分桶
            placed = False
            for name, lo, hi in BUCKETS:
                if hi is None:
                    if h >= lo:
                        bucket_counts[name] += 1
                        placed = True
                        break
                else:
                    if (h >= lo) and (h < hi):
                        bucket_counts[name] += 1
                        placed = True
                        break
            if not placed:
                # 理论上不会走到这里
                pass

            prev_ts = ts_sec

            if total_lines % 500_000 == 0:
                mean_now = gap_sum / len(gap_hours)
                print(f"已处理 {total_lines:,} 行；gap {len(gap_hours):,} 个；当前平均gap {mean_now:.3f} 小时...")

    gap_n = len(gap_hours)
    mean_gap = (gap_sum / gap_n) if gap_n else 0.0
    gap_hours.sort()
    qs = quantiles(gap_hours)

    print("\n================= gap 统计（gap级别）=================")
    print(f"文件：{INPUT_PATH}")
    print(f"总行数：{total_lines:,}；JSON失败：{bad_lines:,}；缺uid：{missing_uid:,}；缺ts：{missing_ts:,}")

    print("\n【1) gap总数】")
    print(f"gap总数（所有用户相邻两次交互形成的gap）：{gap_n:,}")

    print("\n【2) gap平均值】")
    print(f"平均gap：{mean_gap:.4f} 小时（约 {mean_gap/24:.2f} 天）")

    print("\n【3) gap分位数（小时）】")
    print(f"P50={qs[50]:.4f}h（{qs[50]/24:.2f}天）")
    print(f"P75={qs[75]:.4f}h（{qs[75]/24:.2f}天）")
    print(f"P90={qs[90]:.4f}h（{qs[90]/24:.2f}天）")
    print(f"P95={qs[95]:.4f}h（{qs[95]/24:.2f}天）")
    print(f"P99={qs[99]:.4f}h（{qs[99]/24:.2f}天）")
    print(f"MAX={qs[100]:.4f}h（{qs[100]/24:.2f}天）")

    print("\n【4) gap分桶分布】")
    for name, _, _ in BUCKETS:
        c = bucket_counts[name]
        pct = (c / gap_n * 100) if gap_n else 0
        print(f"  - {name}：{c:,} 个，占比 {pct:.2f}%")

    print("\n【5) gap >= k（小时）累计分布】")
    for k in sorted(K_HOURS):
        c = k_counts[k]
        pct = (c / gap_n * 100) if gap_n else 0
        print(f"  - gap >= {k} 小时：{c:,} 个，占比 {pct:.2f}%")

    if negative_gap > 0:
        pct = negative_gap / gap_n * 100 if gap_n else 0
        print(f"\n[注意] 发现同用户时间顺序不一致：{negative_gap:,} 次，占比 {pct:.2f}%")
        print("一般说明：文件并非严格按时间排序，或 TIME_DESC 设错。")

    print("======================================================\n")


if __name__ == "__main__":
    main()
