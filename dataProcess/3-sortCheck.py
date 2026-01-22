# -*- coding: utf-8 -*-
import json

# ====== 你改这俩路径就行 ======
INPUT_PATH = r"Sports_2019_2021_sorted.jsonl"
OUTPUT_PATH = r"Sports_2019_2021_sorted_min10.jsonl"

MIN_INTERACTIONS = 10

# 如果你当时同用户是“新->旧”（timestamp 降序），把这个设为 True
TIME_DESC = False


def main():
    total_lines = 0
    bad_lines = 0
    missing_uid = 0
    missing_ts = 0

    # 排序检查用
    user_order_violations = 0
    time_order_violations = 0
    first_violation_examples = []  # 只存前几条例子，别炸内存
    MAX_EXAMPLES = 5

    prev_uid = None
    prev_ts = None

    # 过滤用（因为按 user_id 聚类，所以一个用户的记录可以用 list 暂存）
    cur_uid = None
    cur_buf = []
    cur_cnt = 0

    kept_users = 0
    removed_users = 0
    kept_lines = 0

    def flush_user(fout):
        nonlocal kept_users, removed_users, kept_lines, cur_uid, cur_buf, cur_cnt
        if cur_uid is None:
            return
        if cur_cnt >= MIN_INTERACTIONS:
            for ln in cur_buf:
                fout.write(ln)
            kept_users += 1
            kept_lines += cur_cnt
        else:
            removed_users += 1
        # reset
        cur_uid = None
        cur_buf = []
        cur_cnt = 0

    with open(INPUT_PATH, "r", encoding="utf-8") as fin, open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, 1):
            if not line.strip():
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

            ts = int(ts)  # 你文件里是 ms，这里直接用 ms 比较即可

            # ====== 排序检查 ======
            if prev_uid is not None:
                if uid < prev_uid:
                    user_order_violations += 1
                    if len(first_violation_examples) < MAX_EXAMPLES:
                        first_violation_examples.append(
                            f"[用户顺序错误] line {line_no}: uid={uid} < prev_uid={prev_uid}"
                        )
                elif uid == prev_uid:
                    # 同用户时间顺序检查
                    if not TIME_DESC:
                        # 旧->新：ts 应该 >= prev_ts
                        if ts < prev_ts:
                            time_order_violations += 1
                            if len(first_violation_examples) < MAX_EXAMPLES:
                                first_violation_examples.append(
                                    f"[时间顺序错误] line {line_no}: ts={ts} < prev_ts={prev_ts} (uid={uid})"
                                )
                    else:
                        # 新->旧：ts 应该 <= prev_ts
                        if ts > prev_ts:
                            time_order_violations += 1
                            if len(first_violation_examples) < MAX_EXAMPLES:
                                first_violation_examples.append(
                                    f"[时间顺序错误] line {line_no}: ts={ts} > prev_ts={prev_ts} (uid={uid})"
                                )

            prev_uid = uid
            prev_ts = ts

            # ====== 过滤：交互次数 < MIN_INTERACTIONS 的用户整段去掉 ======
            if cur_uid is None:
                cur_uid = uid

            if uid != cur_uid:
                flush_user(fout)
                cur_uid = uid

            cur_buf.append(line)  # 保留原始行（包含换行符）
            cur_cnt += 1

            if total_lines % 1_000_000 == 0:
                print(f"已处理 {total_lines:,} 行；当前用户={cur_uid} 缓存={cur_cnt}；已保留用户={kept_users:,}...")

        # flush last user
        flush_user(fout)

    # ====== 输出中文说明 ======
    print("\n================= 检查 + 过滤结果 =================")
    print(f"输入文件：{INPUT_PATH}")
    print(f"输出文件：{OUTPUT_PATH}")
    print(f"总处理行数：{total_lines:,}")
    print(f"JSON 解析失败：{bad_lines:,}")
    print(f"缺失 user_id：{missing_uid:,}")
    print(f"缺失 timestamp：{missing_ts:,}")

    print("\n【排序检查】(按 user_id 升序；同用户时间 {} )".format("新→旧" if TIME_DESC else "旧→新"))
    print(f"用户顺序违规次数：{user_order_violations:,}")
    print(f"同用户时间顺序违规次数：{time_order_violations:,}")
    if first_violation_examples:
        print("示例（前几条）：")
        for s in first_violation_examples:
            print("  " + s)
    else:
        print("未发现违规（看起来已正确排序）。")

    print("\n【过滤（每个用户交互次数 >= {}）】".format(MIN_INTERACTIONS))
    print(f"保留用户数：{kept_users:,}")
    print(f"移除用户数：{removed_users:,}")
    print(f"输出交互行数：{kept_lines:,}")
    print("====================================================\n")


if __name__ == "__main__":
    main()
