# -*- coding: utf-8 -*-
import os
import json
import argparse
import heapq
import tempfile

# --------- timestamp normalize ----------

#   --input Electronics_2019_2021.jsonl --output Electronics_2019_2021_sorted.jsonl


# -*- coding: utf-8 -*-
import os
import json
import heapq
import tempfile
import shutil  # 用于删除非空目录

# ================= 配置区域 =================
# 输入文件路径
INPUT_PATH = r"../rawData/Electronics_2019_2021.jsonl"

# 输出文件路径（排序后的文件）
OUTPUT_PATH = r"Electronics_2019_2021_sorted.jsonl"

# 临时文件夹路径（设为 None 则自动创建系统临时目录，跑完会自动删）
# 如果你的C盘空间不够，可以指定一个大硬盘的路径，比如 r"D:\tmp_sort"
TMP_DIR_PATH = None

# 单次内存处理行数（越大越快，但吃内存。30万行是个保守的安全值）
CHUNK_LINES = 300_000

# 同一个用户内的时间排序：
# False = 旧 -> 新 (升序, 默认)
# True  = 新 -> 旧 (降序)
TIME_DESC = False


# ===========================================


def normalize_ts_ms(ts):
    """
    统一时间戳为毫秒 (int)。
    """
    if ts is None:
        return None
    try:
        ts = int(ts)
    except ValueError:
        return None

    # 如果是秒级时间戳（比如 16xxxxxxxxx），补全为毫秒
    if ts < 10_000_000_000:
        ts *= 1000
    return ts


def make_key(user_id, ts_ms, desc=False):
    """
    生成排序用的 Key。
    格式：user_id + \t + padded_timestamp
    """
    # 如果是降序，用大数减去当前时间，实现“越大越靠前”在字符串排序中变为“越小越靠前”
    if desc:
        ts_ms = 9_999_999_999_999 - ts_ms
    # :013d 保证时间戳对齐，字符串排序逻辑等同于数字大小
    return f"{user_id}\t{ts_ms:013d}"


# --------- 阶段1: 分块排序 (Chunk Sort) ----------
def write_sorted_chunks(input_path, tmp_dir, chunk_lines=300_000, desc=False):
    os.makedirs(tmp_dir, exist_ok=True)
    chunk_files = []
    buf = []
    total = 0
    bad = 0
    missing = 0

    def flush_chunk(buffer, idx):
        buffer.sort()  # 内存内排序（按 key 字符串字典序）
        chunk_path = os.path.join(tmp_dir, f"chunk_{idx:05d}.txt")
        with open(chunk_path, "w", encoding="utf-8", newline="\n") as out:
            # 写入格式：key \t 原始json行
            out.write("\n".join(buffer) + "\n")
        return chunk_path

    print(f"开始读取并分块排序：{input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line_strip = line.rstrip("\n")
            if not line_strip.strip():
                continue

            total += 1
            try:
                obj = json.loads(line_strip)
            except Exception:
                bad += 1
                continue

            uid = obj.get("user_id")
            ts = normalize_ts_ms(obj.get("timestamp"))

            if not uid or ts is None:
                missing += 1
                continue

            # 构造 Key 并与原始行拼接
            key = make_key(uid, ts, desc=desc)
            # 格式：user_id \t padded_ts \t {"user_id":...}
            buf.append(key + "\t" + line_strip)

            if len(buf) >= chunk_lines:
                chunk_path = flush_chunk(buf, len(chunk_files))
                chunk_files.append(chunk_path)
                buf = []
                print(f"  -> 已处理 {total:,} 行，生成分块 {len(chunk_files)}...")

    # 处理剩余的 buffer
    if buf:
        chunk_path = flush_chunk(buf, len(chunk_files))
        chunk_files.append(chunk_path)

    print(f"[分块完成] 总行数：{total:,} (Bad JSON: {bad}, Missing Key: {missing})，分块文件数：{len(chunk_files)}")
    return chunk_files


# --------- 阶段2: K路归并 (K-way Merge) ----------
def merge_chunks(chunk_files, output_path):
    print(f"开始归并 {len(chunk_files)} 个分块到：{output_path}")

    # 打开所有分块文件
    fps = [open(p, "r", encoding="utf-8") for p in chunk_files]
    heap = []

    # 初始化堆：从每个文件读取第一行
    def push_next_line(file_idx):
        line = fps[file_idx].readline()
        if not line:
            return
        line = line.rstrip("\n")

        # 分割：key_part(user_id, ts), json_part
        # make_key 产生 user \t ts，所以我们要切前两个 tab
        parts = line.split("\t", 2)
        if len(parts) < 3:
            return  # 异常行跳过

        user_id, ts_str, jsonline = parts[0], parts[1], parts[2]

        # 构造堆元素：((user_id, ts_str), file_index, json_string)
        # Python 比较元组时会依次比较，user_id 相同则比 ts_str，正是我们要的
        key_tuple = (user_id, ts_str)
        heapq.heappush(heap, (key_tuple, file_idx, jsonline))

    # 初始填充
    for i in range(len(fps)):
        push_next_line(i)

    written = 0
    with open(output_path, "w", encoding="utf-8", newline="\n") as out:
        while heap:
            # 弹出最小的元素（即排序最靠前的）
            _, file_idx, jsonline = heapq.heappop(heap)

            out.write(jsonline + "\n")
            written += 1

            if written % 500_000 == 0:
                print(f"  -> 已归并写入 {written:,} 行...")

            # 从该文件再读下一行补充进堆
            push_next_line(file_idx)

    # 关闭文件句柄
    for fp in fps:
        fp.close()

    print(f"[归并完成] 最终输出行数：{written:,}")


def main():
    # 准备临时目录
    if TMP_DIR_PATH:
        work_tmp_dir = TMP_DIR_PATH
        if not os.path.exists(work_tmp_dir):
            os.makedirs(work_tmp_dir)
        should_cleanup = False  # 用户指定的目录，通常建议不删，或者你要删也行
    else:
        work_tmp_dir = tempfile.mkdtemp(prefix="sort_jsonl_")
        should_cleanup = True

    print(f"使用临时目录：{work_tmp_dir}")
    print(f"同用户内排序顺序：{'新 -> 旧 (Desc)' if TIME_DESC else '旧 -> 新 (Asc)'}")

    try:
        # 1. 分块排序
        chunk_files = write_sorted_chunks(
            input_path=INPUT_PATH,
            tmp_dir=work_tmp_dir,
            chunk_lines=CHUNK_LINES,
            desc=TIME_DESC
        )

        # 2. 归并
        merge_chunks(chunk_files, OUTPUT_PATH)

    finally:
        # 清理临时文件
        if should_cleanup and os.path.exists(work_tmp_dir):
            print("正在清理临时文件...")
            try:
                shutil.rmtree(work_tmp_dir)
            except Exception as e:
                print(f"临时目录清理失败 (可手动删除): {e}")


if __name__ == "__main__":
    main()
