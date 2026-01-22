# -*- coding: utf-8 -*-
import json
import os

# ====== 改这里 ======   categories
INPUT_PATH = r"segment_sample_7d_native_300.jsonl"
OUTPUT_PATH = r"segment_sample_7d_300_categories_only.jsonl"
# ====================

KEEP_FIELDS = [
    "sample_id",
    "segment_type",
    "user_id",
    "unique_item_count",
    "top_category_count",
    "categories",
    "top_category_counter",
]

def main():
    print("当前工作目录：", os.getcwd())
    print("输入文件存在？", os.path.exists(INPUT_PATH), INPUT_PATH)

    n_in = 0
    n_out = 0
    missing_cat = 0

    with open(INPUT_PATH, "r", encoding="utf-8") as fin, open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            obj = json.loads(line)

            out = {k: obj.get(k) for k in KEEP_FIELDS}

            # 兼容：如果脚本里没写 top_category_count，就用 counter 推一下
            if out.get("top_category_counter") and out.get("top_category_count") is None:
                out["top_category_count"] = len(out["top_category_counter"])

            # 如果完全没有类目信息，做个标记
            if not out.get("top_category_counter") and not out.get("top_categories"):
                missing_cat += 1

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"处理完成：输入 {n_in} 行，输出 {n_out} 行")
    print(f"缺少类目信息的行数：{missing_cat}")
    print("输出文件：", OUTPUT_PATH)

if __name__ == "__main__":
    main()
