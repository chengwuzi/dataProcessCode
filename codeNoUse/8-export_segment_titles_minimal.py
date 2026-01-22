# -*- coding: utf-8 -*-
import json
import os

# ========= 你只改这里 =========
INPUT_SAMPLE_PATH = r"segment_sample_7d_native_300.jsonl"   # 你生成的300条sample
META_PATH = r"../dataProcess/meta_Electronics.jsonl"  # 对应类目的meta
OUTPUT_PATH = r"segment_native_300_uid_items_titles.jsonl"  # 输出文件
# =============================


def load_samples(sample_path):
    samples = []
    needed_items = set()

    with open(sample_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            samples.append(obj)
            items = obj.get("items") or obj.get("item_ids") or []
            for it in items:
                if it:
                    needed_items.add(it)
    return samples, needed_items


def build_title_map(meta_path, needed_items):
    """
    只为 sample 里出现的 item 建映射：item_id -> title
    meta 里用 parent_asin 优先，其次 asin
    """
    title_map = {}
    needed = set(needed_items)
    if not needed:
        return title_map

    found = 0
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            item = obj.get("parent_asin") or obj.get("asin")
            if not item or item not in needed:
                continue

            title = obj.get("title")
            if isinstance(title, str) and title.strip():
                title_map[item] = title.strip()
            else:
                title_map[item] = None

            found += 1
            if found >= len(needed):
                break

    return title_map


def main():
    print("当前工作目录：", os.getcwd())
    print("sample 存在？", os.path.exists(INPUT_SAMPLE_PATH), INPUT_SAMPLE_PATH)
    print("meta 存在？", os.path.exists(META_PATH), META_PATH)

    samples, needed_items = load_samples(INPUT_SAMPLE_PATH)
    print(f"sample 条数：{len(samples)}")
    print(f"sample 涉及 item 数（去重）：{len(needed_items)}")

    title_map = build_title_map(META_PATH, needed_items)
    covered = sum(1 for it in needed_items if title_map.get(it))
    print(f"title 覆盖：{covered}/{len(needed_items)} ({covered/len(needed_items)*100:.2f}%)")

    missing_total = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for obj in samples:
            uid = obj.get("user_id")
            items = obj.get("items") or obj.get("item_ids") or []

            # 去重但尽量保序（items 原本是 set 转 list，顺序可能本来就不稳定；这里至少不重复）
            seen = set()
            item_ids = []
            for it in items:
                if not it:
                    continue
                if it in seen:
                    continue
                seen.add(it)
                item_ids.append(it)

            item_titles = []
            miss = 0
            for it in item_ids:
                t = title_map.get(it)
                if not t:
                    miss += 1
                    item_titles.append(None)
                else:
                    item_titles.append(t)

            missing_total += miss

            out_obj = {
                "user_id": uid,
                "item_ids": item_ids,
                "item_titles": item_titles,
            }
            out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    print("输出完成：", OUTPUT_PATH)
    print("所有 segment 合计缺 title 数：", missing_total)


if __name__ == "__main__":
    main()
