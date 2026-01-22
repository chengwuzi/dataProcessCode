# -*- coding: utf-8 -*-
import json
import os

# ================== 你只改这里 ==================
ITEM2INT_PATH = r"../dataset/sports/item2int.json"
META_IN_PATH  = r"meta_Sports_and_Outdoors.jsonl"
# META_OUT_PATH = r"../dataset/electronics/meta_Electronics_filtered.jsonl"
META_OUT_PATH = r"../dataset/sports/meta_Sports_and_Outdoors_filtered.jsonl"

# 可选: "full" | "normal" | "minimal"
MODE = "minimal"
# ===============================================


# ====== 字段策略 ======
# full:    原样输出整条 meta（不做裁剪）
# normal:  常用字段（够做分析/可解释性/后续展示）
# minimal: 最小可用字段（主键 + 标题 + 类目）

FIELDS_NORMAL = [
    # id / key
    "parent_asin", "asin",

    # name & brand
    "title", "store", "brand",

    # category
    "main_category", "categories",

    # rating & price
    "average_rating", "rating_number", "price",

    # text
    "features", "description",

    # details
    "details",

    # media (可选但常用)
    "images", "videos",

    # amazon official extra
    "bought_together",
]

FIELDS_MINIMAL = [
    "parent_asin",
    "title",
    "main_category", "categories",
]


def load_needed_items(item2int_path: str) -> set:
    """item2int.json 的 key 就是训练集里的物品id（你现在统一用 parent_asin）"""
    with open(item2int_path, "r", encoding="utf-8") as f:
        item2int = json.load(f)
    return set(item2int.keys())


def pick_item_id_from_meta(obj: dict) -> str | None:
    """
    按你已确定的逻辑：训练用物品统一用 parent_asin。
    - 有 parent_asin：返回它
    - 没有 parent_asin：返回 None（严格，不再 fallback asin）
    """
    iid = obj.get("parent_asin")
    if isinstance(iid, str) and iid.strip():
        return iid.strip()
    return None


def slim_obj(obj: dict, fields: list[str]) -> dict:
    """按字段白名单裁剪，同时保证至少输出 parent_asin（主键）"""
    out = {}
    for k in fields:
        if k in obj:
            out[k] = obj[k]

    # 保证主键一定存在
    if "parent_asin" not in out:
        pid = pick_item_id_from_meta(obj)
        if pid:
            out["parent_asin"] = pid

    return out


def main():
    print("cwd:", os.getcwd())
    print("MODE:", MODE)
    print("ITEM2INT:", ITEM2INT_PATH)
    print("META_IN :", META_IN_PATH)
    print("META_OUT:", META_OUT_PATH)

    needed = load_needed_items(ITEM2INT_PATH)
    print("needed items:", len(needed))

    total = 0
    kept = 0
    bad = 0
    missing_parent_asin = 0

    with open(META_IN_PATH, "r", encoding="utf-8") as fin, \
         open(META_OUT_PATH, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except Exception:
                bad += 1
                continue

            iid = pick_item_id_from_meta(obj)
            if not iid:
                missing_parent_asin += 1
                continue

            if iid not in needed:
                continue

            if MODE == "full":
                out_obj = obj
            elif MODE == "normal":
                out_obj = slim_obj(obj, FIELDS_NORMAL)
            elif MODE == "minimal":
                out_obj = slim_obj(obj, FIELDS_MINIMAL)
            else:
                raise ValueError(f"Unknown MODE={MODE}, must be full|normal|minimal")

            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            kept += 1

    print("\n[DONE]")
    print("meta scanned:", f"{total:,}")
    print("meta kept   :", f"{kept:,}")
    print("bad json    :", f"{bad:,}")
    print("missing parent_asin:", f"{missing_parent_asin:,}")
    print("output      :", META_OUT_PATH)


if __name__ == "__main__":
    main()
