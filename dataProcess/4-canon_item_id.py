# -*- coding: utf-8 -*-
import json
import os
from collections import Counter

# ================== 你只改这里 ==================
IN_PATH = r"Electronics_2019_2021_sorted_min15.jsonl"
META_PATH = r"meta_Electronics.jsonl"

OUT_PATH = r"Electronics_2019_2021_sorted_min15_canon.jsonl"
OUT_STATS = r"canon_item_id_stats.json"
# ===============================================

# 是否做 meta 覆盖检查（会扫描一遍 meta，但只检查你用到的 item_id）
CHECK_META_COVERAGE = True

# 如果你想让“老代码不改也能用”（老代码用 asin），可以把 asin 覆盖成 parent_asin
# 但我更推荐后续代码统一读 item_id（更干净）
OVERWRITE_ASIN_WITH_PARENT = False

# 进度打印
PRINT_EVERY = 1_000_000


def pick_item_id(obj):
    """统一物品ID：优先 parent_asin，否则 asin"""
    pid = obj.get("parent_asin")
    if isinstance(pid, str) and pid.strip():
        return pid.strip(), "parent_asin"
    aid = obj.get("asin")
    if isinstance(aid, str) and aid.strip():
        return aid.strip(), "asin"
    return None, None


def meta_pick_key(meta_obj):
    """meta 的主键：优先 parent_asin，否则 asin"""
    pid = meta_obj.get("parent_asin")
    if isinstance(pid, str) and pid.strip():
        return pid.strip()
    aid = meta_obj.get("asin")
    if isinstance(aid, str) and aid.strip():
        return aid.strip()
    return None


def main():
    print("cwd:", os.getcwd())
    print("IN exists?", os.path.exists(IN_PATH), IN_PATH)
    print("META exists?", os.path.exists(META_PATH), META_PATH)
    print("OUT:", OUT_PATH)
    print("CHECK_META_COVERAGE:", CHECK_META_COVERAGE)
    print("OVERWRITE_ASIN_WITH_PARENT:", OVERWRITE_ASIN_WITH_PARENT)

    stats = {
        "input_path": IN_PATH,
        "meta_path": META_PATH,
        "output_path": OUT_PATH,
        "checked_lines": 0,
        "written_lines": 0,
        "bad_json": 0,
        "missing_user_or_ts": 0,
        "missing_item_fields": 0,
        "item_id_source_counter": Counter(),  # parent_asin / asin
        "has_asin": 0,
        "has_parent_asin": 0,
        "has_both": 0,
        "asin_eq_parent": 0,
        "asin_ne_parent": 0,
        "unique_item_ids_in_output": 0,

        # meta coverage
        "meta_scanned": 0,
        "meta_hit_items": 0,
        "meta_missing_items": 0,
        "meta_coverage_ratio": None,
    }

    needed_items = set()

    # pass 1: 写 canon 输出 + 收集 needed item_id（只收 unique）
    with open(IN_PATH, "r", encoding="utf-8") as fin, open(OUT_PATH, "w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            stats["checked_lines"] += 1

            try:
                obj = json.loads(line)
            except Exception:
                stats["bad_json"] += 1
                continue

            uid = obj.get("user_id")
            ts = obj.get("timestamp")
            if not uid or ts is None:
                stats["missing_user_or_ts"] += 1
                continue

            asin = obj.get("asin")
            parent = obj.get("parent_asin")

            has_asin = isinstance(asin, str) and asin.strip()
            has_parent = isinstance(parent, str) and parent.strip()

            if has_asin:
                stats["has_asin"] += 1
            if has_parent:
                stats["has_parent_asin"] += 1
            if has_asin and has_parent:
                stats["has_both"] += 1
                if asin.strip() == parent.strip():
                    stats["asin_eq_parent"] += 1
                else:
                    stats["asin_ne_parent"] += 1

            item_id, source = pick_item_id(obj)
            if not item_id:
                stats["missing_item_fields"] += 1
                continue

            stats["item_id_source_counter"][source] += 1
            needed_items.add(item_id)

            # 写出：新增 item_id（可选覆盖 asin）
            obj["item_id"] = item_id
            if OVERWRITE_ASIN_WITH_PARENT and has_parent:
                obj["asin"] = parent.strip()

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            stats["written_lines"] += 1

            if stats["checked_lines"] % PRINT_EVERY == 0:
                print(f"processed {stats['checked_lines']:,}, written {stats['written_lines']:,}, unique_items {len(needed_items):,}")

    stats["unique_item_ids_in_output"] = len(needed_items)

    # pass 2: 扫 meta，只对 needed_items 做命中标记
    if CHECK_META_COVERAGE:
        found = set()
        with open(META_PATH, "r", encoding="utf-8") as fm:
            for line in fm:
                line = line.strip()
                if not line:
                    continue
                stats["meta_scanned"] += 1
                try:
                    mo = json.loads(line)
                except Exception:
                    continue

                key = meta_pick_key(mo)
                if not key:
                    continue
                if key in needed_items:
                    found.add(key)
                    # 早点结束：都找到了就不用继续扫
                    if len(found) >= len(needed_items):
                        break

                if stats["meta_scanned"] % (PRINT_EVERY * 2) == 0:
                    print(f"meta scanned {stats['meta_scanned']:,}, hits {len(found):,}/{len(needed_items):,}")

        stats["meta_hit_items"] = len(found)
        stats["meta_missing_items"] = len(needed_items) - len(found)
        if len(needed_items) > 0:
            stats["meta_coverage_ratio"] = stats["meta_hit_items"] / len(needed_items)

    # dump stats
    stats_to_dump = dict(stats)
    stats_to_dump["item_id_source_counter"] = dict(stats["item_id_source_counter"])

    with open(OUT_STATS, "w", encoding="utf-8") as f:
        json.dump(stats_to_dump, f, ensure_ascii=False, indent=2)

    print("\n[DONE] canon written:", OUT_PATH)
    print("[STATS] saved:", OUT_STATS)
    print("checked_lines:", f"{stats['checked_lines']:,}")
    print("written_lines:", f"{stats['written_lines']:,}")
    print("unique_item_ids_in_output:", f"{stats['unique_item_ids_in_output']:,}")
    print("item_id_source_counter:", dict(stats["item_id_source_counter"]))
    if CHECK_META_COVERAGE:
        print("meta_hit_items:", f"{stats['meta_hit_items']:,}")
        print("meta_missing_items:", f"{stats['meta_missing_items']:,}")
        print("meta_coverage_ratio:", stats["meta_coverage_ratio"])


if __name__ == "__main__":
    main()
