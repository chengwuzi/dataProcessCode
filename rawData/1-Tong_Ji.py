# -*- coding: utf-8 -*-
import json
from datetime import datetime

INPUT_PATH = r"Electronics.jsonl"  # 改成你的路径也行


def get_year(ts):
    """
    ts may be milliseconds (e.g., 1677321053520) or seconds.
    """
    if ts is None:
        return None
    ts = int(ts)
    if ts > 10_000_000_000:   # treat as ms
        ts = ts // 1000
    return datetime.utcfromtimestamp(ts).year


def main():
    # year -> count / set
    year_interactions = {}
    year_users = {}
    year_items = {}

    bad_lines = 0
    missing_ts = 0
    missing_user = 0
    missing_item = 0

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception:
                bad_lines += 1
                continue

            year = get_year(obj.get("timestamp"))
            if year is None:
                missing_ts += 1
                continue

            # interaction count
            year_interactions[year] = year_interactions.get(year, 0) + 1

            # user set
            uid = obj.get("user_id")
            if uid:
                year_users.setdefault(year, set()).add(uid)
            else:
                missing_user += 1

            # item set (prefer parent_asin)
            item = obj.get("parent_asin") or obj.get("asin")
            if item:
                year_items.setdefault(year, set()).add(item)
            else:
                missing_item += 1

            if idx % 1_000_000 == 0:
                print(f"processed {idx} lines...")

    # ====== human-readable report ======
    years = sorted(year_interactions.keys())
    total_inter = sum(year_interactions.values())
    total_users = len(set().union(*[year_users.get(y, set()) for y in years])) if years else 0
    total_items = len(set().union(*[year_items.get(y, set()) for y in years])) if years else 0

    print("\n================= Sports_and_Outdoors 数据统计（按年）=================")
    print(f"总交互条数：{total_inter:,}")
    print(f"全量去重用户数（跨年合并）：{total_users:,}")
    print(f"全量去重物品数（跨年合并）：{total_items:,}")
    if years:
        print(f"时间跨度：{years[0]} ~ {years[-1]}")
    else:
        print("时间跨度：无（没解析到有效 timestamp）")

    print("\n----------------- 各年份详情 -----------------")
    for y in years:
        inter = year_interactions[y]
        u = len(year_users.get(y, set()))
        it = len(year_items.get(y, set()))
        print(f"{y} 年：交互 {inter:,} 条；用户 {u:,} 个；物品 {it:,} 个。")

    print("\n----------------- 数据质量信息 -----------------")
    print(f"无法解析的行（JSON 失败）：{bad_lines:,} 行")
    print(f"缺失 timestamp 的行：{missing_ts:,} 行")
    print(f"缺失 user_id 的行：{missing_user:,} 行")
    print(f"缺失 parent_asin/asin 的行：{missing_item:,} 行")
    print("====================================================================\n")


if __name__ == "__main__":
    main()
