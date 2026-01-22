# 13-check_item_id_fields.py
import json

PATH = r"../dataProcess/Electronics_2019_2021_sorted_min15.jsonl"
N = 2000000

has_parent = 0
has_asin = 0
both = 0
same = 0
diff = 0

with open(PATH, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= N:
            break
        obj = json.loads(line)
        a = obj.get("asin")
        p = obj.get("parent_asin")
        if a: has_asin += 1
        if p: has_parent += 1
        if a and p:
            both += 1
            if a == p:
                same += 1
            else:
                diff += 1

print("checked:", N)
print("has asin:", has_asin)
print("has parent_asin:", has_parent)
print("has both:", both)
print("asin==parent_asin:", same)
print("asin!=parent_asin:", diff)
