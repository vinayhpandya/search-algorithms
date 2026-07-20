import json
from collections import Counter

data = json.loads(open("eval/candidates.json").read())
labels = [c["label"] for entry in data.values() for c in entry["candidates"]]
print(Counter(labels))