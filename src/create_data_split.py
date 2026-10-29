from pathlib import Path
import csv
import random

DATA_ROOT = Path("data/egg.v1i.folder")
OUT = Path("reports/dev_eval_split.csv")
SEED = 42
EVAL_RATIO = 0.20

# Collect images across the existing dataset folders.
extensions = {".jpg", ".jpeg", ".png"}
images = sorted(
    p for p in DATA_ROOT.rglob("*")
    if p.is_file()
    and p.suffix.lower() in extensions
    and not any(part.startswith(".") for part in p.parts)
)

if not images:
    raise SystemExit(f"No images found under {DATA_ROOT}")

# Group Roboflow exports by the shared filename prefix before ".rf.".
# This is a conservative grouping heuristic, not proof of source independence.
groups = {}
for path in images:
    stem = path.stem
    group = stem.split(".rf.")[0] if ".rf." in stem else stem
    groups.setdefault(group, []).append(path)

group_names = sorted(groups)
random.Random(SEED).shuffle(group_names)

target_eval = round(len(images) * EVAL_RATIO)
eval_groups = []
eval_count = 0

for group in group_names:
    if eval_count < target_eval:
        eval_groups.append(group)
        eval_count += len(groups[group])

eval_set = set(eval_groups)

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=["image", "split", "group"]
    )
    writer.writeheader()
    for group in sorted(groups):
        split = "evaluation" if group in eval_set else "development"
        for path in sorted(groups[group]):
            writer.writerow({
                "image": str(path),
                "split": split,
                "group": group,
            })

print("Saved:", OUT)
print("Total images:", len(images))
print("Development:", sum(len(groups[g]) for g in groups if g not in eval_set))
print("Evaluation:", sum(len(groups[g]) for g in eval_set))
print("Unique filename groups:", len(groups))
print("Note: inspect group assignments before claiming leakage is eliminated.")
