import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
TOOLSAVE_DIR = REPO_ROOT / ".cache" / "toolsafe"

# ── ToolSafe ──────────────────────────────────────────────────────────────────
print("=== ToolSafe ===")
ts_root = TOOLSAVE_DIR / "TS-Bench"
if ts_root.exists():
    json_files = list(ts_root.rglob("*.json"))
    print(f"JSON files in TS-Bench: {len(json_files)}")

    total_valid = 0
    skipped = []
    errored = []
    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            rows = data if isinstance(data, list) else [data]
            valid = [r for r in rows if isinstance(r, dict)
                     and (r.get("current_action") or r.get("action") or r.get("tool_call"))
                     and (r.get("instruction") or r.get("task") or r.get("goal"))]
            total_valid += len(valid)
            if not valid:
                skipped.append(jf.name)
        except Exception as e:
            errored.append((jf.name, str(e)))

    print(f"Total valid records:        {total_valid}")
    print(f"Files with 0 valid records: {len(skipped)}  -> {skipped[:8]}")
    print(f"Parse errors:               {len(errored)}  -> {errored[:3]}")
    print(f"Pipeline captured:          7182")
    diff = total_valid - 7182
    print(f"Difference:                 {diff}  ({'OK' if diff == 0 else 'MISSING' if diff > 0 else 'EXTRA'})")
else:
    print(".cache/toolsafe deleted after run. Trusting pipeline log: 7182 records.")

# ── OAS ──────────────────────────────────────────────────────────────────────
print()
print("=== OAS ===")
cache = Path.home() / ".cache/huggingface/hub"
slug  = "datasets--sani903--openagentsafety"
snaps = list((cache / slug / "snapshots").glob("*"))
if snaps:
    tasks_dir  = snaps[0] / "tasks"
    all_tasks  = [f for f in tasks_dir.iterdir() if f.is_dir()]
    with_md    = [t for t in all_tasks if (t / "task.md").exists()]
    without_md = [t.name for t in all_tasks if not (t / "task.md").exists()]
    print(f"Total task folders:           {len(all_tasks)}")
    print(f"Folders with task.md:         {len(with_md)}")
    print(f"Expected synthetic records:   {len(with_md) * 2}")
    print(f"Pipeline generated:           154 (77 tasks x 2)")
    print(f"Folders WITHOUT task.md:      {len(without_md)}  -> {without_md}")
    missing_2 = len(with_md) - 77
    print(f"task.md folders not captured: {missing_2}")
else:
    print("OAS cache not found.")

# ── Output files ──────────────────────────────────────────────────────────────
print()
print("=== Output files ===")
for fname in ["supervisor_train.jsonl", "supervisor_eval.jsonl"]:
    p = DATA_DIR / fname
    if p.exists():
        lines = p.read_text(encoding="utf-8").splitlines()
        records = []
        # Verify JSON integrity
        bad = 0
        for l in lines:
            try:
                records.append(json.loads(l))
            except json.JSONDecodeError:
                bad += 1
        pass_c = sum(1 for r in records if r.get("decision") == "PASS")
        block_c = sum(1 for r in records if r.get("decision") == "BLOCK")
        print(f"{fname}: {len(lines)} records  PASS={pass_c}  BLOCK={block_c}  bad_json={bad}")
    else:
        print(f"{fname}: NOT FOUND")
