from pathlib import Path
import json
from atlassearch.preprocessing.normalize import normalize_text, detect_script

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "data/sample/documents.jsonl"
out = ROOT / "data/sample/documents.normalized.tsv"

rows = []
with src.open(encoding="utf-8") as fh:
    for line in fh:
        obj = json.loads(line)
        text = f"{obj.get('title', '')} {obj.get('text', '')}"
        normalized = normalize_text(text)
        rows.append(f"{int(obj['id'])}\t{normalized}")

out.write_text("\n".join(rows) + "\n", encoding="utf-8")
print(f"wrote {len(rows)} docs to {out}")
