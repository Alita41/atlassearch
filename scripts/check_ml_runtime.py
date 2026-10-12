from __future__ import annotations

import importlib.util
import json
import sys

MODULES = ["torch", "transformers", "sentence_transformers", "faiss"]
status = {name: bool(importlib.util.find_spec(name)) for name in MODULES}
print(json.dumps(status, indent=2))
missing = [name for name, ok in status.items() if not ok]
if missing:
    print("Missing ML runtime packages: " + ", ".join(missing), file=sys.stderr)
    print("Install an appropriate PyTorch build, then: pip install -r requirements-ml.txt", file=sys.stderr)
    raise SystemExit(1)
print("ML runtime dependencies are available.")
