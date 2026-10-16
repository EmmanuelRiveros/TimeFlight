"""Check an API-format workflow against a live ComfyUI's /object_info before spending GPU time.

  python scripts/validate_workflow.py workflows/api/m0_union_depth_api.json \
      --server http://127.0.0.1:8188

Reports: unknown node classes, unknown input names, missing required inputs,
combo values not offered by the server (e.g. a model file that is not downloaded),
and links to missing nodes. Exit code 1 if anything is wrong.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path


def combo_options(spec: list) -> list | None:
    """Combo choices from an /object_info input spec: legacy [[...], {...}] or ["COMBO", {"options": [...]}]."""
    if not spec:
        return None
    if isinstance(spec[0], list):
        return spec[0]
    if spec[0] == "COMBO" and len(spec) > 1 and isinstance(spec[1].get("options"), list):
        return spec[1]["options"]
    return None


def validate(workflow: dict, object_info: dict) -> list[str]:
    problems = []
    for node_id, node in workflow.items():
        cls = node["class_type"]
        where = f"node {node_id} ({node.get('_meta', {}).get('title', cls)})"
        info = object_info.get(cls)
        if info is None:
            problems.append(f"{where}: unknown class {cls}")
            continue
        spec = info.get("input", {})
        required = spec.get("required", {})
        known = {**required, **spec.get("optional", {})}
        for name, value in node["inputs"].items():
            if name not in known:
                problems.append(f"{where}: unknown input {name!r} (known: {sorted(known)})")
                continue
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                if value[0] not in workflow:
                    problems.append(f"{where}: input {name!r} links to missing node {value[0]}")
                continue
            options = combo_options(known[name])
            if options is not None and value not in options:
                shown = options if len(options) < 12 else options[:12] + ["..."]
                problems.append(f"{where}: {name}={value!r} not in server options {shown}")
        for name in required:
            if name not in node["inputs"]:
                problems.append(f"{where}: missing required input {name!r}")
    return problems


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("workflow", type=Path)
    p.add_argument("--server", default="http://127.0.0.1:8188")
    p.add_argument("--ignore-combo", nargs="*", default=["file"],
                   help="inputs whose combo values are set at run time (default: file)")
    args = p.parse_args()

    workflow = json.loads(args.workflow.read_text())
    with urllib.request.urlopen(f"{args.server}/object_info") as r:
        object_info = json.load(r)
    problems = [x for x in validate(workflow, object_info)
                if not any(f": {name}=" in x for name in args.ignore_combo)]
    for x in problems:
        print("PROBLEM", x)
    print(f"{len(workflow)} nodes checked, {len(problems)} problems")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
