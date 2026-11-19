#!/usr/bin/env python3
"""下载 / 更新 openFDA 药品说明书全量导出（公有领域）。官方每周更新；本地记下 last_updated，没变就不重下。

  python scripts/fetch_openfda.py --dir openfda          # 退出码 0 = 有更新并已下载，3 = 没有更新
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

INDEX = "https://api.fda.gov/download.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    d = Path(args.dir)
    d.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(INDEX, timeout=60) as r:
        label = json.load(r)["results"]["drug"]["label"]
    marker = d / ".last_updated"
    if not args.force and marker.exists() and marker.read_text().strip() == label["export_date"]:
        print("openFDA 说明书没有更新：", label["export_date"])
        sys.exit(3)
    tmp = d / ".partial"
    tmp.mkdir(exist_ok=True)
    for i, p in enumerate(label["partitions"]):
        z = tmp / Path(p["file"]).name
        print(f"[{i + 1}/{len(label['partitions'])}] {p['file']}", flush=True)
        urllib.request.urlretrieve(p["file"], z)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(tmp)
        z.unlink()
    for old in d.glob("drug-label-*.json"):
        old.unlink()
    for new in tmp.glob("drug-label-*.json"):
        new.rename(d / new.name)
    tmp.rmdir()
    marker.write_text(label["export_date"])
    print("已更新到", label["export_date"])


if __name__ == "__main__":
    main()
