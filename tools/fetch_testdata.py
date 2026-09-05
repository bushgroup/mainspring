"""Fetch the small UIMF test excerpts PNNL distributes with UIMF-Library into
gitignored external/pnnl-testdata/, for the real-file checks in
tools/check_public.py and the test suite.

Three files, 1-5 MB each, from the Test_Data/ directory of
https://github.com/PNNL-Comp-Mass-Spec/UIMF-Library (Educational Community
License 2.0). They are fetched rather than committed so that this repository
carries no data and no third-party licence text; nothing in the package depends
on them, and every check that uses them reports SKIPPED when they are absent.

Run:  uv run tools/fetch_testdata.py            # fetch what is missing
      uv run tools/fetch_testdata.py --force    # re-download everything
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DEST = os.path.join(ROOT, "external", "pnnl-testdata")

BASE = "https://raw.githubusercontent.com/PNNL-Comp-Mass-Spec/UIMF-Library/master/Test_Data/"
FILES = (
    "9pep_mix_1uM_4bit_50_12Dec11_encoded.uimf",
    "QC_Shew_16_01_Run-3_25Jul16_Oak_16-03-14_Excerpt.uimf",
    "Sarc_MS2_90_6Apr11_Cheetah_11-02-19_Excerpt.uimf",
)


def fetch(name: str, force: bool) -> str:
    path = os.path.join(DEST, name)
    if os.path.isfile(path) and os.path.getsize(path) > 0 and not force:
        print(f"have   {name}  ({os.path.getsize(path):,} bytes)")
        return path
    print(f"fetch  {name} ...", end=" ", flush=True)
    with urllib.request.urlopen(BASE + name, timeout=60) as resp, open(path, "wb") as out:
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            out.write(chunk)
    print(f"{os.path.getsize(path):,} bytes")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="re-download files that are present")
    args = ap.parse_args()
    os.makedirs(DEST, exist_ok=True)
    failed = 0
    for name in FILES:
        try:
            fetch(name, args.force)
        except OSError as exc:
            failed += 1
            print(f"FAILED {name}: {exc}")
    print(f"\n{DEST}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
