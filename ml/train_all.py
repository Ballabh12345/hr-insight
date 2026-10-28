"""Build everything from raw data in one command.

    python ml/train_all.py

Order matters:
    1. synthetic datasets   (needed by the fairness power curve)
    2. attrition model      (MVP-1)
    3. talent matrix        (MVP-4)
    4. fairness audit       (MVP-3, depends on 1 and 2)
    5. assembled dataset    (what the application serves)
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STEPS: list[tuple[str, list[str]]] = [
    ("Synthetic workforce datasets", [sys.executable, "ml/generate_synthetic.py"]),
    ("Attrition risk model", [sys.executable, "ml/train_attrition.py"]),
    ("Talent matrix", [sys.executable, "ml/train_ninebox.py"]),
    ("Fairness audit", [sys.executable, "ml/train_fairness.py"]),
    ("Survival analysis", [sys.executable, "ml/train_survival.py"]),
    ("Application dataset", [sys.executable, "ml/build_dataset.py"]),
]


def main() -> None:
    overall = time.time()
    for index, (name, command) in enumerate(STEPS, start=1):
        print(f"\n{'=' * 72}")
        print(f"STEP {index}/{len(STEPS)}  {name}")
        print("=" * 72)
        started = time.time()
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode != 0:
            print(f"\nFAILED at step {index} ({name}) with code {result.returncode}.")
            sys.exit(result.returncode)
        print(f"[step {index} complete in {time.time() - started:.1f}s]")

    print(f"\n{'=' * 72}")
    print(f"ALL STEPS COMPLETE in {time.time() - overall:.1f}s")
    print("Start the app with:  streamlit run app/main.py")
    print("=" * 72)


if __name__ == "__main__":
    main()
