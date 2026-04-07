from pathlib import Path
import runpy
import sys


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    runpy.run_path(str(root / "inference.py"), run_name="__main__")
