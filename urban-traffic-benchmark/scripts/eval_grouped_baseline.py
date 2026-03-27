"""Compatibility wrapper for grouped evaluator.

Use project-root script `eval_grouped_metrics.py`.
"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    root_script = Path(__file__).resolve().parents[1] / "eval_grouped_metrics.py"
    runpy.run_path(str(root_script), run_name="__main__")
