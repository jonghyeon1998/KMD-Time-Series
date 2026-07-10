"""Append a save_state cell to every NK notebook.

Families:
  univariate           → forecaster.save_state(...)
  univariate_missing   → same, EXCEPT car_parts uses two_model.magnitude
  multivariate         → mv_forecaster.save_state(...)
"""

import json
import pathlib
import sys

EXPERIMENTS = pathlib.Path(__file__).parent

# ── Cell templates ───────────────────────────────────────────────────────────

UNIVARIATE_CELL = """\
# ── Export model state for cross-dataset nested kriging layer ────────────────
from pathlib import Path
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.getcwd()), ".."))

models_dir = Path("../models")
models_dir.mkdir(parents=True, exist_ok=True)

safe_name = DATASET_NAME.replace("/", "__")
save_path = models_dir / f"{safe_name}.npz"
forecaster.save_state(save_path, dataset_name=DATASET_NAME)
print(f"Saved model state → {save_path}  ({len(forecaster._fitted_blocks)} blocks)")
"""

CAR_PARTS_CELL = """\
# ── Export model state for cross-dataset nested kriging layer ────────────────
from pathlib import Path
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.getcwd()), ".."))

models_dir = Path("../models")
models_dir.mkdir(parents=True, exist_ok=True)

safe_name = DATASET_NAME.replace("/", "__")
save_path = models_dir / f"{safe_name}.npz"
if two_model is not None and two_model.magnitude is not None:
    two_model.magnitude.save_state(save_path, dataset_name=DATASET_NAME)
    print(f"Saved magnitude model state → {save_path}  "
          f"({len(two_model.magnitude._fitted_blocks)} blocks)")
else:
    print("Skipped: two_model or magnitude not available.")
"""

MULTIVARIATE_CELL = """\
# ── Export model state for cross-dataset nested kriging layer ────────────────
from pathlib import Path
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.getcwd()), ".."))

models_dir = Path("../models")
models_dir.mkdir(parents=True, exist_ok=True)

safe_name = DATASET_NAME.replace("/", "__")
save_path = models_dir / f"{safe_name}.npz"
mv_forecaster.save_state(save_path, dataset_name=DATASET_NAME)
total_blocks = sum(
    len(fc._fitted_blocks) for fc in mv_forecaster._forecasters
    if fc._fitted_blocks is not None
)
print(f"Saved MV model state → {save_path}  "
      f"({total_blocks} blocks across {mv_forecaster.n_channels} channels)")
"""


def make_code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


MARKER = "# ── Export model state for cross-dataset nested kriging layer"


def already_patched(nb: dict) -> bool:
    for cell in nb["cells"]:
        if cell["cell_type"] == "code" and MARKER in "".join(cell["source"]):
            return True
    return False


def patch_notebook(path: pathlib.Path, cell_source: str) -> bool:
    nb = json.loads(path.read_text())
    if already_patched(nb):
        return False
    nb["cells"].append(make_code_cell(cell_source))
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    return True


def main() -> None:
    patched = skipped = 0

    # Univariate (no missing) ─────────────────────────────────────────────
    for nb_path in sorted(EXPERIMENTS.glob(
        "univariate_nested_kriging_playgrounds/*.ipynb"
    )):
        if patch_notebook(nb_path, UNIVARIATE_CELL):
            print(f"  ✓ {nb_path.name}")
            patched += 1
        else:
            print(f"  - {nb_path.name}  (already patched)")
            skipped += 1

    # Univariate with missing ─────────────────────────────────────────────
    for nb_path in sorted(EXPERIMENTS.glob(
        "univariate_missing_nested_kriging_playgrounds/*.ipynb"
    )):
        if "car_parts" in nb_path.name:
            src = CAR_PARTS_CELL
        else:
            src = UNIVARIATE_CELL
        if patch_notebook(nb_path, src):
            print(f"  ✓ {nb_path.name}")
            patched += 1
        else:
            print(f"  - {nb_path.name}  (already patched)")
            skipped += 1

    # Multivariate ────────────────────────────────────────────────────────
    for nb_path in sorted(EXPERIMENTS.glob(
        "multivariate_nested_kriging_playgrounds/*.ipynb"
    )):
        if patch_notebook(nb_path, MULTIVARIATE_CELL):
            print(f"  ✓ {nb_path.name}")
            patched += 1
        else:
            print(f"  - {nb_path.name}  (already patched)")
            skipped += 1

    print(f"\nDone: {patched} patched, {skipped} already up-to-date.")


if __name__ == "__main__":
    main()
