"""Post-hoc Vinardo rescoring for a single SigmaDock seed directory.

Loads predictions.pt from the given seed directory, scores each pose with
gnina Vinardo, and saves rescoring.pt in the same directory. The output
format matches what sample.py writes inline, so collect_scores() in
sigmadock.chem.statistics can consume it directly.

Args (CLI):
    seed_dir: Path to a seed_<N>/ directory containing predictions.pt.

Returns:
    Writes <seed_dir>/rescoring.pt with keys {"scores": ..., "failed": ...}.
    Exits non-zero if predictions.pt is missing or gnina is not on PATH.
"""

import shutil
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sigmadock" / "src"))

from sigmadock.chem.postprocessor import compute_gnina_score


def main(seed_dir: str) -> None:
    seed_path = Path(seed_dir).resolve()
    predictions_path = seed_path / "predictions.pt"
    rescoring_path = seed_path / "rescoring.pt"

    if not predictions_path.exists():
        print(f"[ERROR] predictions.pt not found in {seed_path}")
        sys.exit(1)

    if shutil.which("gnina") is None:
        print("[ERROR] gnina binary not found on PATH")
        sys.exit(1)

    if rescoring_path.exists():
        print(f"rescoring.pt already exists in {seed_path}, skipping.")
        return

    print(f"Scoring {predictions_path} with Vinardo ...")
    scores, failed = compute_gnina_score(
        predictions_path,
        scoring="vinardo",
        preprocess=False,
        no_gpu=True,
        device=None,
    )

    torch.save({"scores": scores, "failed": failed}, rescoring_path)
    print(f"Saved rescoring.pt to {rescoring_path}")
    print(f"  Scored:  {len(scores)} complexes")
    print(f"  Failed:  {len(failed)}")
    if failed:
        for entry in failed[:5]:
            print(f"    {entry}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python gnina_rescore.py <seed_dir>")
        sys.exit(1)
    main(sys.argv[1])
