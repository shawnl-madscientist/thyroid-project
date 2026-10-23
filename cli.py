import argparse
import os

from .measurement import run_batch
from .compare import compare_to_handscore


def main():
    # Define the options you can type after the command.
    parser = argparse.ArgumentParser(description="Measure thyroid area from photos.")
    parser.add_argument("photos", help="folder of .jpg photos")
    parser.add_argument("--out", default="results", help="folder for output files")
    parser.add_argument("--handscore", help="Excel file of hand-scored areas (optional)")
    args = parser.parse_args()

    # Make the output folder if it doesn't exist yet.
    os.makedirs(args.out, exist_ok=True)

    # Measure every photo; this saves the CSV and the outline pictures.
    df = run_batch(args.photos,
                   os.path.join(args.out, "overlays"),
                   os.path.join(args.out, "thyroid_measurements.csv"))

    # Only compare with hand scores if a hand-score file was given.
    if args.handscore:
        compare_to_handscore(df, args.handscore,
                             os.path.join(args.out, "thyroid_comparison.csv"),
                             os.path.join(args.out, "thyroid_comparison.png"))


if __name__ == "__main__":
    main()