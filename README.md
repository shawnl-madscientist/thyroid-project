## Files
- `thyroid_project/`: package that measures thyroid area from photos and compares it with hand scores
- `notebooks/thyroidarea.ipynb`: hand-scored thyroid area statistics
- `notebooks/thyroid_bw_correlation.ipynb`: thyroid size vs body weight change, group comparisons, sex differences
- `figures/`: saved plots
- `archive/`: early practice files
- `environment.yml`: libraries to run everything

## Usage
conda env create -f environment.yml
conda activate thyroid-env
pip install -e . --no-deps
thyroid-measure PHOTO_FOLDER --out OUTPUT_FOLDER --handscore EXCEL_FILE