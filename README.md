# Thyroid Project
Measure thyroid area from dissected tissues with top view images (1mm-division ruler scale), handscored-data is also compared with automatically generated data for ROI area
## Files
automation3.py Measure thyroid area from pictures and compare with handscored data
thyroidarea.ipynb Stats
environment.yml Libraries to run things
## Usage
thyroid-measure <photo folder> --out <output folder> --handscore <Excel file> 
**Statistics:** open thyroidarea.ipynb in VS Code or JupyterLab, set the data file path in the first cell, and click **Run All**.
## Data
Photos and data files are not included in this repository.
## Setup
1. Install [Anaconda](https://www.anaconda.com/download) if you don't have it.
2. Create and activate the environment:

      bash
   conda env create -f environment.yml
   conda activate thyroid-env
   pip install -e . --no-deps
   