# Thyroid Project
Measure thyroid area from dissected tissues with top view images (1mm-division ruler scale), handscored-data is also compared with automatically generated data for ROI area
## Files
automation3.py Measure thyroid area from pictures and compare with handscored data
thyroidarea.ipynb Stats
environment.yml Libraries to run things
## Usage
**Automatic measurement:** edit the file paths near the bottom of automation3.py, then run:
python automation3.py
Basically it saves a results CSV, outline pictures of each photo, and a comparison with the hand-scored areas (CSV and graph). 
**Statistics:** open thyroidarea.ipynb in VS Code or JupyterLab, set the data file path in the first cell, and click **Run All**.
## Data
Photos and data files are not included in this repository.