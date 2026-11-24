# Exploring Spatial Complexity Using Extended Recurrence Plot Analysis

This repository contains all code used for the creation of my master thesis. The full thesis can be found in 
`latex/thesis.pdf`.  
In the thesis, recurrence plots of two-dimensional data and the Recurrence Quantification Analysis measures
(including the lacunrity) obtained from the respective four-dimensional recurrence plots, are evaluated. 
The data ranges from standardized artificial test patterns, 
through model data represented by a rock-paper-scissors system that models biodiversity, 
to real-world data of river catchments along the Mendocino Triple Junction.

## Structure of the respository

This is a brief overview on the content of each directory in the repository 
(some directories (`scripts/`, `old/` and `tests/`) contain additional READMEs for more information):  

* **data:** contains all data used in the thesis,
can be downloaded here: https://boxup.uni-potsdam.de/s/97GXLpzLsZHKoTP (password: recurrence_plot)
* **img:** contains the results/output
* **old:** contains old obsolete files that were no longer needed but might contain useful code fragments
* **scripts:** contains all code for the Generalized Recurrence Quantification Analysis (for 2d data) (`scripts/grqa`) 
as well as several other helper functions (`scripts/utils`)
* **tests:** contains unit tests, run with `python -m unittest`  
* **thesis:** contains the Master's thesis
 
   
* **load_catchments.py:** prepare the river catchment data by cutting them to the extent of their largest interior rectangle
and save
* **run_grqa.py:** run Recurrence Quantification Analysis
* **grqa_cluster:** batch file to run run_grqa.py on the PIK cluster (on GPU)
* **grqa_cluster_cpu:** batch file to run run_grqa.py on the PIK cluster (on CPU)
* **plot_grqa:** plot results


## Set up Python environment

This repository was created using **Python 3.13**.
* Set up a new python environment: `python -m venv /path/to/new/virtual/environment`
* Install requirements: `pip install -r requirements.txt`
* If there is a problem, consider installing the requirements from `requirements_freeze.txt`, which is a
frozen python environment from 2025-11-19


## Getting started

The jupyter notebook `tutorial.ipynb` contains an introduction to the basic functions of this repository. 
The notebook is useful to understand the workflow used in this work. It is focused on the code and therefore
only contains minimal explanation regarding the theoretical background. It is recommended to read 
``latex/thesis.pdf`` before exploring the notebook.


## Recreate results from thesis

* set up the Python environment
* download the [data](https://boxup.uni-potsdam.de/s/97GXLpzLsZHKoTP) (password: recurrence_plot) and put 
it in a data directory `grqa_mtj/data`
* if you downloaded `dtm` and `slp`, prepare the river catchments by running `python load_catchments.py` (specify your input and output directories)
* create an output directory at `scripts/img/` for the dataset you want to evaluate, e.g. `scripts/img/slp`
* modify the top of `run_grqa.py` by choosing a dataset and setting the desired parameters
  (if GPU is available, it will be detected automatically)
* run:
  * local: `python run_grqa.py`
  * batch: modify `grqa_cluster` to set the desired batch parameters, then submit job with `sbatch grqa_cluster job_name`
* outputs can be found in your specified output directory
* use `plot_grqa.py` to plot the results