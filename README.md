## Installation guide
- To be able to run the provided code you will need Python (tested with Python3.11) along with a few libraries. 
- To install the libraries, run the following command from this directory: `pip install -r requirements.txt`.
- To run the SLIM model, you will need to install the CPLEX solver, you can use your academic email to acquire the full version by signing up [here](https://www.ibm.com/academic/home) and downloading it [here](https://www.ibm.com/academic/topic/data-science). Last step is to configure the [Python API](https://www.ibm.com/docs/en/icos/22.1.1?topic=cplex-setting-up-python-api) ([guide](https://mertbakir.gitlab.io/operations-research/how-to-install-cplex-ibm-academic-initiative/) with more detailed instructions).
- You might also need to install a few additional packages such as the `libtbb-dev` and the `libgmp-dev` package. On Ubuntu, you can do this by running the following command: `sudo apt install libtbb-dev libgmp-dev`.
## Usage guide
- The experiment as a whole can be found in the file `/impl/experiment.ipynb`. Rerunning this notebook takes a significant amount of time.
- We recommend using the other notebook in `/impl/plot.ipynb` to try the interactive plots provided by the `interpret` package. This is also a good place to try any of the models. To do so, you should use the functions `train_final_{modelname}`. 
