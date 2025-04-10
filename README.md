# Calculate the prestress of Emilie membranes.

This script uses an analytical model to simulate the Emilie membrane with electrodes and perforation to estimate the actual prestress on the membrane.
The calculated prestress will not be perfect but better than the analytical estimate from assuming a homogenous membrane. 


## Installation

1. Clone this repository (skip if you received the zipped folder):
    ```bash
    git clone https://github.com/tobias-slowiak/calculate_prestress.git
    cd calculate_prestress
    ```
    

2. (Optional) Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3. Install required dependencies:
    ```bash
    pip install --upgrade numpy
    pip install --upgrade scipy
    pip install --upgrade matplotlib
    pip install --upgrade ngsolve
    pip install --upgrade webgui_jupyter_widgets
    ```

## Usage

The parameters of the membrane are hard-coded in the script, look at the code in `calc_prestress.py`, you can also use `calc_prestress.ipynb` if you prefer notebooks. The comments in the code will guide you to adapt the necessary parameters. Run the script with:
```bash
python calc_prestress.py
```

## Description

The first estimate of the prestress comes from a quadratic fit of simulation results

![Prestress estimation fit](prestress_estimator.png)

using the following parameters:

```python
#All in SI units
p["Esin"], p["nu"] = 250e9, 0.23 
p["sigcr"] = 1e9 #prestress of chrome
p["sigau"] = 40e6
p["rhosin"] = 3440 
p["rhocr"] = 7140
p["rhoau"] = 19320
p["Lside"] = 1e-3

p["hsin"] = 50e-9 #height of membrane
p["hcr"] = 10e-9
p["hau"] = 90e-9

p["Rperf"] = 0.35e-3
p["mpercentage_perf"] = 0.62 #percentage of mass density left on the perforation
```

If you have these parameters, the estimate will already be good enough, if not it serves to make the simulation finish quicker.

After this you will get an output of the calculated prestress resulting from the simulation, this might take a few minutes.