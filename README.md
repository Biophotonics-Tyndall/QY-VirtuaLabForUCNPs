# Numerical simulations for Energy Transfer Upconversion (ETU)

Author: Jean Matias
Email: jean.matias@tyndall.ie

## Setup

The code is fully built in Python v3.9 and it is recommended to start a virtual environment to avoid compatibility issues. The following steps will guide you for that.

1. First install Python 3.9 and the package manager pip
2. Install the virtual environment package in a cmd prompt (Windows) or a 3. terminal (Unix): pip install virtualenv
4. Start a virtual environment: virtualenv -p <path\to\>python3.9 symvenv
5. Activate it: .\symvenv\Scripts\activate
6. Install requirements: pip -r install .\requirements.txt
7. Add the symvenv to the jupyter kernels list: python -m ipykernel install --user --name=symvenv
8. If you need to remove it: jupyter-kernelspec uninstall symvenv
9. Start jupyter notebook: jupyter-notebook
10. Navigate to the notebooks and select the active kernel: symvenv
11. Once you finish, deactivate the symvenv: deactivate

