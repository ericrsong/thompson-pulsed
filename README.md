# Thompson Pulsed Analysis Package

This package was made to streamline the data analysis performed on the strontium pulsed experiment in James Thompson's lab at JILA. It contains:
* parsing functions for common data sources, such as our handheld spectrum analyzer or our FFT machine
* a `Multitrace` object which contains time series data, potentially with multiple dimensions to allow for parallel processing of many shots
	* This object comes with several methods designed to perform common analysis operations on datasets, like averaging or performing Fourier transforms
* experiment-specific preprocessing designed to make writing analysis scripts easier

If there are more things you think would be useful to add, let me know! Or do it yourself. Your pick.

## Installation Notes

Installing on a Windows 10 conda-based distribution requires access to the git CLI, which you can install with `conda install git`. Then, perform the following in your Anaconda Powershell Prompt to clone the project repository and install the latest version of `thompson_pulsed`:

```
cd <parent directory for local repo>
git clone <HTTPS url for project>
cd thompson_pulsed
pip install .
```

Running `thompson_pulsed` requires `numpy`, `scipy`, `matplotlib`, and `pandas` to be installed at a minimum on your machine.