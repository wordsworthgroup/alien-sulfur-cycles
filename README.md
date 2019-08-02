# Alien Sulfur Cycles
Code associated with “Sulfate Aerosol Hazes and SO2 Gas as Constraints on Rocky Exoplanets' Surface Liquid Water” by Loftus, Wordsworth, &amp; Morley (2019), submitted ApJ. (Henceforth, LoWoMo19.)

### Logistics
The code is written in Python 3.
It requires two non-standard packages to run: PrettyTable and cycler. You can easily install these via the command line using pip:

```pip install PrettyTable```

```pip install cycler```.

After this (one time) installation and the repo downloaded, you should be good to go.

### Paper Results
To reproduce the calculations and figures in LoWoMo19, simply run:

```python lowomo19.py```

within this repo’s directory. Results relevant to LoWoMo19 analysis are printed. Figures included in the paper are saved under figs/ in the directory you're running lowomo19 from. (Note Figure 1 is not reproduced as it is a schematic illustration.)  Additional figures relevant to analysis but not included in the paper are saved under figs_sup/. 

Inputs for spectra are saved under spec_inputs/. The code for generating spectra from inputs is not included in this repo but is described in [Morley et al. 2017](https://arxiv.org/pdf/1708.04239.pdf). Resulting spectra by Caroline Morley for each generated input are included under data/simtransspec/. 

### Sulfur Model
To use LoWoMo19's sulfur model with your own inputs, you can utilize main.py. Make your desired adjustments for planet and model parameters within main.py as instructed by comments. Then run:

```python main.py```

within this repo’s directory. Results will be saved under my_results/ as described in main.py.