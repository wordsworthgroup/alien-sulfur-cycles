# Alien Sulfur Cycles
Code associated with Loftus, Wordsworth, &amp; Morley (2019), in prep. (Henceforth, LoWoMo19.)

### Logistics
The code is written in Python 3.
It requires two non-standard packages to run: PrettyTable and cycler. You can easily install these via the command line using pip:

```pip install PrettyTable```

```pip install cycler```.

After this (one time) installation and the repo downloaded, you should be good to go.

### Paper Results
To reproduce the calculations and figures in LoWoMo19, simply run:

```python lowomo19.py```.

Results relevant to LoWoMo19 analysis are printed. Figures included in the paper are saved under figs/ in the directory you're running lowomo19 from. (Note Figure 1 is not reproduced as it is a schematic illustration.)  Additional figures relevant to analysis but not included in the paper are saved under figs_sup/. 

Inputs for spectra are saved under spec_inputs/. The code for generating spectra from inputs is not included in this repo. However, resulting spectra for each generated input are included under data/simtransspec/. 

### Sulfur Model
To use LoWoMo19's sulfur model with your own inputs, you can utilize main.py. 