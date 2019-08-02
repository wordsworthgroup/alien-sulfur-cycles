################################################################
# use the model of Loftus, Wordsworth, & Morley, 2019
# to calculate the observability of atmospheric sulfur in a
# planetary atmosphere of your choosing
#
# code by Kaitlyn Loftus (2019)
################################################################

'''
CAVEAT EMPTOR
I have extensively tested the model within
the regimes outlined in LoWoMo19's sensitivity tests of
Section 4.2 and Figure 4
outside of these parameters, I am confidently hopeful that
the model should run, but if you encounter nonsensical
results outside these inputs please let me know

I hopefully have put sufficient checks in place that you
cannot run the model by entering in bogus inputs
(e.g., mixing ratios that don't sum to 1)
or unintended combinations of inputs
(e.g., two booleans set to True where only one should be)
but I am, alas, not omnipotent,
so there may be more issues or ways to crash the program
than I have dreamt of in my philosophy
I'd recommend self checking what you are inputting makes
physical sense and carefully follows the outlined
instructions, rather than relying on my anticipation
of knowing such mistakes might be made and throwing errors

also note I have NOT put any checks in place to ensure user
use is confined to planetary parameters where the
assumptions of the paper hold
(e.g., you CAN make a hot Jupiter or a planet with a super
reduced atmosphere and have the code output results,
but obviously the tenets of the paper no longer hold)
again, check yourself b4 you wreck yourself

finally, pay attention to units requested of inputs as some are non-SI
(I know--it's an atrocity to mankind
my sincere apologies to the science gods)

Kaitlyn Loftus -- kloftus@g.harvard.edu
1 Aug 2019
'''


import numpy as np
import matplotlib.pyplot as plt
import src.user_run as user_run

# CONSTANTS
s_in_day = 3600.*24 # [s]
s_in_yr = 365.25*s_in_day # [s]

# SET PLANETARY PARAMETERS HERE

# planet radius in Earth radii
R_p = 1 # [Earth radii]

# planet mass in Earth masses
# if set to None, then will determine a scaled planet mass from radius
# following Valencia et al. 2006
M_p = None # [Earth masses]

# planet surface temperature in K
T_surf = 300 # [K]

# planet isothermal stratospheric temperature in K
T_strat = 200 # [K]

# planet dry surface pressure in Pa
p_surf = 1.e5 # [Pa]

# dry atmospheric composition in volume mixing ratio
# you can include H2, He, N2, O2, and CO2
# (in addition to H2O, which is added later via a different process)
# values MUST add to 1 or your program will crash

# mixing ratio of H2
f_h2 = 0. # [vmr]
# mixing ratio of He
f_he = 0. # [vmr]
# mixing ratio of N2
f_n2 = 1. # [vmr]
# mixing ratio of O2
f_o2 = 0. # [vmr]
# mixing ratio of CO2
f_co2 = 0. # [vmr]

# surface relative humidity, float between 0 and 1
# this input determines water vapor content of your atmosphere
# pH2O = RH_surf * pH2Osat(Tsurf)
# equivalently, fH2O = RH_surf * pH2Osat(Tsurf)/(pH2O + psurf_dry)
# this setup ensures that
# pH2O will realistically scale with surface temperature
RH_surf = 0.5 # []

# -------------------------------------------------------------- #
# -------------------------------------------------------------- #

# SET MODEL PARAMETERS HERE

# first, choose whether looking to test observability of
# 1) H2SO4-H2O aerosols ('aero') or
# 2) SO2 gas ('gas')
# NOTE, only these two options are available
# any other option will crash program
type_obs = 'aero' # or 'gas'

# now, set model parameters
# you can either choose one of our two default options or
# set the parameters yourself

# for model parameters used as "best guess" inputs from LoWoMo19
# make this boolean True
# otherwise set to False
is_best_params = True # [bool]

# for model parameters used as "limiting" inputs from LoWoMo19
# make this boolean True
# otherwise set to False
# NOTE, only one of is_best_params and is_limiting_params can be set to True
# or program will crash
is_limiting_params = False

# see Table 1 for best guess and limiting inputs
# Section 3 discusses why we chose each of these inputs
# NOTE, if you have 'aero' observation type
# you must also choose which type of star you're interested in
# with is_G or is_M with these is_*_params
# all other parameters below will be ignored EVEN IF YOU SET THEM
# if either of these booleans are True

# you can also input your own choices for these model parameters below
# in which case set BOTH these booleans to False

# -------------------------------------------------------------- #

# FOR AEROSOL OBSERVATION

# if you want to set aerosol model parameters individually
# update variables here
# to use these variables, is_best_params and is_limiting_params
# must BOTH be set to False

# vertical optical depth for aerosol observation
tau_for_obs = 0.1 # []

# average H2SO4-H2O aerosol size in meters
r = 1e-6 # [m]

# weight fraction H2SO4 in H2SO4-H2O aerosol mixture
# if you adjust w from 0.75, you will need to adjust index of refraction
# (m_aero) to correspond if you wish to be self consistent
w = 0.75 # [kg/kg]

# mixing timescale between troposphere and stratosphere in seconds
t_mix = s_in_yr # [s]

# conversion timescale of SO2 to H2SO4 in seconds
t_convert = 30*s_in_day # [s]

# ratio mixing ratio of SO2 of tropopause to surface
# alpha = f_SO2(z=tropopause)/f_SO2(z=surface) in [0,1]
alpha = 0.1 # []

# Mie scattering parameters
# to calculate scattering need to know index of refraction of aerosol and
# wavelength of incident light
# option 1 is to choose "default" G or M star settings
# option 2 is to set these parameters yourself

# for default settings use the below booleans is_G and is_M
# set your desired star type to True and the other to False
# ONLY ONE is allowed to be set to True
is_G = True # [bool]
is_M = False # [bool]

# you can set index of refraction and stellar wavelength yourself
# if BOTH is_G and is_M are set to False

# peak stellar incident light in meters
lambda_stell = 1e-6 # [m]

# index of refraction of H2SO4-H2O aerosols
# dependent on incident light wavelength and weight fraction
# H2SO4 in H2SO4-H2O mixture
# to determine m_aero for your desired wavelength and w, see
# https://www.cfa.harvard.edu/HITRAN/HITRAN2012/Aerosols/ascii/single_files/palmer_williams_h2so4.dat
m_aero = complex(1,0) # []


# -------------------------------------------------------------- #

# FOR GAS OBSERVATION

# if you want to set gas model parameters individually
# update variable here
# to use this variable, is_best_params and is_limiting_params
# must BOTH be set to False

# vertical mass column of SO2 necessary for observable SO2
# in kilograms per meter squared
u_so2_for_obs = 2.3e-2 # [kg/m2]
# (if you know your mixing ratio of SO2, to convert to uSO2
# see LoWoMo19 Section 3.1)
# (this original u_so2_for_obs value also corresponds to
# 1 ppm SO2, so you can scale from that knowledge)

# -------------------------------------------------------------- #
# -------------------------------------------------------------- #

# SET OCEAN PARAMETERS HERE
# the pH and size of your ocean
# you can either run for one set of ocean parameters,
# a range of one ocean parameter and a single other,
# or a range of both ocean parameters
# NOTE, if you wish to use arrays for both ocean parameters,
# the run function handles gridding all combinations of pH and ocean mass
# only input 1D arrays

# ocean pH
pH = 7 # [log10(M)]
# example array: pH = np.linspace(1,14,100) # [log10(M)]
# ocean mass in Earth ocean masses
oc_mass = 1e-3 # [Earth ocean masses]
# example array: oc_mass = np.logspace(-4,0,100) # [Earth ocean masses]


# SET EXPECTED SULFUR PARAMETERS HERE
# rate of mass sulfur outgassing per year in average modern Earth S outgassing rate
S_outgass = 1 # [avg modern Earth S outgassing rate = 10e9 kg S/yr]

# SET RUN NAME HERE
# run name under which your results will be saved
# if you set the same name twice, your results will be overwritten
# presently, results are simply printed to terminal and saved to a csv
# planet parameters are also saved to a csv in case you are forgetful
name = 'arrakis' # https://en.wikipedia.org/wiki/Arrakis


# -------------------------------------------------------------- #
# -------------------------------------------------------------- #
# -------------------------------------------------------------- #

# this function performs your calculation for t_SIV*
# (critical decay timescale of S(IV) in ocean for observable sulfur)
# pending how you entered in ocean parameters, you will either
# get the results back as a printed output in the terminal
# or as a plot saved in ./my_results/
# (if you haven't edited the file setup structure)

# for the more intrepid user, this function returns a
# Sulfur_Cycle object from the calculations from which you can access
# many stages of potential interest from the calculation . . .
# if you read the documentation of ./src/sulfur.py to figure out the setup :)

# check out ./src/user_run.py for more information on how this function works
# if you're having issues
s_cyc = user_run.user_run(R_p,M_p,T_surf,T_strat,p_surf,f_h2,f_he,f_n2,f_o2,f_co2,
                          RH_surf,type_obs,is_best_params,is_limiting_params,
                          tau_for_obs,r,w,t_mix,t_convert,alpha,is_G,is_M,m_aero,lambda_stell,
                          u_so2_for_obs,pH,oc_mass,S_outgass,name)
