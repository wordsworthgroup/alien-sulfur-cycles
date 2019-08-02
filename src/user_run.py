################################################################
# basically just the function for a non-Kait user to easily run
# LoWoMo19's sulfur cycle model
################################################################

import src.sulfur as sulfur
from src.planet import Planet
import src.atm_pro as atm_pro
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# CONSTANTS
s_in_day = 3600.*24 # [s]
s_in_yr = 365.25*s_in_day # [s]

def user_run(R_p,M_p,T_surf,T_strat,p_surf,f_h2,f_he,f_n2,f_o2,f_co2,
             RH_surf,type_obs,is_best_params,is_limiting_params,
             tau_for_obs,r,w,t_mix,t_convert,alpha,is_G,is_M,m_aero,lambda_stell,
             u_so2_for_obs,pH,oc_mass,S_outgass,name):

    # tell user what's up
    print('\nALIEN SULFUR CYCLE INITIATED.')
    print('calculating critical decay timescale of S(IV) (t_SIV*) for observable sulfur.\n')

    # confirm inputs are correct
    if is_best_params==True and is_limiting_params==True:
        raise Exception('You have set both is_best_params & is_limiting_params as True. You can only set up to one of these booleans as True.')
    if is_G==True and is_M==True:
        raise Exception('You have set both is_G & is_M as True. You can only set up to one of these booleans as True.')

    # now proceed
    # make Planet instance
    atm_comp = np.array([f_h2,f_he,f_n2,f_o2,f_co2]) # do not adjust this ordering
    planet = Planet(R_p,T_surf,T_strat,p_surf,atm_comp)
    # make Atm instance
    atm = atm_pro.Atm(planet,RH_surf)
    # set up atmosphere structure
    atm.set_up_atm_pro()

    # add outgassing as planet mass dependent
    # set up sulfur model parameters
    if is_best_params:
        # vertical optical depth for aerosol observation
        tau_for_obs = 0.1 # []
        # average H2SO4-H2O aerosol size in meters
        r = 1e-6 # [m]
        # weight fraction H2SO4 in H2SO4-H2O aerosol mixture
        w = 0.75 # [kg/kg]
        # mixing timescale between troposphere and stratosphere in seconds
        t_mix = s_in_yr # [s]
        # conversion timescale of SO2 to H2SO4 in seconds
        t_convert = 30*s_in_day # [s]
        # ratio of mixing ratio of SO2 of tropopause to surface
        alpha = 0.1 # []
    elif is_limiting_params:
        # vertical optical depth for aerosol observation
        tau_for_obs = 0.1 # []
        # average H2SO4-H2O aerosol size in meters
        r = 1e-7 # [m]
        # weight fraction H2SO4 in H2SO4-H2O aerosol mixture
        w = 0.75 # [kg/kg]
        # mixing timescale between troposphere and stratosphere in seconds
        t_mix = s_in_yr # [s]
        # conversion timescale of SO2 to H2SO4 in seconds
        t_convert = 1.7*s_in_day # [s]
        # adjust timescales if M star
        if is_M:
            t_convert = 1.25*s_in_day # [s]
        # ratio of mixing ratio of SO2 of tropopause to surface
        alpha = 1. # []

    # make Sulfur_Cycle instance
    S_cyc = sulfur.Sulfur_Cycle(atm,type_obs,tau_for_obs,r,w,t_mix,t_convert,
                                alpha,is_G,is_M,m_aero,lambda_stell,
                                u_so2_for_obs)

    # see shape of ocean parameters inputted
    is_single_pH = False
    is_single_oc_mass = False
    try:
        len(pH)
    except TypeError:
        is_single_pH = True

    try:
        len(oc_mass)
    except TypeError:
        is_single_oc_mass = True

    # if vary both ocean pH and size
    # make a grid to have all possible combinations
    if not is_single_pH and not is_single_oc_mass:
        oc_mass,pH = np.meshgrid(oc_mass,pH)

    # calculate S in ocean for atm S observation
    S_cyc.calc_oc_S(oc_mass,pH)
    # calculate t_SIV_crit
    t_SIV_crit = S_cyc.calc_t_SIV(S_outgass)

    # update user that run worked
    print('successful run.\n')

    # SAVE RESULTS

    # make directory for saving results
    usr_results_dir = './my_results/'
    os.makedirs(usr_results_dir, exist_ok=True)

    # flatten arrays if have 2D arrays
    if not is_single_pH and not is_single_oc_mass:
        pH = pH.flatten()
        oc_mass = oc_mass.flatten()
        t_SIV_crit = t_SIV_crit.flatten()


    # output planetary & model parameters
    fname_pl = usr_results_dir + name + '_params.txt'

    f = open(fname_pl,'w')
    f.write('PLANET PARAMETERS\n\n')
    f.write('R\t\t%1.3E m\n'%atm.planet.R)
    f.write('M\t\t%1.3E kg\n'%atm.planet.M)
    f.write('T_surf\t\t%1.3F K\n'%T_surf)
    f.write('T_strat\t\t%1.3F K\n'%T_strat)
    f.write('p_surf_dry\t%1.3E Pa\n'%p_surf)
    f.write('f_h2_dry\t%1.3F\n'%f_h2)
    f.write('f_he_dry\t%1.3F\n'%f_he)
    f.write('f_n2_dry\t%1.3F\n'%f_n2)
    f.write('f_o2_dry\t%1.3F\n'%f_o2)
    f.write('f_co2_dry\t%1.3F\n'%f_co2)
    f.write('RH_surf\t\t%1.3F\n'%RH_surf)
    f.write('p_surf\t\t%1.3E Pa\n\n\n'%atm.planet.p_surf)

    f.write('MODEL PARAMETERS\n\n')
    f.write('type_obs\t\t%s\n'%type_obs)
    if type_obs=='aero':
        f.write('tau_for_obs\t\t%1.2E\n'%S_cyc.tau)
        f.write('r\t\t\t%1.2E m\n'%S_cyc.r)
        f.write('w\t\t\t%1.2F kg/kg\n'%S_cyc.w)
        f.write('t_mix\t\t\t%1.3E s\n'%S_cyc.t_mix)
        f.write('t_convert\t\t%1.3E s\n'%S_cyc.t_convert)
        f.write('m_aero\t\t\t%1.2F, %1.2F i\n'%(S_cyc.m_aero.real,S_cyc.m_aero.imag))
        f.write('alpha\t\t\t%1.2F\n'%S_cyc.alpha)
        f.write('lambda_stell\t\t%1.3E m\n'%S_cyc.lambda_stell)
        f.write('S_outgass\t\t%1.3E kg S/yr'%S_cyc.m_outgass_SIV)
    elif type_obs=='gas':
        f.write('u_so2_for_obs\t\t%1.3E kg/m2'%S_cyc.u_so2)
    f.close()

    # output results
    # use pandas to make writing out to csv easy
    if is_single_pH and is_single_oc_mass:
        results = np.array([[pH,oc_mass,t_SIV_crit[0]]])
    elif is_single_pH:
        results = np.zeros((oc_mass.shape[0],3))
        results[:,0].fill(pH)
        results[:,1] = oc_mass
        results[:,2] = t_SIV_crit
    elif is_single_oc_mass:
        results = np.zeros((pH.shape[0],3))
        results[:,0] = pH
        results[:,1].fill(oc_mass)
        results[:,2] = t_SIV_crit
    else:
        results = np.zeros((pH.shape[0],3))
        results[:,0] = pH
        results[:,1] = oc_mass
        results[:,2] = t_SIV_crit
    # print(results.shape)
    results_df = pd.DataFrame(results, columns=['pH','ocean_mass_earth','t_SIV*_yr'])
    fname_t_SIV = usr_results_dir + name + '_results.csv'
    results_df.to_csv(fname_t_SIV,index=False)

    # tell user where results are saved
    print('planet parameters for run saved under:')
    print('\t %s'%fname_pl)
    print('results of t_SIV* vs ocean parameters for run saved under:')
    print('\t %s\n\n'%fname_t_SIV)


    return S_cyc
