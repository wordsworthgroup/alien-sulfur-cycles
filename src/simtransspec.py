################################################################
# set up inputs for simulated transit spectra
################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import src.mie as mie
import src.atm_pro as atm_pro
import src.sulfur as sulfur

# CONSTANTS
s_in_yr = 365.25*3600.*24. # [s/yr]

k_B = 1.38065e-23 #J/K
N_A = 6.0221409e23 #[particles/mol]
R_gas = 8.31446 #[J/mol/K]

m_H2SO4 = 1.628e-25 # [kg]
mu_h2so4 = 0.098078 # [kg/mol]
rho_h2so4 = 1830.5 # [kg/m3]
mu_h2o = 0.018015 #[kg/mol]
rho_h2o = 1000 #[kg/m3]

# EARTH SPECIFIC VALUES
eta_air = 1.6e-5 #[Pa/s]
mu_air = 0.02896 #[kg/mol]
c_p_air=1.003*1000 #[J/kg/K]
R_air = 287.058 #J/kg/K
ep = mu_h2o/mu_air # []
g_earth = 9.81 # [m/s2]

def h2so4h2o_profile(pH2SO4,r,T_strat,w_h2so4,n,p_strat,p):
    '''
    create the profile of H2SO4-H2O aerosols
    H2SO4-H2O aerosol number density at each pressure layer
    assumes number density of H2SO4-H2O aerosol decays exponentially
    from peak at tropopause
    inputs:
        * pH2SO4 [Pa] - partial pressure of H2SO4 profile
        * r [m] - radius of H2SO4-H2O aerosol
        * T_strat [K] - (isothermal) stratosphere temperature
        * w_h2so4 [kg/kg] - weight percentage H2SO4 in aerosol
        * n [#] - number of pressure layers in profile
        * p_strat [Pa] - pressure at tropopause (beginning of stratosphere)
        * p [Pa] - pressure profile

    output:
        * n_aero [# water particles/m3] - number density of H2SO4-H2O aerosols
                                          at each pressure level

    '''
    # empty H2SO4 profile
    n_h2so4 = np.zeros(n)
    for i in range(n):
        if pH2SO4[i]/sulfur.calc_p_sat_h2so4(T_strat)>1. and p[i]<p_strat:
            n_h2so4[i] = pH2SO4[i]/k_B/T_strat
    # H2SO4 mass per aerosol
    m_h2so4_per_aero = w_h2so4*np.pi*r**3*4./3*rho_h2so4
    # number of H2SO4 molecules per aerosol
    n_h2so4_per_aero = m_h2so4_per_aero/mu_h2so4*N_A
    # create aerosol profile from h2so4 profile
    n_aero = n_h2so4/n_h2so4_per_aero
    return n_aero

def water_cloud_profile(p,p_start,n,n_water,cloud_thickness,f_t_p,mu_air):
    '''
    create the profile of water cloud particles
    water cloud particle density at each pressure layer
    assumes number density of water/ice particles within cloud is constant
    inputs:
        * p [Pa] - pressure profile
        * p_start [Pa] - pressure at which cloud starts
        * n [#] - number of pressure layers in profile
        * n_water [# water particles/m3] - number density of water
                                           particles within cloud
        * cloud_thickness [m] - height of cloud
        * f_t_p [function] - function that converts pressure to temperature
                             in given atmosphere
        * mu_air [kg/mol] - average molar mass of air
    output:
        * n_h2o [# water particles/m3] - number density of water particles
                                         at each pressure level
    '''
    # empty cloud profile
    n_h2o = np.zeros(n)
    # T at pressure where cloud starts
    T_start = f_t_p(p_start)
    # local scale height to convert cloud thickness in m to Pa
    H = R_gas*T_start/g_earth/mu_air
    # top of cloud -- convert cloud thickness to a pressure
    p_top_cloud = p_start*np.exp(-cloud_thickness/H)
    for i in range(n):
        if p[i]<=p_start and p[i]>p_top_cloud:
            n_h2o[i] = n_water
    return n_h2o

def input_pro(p_surf,T_surf,T_strat,delta_T,n,tau,r_h2so4h2o,r_cloud=0,
        is_high_clouds=False,w=0.75,R_air=287.,
        c_p_air=1.003e3,p_min=1,RH_h2o_surf=0.75):
    '''
    generate profile for all the constituent parts of the atmosphere
    that (can) affect the transmission spectra: gases and particles
    gases in bars and particle number densities in # particles/cm3
    assumes Earth-like atmospheric composition
    assume atmosphere follows a dry adiabat until H2O becomes saturated
    and then a moist adiabat until the stratospheric temperature is
    reached
    H2SO4-H2O aerosols can be toggled on or off
    convective water or high atmosphere ice clouds can be toggled on or off
    inputs:
        * p_surf [Pa] - surface pressure
        * T_surf [K] - surface temperature
        * T_strat [K] - (isothermal) stratosphere temperature
        * delta_T [K] -
        * n [#] - number of natural-log-spaced pressure layers to include
        * tau [] - vertical optical depth of H2SO4-H2O aerosols
        * r_h2so4h2o [m] - radius of H2SO4-H2O aerosol particles
        * r_cloud [m] - radius of H2O cloud particles
        * is_high_clouds [boolean] - True => high clouds, false => low clouds
        * w [kg/kg] - weight percent H2SO4 in aerosol
        * R_air [] - specific gas constant of air
        * c_p_air [] - specific heat for constant pressure of air
        * p_min [Pa] - minimum pressure of atmosphere profile
        * RH_h2o_surf [] - relative humidity of water at the surface
    outputs:
        * nothing but saves profile as a csv file of profile
          named according to inputs
    '''
    # surface water mixing ratio
    f_h2o_surf = RH_h2o_surf*atm_pro.p_h2osat(T_surf)/p_surf # []

    # determine where moist adiabat starts
    T_transition_moist = brentq(atm_pro.T_transition_moist0,T_surf, T_strat, args=(p_surf,T_surf,f_h2o_surf,R_air,c_p_air))
    kappa = R_air/c_p_air
    p_transition_moist = p_surf*(T_transition_moist/T_surf)**(1./kappa)

    # Earth-like composition
    f_n2_dry = 0.7809
    f_o2_dry = 0.2095
    f_co2_dry = 400.e-6

    # integrate to get moist adiabat
    T_Tspaced,p_Tspaced = atm_pro.calc_moist(delta_T,T_transition_moist,p_transition_moist,T_strat,f_h2o_surf,p_surf,T_surf,f_n2_dry,f_o2_dry,f_co2_dry)

    #calculate T for a given P in moist adiabat by interpolating from diff eq solution
    tp_pro_moist = interp1d(p_Tspaced,T_Tspaced)
    tp_pro_moist_tofp = interp1d(T_Tspaced,p_Tspaced)
    p_strat = tp_pro_moist_tofp(T_strat)
    f_h2o_strat = atm_pro.p_h2osat(T_strat)/p_strat
    f_nonh2o_strat = 1 - f_h2o_strat

    # create atmosphere profile to be saved to csv
    profile = np.zeros((n,9))
    # pressure
    profile[:,0] = np.logspace(np.log10(p_surf), np.log10(p_min), n)
    # temperature
    profile[:,1] = list(map(lambda x: atm_pro.tp_pro(x,p_transition_moist,p_strat,p_surf,T_surf,T_strat,tp_pro_moist),profile[:,0]))
    # pH2O
    profile[:,2] = list(map(lambda x: atm_pro.h2o_pro(x,p_transition_moist,p_strat,f_h2o_surf,f_h2o_strat,tp_pro_moist),profile[:,0]))
    # p non condensing
    p_nonh2o = (profile[:,0] - profile[:,2])

    # calculate
    if r_h2so4h2o!=0:
        # calculate extinction efficiency of H2SO4-H2O aerosols to
        # calculate proper optical depth
        f_so2_dry, f_h2so4_dry = sulfur.crit_S(r_h2so4h2o, tau, T_surf, T_strat,
                                               p_strat, w=w,t_mix=s_in_yr,
                                               t_convert=3600.*24.*30)
        # pSO2
        profile[:,3] = f_so2_dry*p_nonh2o

    #pN2
    profile[:,4] = f_n2_dry*p_nonh2o
    #pO2
    profile[:,5] = f_o2_dry*p_nonh2o
    #pCO2
    profile[:,6] = f_co2_dry*p_nonh2o

    # number density of H2SO4-H2O aerosols of radius r
    if r_h2so4h2o!=0:
        profile[:,7] = h2so4h2o_profile(f_h2so4_dry*p_nonh2o,
                                          r_h2so4h2o,T_strat,w,n,p_strat,
                                          profile[:,0])

    # number density of water cloud particles
    if r_cloud!=0:
        if is_high_clouds:
            p_start = p_surf*np.exp(-0.7*np.log(p_surf/p_strat))
            profile[:,8] = water_cloud_profile(profile[:,0],p_start,n,30.e-3,1.5e3,tp_pro_moist,mu_air)
        else:
            profile[:,8] = water_cloud_profile(profile[:,0],p_transition_moist,n,100,1.e3,tp_pro_moist,mu_air)

    # convert Pa to bars for pressure columns
    profile[:,2:7] = profile[:,2:7]*1e-5
    profile[:,0] = profile[:,0]*1e-5

    # convert number density of H2SO4-H2O aerosols from particles/m3 to particles/cm3
    profile[:,7] = profile[:,7]*1e-6
    # make profile into dataframe for easy conversion to csv file
    profile_df = pd.DataFrame(profile, columns=['p','T','pH2O','pSO2','pN2','pO2','pCO2','n_s_aero_r_%1.e'%r_h2so4h2o,'n_water_cloudrop_r_%1.e'%r_cloud])
    # name output file based on inputs
    atm_name = 'tau_%1.e_r_h2so4h2o_%1.e_r_water_%1.e'%(tau,r_h2so4h2o,r_cloud)
    # save profile as csv
    profile_df.to_csv('./spec_inputs/atm_pro_'+atm_name+'.csv',index=False)

def calc_avg_spec(fname,n_chunks=500):
    '''
    average spectrum over log spaced bins
    inputs:
        * fname [string] - file name containing spectrum
        * n_chunks [] - number of bins to average to
    outputs:
        * avg_wvlngth [um] - wavelengths corresponding to averaged spectrum
        * avg_spec [ppm] - averaged spectrum
    '''
    spect = np.genfromtxt(fname)
    spect = spect[np.where(spect[:,0]<=55)]
    wvlngth = spect[:,0]
    spct = spect[:,1]
    m = wvlngth.shape[0]
    avg_wvlngth = np.logspace(np.log10(np.amax(wvlngth)),np.log10(np.amin(wvlngth)),n_chunks+1)
    i = 0
    avg_spec = np.zeros(n_chunks)
    for j,w in enumerate(avg_wvlngth):
        spec_avger = []
        if j!=0:
            if i<m:
                while wvlngth[i]>=w:
                    spec_avger.append(spct[i])
                    i+=1
                    if i>=m:
                        break

            avg_spec[j-1] = np.mean(spec_avger)
    avg_wvlngth = avg_wvlngth[:-1]
    return avg_wvlngth, avg_spec*1e6
