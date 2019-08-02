################################################################
# set up inputs for simulated transit spectra
################################################################

import numpy as np
import pandas as pd
import src.sulfur as sulfur

# CONSTANTS
k_B = 1.38065e-23 #J/K
N_A = 6.0221409e23 #[particles/mol]
R_gas = 8.31446 #[J/mol/K]

m_H2SO4 = 1.628e-25 # [kg]
mu_h2so4 = 0.098078 # [kg/mol]
rho_h2so4 = 1830.5 # [kg/m3]

def h2so4h2o_profile(pH2SO4,r,T_strat,w_h2so4,n,p_strat,p,H,haze_h):
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
        * H [m] - scale height of stratosphere
        * haze_h [km] - haze vertical extent cutoff (distance from tropopause to n_haze=0)

    output:
        * n_aero [# water particles/m3] - number density of H2SO4-H2O aerosols
                                          at each pressure level

    '''
    # empty H2SO4 profile
    n_h2so4 = np.zeros(n)
    # pressure at top of haze (if have a haze cutoff)
    if haze_h!=None:
        p_top = p_strat*np.exp(-haze_h*1e3/H)
    else:
        # if have no haze cutoff
        p_top = 0.
    for i in range(n):
        if p[i]>p_top and p[i]<p_strat:
            n_h2so4[i] = pH2SO4[i]/k_B/T_strat
            # print('in haze, p=',p[i])
    # H2SO4 mass per aerosol
    m_h2so4_per_aero = w_h2so4*np.pi*r**3*4./3*rho_h2so4
    # number of H2SO4 molecules per aerosol
    n_h2so4_per_aero = m_h2so4_per_aero/mu_h2so4*N_A
    # create aerosol profile from h2so4 profile
    n_aero = n_h2so4/n_h2so4_per_aero
    return n_aero

def water_cloud_profile(p,p_start,n,n_water,cloud_thickness,atm):
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
        * atm [Atm] - Atm object with atmospheric properties

    output:
        * n_h2o [# water particles/m3] - number density of water particles
                                         at each pressure level
    '''
    # empty cloud profile
    n_h2o = np.zeros(n)
    # T at pressure where cloud starts
    T_start = atm.p2T(p_start)
    # local scale height to convert cloud thickness in m to Pa
    H = R_gas*T_start/atm.planet.g/atm.p2mu(p_start)
    # top of cloud -- convert cloud thickness to a pressure
    p_top_cloud = p_start*np.exp(-cloud_thickness/H)
    for i in range(n):
        if p[i]<=p_start and p[i]>p_top_cloud:
            n_h2o[i] = n_water
    return n_h2o

def input_pro(atm,n,tau,r_h2so4h2o,r_cloud=0,
              is_high_clouds=False,w=0.75,p_min=1,haze_h=None):
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
        * atm [Atm] - Atm object with atmospheric profile calculated
        * n [#] - number of natural-log-spaced pressure layers to include
        * tau [] - vertical optical depth of H2SO4-H2O aerosols
        * r_h2so4h2o [m] - radius of H2SO4-H2O aerosol particles
        * r_cloud [m] - radius of H2O cloud particles
        * is_high_clouds [boolean] - True => high clouds, false => low clouds
        * w [kg/kg] - weight percent H2SO4 in aerosol
        * p_min [Pa] - minimum pressure of atmosphere profile
        * haze_h [km] - optional, vertical thickness of haze (distance from tropopause to n_haze=0)

    outputs:
        * nothing but saves profile as a csv file of profile
          named according to inputs
    '''
    # create atmosphere profile to be saved to csv
    profile = np.zeros((n,9))
    # pressure
    profile[:,0] = np.logspace(np.log10(atm.planet.p_surf), np.log10(p_min), n)
    # temperature
    profile[:,1] = atm.p2T(profile[:,0])
    # pH2O
    profile[:,2] = atm.p2p_h2o(profile[:,0])
    # p non condensing
    p_nonh2o = (profile[:,0] - profile[:,2])

    # calculate
    if r_h2so4h2o!=0:
        # calculate extinction efficiency of H2SO4-H2O aerosols to
        # calculate proper optical depth
        aero = sulfur.Sulfur_Cycle(atm,'aero',tau=tau,r=r_h2so4h2o)
        f_so2, f_h2so4 = aero.calc_f_S()
        # pSO2
        profile[:,3] = f_so2*profile[:,0]

    #pN2
    profile[:,4] = atm.planet.atm_comp_dry[2]*p_nonh2o
    #pO2
    profile[:,5] = atm.planet.atm_comp_dry[3]*p_nonh2o
    #pCO2
    profile[:,6] = atm.planet.atm_comp_dry[4]*p_nonh2o

    # number density of H2SO4-H2O aerosols of radius r
    if r_h2so4h2o!=0:
        # scale height in straosphere
        mu_air = atm.p2mu(atm.p_transition_strat)
        H = R_gas*atm.planet.T_strat/atm.planet.g/mu_air # [m]
        profile[:,7] = h2so4h2o_profile(f_h2so4*profile[:,0],
                                          r_h2so4h2o,atm.planet.T_strat,w,n,atm.p_transition_strat,
                                          profile[:,0],H,haze_h)

    # number density of water cloud particles
    if r_cloud!=0:
        if is_high_clouds:
            p_start = atm.planet.p_surf*np.exp(-0.7*np.log(atm.planet.p_surf/atm.p_transition_strat))
            profile[:,8] = water_cloud_profile(profile[:,0],p_start,n,30.e-3,1.5e3,atm)
        else:
            profile[:,8] = water_cloud_profile(profile[:,0],atm.p_transition_moist,n,100,1.e3,atm)

    # convert Pa to bars for pressure columns
    profile[:,2:7] = profile[:,2:7]*1e-5
    profile[:,0] = profile[:,0]*1e-5

    # convert number density of H2SO4-H2O aerosols from particles/m3 to particles/cm3
    profile[:,7] = profile[:,7]*1e-6
    # make profile into dataframe for easy conversion to csv file
    profile_df = pd.DataFrame(profile, columns=['p','T','pH2O','pSO2','pN2','pO2','pCO2','n_s_aero_r_%1.e'%r_h2so4h2o,'n_water_cloudrop_r_%1.e'%r_cloud])
    # name output file based on inputs
    if haze_h != None:
        atm_name = 'tau_%1.e_r_h2so4h2o_%1.e_r_water_%1.e_haze_h_%1.f'%(tau,r_h2so4h2o,r_cloud,haze_h)
    else:
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
