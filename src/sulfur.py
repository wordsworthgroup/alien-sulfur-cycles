import numpy as np
import src.atm_pro as atm_pro
import src.mie as mie

#############################################################
# CONSTANTS
#############################################################

s_in_yr = 365.25*3600.*24. # [s/yr]

G = 6.674*10**(-11) # [Nm2/kg2]
R_gas = 8.31446 # [J/mol/K]
N_A = 6.0221409e23 # [particles/mol]

mu_h2o = 0.018015 # [kg/mol]
rho_h2o = 1000 # [kg/m3]
mol_per_L_h2o = 1./mu_h2o # [mol/L]

mu_so2 = 0.06406 # [kg/mol]
m_SO2 = 1.064e-25 # [kg]

mu_S = 0.03206 # [kg/mol]

m_H2SO4 = 1.628e-25 # [kg]
mu_h2so4 = 0.098078 # [kg/mol]
rho_h2so4 = 1830.5 # [kg/m3]

# MIE
# index of refraction of H2SO4-H2O for w=75%
# source:
# https://www.cfa.harvard.edu/HITRAN/HITRAN2012/Aerosols/ascii/single_files/palmer_williams_h2so4.dat
m_r_G = 1.4315 # []
m_i_G = 0. # []
m_r_M = 1.422 # []
m_i_M = 1.53e-6 # []
lambda_G = 0.556 #[um] wavelength for a Sun-like star
lambda_M = 1. #[um] wavelength for a M-dwarf

# EARTH SPECIFIC VALUES
M_earth = 5.9721986*10**24 # [kg]
R_earth = 6.371*10**6 # [m]
mass_earth_ocean = 1.4*10**21 # [kg]

eta_air = 1.6e-5 # [Pa/s]
mu_air = 0.02896 #[kg/mol]
c_p_air=1.003*1000 # [J/kg/K]
R_air = 287.058 # [J/kg/K]

def calc_rho_p(w):
    '''
    calculate the density of a H2SO4-H2O aerosol for a given
    weight percentage (w) H2SO4
    assumes density of H2SO4-H2O mixture is a linear function of
    H2SO4 and H2O densities

    inputs:
        * w - weight percentage H2SO4

    output:
        * rho - density of aerosol
    '''
    return w*rho_h2so4 + (1-w)*rho_h2o # [kg/m3]

def calc_Cc(r,p,T,mu_atm,eta_atm=eta_air):
    '''
    calculate the Cunningham-Stokes correction factor
    dimensionless number to account for drag on small particles
    in between continuum and free molecular flow regimes

    inputs:
        * r [m] - radius of the falling particle
        * p [Pa] - local air pressure
        * T [K] - local air temperature
        * mu_atm [mol/kg] - average molecular mass of air
        * eta_atm [Pa s] - dynamic visocity of air

    output:
        * Cc [] - Cunningham-Stokes correction factor
    '''
    # mean free path of air
    mfp = 2.*eta_atm/(p*(8.*mu_atm/(np.pi*R_gas*T))**0.5) # [m]
    # Knudsen number
    Kn = mfp/r # []
    # Cunningham-Stokes correction factor
    # assuming aerosol is liquid
    Cc = 1 + Kn*(1.207+0.440*np.exp(-0.596/Kn)) #[]
    return Cc

################################################################
# AQUEOUS S(IV) CHEMISTRY
################################################################

def K1(T):
    '''
    first acid dissociation constant of H2SO3
    in theory dependent on temperature but neglected here
    source:

    input
        * T [K] - temperature (not actually used)
    output
        * K1 [log10(mol/L)] - first acid dissociation constant
    '''
    K = 10**(-1.85) # [log10(M)]
    return K


def K2(T):
    '''
    second acid dissociation constant of H2SO3
    in theory dependent on temperature but neglected here
    source:

    input
        * T [K] - temperature (not actually used)
    output
        * K2 [log10(M)] - first acid dissociation constant
    '''
    K = 10**(-7.2) # [log10(M)]
    return K


def K_H_SO2(T):
    '''
    Henry's constant of SO2 as a function of temperature
    source: Principles of Planetary Climate, Pierrehumbert, 2010
    note, need to convert units from mol/mol to M = mol/L

    input:
        * T [K] - water temperature
    output:
        * K_H [Pa/M] - Henry's constant

    '''
    K0 = 4*10**6 # [Pa]
    C_H = 2900 # [K]
    T0 = 289 # [K]
    K_H = K0*np.exp(-C_H*(1./T - 1./T0)) #[ Pa]
    K_H = K_H/mol_per_L_h2o # [Pa/M] (note M = mol/L)
    return K_H

# convert pH to concentration of H+
def H(pH):
    '''
    convert pH to concentration of H+

    input:
        * pH
    output:
        * h [M]- [H+]
    '''
    h = 10**(-pH) # [M]
    return h

def SO2_aq(p_SO2,T):
    '''
    calculate [SO2(aq)]
    assuming ocean in equilibrium with atmosphere from Henry's Law

    inputs:
        * p_SO2 [Pa] - partial pressure of SO2 at ocean-atmosphere interface
        * T [K] - temperature of ocean

    output:
        * so2 [M] - concentration of SO2(aq)
    '''
    so2 = p_SO2/K_H_SO2(T) # [M]
    return so2

def HSO3(p_SO2,T,pH):
    '''
    calculate [HSO3-]
    assuming ocean in equilibrium with atmosphere from Henry's Law,
    assuming saturation of S(IV)

    inputs:
        * p_SO2 [Pa] - partial pressure of SO2 at ocean-atmosphere interface
        * T [K] - temperature of ocean
        * pH - pH of ocean

    output:
        * hso3 [M] - [HSO3-]
    '''
    hso3 = K1(T)*SO2_aq(p_SO2,T)/H(pH)
    return hso3

def SO3(p_SO2,T,pH):
    '''
    calculate [SO3--]
    assuming ocean in equilibrium with atmosphere from Henry's Law,
    assuming saturation of S(IV)

    inputs:
        * p_SO2 [Pa] - partial pressure of SO2 at ocean-atmosphere interface
        * T [K] - temperature of ocean
        * pH - pH of ocean

    output:
        * so3 [M] - [SO3--]
    '''
    so3 = K2(T)*HSO3(p_SO2,T,pH)/H(pH)
    return so3

def S_aq_concentrations(pH,T,p_SO2=1.):
    '''
    calculate concentrations of S(IV) species (SO2(aq), HSO3-, SO3--)
    assuming ocean in equilibrium with atmosphere from Henry's Law,
    assuming saturation of S(IV)

    inputs:
        * p_SO2 [Pa] - partial pressure of SO2 at ocean-atmosphere interface
        * T [K] - temperature of ocean
        * pH [log10(M)] - pH of ocean

    outputs:
        * so2 [M] - [SO2(aq)]
        * hso3 [M] - [HSO3-]
        * so3 [M] - [SO3--]
    '''
    so2 = SO2_aq(p_SO2,T)
    hso3 = HSO3(p_SO2,T,pH)
    so3 = SO3(p_SO2,T,pH)
    return so2, hso3, so3


def S_aq_fractions(pH,T,p_SO2=1.):
    '''
    calculate S(IV) distribution among S(IV) species (SO2(aq), HSO3-, SO3--)
    assuming ocean in equilibrium with atmosphere from Henry's Law,
    assuming saturation of S(IV)

    inputs:
        * p_SO2 [Pa] - partial pressure of SO2 at ocean-atmosphere interface
        * T [K] - temperature of ocean
        * pH [log10(M)] - pH of ocean

    outputs:
        * frac_so2 [] - [SO2(aq)]/[S(IV)]
        * frac_hso3 [] - [HSO3-]/[S(IV)]
        * frac_so3 [] - [SO3--]/[S(IV)]
    '''
    so2 = SO2_aq(p_SO2,T)
    hso3 = HSO3(p_SO2,T,pH)
    so3 = SO3(p_SO2,T,pH)
    total_s = so2 + hso3 + so3
    frac_so2 = so2/total_s
    frac_hso3 = hso3/total_s
    frac_so3 = so3/total_s
    return frac_so2, frac_hso3, frac_so3

def S_atm_ocean_frac(pH,num_earth_oceans,mu_atm=mu_air,T_surf=288,R_p_earth=1,M_p_earth=1):
    '''
    calculate ratio of S in atmosphere (SO2) to
    S in ocean (SO2(aq), HSO3-, SO3--)
    assuming ocean in equilibrium with atmosphere from Henry's Law,
    assuming saturation of S(IV)

    inputs:
        * pH [log10(M)] - pH of ocean
        * num_earth_oceans [] - mass of ocean in Earth ocean masses
        * mu_atm [kg/mol] - average molecular mass of air
        * T_surf [K] - assumed ocean temperature
        * R_p_earth [] - radius of planet in Earth radii
        * M_p_earth [] - mass of planet in Earth masses

    outputs:
        * ratio [] - S atmosphere / S ocean
    '''
    # radius of planet
    R_p = R_p_earth*R_earth #[m]
    # mass of planet
    M_p = M_p_earth*M_earth #[kg]
    # surface gravity of planet
    g = M_p*G/R_p**2 #[m/s2]
    SIV = 1./K_H_SO2(T_surf)*(1 + K1(T_surf)/H(pH) + K1(T_surf)*K2(T_surf)/H(pH)**2) #[mol/L]
    # critical moles of S in the atmosphere
    mol_SIV_atm = 4*np.pi*R_p**2/mu_so2/g #[moles]
    # critical moles of S in the ocean
    mol_SIV_ocean = SIV*num_earth_oceans*mass_earth_ocean/rho_h2o*1000. #[moles]
    ratio = mol_SIV_atm/mol_SIV_ocean
    return ratio

################################################################
# OBSERVABLE SULFUR
################################################################

def calc_N_SIV(tau, r, T_surf, T_strat, p_surf, R_p_earth, M_p_earth,
                num_earth_oceans, pH, w=0.75, t_mix=s_in_yr,
                t_convert=3600.*24., alpha=1,RH_h2o_surf=0,
                f_n2=0.7809,f_o2=0.2095,f_co2=0.,is_G=True):
    '''
    calculate critical total molecules of S
    in atmosphere-ocean for an observable haze layer
    follows eqs X-Y in LWM19
    assume lower atmosphere (troposphere) follows an adiabat (dry or moist as implied by f_h2o)
    assume upper atmosphere (stratosphere) is isothermal
    INPUTS
    inputs:
        * tau [] - critical optical depth for observable haze layer
        * r [m] - average radius of H2SO4-H2O aerosol
        * Qe [] - extinction efficiency of H2SO4-H2O aerosol from Mie theory
        * T_surf [K] - surface temperature
        * T_strat [K] - temperature of the (isothermal) stratosphere
        * p_surf [Pa] - surface atmosphere pressure
        * R_p_earth [R_earth] - radius of planet in Earth radii
        * M_p_earth [M_earth] - mass of planet in Earth masses
        * num_earth_oceans [mass_earth_ocean] - mass of ocean water in Earth oceans
        * pH [log10(mol/kg)] - pH of ocean water
        * w [kg/kg] - weight percentage H2SO4 of H2SO4-H2O aerosol
        * t_mix [s] - time for mixing between stratosphere and troposphere
        * alpha [] - fSO2(surface)/fSO2(tropopause), change in mixing ratio
               of SO2 between surface and stratosphere
        * RH_h2o_surf [] - relative humidity of water at the surface
        * mu_atm [mol/kg] - average molar mass of atmosphere
        * c_p_atm [] - specific heat at constant pressure of atmosphere
    output:
        * [# atoms] critical atoms of S in ocean-atmosphere for an observable haze layer
    '''
    # density of aerosol particle
    rho_p = calc_rho_p(w) # [kg/m3]
    # radius of planet
    R_p = R_p_earth*R_earth # [m]
    # mass of planet
    M_p = M_p_earth*M_earth # [kg]
    # surface gravity of planet
    g = M_p*G/R_p**2 # [m/s2]
    # extinction efficiency
    if is_G:
        Qe = mie.mie_scatter(m_r_G, m_i_G, x0=2.*np.pi*r*1e6/lambda_G)[2]
    else:
        Qe = mie.mie_scatter(m_r_M, m_i_M, x0=2.*np.pi*r*1e6/lambda_M)[2]
    # critical molecules of H2SO4 for observable haze
    N_H2SO4 = 16.*np.pi/3.*tau/Qe*r*rho_p*R_p**2*w/m_H2SO4 # [# molecules]
    # average molar mass of atmosphere
    mu_atm = atm_pro.calc_mu_atm_dry(f_n2,f_o2,f_co2) # [kg/mol]
    # pressure of the tropopause (transition to stratosphere)
    p_strat = atm_pro.calc_p_strat_moist(p_surf,T_surf,T_strat,0.001,RH_h2o_surf,
                                         f_n2,f_o2,f_co2) # [Pa]
    # scale height of atmosphere in stratosphere
    # assumed to be the average distance an aerosol has to fall
    z_fall = R_gas*T_strat/mu_atm/g
    # stokes velocity of falling aerosol
    v_stokes = 2./9.*r**2*rho_p*g*calc_Cc(r,p_strat,T_strat,mu_atm)/eta_air # [m/s]
    # timescale for aerosol to fall to tropopause
    t_fall = z_fall/v_stokes # [s]
    # lifetime of aerosol in stratosphere
    # whichever of falling or mixing is faster
    # (will depend on size of particle)
    t_life = min(t_fall,t_mix) # [s]
    # critical partial pressure of SO2 at the tropopause
    p_so2_boundary = g/4./np.pi/R_p**2*N_H2SO4*m_SO2*t_convert/t_life #[Pa]
    # critical partial pressure of SO2 at the surface
    p_so2_surf = p_so2_boundary/p_strat*p_surf/alpha # [Pa]
    # critical concentration of S(IV) in the ocean
    SIV = p_so2_surf/K_H_SO2(T_surf)*(1 + K1(T_surf)/H(pH) + K1(T_surf)*K2(T_surf)/H(pH)**2) #[mol/L]
    # critical moles of S in the atmosphere
    # with SO2
    mol_SIV_atm = 4*np.pi*R_p**2*p_so2_surf/mu_so2/g # [moles]
    # with H2SO4
    mol_SIV_atm += N_H2SO4/N_A # [moles]
    # critical moles of S(IV) in the ocean
    mol_SIV_ocean = SIV*num_earth_oceans*mass_earth_ocean/rho_h2o*1000. #[moles]
    # critical number of S(IV) molecules in the atmosphere-ocean system
    # N_SIV_tot_crit = (mol_SIV_atm + mol_SIV_ocean)*N_A #[# molecules]
    N_SIV_tot_crit = mol_SIV_ocean*N_A # [# molecules]
    return N_SIV_tot_crit

def calc_t_SIV(tau, r, T_surf, T_strat, p_surf, R_p_earth,
                        M_p_earth, num_earth_oceans, pH, w, t_mix,
                        t_convert, alpha, n_m_outgass_earth,
                        RH_h2o_surf,f_n2,f_o2,f_co2,is_G):
    N_SIV_crit = calc_N_SIV(tau, r, T_surf, T_strat, p_surf, R_p_earth,
                            M_p_earth, num_earth_oceans, pH, w, t_mix,
                            t_convert, alpha,RH_h2o_surf,f_n2,f_o2,f_co2,is_G)
    m_outgass_SIV = 10e12*1e-3*n_m_outgass_earth # [kg S/yr]
    N_SIV_outgass = m_outgass_SIV/mu_S*N_A # [molecules S/yr]
    t_SIV_crit = N_SIV_crit/N_SIV_outgass # [yr]
    return t_SIV_crit

def calc_crit_S_atm_obs_haze(tau, r, T_surf, T_strat, p_surf, R_p_earth, M_p_earth,
                w=0.75, t_mix=s_in_yr,
                t_convert=3600.*24., alpha=1,RH_h2o_surf=0,
                f_n2=0.7809,f_o2=0.2095,f_co2=0.,is_G=True):
    # density of aerosol particle
    rho_p = calc_rho_p(w) # [kg/m3]
    # radius of planet
    R_p = R_p_earth*R_earth # [m]
    # mass of planet
    M_p = M_p_earth*M_earth # [kg]
    # surface gravity of planet
    g = M_p*G/R_p**2 # [m/s2]
    # average molar mass of atmosphere
    mu_atm = atm_pro.calc_mu_atm_dry(f_n2,f_o2,f_co2) # [kg/mol]
    # extinction efficiency
    if is_G:
        Qe = mie.mie_scatter(m_r_G, m_i_G, x0=2.*np.pi*r*1e6/lambda_G)[2]
    else:
        Qe = mie.mie_scatter(m_r_M, m_i_M, x0=2.*np.pi*r*1e6/lambda_M)[2]
    # critical molecules of H2SO4 for observable haze
    N_H2SO4 = 16.*np.pi/3.*tau/Qe*r*rho_p*R_p**2*w/m_H2SO4 # [# molecules]
    # pressure of the tropopause (transition to stratosphere)
    p_strat = atm_pro.calc_p_strat_moist(p_surf,T_surf,T_strat,0.1,RH_h2o_surf,
                                         f_n2,f_o2,f_co2) # [Pa]
    # print('p_strat = %1.3E'%p_strat, 'T_surf = %1.3F'%T_surf)
    # scale height of atmosphere in stratosphere
    # assumed to be the average distance an aerosol has to fall
    z_fall = R_gas*T_strat/mu_atm/g
    # stokes velocity of falling aerosol
    v_stokes = 2./9.*r**2*rho_p*g*calc_Cc(r,p_strat,T_strat,mu_atm)/eta_air #[m/s]
    # timescale for aerosol to fall to tropopause
    t_fall = z_fall/v_stokes # [s]
    # lifetime of aerosol in stratosphere
    # whichever of falling or mixing is faster
    # (will depend on size of particle)
    t_life = min(t_fall,t_mix) # [s]
    # critical partial pressure of SO2 at the tropopause
    p_so2_boundary = g/4./np.pi/R_p**2*N_H2SO4*m_SO2*t_convert/t_life #[Pa]
    # critical partial pressure of SO2 at the surface
    p_so2_surf = p_so2_boundary/p_strat*p_surf/alpha # [Pa]
    # critical moles of S in the atmosphere
    mol_SIV_atm = 4*np.pi*R_p**2*p_so2_surf/mu_so2/g # [moles]
    mol_SIV_atm += N_H2SO4/N_A
    return mol_SIV_atm*N_A,  mol_SIV_atm, mol_SIV_atm*mu_S # [# S molecules, # S moles, kg S]

def calc_t_SIV_SO2(u_so2_obs,p_surf,T_surf,pH,num_earth_oceans,
                   n_m_outgass_earth,R_p_earth,M_p_earth,f_n2,f_o2,f_co2):
    # radius of planet
    R_p = R_p_earth*R_earth # [m]
    # mass of planet
    M_p = M_p_earth*M_earth # [kg]
    # surface gravity of planet
    g = M_p*G/R_p**2 #[m/s2]
    # average molar mass of atmosphere
    mu_atm = atm_pro.calc_mu_atm_dry(f_n2,f_o2,f_co2) # [kg/mol]
    # critical partial pressure of SO2 at the surface
    p_so2_surf = u_so2_obs*g*mu_atm/mu_so2 # [Pa]
    # critical concentration of S(IV) in the ocean
    SIV = p_so2_surf/K_H_SO2(T_surf)*(1 + K1(T_surf)/H(pH) + K1(T_surf)*K2(T_surf)/H(pH)**2) # [mol/L]
    # critical moles of S(IV) in the atmosphere
    mol_SIV_atm = 4*np.pi*R_p**2*p_so2_surf/mu_so2/g # [moles]
    # critical moles of S(IV) in the ocean
    mol_SIV_ocean = SIV*num_earth_oceans*mass_earth_ocean/rho_h2o*1000. #[moles]
    # critical number of S(IV) molecules in the atmosphere-ocean system
    N_SIV_tot_crit = (mol_SIV_atm + mol_SIV_ocean)*N_A # [# molecules]
    m_outgass_SIV = 10e12*1e-3*n_m_outgass_earth # [kg S/yr]
    N_SIV_outgass = m_outgass_SIV/mu_S*N_A # [molecules S/yr]
    t_SIV_crit = N_SIV_tot_crit/N_SIV_outgass # [yr]
    t_SIV_crit = mol_SIV_ocean*N_A/N_SIV_outgass # [yr]
    return t_SIV_crit

def calc_crit_S_atm_obs_SO2(u_so2_obs,p_surf,T_surf,R_p_earth,M_p_earth,f_n2,f_o2,f_co2):
    # radius of planet
    R_p = R_p_earth*R_earth # [m]
    # mass of planet
    M_p = M_p_earth*M_earth # [kg]
    # surface gravity of planet
    g = M_p*G/R_p**2 # [m/s2]
    # average molar mass of atmosphere
    mu_atm = atm_pro.calc_mu_atm_dry(f_n2,f_o2,f_co2) # [kg/mol]
    # critical partial pressure of SO2 at the surface
    p_so2_surf = u_so2_obs*g*mu_atm/mu_so2 # [Pa]
    # critical moles of S(IV) in the atmosphere
    mol_SIV_atm = 4*np.pi*R_p**2*p_so2_surf/mu_so2/g # [moles]
    return mol_SIV_atm*N_A, mol_SIV_atm, mol_SIV_atm*mu_S # [# S molecules, # S moles, kg S]

def crit_S(r, tau, T_surf, T_strat, p_strat, R_p_earth=1, M_p_earth=1,w=0.75, t_mix=s_in_yr,
           t_convert=3600.*24., mu_atm=mu_air, c_p_atm=c_p_air,is_G=True):
    # density of aerosol particle
    rho_p = calc_rho_p(w) #[kg/m3]
    # radius of planet
    R_p = R_p_earth*R_earth #[m]
    # mass of planet
    M_p = M_p_earth*M_earth #[kg]
    # surface gravity of planet
    g = M_p*G/R_p**2 #[m/s2]
    # specific gas constant for atmosphere
    R_atm = R_gas/mu_atm #[]

    # extinction efficiency
    if is_G:
        Qe = mie.mie_scatter(m_r_G, m_i_G, x0=2.*np.pi*r*1e6/lambda_G)[2]
    else:
        Qe = mie.mie_scatter(m_r_M, m_i_M, x0=2.*np.pi*r*1e6/lambda_M)[2]

    # critical molecules of H2SO4 for observable haze
    N_H2SO4 = 16.*np.pi/3./Qe*r*rho_p*R_p**2*w/m_H2SO4*tau #[# molecules]
    p_h2so4_boundary = g/4./np.pi/R_p**2*N_H2SO4*m_H2SO4 #[Pa]

    # scale height of atmosphere in stratosphere
    # assumed to be the average distance an aerosol has to fall
    z_fall = R_gas*T_strat/mu_atm/g
    # stokes velocity of falling aerosol
    eta_air = 1.8325e-5*(416.16/(T_strat+120.))*(T_strat/296.15)**1.5
    v_stokes = 2./9.*r**2*rho_p*g*calc_Cc(r,p_strat,T_strat,eta_air)/eta_air #[m/s]
    # timescale for aerosol to fall to tropopause
    t_fall = z_fall/v_stokes #[s]

    # lifetime of aerosol in stratosphere
    # whichever of falling or mixing is faster
    # (will depend on size of particle)
    t_life = min(t_fall,t_mix) #[s]

    # critical partial pressure of SO2 at the tropopause
    p_so2_boundary = g/4./np.pi/R_p**2*N_H2SO4*m_SO2*t_convert/t_life #[Pa]
    f_so2 = p_so2_boundary/p_strat
    f_h2so4 = p_h2so4_boundary/p_strat
    return f_so2,f_h2so4

def calc_p_sat_h2so4(T,T0=360):
    '''
    calculate saturation partial pressure of H2SO4 for a given temperature
    source: Kulmala and Laaksonen (1990)
    inputs:
        * T [K] - local temperature
        * T0 [K] - reference temperature
    output:
        * p_sat [Pa] - saturation pressure of H2SO4 at given T
    '''
    p_sat0 = np.exp(-(10156/T0) + 16.259)*101325
    p_sat = p_sat0*np.exp(10156*(-1/T + 1/T0 + 0.38/(905 - T0)*(1 + np.log(T0/T) - T0/T)))
    return p_sat

def calc_uSO2_boundary(tau, r, T_surf, T_strat, p_surf, R_p_earth, M_p_earth,
                w=0.75, t_mix=s_in_yr,
                t_convert=3600.*24., alpha=1,RH_h2o_surf=0,
                f_n2=0.7809,f_o2=0.2095,f_co2=0.,is_G=True):
    '''
    calculate critical total molecules of S
    in atmosphere-ocean for an observable haze layer
    follows eqs X-Y in LWM19
    assume lower atmosphere (troposphere) follows an adiabat (dry or moist as implied by f_h2o)
    assume upper atmosphere (stratosphere) is isothermal
    INPUTS
    inputs:
        * tau [] - critical optical depth for observable haze layer
        * r [m] - average radius of H2SO4-H2O aerosol
        * Qe [] - extinction efficiency of H2SO4-H2O aerosol from Mie theory
        * T_surf [K] - surface temperature
        * T_strat [K] - temperature of the (isothermal) stratosphere
        * p_surf [Pa] - surface atmosphere pressure
        * R_p_earth [R_earth] - radius of planet in Earth radii
        * M_p_earth [M_earth] - mass of planet in Earth masses
        * num_earth_oceans [mass_earth_ocean] - mass of ocean water in Earth oceans
        * pH [log10(mol/kg)] - pH of ocean water
        * w [kg/kg] - weight percentage H2SO4 of H2SO4-H2O aerosol
        * t_mix [s] - time for mixing between stratosphere and troposphere
        * alpha [] - fSO2(surface)/fSO2(tropopause), change in mixing ratio
               of SO2 between surface and stratosphere
        * RH_h2o_surf [] - relative humidity of water at the surface
        * mu_atm [mol/kg] - average molar mass of atmosphere
        * c_p_atm [] - specific heat at constant pressure of atmosphere
    outout:
        * [# atoms] critical atoms of S in ocean-atmosphere for an observable haze layer
    '''
    # density of aerosol particle
    rho_p = calc_rho_p(w) # [kg/m3]
    # radius of planet
    R_p = R_p_earth*R_earth # [m]
    # mass of planet
    M_p = M_p_earth*M_earth # [kg]
    # surface gravity of planet
    g = M_p*G/R_p**2 # [m/s2]
    # extinction efficiency
    if is_G:
        Qe = mie.mie_scatter(m_r_G, m_i_G, x0=2.*np.pi*r*1e6/lambda_G)[2]
    else:
        Qe = mie.mie_scatter(m_r_M, m_i_M, x0=2.*np.pi*r*1e6/lambda_M)[2]
    # critical molecules of H2SO4 for observable haze
    N_H2SO4 = 16.*np.pi/3.*tau/Qe*r*rho_p*R_p**2*w/m_H2SO4 # [# molecules]
    # average molar mass of atmosphere
    mu_atm = atm_pro.calc_mu_atm_dry(f_n2,f_o2,f_co2) # [kg/mol]
    # pressure of the tropopause (transition to stratosphere)
    p_strat = atm_pro.calc_p_strat_moist(p_surf,T_surf,T_strat,0.001,RH_h2o_surf,
                                         f_n2,f_o2,f_co2) # [Pa]
    # scale height of atmosphere in stratosphere
    # assumed to be the average distance an aerosol has to fall
    z_fall = R_gas*T_strat/mu_atm/g
    # stokes velocity of falling aerosol
    v_stokes = 2./9.*r**2*rho_p*g*calc_Cc(r,p_strat,T_strat,mu_atm)/eta_air # [m/s]
    # timescale for aerosol to fall to tropopause
    t_fall = z_fall/v_stokes # [s]
    # lifetime of aerosol in stratosphere
    # whichever of falling or mixing is faster
    # (will depend on size of particle)
    t_life = min(t_fall,t_mix) # [s]
    # critical partial pressure of SO2 at the tropopause
    p_so2_boundary = g/4./np.pi/R_p**2*N_H2SO4*m_SO2*t_convert/t_life #[Pa]
    # convert p_so2_boundary to mass column in units of molecules/cm2
    u_so2_boundary = p_so2_boundary/g*mu_so2/mu_atm*N_A/mu_so2*1e-4 #[molecules/cm2]
    return u_so2_boundary
