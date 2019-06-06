import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d

#############################################################
# CONSTANTS
#############################################################
R_gas = 8.31446 #[J/mol/K]

mu_h2o = 0.018015 #[kg/mol]
L_h2o = 2496.5e3 #J/kg
R_h2o = R_gas/mu_h2o

mu_o2 = 0.031998 #kg/mol
mu_n2 = 0.028014 #kg/mol
mu_co2 = 0.04401 #kg/mol
R_N2 = R_gas/mu_n2
R_O2 = R_gas/mu_o2
R_CO2 = R_gas/mu_co2


# EARTH SPECIFIC VALUES
eta_air = 1.6e-5 #[Pa/s]
mu_air = 0.02896 #[kg/mol]
c_p_air=1.003*1000 #[J/kg/K]
R_air = 287.058 #J/kg/K
ep = mu_h2o/mu_air

def p_h2osat(T):
    '''
    calculate saturation partial pressure of water for a given temperature
    follows Buck equation
    input:
        * T [K] - local temperature
    output:
        * p [Pa] - saturation partial pressure of water
    '''
    T = T - 273.15
    p_water = 0.61121*np.exp((18.678-T/234.5)*(T/(257.14+T)))*1000 # over water
    p_ice = 0.61094*np.exp(17.27*T/(T+237.3))*1000 # over ice
    p = np.where(T<=0,p_ice,p_water)
    return p

#
def cp_N2(T):
    '''
    calculate specific heat for constant pressure for N2 as a function of
    temperature
    source:
    input:
        * T [K] - local temperature
    output:
        * [J/kg/K] specific heat of N2
    '''
    return 1018.7 + 0.078*T
def cp_O2(T):
    '''
    calculate specific heat for constant pressure for O2 as a function of
    temperature
    source:
    input:
        * T [K] - local temperature
    output:
        * [J/kg/K] specific heat of O2
    '''
    return 0.1369697*T + 880.9
def cp_CO2(T):
    '''
    calculate specific heat for constant pressure for CO2 as a function of
    temperature
    source:
    input:
        * T [K] - local temperature
    output:
        * [J/kg/K] specific heat of CO2
    '''
    return 1.03709*T + 530.43636


def c_p_nonc(T,f_n2,f_o2,f_co2):
    '''
    calculate specific heat for constant pressure for dry atmosphere
    assume linear combination of constituent gases' specfic heats
    weighted by mixing ratio
    inputs:
        * T [K] - local temperature
        * f_n2 [] - mixing ratio of N2 (dry)
        * f_o2 [] - mixing ratio of O2 (dry)
        * f_co2 [] - mixing ratio of CO2 (dry)

    output:
        * c [J/kg/K] - specific heat constant pressure for dry atmosphere
    '''
    c = f_n2*(cp_N2(T)) + f_o2*(cp_O2(T)) + f_co2*(cp_CO2(T))
    return c

def calc_mu_atm_dry(f_n2,f_o2,f_co2):
    '''
    calculate the average molar mass of the dry atmosphere
    inputs:
        * f_n2 [] - mixing ratio of N2 (dry)
        * f_o2 [] - mixing ratio of O2 (dry)
        * f_co2 [] - mixing ratio of CO2 (dry)
    output:
        * [kg/mol] average molar mass of dry atmosphere
    '''
    return f_n2*mu_n2 + f_o2*mu_o2 + f_co2*mu_co2

def calc_dlnpdlnT(dT,T,p,ph2o_last,f_n2,f_o2,f_co2):
    '''
    calculate d ln p / d ln T for a moist adiabat
    following Wordsworth & Pierrehumbert 2013b
    inputs:
        * dT [K] - change in local temperature
        * T [K] - local temperature
        * p [Pa] - local pressure
        * ph2o_last [Pa] - local partial pressure of water at last T step
        * f_n2 [] - mixing ratio of N2 (dry)
        * f_o2 [] - mixing ratio of O2 (dry)
        * f_co2 [] - mixing ratio of CO2 (dry)
    outputs:
        * dlnpdlnT [ln(Pa)/ln(K)] - derivative to advance moist adiabat
        * ph2o [Pa] - local partial pressure of water
    '''
    ph2o = p_h2osat(T)
    pnonc = p - ph2o
    rho_h2o = ph2o/R_h2o/T
    rho_h2o_last = ph2o_last/R_h2o/(T - dT)
    rho_nonc = pnonc/R_air/T
    alpha = rho_h2o/rho_nonc
    dlnph2odlnT = 1./dT*T*(np.log(ph2o)-np.log(ph2o_last))
    dlnrhoh2odlnT = 1./dT*T*(np.log(rho_h2o)-np.log(rho_h2o_last))
    dlnalphadlnT = T*((alpha+ep)/T - c_p_nonc(T,f_n2,f_o2,f_co2)/L_h2o)/(alpha + R_air*T/L_h2o)
    dlnpdlnT = ph2o/p*dlnph2odlnT + pnonc/p*(1 + dlnrhoh2odlnT - dlnalphadlnT)
    return dlnpdlnT, ph2o

def calc_p_dry(p,T_strat,T_s,p_s,R_air=287.,c_p_air=1.003*1000):
    '''
    calculate the temperature for a local pressure for a dry atmosphere
    inputs:
        * P [pa] - local pressure
        * T_strat [K] - stratosphere temperature
        * T_s [K] - surface temperature
        * p_s [Pa] - surface pressure
        * R_air [J/kg/K] - specific gas constant for air
        * c_p_air [J/kg/K] - specific heat for constant pressure for air
    output:
        * T [K] - local temperature
    '''
    kappa = R_air/c_p_air
    p_strat = p_s*(T_strat/T_s)**(1./kappa)
    T = np.where(p>p_strat,T_s*(p/p_s)**kappa,T_strat)
    return T

def tp_pro(p,p_transition_moist,p_transition_strat,p_surf,T_surf,T_strat,
           tp_pro_moist,R_air=287.,c_p_air=1.003*1000):
    '''
    calculate the temperature for a local pressure for a atmosphere with
    a (partial) moist adiabat
    inputs:
        * p [Pa] - local pressure
        * p_transition_moist [Pa] - pressure at which atmospheric structure
                                    transitions to a moist adiabat
        * p_transition_strat [Pa] - pressure at which atmosphere becomes
                                    isothermal
        * p_surf [Pa] - surface pressure
        * T_surf [K] - surface temperature
        * T_strat [K] - (isothermal) stratosphere temperature
        * tp_pro_moist [function] - a function that calculates temperature
                                    as a function of pressure within a given
                                    atmosphere's moist adiabat
        * R_air [J/kg/K] - specific gas constant for air
        * c_p_air [J/kg/K] - specific heat for constant pressure for air
    output:
        * T [K] - local temperature (from given pressure)
    '''
    #dry adiabat
    if p>=p_transition_moist:
        kappa = R_air/c_p_air
        T = T_surf*(p/p_surf)**kappa
    #isothermal
    elif p<=p_transition_strat:
        T = T_strat
    #moist adiabat
    else:
        T = tp_pro_moist(p)
    return T

# vectorized verision of tp_pro function
tp_pro_vec = np.vectorize(tp_pro)

def h2o_pro(p,p_transition_moist,p_transition_strat,f_h2o_surf,f_h2o_strat,tp_pro_moist):
    '''
    calculate the local partial pressure of H2O for a local pressure
    for a atmosphere with a (partial) moist adiabat
    inputs:
        * p [Pa] - local pressure
        * p_transition_moist [Pa] - pressure at which atmospheric structure
                                    transitions to a moist adiabat
        * p_transition_strat [Pa] - pressure at which atmosphere becomes
                                    isothermal
        * f_h2o_surf [] - mixing ratio of H2O at the surface
        * f_h2o_strat [] - mixing ratio of H2O in the stratosphere
        * tp_pro_moist [function] - a function that calculates temperature
                                    as a function of pressure within a given
                                    atmosphere's moist adiabat
    output:
        * ph2o [Pa] - local partial pressure H2O (from given pressure)
    '''
    #dry adiabat
    if p>=p_transition_moist:
        ph2o = p*f_h2o_surf
    #isothermal
    elif p<=p_transition_strat:
        ph2o = p*f_h2o_strat
    #moist adiabat
    else:
        ph2o = p_h2osat(tp_pro_moist(p))
    return ph2o

# vectorized verison of h2o_pro function
h2o_pro_vec = np.vectorize(h2o_pro)

def calc_moist(delta_T,T_transition_moist,p_transition_moist,T_strat,f_h2o_surf,p_surf,T_surf,f_n2,f_o2,f_co2):
    '''
    calculate pressure-temperature profile in region of atmospheric following a
    moist adiabat
    inputs:
        * delta_T [K] - temperature step size
        * T_transition_moist [K] - temperature at which atmospheric profile transitions to a moist adiabat
        * p_transition_moist [Pa] - pressure at which atmospheric profile transitions to a moist adiabat
        * T_strat [K] - temperature of stratosphere
        * f_h2o_surf [] - mixing ratio of H2O at surface
        * p_surf [Pa] - atmospheric pressure at surface
        * T_surf [K] - temperature at surface
        * f_n2 [] - mixing ratio of N2 (dry)
        * f_o2 [] - mixing ratio of O2 (dry)
        * f_co2 [] - mixing ratio of CO2 (dry)
    outputs:
        * Ts [K] - temperatures
        * np.exp(lnp)=p [Pa] - pressures
    '''
    Ts = [T_transition_moist]
    lnp = [np.log(p_transition_moist)]

    i = 0

    T_last = T_transition_moist + delta_T
    kappa = R_air/c_p_air
    p_last = p_surf*(T_last/T_surf)**(1./kappa)
    ph2o_last = p_last*f_h2o_surf
    while Ts[i]>T_strat-delta_T:
        if i==0:
            dlnpdlnT, ph2o_last = calc_dlnpdlnT(-delta_T,Ts[i],p_last,ph2o_last,f_n2,f_o2,f_co2)
        else:
            dlnpdlnT, ph2o_last = calc_dlnpdlnT(-delta_T,Ts[i],np.exp(lnp[i]),ph2o_last,f_n2,f_o2,f_co2)
        lnp.append(lnp[i] - 1./Ts[i]*delta_T*dlnpdlnT)
        Ts.append(Ts[i] - delta_T)
        i+=1
    return Ts, np.exp(lnp)

def T_transition_moist0(T,p_surf,T_surf,f_h2o=0.01,R_air=287.,c_p_air=1.003e3):
    '''
    calculate temperature at which RH = 1 and atmosphere transitions
    to a moist adiabat
    inputs:
        * T [K] - local temperature
        * p_surf [Pa] - surface pressure
        * T_surf [K] - surface temperature
        * f_h2o [] - mixing ratio of H2O at the surface
        * R_air [J/kg/K] - specific gas constant for air
        * c_p_air [J/kg/K] - specific heat for constant pressure for air
    output:
        * difference between local pH2O and saturated pH2O at local T [Pa]
    '''
    kappa = R_air/c_p_air # []
    return f_h2o*p_surf*(T/T_surf)**(1./kappa) - p_h2osat(T)


def calc_p_strat_dry(T_strat,T_surf,p_surf,R_atm,c_p_atm):
    '''
    calculate pressure of tropopause (beginning of stratosphere)
    assume dry adiabt in troposphere
    inputs:
        * T_strat [K] - temperature of isothermal stratosphere
        * p_surf [Pa] - surface pressure
        * T_surf [K] - surface temperature
        * R_air [J/kg/K] - specific gas constant for air
        * c_p_air [J/kg/K] - specific heat for constant pressure for air
    output:
        * [Pa] pressure at tropopause
    '''
    kappa = R_atm/c_p_atm
    return p_surf*(T_strat/T_surf)**(1./kappa) #[Pa]

def calc_p_strat_moist(p_surf,T_surf,T_strat,delta_T,RH_h2o_surf,
                  f_n2=0.7809,f_o2=0.2095,f_co2=0.):
    '''
    calculate pressure at the tropopause and average molecular weight
    for a given atmosphere
    assume atmosphere follows a dry adiabat until H2O becomes saturated
    and then a moist adiabat until the stratospheric temperature is
    reached
    (if no/limited H2O in atmosphere, will just follow a dry adiabat
    in the troposphere)
    inputs:
        * p_surf [Pa] - surface pressure
        * T_surf [K] - surface temperature
        * T_strat [K] - (isothermal) stratosphere temperature
        * delta_T [K] - size of T step in integration
        * RH_h2o_surf [] - relative humidity of water at the surface
        * f_n2 [] - mixing ratio of N2 (dry)
        * f_o2 [] - mixing ratio of O2 (dry)
        * f_co2 [] - mixing ratio of CO2 (dry)
    output:
        * p_strat [Pa] - pressure at the tropopause
        * mu_atm [kg/mol] - average molar mass of the atmosphere
    '''
    # calculate atmospheric parameters
    c_p_atm = c_p_nonc(T_surf,f_n2,f_o2,f_co2)
    mu_atm = calc_mu_atm_dry(f_n2,f_o2,f_co2)
    R_atm = R_gas/mu_atm
    kappa = R_atm/c_p_atm
    # surface water mixing ratio
    f_h2o_surf = RH_h2o_surf*p_h2osat(T_surf)/p_surf # []
    # check if troposphere is all dry or all moist adiabat
    isdry = False
    ismoistatsurf = False
    if T_transition_moist0(T_strat,p_surf,T_surf,f_h2o_surf,R_atm,c_p_atm)<=0:
        isdry = True
    if T_transition_moist0(T_surf,p_surf,T_surf,f_h2o_surf,R_atm,c_p_atm)>0:
        ismoistatsurf = True
    if isdry:
        p_strat = calc_p_strat_dry(T_strat,T_surf,p_surf,R_atm,c_p_atm)
    else:
        # determine where moist adiabat starts
        if ismoistatsurf:
            T_transition_moist = T_surf
            p_transition_moist = p_surf
        else:
            T_transition_moist = brentq(T_transition_moist0,T_surf, T_strat, args=(p_surf,T_surf,f_h2o_surf,R_atm,c_p_atm))
            p_transition_moist = p_surf*(T_transition_moist/T_surf)**(1./kappa)
        # integrate to get moist adiabat
        T_Tspaced,p_Tspaced = calc_moist(delta_T,T_transition_moist,p_transition_moist,T_strat,f_h2o_surf,p_surf,T_surf,f_n2,f_o2,f_co2)
        #calculate T for a given p in moist adiabat by interpolating from diff eq solution
        tp_pro_moist = interp1d(p_Tspaced,T_Tspaced)
        tp_pro_moist_tofp = interp1d(T_Tspaced,p_Tspaced)
        p_strat = tp_pro_moist_tofp(T_strat)
    return p_strat
