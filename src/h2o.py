################################################################
# H2O properties
################################################################

import numpy as np

# molar mass
mu = 0.01801528 # [kg/mol]
# specific gas constant
R = 8.31446/mu # [J/kg/K]
# density
rho = 1000 # [kg/m3]
# technically depends on T but only varies by <5% from T = 0 C to 100 C

def p_sat(T):
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

def L(T):
    '''
    calculate latent heat of vaporization
    currently not temperature dependent

    input:
        * T [K] - local temperature

    output:
        * Lv [J/kg] - water's latent heat of vaporization
    '''
    return 2496.5e3 # [J/kg]
