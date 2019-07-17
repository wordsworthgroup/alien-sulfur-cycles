import numpy as np

# CONSTANTS
G = 6.674e-11 # [Nm2/kg2]

# EARTH SPECIFIC VALUES
M_earth = 5.9721986e24 # [kg]
R_earth = 6.371e6 # [m]

class Planet:
    '''
    Planet class (really a structure)
    planetary properties
    attributes:
        * R [m] - planet radius
        * M [kg] - planet mass
        * g [m/s2] - surface gravity
        * T_surf [K] - surface temperature
        * T_strat [K] - isothermal stratosphere temperature
        * p_surf_dry [Pa] - surface pressure of dry gases
        * p_surf [Pa] - surface pressure total (will be adjusted later to include water vapor)
        * atm_comp_dry [volume mixing ratio] - dry atmospheric composition array
                                               [H2, He, N2, O2, CO2]
                                               assume constant mixing ratios of
                                               dry species throughout atm
        * mu_dry [kg/mol] - average dry atmospheric molar mass
    methods:
        * __init__ - constructor
        * calc_mu_dry - calculate the average dry atmospheric molar mass
    '''
    def __init__(self,Rp_earth,T_surf,T_strat,p_surf,atm_comp,Mp_earth=None):
        '''
        constructor for Planet class
        initilize Planet object
        inputs:
            * Rp_earth [Earth radii] - planet radius
            * T_surf [K] - surface temperature
            * T_strat [K] - stratospheric temperature ** perhaps unneccessary?
            * p_surf [Pa] - surface pressure
            * atm_comp [volume mixing ratio] - dry atmospheric composition array
                                               [H2, He, N2, O2, CO2]
            * Mp_earth [Earth masses] - planet mass
        '''
        self.R = Rp_earth*R_earth # [m] planetary radius
        # set planetary mass
        # if no input, scale mass from radius using Valencia et al. 2006 scaling
        if Mp_earth==None:
            # Valencia et al. 2006 scaling
            conversion_V2006 = M_earth/R_earth**(1./0.27) # [whatever makes below eq have correct units]
            self.M = conversion_V2006*self.R**(1./0.27) # [kg]
        else:
            self.M = Mp_earth*M_earth # [kg]
        self.g = G*self.M/self.R**2 # [m/s2] surface gravity

        self.T_surf = T_surf # [K] surface temperature
        self.T_strat = T_strat # [K] isothermal stratosphere temperature
        self.p_surf_dry = p_surf # [Pa] surface pressure
        self.p_surf = p_surf # [Pa] surface pressure

        # confirm atmospheric composition sums to 1 to ~ machine error
        # if not, terminate proceedings
        if abs(np.sum(atm_comp) - 1.) > 1e-9:
            raise Exception('atmospheric composition does not sum to 1')
        if atm_comp.shape[0] != 5:
            raise Exception('atmospheric composition does not have the correct # of component dry gases')
        self.atm_comp_dry = atm_comp #[volume mixing ratio] dry atmospheric composition array
        self.mu_dry = 0. # [kg/mol] average dry atmospheric molar mass, will need to be set by calling calc_mu_dry

    def calc_mu_dry(self, gas_properties):
        '''
        calculate the average dry atmospheric molar mass from gas properties
        input:
            * self - Planet object
            * gas_properties - array of gas properties, col index 1 is molar mass
        '''
        self.mu_dry = np.sum(gas_properties['molar_mass'][:-1]*self.atm_comp_dry) # [kg/mol]
