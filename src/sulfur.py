import numpy as np
import src.atm_pro as atm_pro
import src.mie as mie
from src.planet import Planet

#############################################################
# CONSTANTS
#############################################################
s_in_day = 3600.*24 # [s/day]
s_in_yr = 365.25*s_in_day # [s/yr]

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
rho_h2so4 = 1830.5 # [kg/m3]

# EARTH SPECIFIC VALUES
mass_earth_ocean = 1.4*10**21 # [kg]

def calc_rho_aero(w):
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

def calc_Cc(r,atm):
    '''
    calculate the Cunningham-Stokes correction factor
    dimensionless number to account for drag on small particles
    in between continuum and free molecular flow regimes

    inputs:
        * r [m] - radius of the falling particle
        * atm [Atm] - Atm object with atmospheric properties

    output:
        * Cc [] - Cunningham-Stokes correction factor
    '''
    p = atm.p_transition_strat # [Pa] local p at tropopause
    T = atm.planet.T_strat # [K] T in strat
    mu_air = atm.p2mu(p) # [mol/kg] - average molecular mass of air in stratosphere
    eta_air = atm.p2eta(p) # [Pa s] - dynamic visocity of air at tropopause
    # mean free path of air
    mfp = 2.*eta_air/(p*(8.*mu_air/(np.pi*R_gas*T))**0.5) # [m]
    # Knudsen number
    Kn = mfp/r # []
    # Cunningham-Stokes correction factor
    # assuming aerosol is liquid
    Cc = 1 + Kn*(1.207+0.440*np.exp(-0.596/Kn)) #[]
    return Cc

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

################################################################
# AQUEOUS S(IV) CHEMISTRY
################################################################

def K1(T):
    '''
    first acid dissociation constant of H2SO3
    in theory dependent on temperature but neglected here
    source: Neta & Huie (1985)

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
    source: Neta & Huie (1985)

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

def H(pH):
    '''
    convert pH to concentration of H+

    input:
        * pH [log10(M)] - pH of water
    output:
        * h [M] - [H+]
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
        * pH [log10(M)] - pH of ocean

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

def S_atm_ocean_frac(pH,num_earth_oceans,planet):
    '''
    calculate ratio of S in atmosphere (SO2) to
    S in ocean (SO2(aq), HSO3-, SO3--)
    assuming ocean in equilibrium with atmosphere from Henry's Law,
    assuming saturation of S(IV)

    inputs:
        * pH [log10(M)] - pH of ocean
        * num_earth_oceans [] - mass of ocean in Earth ocean masses
        * planet [Planet] - Planet object

    outputs:
        * ratio [] - S atmosphere / S ocean
    '''
    T_surf = planet.T_surf # [K] - assumed ocean temperature
    SIV = 1./K_H_SO2(T_surf)*(1 + K1(T_surf)/H(pH) + K1(T_surf)*K2(T_surf)/H(pH)**2) #[mol/L]
    # critical moles of S in the atmosphere
    mol_SIV_atm = 4*np.pi*planet.R**2/mu_so2/planet.g #[moles]
    # critical moles of S in the ocean
    mol_SIV_ocean = SIV*num_earth_oceans*mass_earth_ocean/rho_h2o*1000. #[moles]
    ratio = mol_SIV_atm/mol_SIV_ocean
    return ratio

################################################################
# OBSERVABLE SULFUR
################################################################

class Sulfur_Cycle:
    def __init__(self,atm,type_obs,tau=0.1,r=1e-6,w=0.75,
                 t_mix=s_in_yr,t_convert=30*s_in_day,alpha=0.1,
                 is_G=True,is_M=False,m_aero=None,lambda_stell=None,
                 u_so2=2.3e-2):
        '''
        constructor for Sulfur_Cycle class
        initialize Sulfur_Cycle object

        inputs:
            * atm [Atm] - Atm object with atmospheric properties
            * type_obs [string] - two options either 'aero' (H2SO4-H2O) or 'gas' (SO2)
                                  otherwise program will crash
            MODEL PARAMETERS
            parameters default values set to "best" guess values, listed in Table 1
            ~~ aerosol parameters ~~
            * tau [] - optical depth necessary for aerosol observation
            * r [m] - radius of H2SO4-H2O aerosol
            * w [kg/kg] - weight percent H2SO4 in aerosol
            * t_mix [s] - timescale for dynamic exchange between troposphere and stratosphere
            * t_convert [s] - timescale for conversion of SO2 to H2SO4
            * alpha [] - ratio of mixing ratio of SO2 (f_SO2) from tropopause to surface
            * is_G [bool] - star is G star, set m_aero & lambda_stell to default G values, assuming w = 0.75
            * is_M [bool] - star is M star, set m_aero & lambda_stell to default M values, assuming w = 0.75
                source for setting index of refraction of H2SO4-H2O particles for various wavelengths & weight percents
                https://www.cfa.harvard.edu/HITRAN/HITRAN2012/Aerosols/ascii/single_files/palmer_williams_h2so4.dat
            * m_aero [] - complex index of refraction of aerosol
            * lambda_stell [m] - wavelength of peak stellar spectrum
            ~~ gas parameters ~~
            * u_so2 [kg/m2] - mass column SO2 necessary for gas observation
        '''
        self.atm = atm # Atm object to store info about planet and its atmosphere
        # whether looking to observe H2SO4-H2O aerosols ('aero')
        # or SO2 gas ('gas')
        self.type_obs = type_obs
        if type_obs=='aero':
            # set up aero model parameters
            self.tau = tau
            self.r = r
            self.w = w
            self.t_mix = t_mix
            self.t_convert = t_convert
            self.alpha = alpha
            self.rho_aero = calc_rho_aero(self.w)

            # Mie parameters
            # default G star
            if is_G==True:
                self.lambda_stell = 0.556e-6 # [m]
                self.m_aero = complex(1.4315,0.) # []
            # default M star
            elif is_M==True:
                self.lambda_stell = 1e-6 # [m]
                self.m_aero = complex(1.422,1.53e-6) # []
            # specific values
            else:
                self.lambda_stell = lambda_stell # [m]
                self.m_aero = m_aero # []
            self.Qe = None

        elif type_obs=='gas':
            # set up gas model parameters
            self.u_so2 = u_so2
        else:
            raise Exception('must choose either "aero" or "gas" as type_obs input')
        # pressure of the tropopause (transition to stratosphere)
        self.p_strat = self.atm.p_transition_strat # [Pa]
        # calculate critical S neccessary for S observation in atm
        self.calc_atm_S()

    def calc_atm_S(self):
        '''
        calculate the atmospheric S necessary for atmospheric sulfur observation
        for aero obs: follows Sections 3.2-3.5 in LoWoMo19
        for gas obs: follows Section 3.1 in LoWoMo19

        inputs:
            * self
        '''
        # to observe H2SO4-H2O aerosols
        if self.type_obs == 'aero':
            # extinction efficiency
            self.Qe = mie.mie_scatter(self.m_aero,x0=2.*np.pi*self.r/self.lambda_stell)[2]
            # critical molecules of H2SO4 for observable haze
            self.N_H2SO4 = 16.*np.pi/3.*self.tau/self.Qe*self.r*self.rho_aero*self.atm.planet.R**2*self.w/m_H2SO4 # [# molecules]

            # average molar mass of atmosphere in stratosphere
            mu_atm = self.atm.p2mu(self.p_strat) # [kg/mol]
            # dynamic visocity in stratosphere
            eta_atm = self.atm.p2eta(self.p_strat) # [Pa/s]

            # scale height of atmosphere in stratosphere
            # assumed to be the average distance an aerosol has to fall
            z_fall = R_gas*self.atm.planet.T_strat/mu_atm/self.atm.planet.g
            # stokes velocity of falling aerosol
            v_stokes = 2./9.*self.r**2*self.rho_aero*self.atm.planet.g*calc_Cc(self.r,self.atm)/eta_atm # [m/s]
            # timescale for aerosol to fall to tropopause
            self.t_fall = z_fall/v_stokes # [s]
            # lifetime of aerosol in stratosphere
            # whichever of falling or mixing is faster
            # (will depend on size of particle)
            self.t_life = min(self.t_fall,self.t_mix) # [s]
            # critical partial pressure of SO2 at the tropopause
            self.p_so2_boundary = self.atm.planet.g/4./np.pi/self.atm.planet.R**2*self.N_H2SO4*m_SO2*self.t_convert/self.t_life #[Pa]
            # critical partial pressure of SO2 at the surface
            self.p_so2_surf = self.p_so2_boundary/self.p_strat*self.atm.planet.p_surf/self.alpha # [Pa]


        # to observe SO2
        elif self.type_obs=='gas':
            # average molar mass of atmosphere at surface
            mu_atm = self.atm.p2mu(self.atm.planet.p_surf) # [kg/mol]
            # critical partial pressure of SO2 at the surface
            self.p_so2_surf = self.u_so2*self.atm.planet.g*mu_atm/mu_so2 # [Pa]

        # calculate moles of S in atmosphere
        # SO2 S
        self.mol_S_atm = 4*np.pi*self.atm.planet.R**2*self.p_so2_surf/mu_so2/self.atm.planet.g # [moles]
        if self.type_obs=='aero':
            # add H2SO4 S
            self.mol_S_atm += self.N_H2SO4/N_A # [moles]
        # number of S atoms in atm
        self.N_S_atm = self.mol_S_atm*N_A # [# S atoms]
        # mass of S in atm
        self.mass_S_atm = self.mol_S_atm*mu_S # [kg S]


    def calc_oc_S(self,num_oceans_earth,pH):
        '''
        calculate S in ocean necessary for atmospheric sulfur observation
        follows Section 3.7 in LoWoMo19
        must have called calc_atm_S before

        inputs:
            * self
            * num_oceans_earth [#] - ocean mass in Earth ocean masses
            * pH [log10(M)] - pH of ocean water
        '''
        T_surf = self.atm.planet.T_surf # [K]
        # critical concentration of S(IV) in the ocean
        self.SIV_aq_conc = self.p_so2_surf/K_H_SO2(T_surf)*(1 + K1(T_surf)/H(pH) + K1(T_surf)*K2(T_surf)/H(pH)**2) #[mol/L]
        # critical moles of S(IV) in the ocean
        self.mol_S_oc = self.SIV_aq_conc*num_oceans_earth*mass_earth_ocean/rho_h2o*1000. # [moles]
        # critical number of S atoms in ocean
        self.N_S_oc = self.mol_S_oc*N_A # [# S atoms]

    def calc_t_SIV(self,m_outgass_S_earth,is_include_atm=False,is_output=True):
        '''
        calculate critical decay timescale for S(IV) to have observable S
        follows Section 3.8 in LoWoMo19
        must have called calc_atm_S & calc_oc_S before

        inputs:
            * self
            * m_outgass_S_earth [Earth avg S outgassing] - S outgassing rate
            * is_include_atm [bool] - whether to include atmospheric S in N_SIV_crit
            * is_output [bool] - whether to output t_SIV_crit at end of method

        output (optional):
            * t_SIV_crit [s] - critical S(IV) decay timescale for S observation
        '''
        # mass of S outgassed
        m_outgass_SIV = 10e12*1e-3*m_outgass_S_earth # [kg S/yr]
        # # S atoms outgassed
        self.N_SIV_outgass = m_outgass_SIV/mu_S*N_A # [atoms S/yr]
        # add ocean S to critical S
        self.N_S_crit = self.N_S_oc
        # whether to include atmospheric S in critical S total
        if is_include_atm:
            self.N_S_crit += self.N_S_atm
        # calculate critical decay timescale for S(IV) to have observable S
        self.t_SIV_crit = self.N_S_crit/self.N_SIV_outgass # [yr]
        # whether to return t_SIV_crit when method is called
        if is_output:
            return self.t_SIV_crit

    def calc_uSO2_boundary(self):
        '''
        calculate mass column SO2
        mainly intended for photochemical calculation
        must have already called calc_atm_S

        inputs:
            * self

        output:
            * u_so2_boundary - [molecules/cm2] critical atoms of S in ocean-atmosphere for an observable haze layer
        '''
        # average molar mass of atmosphere at surface
        mu_atm = self.atm.p2mu(self.atm.planet.p_surf) # [kg/mol]
        # convert p_so2_boundary to mass column in units of molecules/cm2
        u_so2_boundary = self.p_so2_boundary/self.atm.planet.g*mu_so2/mu_atm*N_A/mu_so2*1e-4 #[molecules/cm2]
        return u_so2_boundary

    def calc_f_S(self):
        '''
        calculate mixing ratios in stratosphere of SO2 and eq H2SO4
        mainly intended for generating simulated transit spectra inputs
        must have already called calc_atm_S

        input:
            * self

        outputs:
            * f_so2 [vmr] - stratospheric mixing ratio of SO2
            * f_h2so4 [vmr] - stratospheric mixing ratio of H2SO4
        '''
        # mixing ratio SO2 at tropopause
        f_so2 = self.p_so2_boundary/self.p_strat # [vmr]
        # equivalent H2SO4 partial pressure at tropopause
        p_h2so4_boundary = self.atm.planet.g/4./np.pi/self.atm.planet.R**2*self.N_H2SO4*m_H2SO4 #[Pa]
        # equivalent mixing ratio H2SO4 at tropopause
        f_h2so4 = p_h2so4_boundary/self.p_strat # [vmr]
        return f_so2,f_h2so4
