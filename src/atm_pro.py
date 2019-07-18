import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import src.h2o as h2o

#############################################################
# CONSTANTS
#############################################################
# ideal gas constant
R_gas = 8.31446 #[J/mol/K]
# Avogadro's number
N_A = 6.022141e23 # [particles/mol]

# read in gas data
gas_properties = np.genfromtxt('./data/gas_properties.csv',delimiter=',',names=True)

def T_transition_moist0(T,p_surf,T_surf,f_h2o,kappa):
    '''
    calculate temperature at which RH = 1 and atmosphere transitions
    to a moist adiabat

    inputs:
        * T [K] - local temperature
        * p_surf [Pa] - surface pressure
        * T_surf [K] - surface temperature
        * f_h2o [] - mixing ratio of H2O at the surface
        * kappa [J/kg/K] - R_air/c_p_air

    output:
        * difference between local pH2O and saturated pH2O at local T [Pa]
    '''
    return f_h2o*p_surf*(T/T_surf)**(1./kappa) - h2o.p_sat(T)


class Atm:
    '''
    Atm class
    atmospheric properties and atmospheric structure

    attributes:
        * planet - Planet object containing planetary properties
        * RH_surf [] - relative humidity at surface, set from input value
        * f_h2o_surf [vmr] - mixing ratio of water at the surface
        * R_air_dry [J/kg/K] - specific gas constant for dry air
        * R_air_surf [J/kg/K] - specific gas constant for air at surface
        * ep [] - ratio of h2o molar mass to dry atmosphere molar mass
        * c_p_surf [J/kg/K] - specific heat at surface water levels
        * kappa [] - exponent for dry adiabat
        * delta_T [K] - temperature step size for integrate moist adiabat
        * T_transition_moist [K] - temperature at which
        * p_transition_moist [Pa] - pressure at which moist adiabat starts
        * p_transition_strat [Pa] - pressure of tropopause
        * f_h2o_strat [vmr] - mixing ratio of water in stratosphere
        * standard_ps [Pa] - pressure array that captures changes in atmospheric profile

    methods:
        * __init__ - constructor
        * set_up_atm_pro - calculate atmospheric profile
        * tp_pro - calculate T from pressures
        * tp_pro_vec - vectorized form of tp_pro
        * h2o_pro - calculate p_h2o from pressures
        * h2o_pro_vec - vectorized form of h2o_pro
        * p2z - convert pressure to altitude
        * z2p - convert altitude to pressure
        * p2T - convert pressure to temperature
        * p2p_h2o - convert pressure to water partial pressure
        * p2atm_comp - convert pressure to atmospheric composition
        * p2mu - convert pressure to average molar mass of atmosphere
        * p2RH - convert pressure to relative humidity
        * calc_c_p - calculate specific heat capacity
        * calc_dlnpdlnT - calculate key derivative for moist adiabat integration
        * calc_z_v_p - integrate to get z(p)
        * plot_tp_pro - plot p vs T
        * plot_f_h2o_pro - plot p vs f_h2o

    '''
    def __init__(self,pl,RH_surf,delta_T=0.1):
        '''
        constructor for Atm class
        initialize Atm object

        inputs:
            * pl [Planet object] - Planet object containing planetary properties
            * RH_surf [] - relative humidity at the surface
            * delta_T [K] - temperature step size for integrate moist adiabat, optional
        '''
        self.planet = pl # Planet object containing planetary properties
        # set average dry molar mass
        self.planet.calc_mu_dry(gas_properties)

        self.RH_surf = RH_surf # [] relative humidity at the surface
        # set water mixing ratio at surface from prescribed RH and T_surf
        p_h2o_surf =  self.RH_surf*h2o.p_sat(self.planet.T_surf)
        self.planet.p_surf = self.planet.p_surf_dry + p_h2o_surf
        self.f_h2o_surf = p_h2o_surf/self.planet.p_surf # [vmr] water mixing ratio at surface
        # specific gas constant for dry air
        self.R_air_dry = R_gas/self.planet.mu_dry # [J/kg/K]
        # specific gas constant for air with surface f_h2o
        self.R_air_surf = (1-self.f_h2o_surf)*self.R_air_dry + self.f_h2o_surf*h2o.R # [J/kg/K]

        # used in moist adiabat calculation
        self.ep = h2o.mu/self.planet.mu_dry # []
        # calculate specific heat at surface
        # used for dry adiabat
        self.c_p_surf = self.calc_c_p(self.planet.T_surf,1-self.f_h2o_surf) # [J/kg/K]
        # calculate kappa for dry adiabat
        self.kappa = self.R_air_surf/self.c_p_surf
        # temperature step for moist adiabat calculation
        self.delta_T = delta_T # [K]


        # check if troposphere is too dry or oversaturated
        self.isdry = False
        self.issatsurf = False
        if T_transition_moist0(self.planet.T_strat,self.planet.p_surf,self.planet.T_surf,self.f_h2o_surf,self.kappa)<=0:
            self.isdry = True
        if T_transition_moist0(self.planet.T_surf,self.planet.p_surf,self.planet.T_surf,self.f_h2o_surf,self.kappa)>0:
            self.issatsurf = True

        self.standard_ps = np.zeros(150)

        self.tp_pro_vec = np.vectorize(self.tp_pro)
        self.h2o_pro_vec = np.vectorize(self.h2o_pro)


    def calc_c_p(self,T,f_dry=1):
        '''
        calculate specific heat capacity at constant pressure for given T
        source: A.6 of Fundamentals of Thermodynamics Borgnakke & Sonntag 2009
        c_p_i(T) = 1000*(sum over j of c_p_j*T^j/1000) for j=0-3
        c_p(T) = sum over i of c_p_i(T)*q_i

        inputs:
            * T [K] - local temperature
            * f_dry [volume mixing ratio] - volume mixing ratio of dry species

        outputs:
            * c_p [J/kg/K] - specific heat capacity
        '''
        mu_avg = self.planet.mu_dry*f_dry + (1-f_dry)*h2o.mu
        theta = T/1000.
        # calculate dry component of heat capacity
        c_p = f_dry/mu_avg*np.sum(gas_properties['molar_mass'][:-1]*self.planet.atm_comp_dry*(gas_properties['c_p_0'][:-1] + gas_properties['c_p_1'][:-1]*theta + gas_properties['c_p_2'][:-1]*theta**2 + gas_properties['c_p_3'][:-1]*theta**3))
        # if necessary add water's contribution to heat capacity
        if f_dry!= 1:
            c_p += h2o.mu/mu_avg*(1-f_dry)*(gas_properties['c_p_0'][-1] + gas_properties['c_p_1'][-1]*theta + gas_properties['c_p_2'][-1]*theta**2 + gas_properties['c_p_3'][-1]*theta**3)
        return c_p*1000


    def calc_dlnpdlnT(self,dT,T,p,p_h2o_last):
        '''
        calculate d ln p / d ln T for a moist adiabat
        following Wordsworth & Pierrehumbert 2013b

        inputs:
            * dT [K] - change in local temperature
            * T [K] - local temperature
            * p [Pa] - local pressure
            * ph2o_last [Pa] - local partial pressure of water at last T step

        outputs:
            * dlnpdlnT [ln(Pa)/ln(K)] - derivative to advance moist adiabat
            * ph2o [Pa] - local partial pressure of water
        '''
        p_h2o = h2o.p_sat(T)
        p_nonc = p - p_h2o
        rho_h2o = p_h2o/h2o.R/T
        rho_h2o_last = p_h2o_last/h2o.R/(T - dT)
        rho_nonc = p_nonc/self.R_air_dry/T
        alpha = rho_h2o/rho_nonc
        dlnph2odlnT = 1./dT*T*(np.log(p_h2o)-np.log(p_h2o_last))
        dlnrhoh2odlnT = 1./dT*T*(np.log(rho_h2o)-np.log(rho_h2o_last))
        dlnalphadlnT = T*((alpha+self.ep)/T - self.calc_c_p(T)/h2o.L(T))/(alpha + self.R_air_dry*T/h2o.L(T))
        dlnpdlnT = p_h2o/p*dlnph2odlnT + p_nonc/p*(1 + dlnrhoh2odlnT - dlnalphadlnT)
        return dlnpdlnT, p_h2o

    def calc_moist(self):
        '''
        calculate pressure-temperature profile in region of atmosphere following a
        pseudo moist adiabat

        inputs:
            * self

        outputs:
            * Ts [K] - temperatures
            * np.exp(lnp)=p [Pa] - pressures
        '''
        Ts = [self.T_transition_moist]
        lnp = [np.log(self.p_transition_moist)]

        i = 0

        T_last = self.T_transition_moist + self.delta_T
        p_last = self.planet.p_surf*(T_last/self.planet.T_surf)**(1./self.kappa)
        p_h2o_last = p_last*self.f_h2o_surf
        while Ts[i]>self.planet.T_strat-self.delta_T:
            if i==0:
                dlnpdlnT, p_h2o_last = self.calc_dlnpdlnT(-self.delta_T,Ts[i],p_last,p_h2o_last)
            else:
                dlnpdlnT, p_h2o_last = self.calc_dlnpdlnT(-self.delta_T,Ts[i],np.exp(lnp[i]),p_h2o_last)
            lnp.append(lnp[i] - 1./Ts[i]*self.delta_T*dlnpdlnT)
            Ts.append(Ts[i] - self.delta_T)
            i+=1
        return Ts, np.exp(lnp)


    def tp_pro(self,p,tp_pro_moist):
        '''
        calculate the temperature for a local pressure for a atmosphere with
        a (partial) moist adiabat

        inputs:
            * self
            * p [Pa] - local pressure
            * tp_pro_moist [function] - a function that calculates temperature
                                        as a function of pressure within a given
                                        atmosphere's moist adiabat

        output:
            * T [K] - local temperature (from given pressure)
        '''
        #dry adiabat
        if p>=self.p_transition_moist:
            T = self.planet.T_surf*(p/self.planet.p_surf)**self.kappa
        #isothermal
        elif p<=self.p_transition_strat:
            T = self.planet.T_strat
        #moist adiabat
        else:
            T = tp_pro_moist(p)
        return T

    def h2o_pro(self,p,tp_pro_moist):
        '''
        calculate the local partial pressure of H2O for a local pressure
        for a atmosphere with a (partial) moist adiabat

        inputs:
            * self
            * p [Pa] - local pressure
            * tp_pro_moist [function] - a function that calculates temperature
                                        as a function of pressure within a given
                                        atmosphere's moist adiabat

        output:
            * p_h2o [Pa] - local partial pressure H2O (from given pressure)
        '''
        #dry adiabat
        if p>=self.p_transition_moist:
            p_h2o = p*self.f_h2o_surf
        #isothermal
        elif p<=self.p_transition_strat:
            p_h2o = p*self.f_h2o_strat
        #moist adiabat
        else:
            p_h2o = h2o.p_sat(tp_pro_moist(p))
        return p_h2o

    def set_up_atm_pro(self):
        '''
        set up atmospheric profile

        input:
            * self
        '''
        # calculate T at which RH first exceeds 100%
        if self.isdry:
            self.p_transition_strat = self.planet.p_surf*(self.planet.T_strat/self.planet.T_surf)**(1./self.kappa) #[Pa]
            self.p_transition_moist = 0 # [Pa]

            tp_pro_moist = None

            # set up pressure levels that will resolve different regions of the atmosphere well
            # dry adiabat
            self.standard_ps[:100] = np.logspace(np.log10(self.planet.p_surf),np.log10(self.p_transition_strat + 0.1),100)
            # stratosphere
            self.standard_ps[100:] = np.logspace(np.log10(self.p_transition_strat),np.log10(self.p_transition_strat*0.1),50)
        else:
            if self.issatsurf:
                self.T_transition_moist = self.planet.T_surf # [K]
                self.p_transition_moist = self.planet.p_surf # [Pa]
            else:
                self.T_transition_moist = brentq(T_transition_moist0,self.planet.T_surf,
                                            self.planet.T_strat, args=(self.planet.p_surf,
                                            self.planet.T_surf,self.f_h2o_surf,
                                            self.kappa))
                # convert T transition to a pseudo moist adiabat to a p
                self.p_transition_moist = self.planet.p_surf*(self.T_transition_moist/self.planet.T_surf)**(1./self.kappa)
            # integrate to get moist adiabat
            T_Tspaced,p_Tspaced = self.calc_moist()
            #calculate T for a given p in moist adiabat by interpolating from diff eq solution
            tp_pro_moist = interp1d(p_Tspaced,T_Tspaced,fill_value='extrapolate')
            # fill_value='extrapolate' used to ensure small rounds errors in
            # log conversions don't cause interpolation to fail
            tp_pro_moist_tofp = interp1d(T_Tspaced,p_Tspaced)
            # calculate pressure level at which stratosphere begins
            self.p_transition_strat = tp_pro_moist_tofp(self.planet.T_strat)

            # set up pressure levels that will resolve different regions of the atmosphere well
            # dry adiabat
            self.standard_ps[:50] = np.logspace(np.log10(self.planet.p_surf),np.log10(self.p_transition_moist + 0.1),50)
            # moist adiabat
            self.standard_ps[50:100] = np.logspace(np.log10(self.p_transition_moist),np.log10(self.p_transition_strat + 0.1),50)
            # stratosphere
            self.standard_ps[100:] = np.logspace(np.log10(self.p_transition_strat),np.log10(self.p_transition_strat*0.1),50)

        # calculate mixing ratio of water in stratosphere
        self.f_h2o_strat = h2o.p_sat(self.planet.T_strat)/self.p_transition_strat
        # calculate mixing ratio of dry gases in stratosphere
        self.f_dry_strat = 1 - self.f_h2o_strat

        # calculate temperatures associated with these standard pressure levels
        standard_Ts = self.tp_pro_vec(self.standard_ps,tp_pro_moist)
        # create interpolating function from these standard values to
        # be able to calculate T from any p easily
        self.p2T = interp1d(self.standard_ps,standard_Ts,fill_value='extrapolate')

        # calculate partial pressures of H2O associated with standard pressure levels
        standard_p_h2o = self.h2o_pro_vec(self.standard_ps,tp_pro_moist)
        # create interpolating function from these standard values to
        # be able to calculate p_h2o from any p easily
        self.p2p_h2o = interp1d(self.standard_ps,standard_p_h2o,fill_value='extrapolate')

    def p2atm_comp(self,p):
        '''
        calculate atmospheric composition accounting for varying water content
        as a function of pressure
        only will work once set_up_atm_pro has been called!!

        inputs:
            * self
            * p [Pa] - local pressure

        output:
            * X [vmr] - mixing ratios
        '''
        X = np.zeros(6)
        # calculate f_h2o
        X[-1] = self.p2p_h2o(p)/p
        # adjust known dry vmr for water content
        X[:-1] = (1-X[-1])*self.planet.atm_comp_dry
        return X

    def p2mu(self,p):
        '''
        calculate average atmospheric molar mass (mu) properly accounting for water vapor content

        inputs:
            * self
            * p [Pa] - local pressure

        output:
            * mu [kg/mol] - avg atmospheric molar mass @ pressure p
        '''
        # weight avg molar mass by mixing ratio
        mu = np.sum(self.p2atm_comp(p)*gas_properties['molar_mass'])
        return mu


    def p2eta(self,p):
        '''
        calculate viscosity of air (eta) as a function of local pressure
        follow Sutherland's law for temperature dependence of viscosity
        follow Wilkes' rule for mixtures for combining different gases into "air"

        inputs:
            * self
            * p [Pa] - local pressure

        output:
            * eta [Pa s] - dynamic viscosity
        '''
        T = self.p2T(p)
        X = self.p2atm_comp(p)
        n = X.shape[0]
        # calc eta for individual gases based on Sutherland's Law
        eta_i = (gas_properties['eta_T0'] + gas_properties['eta_C'])/(T + gas_properties['eta_C']) * gas_properties['eta_eta0']* (T/gas_properties['eta_T0'])**1.5
        eta = 0.
        for i in range(n):
            denom = 0.
            for j in range(n):
                phi = (1 + (eta_i[i]/eta_i[j])**0.5*(gas_properties['molar_mass'][j]/gas_properties['molar_mass'][i])**0.25)**2/(4./np.sqrt(2)*(1 + gas_properties['molar_mass'][i]/gas_properties['molar_mass'][j])**0.5)
                denom += X[j]*phi
            eta += X[i]*eta_i[i]/denom
        return eta


    def p2RH(self,p):
        '''
        calculate relative humidity of air (RH) at local pressure level

        inputs:
            * self
            * p [Pa] - local pressure

        output:
            * RH [] - local relative humiduty
        '''
        p_sat_loc = h2o.p_sat(self.p2T(p))
        RH = self.p2p_h2o(p)/p_sat_loc
        return RH


    def setup_z(self):
        '''
        set up functions to translate p to z and vice versa

        input:
            * self
        '''
        # integrate to get altitude as a function of pressure now that can calculate
        # T(p) and atmospheric composition as a function of p
        p_pspaced,z_pspaced = self.calc_z_v_p()
        # create interpolating function from this integration to
        # be able to calculate z from any p easily
        self.p2z = interp1d(p_pspaced,z_pspaced,fill_value='extrapolate')
        # and be able to calculate p from any z easily
        self.z2p = interp1d(z_pspaced,p_pspaced,fill_value='extrapolate')

    def calc_z_v_p(self):
        '''
        calculate z as a function of p
        assuming HSE and ideal gas law:
        dz = - (R T)/(p g mu) dp
        integrate this expression from surface using Euler's method
        up into stratosphere

        inputs:
            * self

        outputs:
            * p [Pa] - pressures
            * z [m] - altitudes
        '''
        # initial conditions from surface
        z = [0]
        p = [self.planet.p_surf]
        while p[-1]>=self.standard_ps[-1]:
            # scale dp with pressure but don't get too bogged down in small steps
            dp = p[-1]*1e-3
            if p[-1]-dp<1:
                dp = self.standard_ps[-1]*0.5
            # calculate molar mass at given pressure height
            mu = self.p2mu(p[-1])
            # calculate dz|z=z_i
            dz = R_gas*self.p2T(p[-1])/p[-1]/mu/self.planet.g*dp
            # z_i+1 = z_i + dz|z=z_i
            z.append(z[-1]+dz)
            # p_i+1 = p_i + dp|p=p_i (negative dp here bc dp variable positive)
            p.append(p[-1]-dp)
        return p,z

    def p2rho(self,p):
        '''
        calculate atmospheric density (rho) assuming ideal gas law
        => rho = p*mu/R_gas/T

        inputs:
            * self
            * p [Pa] - local pressure

        outputs:
            * rho [kg/m3] - density
        '''
        rho = p*self.p2mu(p)/R_gas/self.p2T(p)
        return rho


    # PLOTTING METHODS
    def plot_tp_pro(self):
        '''
        plot pressure vs temperature
        '''
        standard_Ts = self.p2T(self.standard_ps)
        plt.plot(standard_Ts,self.standard_ps,c='k')
        plt.axhline(self.p_transition_moist,ls='--',c='0.5')
        plt.axhline(self.p_transition_strat,ls='--',c='0.5')
        plt.ylim(self.planet.p_surf,min(self.standard_ps))
        plt.yscale('log')
        plt.ylabel(r'$p$ [Pa]')
        plt.xlabel(r'$T$ [K]')
        plt.show()

    def plot_f_h2o_pro(self):
        '''
        plot pressure vs mixing ratio of water
        '''
        standard_p_h2o = self.p2p_h2o(self.standard_ps)
        plt.plot(standard_p_h2o/self.standard_ps,self.standard_ps,c='k')
        plt.axhline(self.p_transition_moist,ls='--',c='0.5')
        plt.axhline(self.p_transition_strat,ls='--',c='0.5')
        plt.ylim(self.planet.p_surf,min(self.standard_ps))
        plt.yscale('log')
        plt.xscale('log')
        plt.xlim(1e-6,1)
        plt.xlabel(r'H$_2$O mixing ratio []')
        plt.ylabel(r'$p$ [Pa]')
        plt.show()
