################################################################
# results & plots of Loftus, Wordsworth, & Morley, 2019
# Kaitlyn Loftus (2019)
################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import scipy.integrate as integrate
from matplotlib.patches import Rectangle
from prettytable import PrettyTable
import src.atm_pro as atm_pro
import src.sulfur as sulfur
import src.mie as mie
import src.simtransspec as sts
import src.photochem as pc

################################################################
# SETUP
################################################################

# create directory to store figures in
os.makedirs('./figs', exist_ok=True)
os.makedirs('./figs_sup', exist_ok=True)
os.makedirs('./spec_inputs', exist_ok=True)

# EARTH SPECIFIC VALUES
M_earth = 5.9721986*10**24 # [kg]
R_earth = 6.371*10**6 # [m]
mass_earth_ocean = 1.4*10**21 # [kg]

eta_air = 1.6e-5 # [Pa/s]
mu_air = 0.02896 #[kg/mol]
c_p_air=1.003*1000 # [J/kg/K]
R_air = 287.058 # [J/kg/K]

pH_earth = 8.14
T_earth = 288 # [K]
p_SO2_earth = 1.01325e5*1e-10 # [Pa]

s_in_yr = 365.25*3600.*24. # [s/yr]

mu_o2 = 0.031998 #kg/mol
mu_n2 = 0.028014 #kg/mol
mu_co2 = 0.04401 #kg/mol
mu_h2o = 0.018015 #[kg/mol]
rho_h2o = 1000 #[kg/m3]

# color scheme where 3 colors are neccessary
colors3 = ['#9e9e9e','#135b1b','#0D19B6']

print('\nRESULTS FROM LOFTUS, WORDSWORTH, & MORLEY (2019)')
print('figures in paper saved in directory ./figs')
print('additional figures saved in directory ./figs_sup')
print('inputs for transit spectra saved in directory ./spec_inputs')

################################################################
# MIE SCATTERING
# Figure 2 & discussion in Section 3.2
################################################################

print('\n------------------------------------\n'
      +'MIE SCATTERING\n------------------------------------')

print('creating Figure 2')
# calculate scattering and extinction efficiencies for Sun-like and M-dwarf light
m_medium = 1. # assume index of refraction of air is 1
lambda_sol = 0.556 #[um] wavelength for a Sun-like star
lambda_M = 1. #[um] wavelength for a M-dwarf

# index of refraction of H2SO4-H2O for w=75%
# source:
# https://www.cfa.harvard.edu/HITRAN/HITRAN2012/Aerosols/ascii/single_files/palmer_williams_h2so4.dat
m_r_G = 1.4315
m_i_G = 0.
m_r_M = 1.422
m_i_M = 1.53e-6

r_min = 0.01 #[um]
r_max = 10 #[um]
r = np.logspace(r_min,r_max,500)

x_sol, Qs_sol, Qe_sol = mie.mie_scatter(m_r_G, m_i_G,
                                        xparams=[r_min,r_max,m_medium,lambda_sol],
                                        vary_lambda=False)
x_M, Qs_M, Qe_M = mie.mie_scatter(m_r_M, m_i_M,
                                  xparams=[r_min,r_max,m_medium,lambda_sol],
                                  vary_lambda=False)
Qe_sol_ray = mie.Rayleigh(x_sol,m_r_G)
Qe_M_ray = mie.Rayleigh(x_M,m_r_M)
r_sol = x_sol/2./np.pi*lambda_sol
r_M = x_M/2./np.pi*lambda_M

# FIGURE 2
# extinction effeciency vs particle radius
plt.plot(r_sol,Qe_sol,c='k',label='Mie, Sun-like')
plt.plot(r_M,Qe_M,c='r',label='Mie, M-dwarf')
plt.plot(r_sol,Qe_sol_ray,c='k',ls='--',label='Rayleigh, Sun-like')
plt.plot(r_M,Qe_M_ray,c='r',ls='--',label='Rayleigh, M-dwarf')
plt.xlabel(r'r [$\mu$m]')
plt.xscale('log')
plt.xlim(5e-2,4)
plt.ylabel(r'$Q_\mathrm{e}$ []')
plt.ylim(5e-2,10)
plt.yscale('log')
plt.legend()
plt.savefig('figs/fig02.pdf',bbox_inches='tight',transparent=True)
plt.close()
print('Figure 2 saved')

# print Qe for particle size of interest for each star
r_sol_sing = 0.1 # [um]
r_M_sing = 0.2 # [um]
Qe_sol_sing = mie.mie_scatter(m_r_G, m_i_G, x0=2.*np.pi*r_sol_sing/lambda_sol)[2]
r_sol_sing2 = 1. # [um]
Qe_sol_sing2 = mie.mie_scatter(m_r_G, m_i_G, x0=2.*np.pi*r_sol_sing2/lambda_sol)[2]
Qe_M_sing = mie.mie_scatter(m_r_M, m_i_M, x0=2.*np.pi*r_M_sing/lambda_M)[2]

t = PrettyTable(['star','lambda [um]','r [um]','Qe []'])
t.add_row(['G','0.556','1','%1.3F'%Qe_sol_sing2])
t.add_row(['G','0.556','0.1','%1.3F'%Qe_sol_sing])
t.add_row(['M','1.0','0.2','%1.3F'%Qe_M_sing])
print(t)

################################################################
# LIMITING PHOTOCHEMICAL TIMESCALE
# discussion in Section 3.4
################################################################

print('\n------------------------------------\n'
      +'LIMITING PHOTOCHEMICAL TIMESCALE\n------------------------------------')

# calculate limiting timescale for SO2 to H2SO4 conversion
# for a G star and M star

cross_w_SO2, cross_max, spectrum_photo_G = pc.set_up_photochem()
spectrum_photo_M = pc.set_up_photochem(f_XUV=10.,f_UV=0.1)[2]
t_G = (0.5*integrate.simps(cross_max*spectrum_photo_G[:,1],spectrum_photo_G[:,0],even='last'))**(-1) #[s]
t_M = (0.5*integrate.simps(cross_max*spectrum_photo_M[:,1],spectrum_photo_M[:,0],even='last'))**(-1) #[s]
pc.plot_stellar_spectrum(spectrum_photo_G)
pc.plot_stellar_spectrum(spectrum_photo_M,fig_name='stellar_spec_M')
pc.plot_cross_section(spectrum_photo_G,cross_w_SO2,cross_max)
pc.plot_cross_section(spectrum_photo_G,cross_w_SO2,cross_max,is_SO2=False)

# establish SO2 is not optically thick and thus should not contribute to
# the absorbtion cross section
pc.plot_SO2_tau(spectrum_photo_G,cross_w_SO2)

# print results of limiting timescales without SO2
t = PrettyTable(['star','t [s]', 't [days]'])
t.add_row(['G','%1.F'%t_G,'%1.2F'%(t_G/3600./24.)])
t.add_row(['M','%1.F'%t_M,'%1.2F'%(t_M/3600./24.)])
print(t)


################################################################
# SIMULATED TRANSMISSION SPECTRA
# Figure 3 & discussion in Section 3.6 & results in Section 4.1
# also inputs that generate simulated transmission spectra
################################################################
print('\n------------------------------------\n'
      +'SIMULATED TRANSMISSION SPECTRA\n------------------------------------')
print('creating inputs for transmission spectra')
taus = np.logspace(-4,1,6)
for tau in taus:
    sts.input_pro(1.01325e5,288,200,0.01,1000,tau,1.e-6,0,False)
    print('input for tau_h2so4 = %1.1F, r_h2so4 = 1 um saved'%tau)
rs = np.linspace(1,10,10)*1.e-7
for r in rs:
    sts.input_pro(1.01325e5,288,200,0.01,1000,0.1,r,0,False)
    print('input for tau_h2so4 = 0.1, r_h2so4 = %1.1F um saved'%(r*1e6))
#clear
sts.input_pro(1.01325e5,288,200,0.1,1000,0.1,0,0,False)
print('input for clear sky saved')
# low clouds
sts.input_pro(1.01325e5,288,200,0.01,1000,0.1,0,5.e-6,False)
print('input for low water clouds saved')
# high clouds
sts.input_pro(1.01325e5,288,200,0.01,1000,0.1,0,100e-6,True)
print('input for high water clouds saved')

print('creating Figure 3')

# FIGURE 3
print('Figure 3 saved')


################################################################
# AQUEOUS S(IV) CHEMISTRY
# Figures 5-6 & results in Section 4.3
################################################################

print('\n------------------------------------\n'
      +'AQUEOUS S(IV) CHEMISTRY\n------------------------------------')

# FIGURE 5
# plot distribution of S(IV) among SO2(aq), HSO3-, SO3-- vs pH
# ASSUMING S(IV) saturation
print('creating Figure 5')
pHs = np.linspace(1,14,100)
frac_so2, frac_hso3, frac_so3 = sulfur.S_aq_fractions(pHs,T_earth)
plt.plot(pHs,frac_so2, lw=3, label=r'SO$_2$(aq)', c=colors3[0])
plt.plot(pHs, frac_hso3, lw=1, label=r'HSO$_3^-$',c=colors3[1])
plt.plot(pHs,frac_so3, lw=1, label=r'SO$_3^{2-}$', c=colors3[2])
plt.legend(loc = 4)
plt.xlabel('pH')
plt.xlim(1,14)
plt.ylabel('Fraction Total Ocean S(IV)')
plt.yscale('log')
plt.ylim(1e-3,1.1)
plt.savefig('figs/fig05.pdf',bbox_inches='tight',transparent=True)
plt.close()
print('Figure 5 saved')

# FIGURE 6
# ratio of sulfur in the atmosphere to sulfur in the ocean vs pH
# for oceans with masses m_oc = 0.001, 1, 1000 Earth oceans
# ASSUMING S(IV) saturation
print('creating Figure 6')
for i,m_oc in enumerate([0.001,1,1000]):
    f = sulfur.S_atm_ocean_frac(pHs,m_oc)
    plt.plot(pHs,f,label=r'M$_\mathrm{ocean}$ = '+str(m_oc)+
             r'M$_{\bigoplus \mathrm{ocean}}$',c=colors3[i])
plt.xlabel('pH')
plt.xlim(1,14)
plt.ylabel('Atmosphere S / Ocean S')
plt.yscale('log')
plt.legend()
plt.savefig('figs/fig06.pdf',bbox_inches='tight',transparent=True)
plt.close()
print('Figure 6 saved\n')

# print S(IV) concentrations for modern earth pH
# ASSUMING S(IV) saturation
so2, hso3, so3 = sulfur.S_aq_concentrations(pH_earth,T_earth,p_SO2_earth)

# print % S(IV) concentrations for modern earth pH
# ASSUMING S(IV) saturation
frac_so2, frac_hso3, frac_so3 = sulfur.S_aq_fractions(pH_earth,T_earth)
print('for modern earth pH = 8.14, fSO2 = 1E-10, & S(IV) saturation:\n'
       '[SO2(aq)] = %1.3E mol/L\n'%so2
       + '[SO2(aq)]/[S(IV)] = %1.3E\n\n'%frac_so2
       + '[HSO3-] = %1.3E mol/L\n'%hso3
       + '[HSO3-]/[S(IV)] = %1.3F\n\n'%frac_hso3
       + '[SO3--] = %1.3E mol/L\n'%so3
       + '[SO3--]/[S(IV)] = %1.3F\n'%frac_so3)
# print ratio of sulfur in atmosphere vs ocean for modern earth pH
# ASSUMING S(IV) saturation
print('atmosphere S / ocean S = %1.3E'%sulfur.S_atm_ocean_frac(pH_earth,1))


################################################################
# SULFUR IN THE ATMOSPHERE
# Figure 4 & results in Section 4.2
################################################################
print('\n------------------------------------\n'
      +'ATMOSPHERIC SULFUR\n------------------------------------')

n = 50
mu_atm = 0.02896 #kg/mole, air

pH = np.linspace(1,14,n)
oceans = np.logspace(-3.03,0.1,n)
oceans_ex0 = np.logspace(-6,0.1,n)
oceans_ex = np.logspace(-9,0.1,n)
oceans_ex2 = np.logspace(-11,0.1,n)
ocgrid, pHgrid  = np.meshgrid(oceans,pH)
ocgrid_ex0, pHgrid_ex0  = np.meshgrid(oceans_ex0,pH)
ocgrid_ex, pHgrid_ex  = np.meshgrid(oceans_ex,pH)
ocgrid_ex2, pHgrid_ex2  = np.meshgrid(oceans_ex2,pH)
R_p_earth = 1
M_p_earth = 1
p_surf = 1.01325e5
T_surf = 288.
T_strat = 200.
r_b = 1.e-6
r_G_lim = 1.e-7
r_M_lim = 2.e-7
tau = 0.1
is_G = True
is_M = False
t_convert_M_lim =  3600.*24.*1.3
t_convert_G_lim =  3600.*24.*1.3
t_convert_b =      3600.*24.*30.
alpha_lim = 1
alpha_b = 0.01
w = 0.75
t_mix = 1.*s_in_yr
n_outgass_lim = 200.
n_outgass_b = 1.
f_n2 = 0.7809
f_o2 = 0.2095
f_co2 = 400.e-6
RH_h2o_surf_b = 0.75
RH_h2o_surf_lim = 0.

u_so2_lim = 0.0001
u_so2_b = 0.1

t = PrettyTable(['obs S','model param',' # S atoms','# S moles', 'S kg'])

atoms, moles, kg = sulfur.calc_crit_S_atm_obs_haze(tau, r_b, T_surf, T_strat, p_surf,
                      R_p_earth, M_p_earth, w, t_mix,
                      t_convert_b, alpha_b,RH_h2o_surf_b,f_n2,f_o2,f_co2)
t.add_row(['aerosol','best','%1.1E'%atoms,'%1.3E'%moles, '%1.3E'%kg])

atoms, moles, kg = sulfur.calc_crit_S_atm_obs_haze(tau, r_M_lim, T_surf, T_strat, p_surf,
                            R_p_earth, M_p_earth, w, t_mix,
                            t_convert_M_lim, 1.,0.,f_n2,f_o2,f_co2)
t.add_row(['aerosol','limiting M','%1.1E'%atoms,'%1.3E'%moles, '%1.3E'%kg])

atoms, moles, kg = sulfur.calc_crit_S_atm_obs_haze(tau, r_G_lim, T_surf, T_strat, p_surf,
                            R_p_earth, M_p_earth, w, t_mix,
                            t_convert_G_lim, 1.,0.,f_n2,f_o2,f_co2)
t.add_row(['aerosol','limiting G','%1.1E'%atoms,'%1.3E'%moles, '%1.3E'%kg])

atoms, moles, kg = sulfur.calc_crit_S_atm_obs_SO2(u_so2_b,p_surf,T_surf,R_p_earth,M_p_earth,f_n2,f_o2,f_co2)
t.add_row(['gas','best','%1.1E'%atoms,'%1.3E'%moles, '%1.3E'%kg])

atoms, moles, kg = sulfur.calc_crit_S_atm_obs_SO2(u_so2_lim,p_surf,T_surf,R_p_earth,M_p_earth,f_n2,f_o2,f_co2)
t.add_row(['gas','limiting','%1.1E'%atoms,'%1.3E'%moles, '%1.3E'%kg])

print('critical total atmospheric sulfur to be observable')
print(t)

# FIGURE 4
# sensitivities to planetary parameters for critical sulfur required
# in the atmosphere for observation
print('creating Figure 4')
fig, axarr = plt.subplots(2,3,sharey=True,figsize=(16, 12))
axarr[0,0].set_title(r'$T_\mathrm{surf}$')
axarr[0,0].set_xlabel(r'$T_\mathrm{surf}$ [K]')
T_surfs = np.linspace(250,400,n)
N_S_T_surfs = np.zeros((n,2))
for i,T in enumerate(T_surfs):
    N_S_T_surfs[i,0] = sulfur.calc_crit_S_atm_obs_haze(tau, r_b, T, T_strat, p_surf,
                                R_p_earth, M_p_earth, w, t_mix,
                                t_convert_b, alpha_b,RH_h2o_surf_b,f_n2,f_o2,f_co2,is_G)[0]
    N_S_T_surfs[i,1] = sulfur.calc_crit_S_atm_obs_SO2(u_so2_b,p_surf,T,R_p_earth,M_p_earth,f_n2,f_o2,f_co2)[0]
N_S_base0 = sulfur.calc_crit_S_atm_obs_haze(tau, r_b, T_surf, T_strat, p_surf,
                            R_p_earth, M_p_earth, w, t_mix,
                            t_convert_b, alpha_b,RH_h2o_surf_b,f_n2,f_o2,f_co2,is_G)[0]
N_S_base1 = sulfur.calc_crit_S_atm_obs_SO2(u_so2_b,p_surf,T_surf,R_p_earth,M_p_earth,f_n2,f_o2,f_co2)[0]
axarr[0,0].plot(T_surfs,N_S_T_surfs[:,0]/N_S_base0,c='0.5',label='aerosol')
axarr[0,0].plot(T_surfs,N_S_T_surfs[:,1]/N_S_base1,c='b',label='gas')
axarr[0,0].axvline(T_surf,ls='--',c='0.8')

axarr[0,0].set_ylabel(r'$N^\ast_{\mathrm{S}}/N^\ast_{\mathrm{S,}\oplus}$')
axarr[1,0].set_ylabel(r'$N^\ast_{\mathrm{S}}/N^\ast_{\mathrm{S,}\oplus}$')
axarr[1,0].set_title(r'$T_\mathrm{strat}$')
T_strats = np.linspace(150,215,n)
N_S_T_strats = np.zeros((n,2))
for i,T in enumerate(T_strats):
    N_S_T_strats[i,0] = sulfur.calc_crit_S_atm_obs_haze(tau, r_b, T_surf, T, p_surf,
                                R_p_earth, M_p_earth, w, t_mix,
                                t_convert_b, alpha_b,RH_h2o_surf_b,f_n2,f_o2,f_co2,is_G)[0]
    N_S_T_strats[i,1] = sulfur.calc_crit_S_atm_obs_SO2(u_so2_b,p_surf,T_surf,R_p_earth,M_p_earth,f_n2,f_o2,f_co2)[0]
axarr[1,0].plot(T_strats,N_S_T_strats[:,0]/N_S_base0,c='0.5',label='aerosol')
axarr[1,0].plot(T_strats,N_S_T_strats[:,1]/N_S_base1,c='b',label='gas')
axarr[1,0].axvline(T_strat,ls='--',c='0.8')
axarr[1,0].set_xlabel(r'$T_\mathrm{strat}$ [K]')

axarr[0,1].set_title(r'$p_\mathrm{surf}$')
axarr[0,1].set_xlabel(r'$p_\mathrm{surf}$ [Pa]')
p_surfs = np.logspace(-2,2,n)*1.01325e5
N_S_p_surfs = np.zeros((n,2))
for i,p in enumerate(p_surfs):
    N_S_p_surfs[i,0] = sulfur.calc_crit_S_atm_obs_haze(tau, r_b, T_surf, T_strat, p,
                                R_p_earth, M_p_earth, w, t_mix,
                                t_convert_b, alpha_b,RH_h2o_surf_b,f_n2,f_o2,f_co2,is_G)[0]
    N_S_p_surfs[i,1] = sulfur.calc_crit_S_atm_obs_SO2(u_so2_b,p,T_surf,R_p_earth,M_p_earth,f_n2,f_o2,f_co2)[0]
axarr[0,1].plot(p_surfs,N_S_p_surfs[:,0]/N_S_base0,c='0.5',label='aerosol')
axarr[0,1].plot(p_surfs,N_S_p_surfs[:,1]/N_S_base1,c='b',label='gas')
axarr[0,1].axvline(p_surf,ls='--',c='0.8')
axarr[0,1].set_xscale('log')

axarr[1,1].set_title(r'$R_\mathrm{P}$')
axarr[1,1].set_xlabel(r'$R_\mathrm{P}$ [$R_\oplus$]')
R_ps = np.linspace(0.25,1.6,n)*R_earth
convert = M_earth/R_earth**(1./0.27)
M_ps = convert*R_ps**(1./0.27)/M_earth
R_ps = R_ps/R_earth
N_S_size = np.zeros((n,2))
for i,R in enumerate(R_ps):
    N_S_size[i,0] = sulfur.calc_crit_S_atm_obs_haze(tau, r_b, T_surf, T_strat, p_surf,
                                R, M_ps[i], w, t_mix,
                                t_convert_b, alpha_b,RH_h2o_surf_b,f_n2,f_o2,f_co2,is_G)[0]/M_ps[i]
    N_S_size[i,1] = sulfur.calc_crit_S_atm_obs_SO2(u_so2_b,p_surf,T_surf,R,M_ps[i],f_n2,f_o2,f_co2)[0]/M_ps[i]
axarr[1,1].plot(R_ps,N_S_size[:,0]/N_S_base0,c='0.5',label='aerosol')
axarr[1,1].plot(R_ps,N_S_size[:,1]/N_S_base1,c='b',label='gas')
axarr[1,1].axvline(1,ls='--',c='0.8')


axarr[0,2].set_title('composition')
axarr[0,2].set_xlabel(r'$\mu$ [g/mol]')
percent_x = np.linspace(0,1,n)
mus = np.zeros(n)
N_S_mus = np.zeros((n,2))
for i,x in enumerate(percent_x):
    mus[i] = ((1-x)*mu_n2 + x*mu_co2)
    N_S_mus[i,0] = sulfur.calc_crit_S_atm_obs_haze(tau, r_b, T_surf, T_strat, p_surf,
                                R_p_earth, M_p_earth, w, t_mix,
                                t_convert_b, alpha_b,RH_h2o_surf_b,1-x,0,x,is_G)[0]
    N_S_mus[i,1] = sulfur.calc_crit_S_atm_obs_SO2(u_so2_b,p_surf,T_surf,R_p_earth,M_p_earth,1-x,0,x)[0]
axarr[0,2].plot(mus*1e3,N_S_mus[:,0]/N_S_base0,c='0.5',label='aerosol')
axarr[0,2].plot(mus*1e3,N_S_mus[:,1]/N_S_base1,c='b',label='gas')
axarr[0,2].axvline(mu_air*1e3,ls='--',c='0.8')


axarr[1,2].set_title(r'$f_{\mathrm{H}_2\mathrm{O}}$')
axarr[1,2].set_xlabel(r'$f_{\mathrm{H}_2\mathrm{O}}$')
RH_h2o_surfs = np.logspace(-5,0,n)
N_S_f_h2os = np.zeros((n,2))
for i,RH in enumerate(RH_h2o_surfs):
    N_S_f_h2os[i,0] = sulfur.calc_crit_S_atm_obs_haze(tau, r_b, T_surf, T_strat, p_surf,
                                R_p_earth, M_p_earth, w, t_mix,
                                t_convert_b, alpha_b,RH,f_n2,f_o2,f_co2)[0]
    N_S_f_h2os[i,1] = sulfur.calc_crit_S_atm_obs_SO2(u_so2_b,p_surf,T_surf,R_p_earth,M_p_earth,f_n2,f_o2,f_co2)[0]
f_h2o_surfs = RH_h2o_surfs*atm_pro.p_h2osat(T_surf)/p_surf # []
f_h2o_surf_b = RH_h2o_surf_b*atm_pro.p_h2osat(T_surf)/p_surf # []
axarr[1,2].plot(f_h2o_surfs,N_S_f_h2os[:,0]/N_S_base0,c='0.5',label='aerosol')
axarr[1,2].plot(f_h2o_surfs,N_S_f_h2os[:,1]/N_S_base1,c='b',label='gas')
axarr[1,2].axvline(f_h2o_surf_b,ls='--',c='0.8')
axarr[1,2].set_xscale('log')
plt.ylim(1e-2,1e2)
plt.yscale('log')
for i in range(2):
    for j in range(3):
        axarr[i,j].axhspan(0.01,0.1,color='r',alpha=0.8,label='\n<10%\n' r'$N^\ast_{\mathrm{S}}/N^\ast_{\mathrm{S,}\oplus}$')

fig.subplots_adjust(hspace=0.35,wspace=0.05)
handles, labels = axarr[1,2].get_legend_handles_labels()
fig.legend(handles, labels, loc=7)
plt.savefig('figs/fig04.pdf',bbox_inches='tight',transparent=True)
plt.close()
print('Figure 4 saved')


################################################################
# SULFUR OBSERVABILITY GIVEN OCEAN PARAMETERS
# Figures 7-10 & results in Section 4.4
################################################################

print('\n------------------------------------\n'
      +'SULFUR OBSERVABILITY GIVEN OCEAN PARAMETERS\n------------------------------------')

print('creating Figures 7-10')
t_SIV_M_lim = sulfur.calc_t_SIV(tau, r_M_lim, T_surf, T_strat, p_surf,
                         R_p_earth, M_p_earth, ocgrid, pHgrid, w, t_mix,
                         t_convert_M_lim, alpha_lim, n_outgass_lim,
                         RH_h2o_surf_lim,f_n2,f_o2,f_co2, is_M)

t_SIV_G_lim = sulfur.calc_t_SIV(tau, r_G_lim, T_surf, T_strat, p_surf,
                         R_p_earth, M_p_earth, ocgrid, pHgrid, w, t_mix,
                         t_convert_G_lim, alpha_lim, n_outgass_lim,
                         RH_h2o_surf_lim,f_n2,f_o2,f_co2,is_G)

t_SIV_b = sulfur.calc_t_SIV(tau, r_b, T_surf, T_strat, p_surf,
                         R_p_earth, M_p_earth, ocgrid_ex, pHgrid_ex, w, t_mix,
                         t_convert_b, alpha_b, n_outgass_b,
                         RH_h2o_surf_b,f_n2,f_o2,f_co2,is_G)

u_so2_obs = 0.1
u_so2_obs_lim = u_so2_obs*0.001

t_SIV_gas = sulfur.calc_t_SIV_SO2(u_so2_obs,p_surf,T_surf,
                                  pHgrid_ex2,ocgrid_ex2,n_outgass_b,R_p_earth,
                                  M_p_earth,f_n2,f_o2,f_co2)
t_SIV_gas_lim = sulfur.calc_t_SIV_SO2(u_so2_obs_lim,p_surf,T_surf,
                                      pHgrid_ex0,ocgrid_ex0,n_outgass_lim,
                                      R_p_earth,M_p_earth,f_n2,f_o2,f_co2)

#place in log10 years where t_SIV becomes reasonable
likely = -1
# set up contour levels and their colors
levels = [-2,-1,0,1,2,4,10]
oranges = matplotlib.cm.get_cmap('Wistia')
norm_orange = matplotlib.colors.Normalize(vmin=-5, vmax=-0.99)
blues = matplotlib.cm.get_cmap('Blues')
norm_blues= matplotlib.colors.Normalize(vmin=-1, vmax=8)
colors = [oranges(norm_orange(-2)),oranges(norm_orange(-1)),
          blues(1-norm_blues(-1)),blues(1-norm_blues(0)),
          blues(1-norm_blues(1)),blues(1-norm_blues(2)),
          blues(1-norm_blues(4)),blues(1-norm_blues(7))]



f, (ax1, ax2) = plt.subplots(1,2,sharey=True,figsize=(12,4.5))
ax1.set_yscale('log')
ax2.set_yscale('log')
cs1 = ax1.contourf(pHgrid, ocgrid, np.log10(t_SIV_M_lim),levels=levels,colors=colors,extend='both')
cs2 = ax2.contourf(pHgrid, ocgrid, np.log10(t_SIV_G_lim),levels=levels,colors=colors,extend='both')
cs2_ = f.colorbar(cs2,ticks=levels,ax=[ax1,ax2])
cs2_.set_label(r'$\log_{10}$($\tau_\mathrm{S(IV)}^\ast$ [yr])')
# remove white lines between contours for vector image
cs2_.solids.set_edgecolor('face')
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
zeromark = ax1.contour(pHgrid, ocgrid, np.log10(t_SIV_M_lim),levels=[likely],linewidths=3,colors='w')
ax2.contour(pHgrid, ocgrid, np.log10(t_SIV_G_lim),levels=[likely],linewidths=3,colors='w')
cs2_.add_lines(zeromark)
ax1.set_xlabel('pH')
interesting_oc = Rectangle((6,1e-3),7.9,1.2,fill=False,edgecolor='0.8',lw=3,ls='--',label='\n<10%\n' r'$N^\ast_{\mathrm{S}}/N^\ast_{\mathrm{S,}\oplus}$')
interesting_oc2 = Rectangle((6,1e-3),7.9,1.2,fill=False,edgecolor='0.8',lw=3,ls='--',label='\n<10%\n' r'$N^\ast_{\mathrm{S}}/N^\ast_{\mathrm{S,}\oplus}$')

ax1.add_patch(interesting_oc)
ax2.add_patch(interesting_oc2)

oc_aero_lM = 2.5e-3 # [Earth oc]
oc_aero_lG = 4e-3 # [Earth oc]
h_aero_lM = oc_aero_lM*mass_earth_ocean/rho_h2o/4./np.pi/R_earth**2 # [m]
h_aero_lG = oc_aero_lG*mass_earth_ocean/rho_h2o/4./np.pi/R_earth**2 # [m]

ax2.set_xlabel('pH')
ax1.set_ylabel(r'# Earth oceans [$M_{\oplus\mathrm{ocean}}$]')
ax1.scatter(8.14,1,color='k',s=10,zorder=10)
ax1.annotate('modern Earth ocean', (8.14,0.7))
ax2.scatter(8.14,1,color='k',s=10,zorder=10)
ax2.annotate('modern Earth ocean', (8.14,0.7))
ax1.set_title('M-dwarf')
ax2.set_title('Sun-like')

ax1.arrow(0.35, 0.35, 0.5, 0.4, transform=ax1.transAxes,
          length_includes_head=True,head_width=0.02, head_length=0.03,fc='k')
ax1.annotate('observable \n' r'H$_2$SO$_4$-H$_2$O haze'
             '\nincreasingly \nunlikely',(0.3,0.425),xycoords='figure fraction')
ax1.annotate('observable \n' r'H$_2$SO$_4$-H$_2$O' '\nhaze'
             '\npossible',(0.09,0.15),xycoords='figure fraction',
             color='k')
ax2.arrow(0.4, 0.4, 0.45, 0.35, transform=ax2.transAxes,
          length_includes_head=True,head_width=0.02, head_length=0.03,fc='k')
ax2.annotate('observable \n' r'H$_2$SO$_4$-H$_2$O haze'
             '\nincreasingly \nunlikely',(0.725,0.375),xycoords='figure fraction')
ax2.annotate('observable \n' r'H$_2$SO$_4$-H$_2$O' '\nhaze'
             '\npossible',(0.525,0.2),xycoords='figure fraction',
             color='k')

plt.savefig('figs/fig10.pdf',bbox_inches='tight',transparent=True)
plt.close()

fig, ax1 = plt.subplots(figsize=(4.5,6.5))
plt.yscale('log')
cs1 = plt.contourf(pHgrid_ex, ocgrid_ex, np.log10(t_SIV_b),levels=levels,colors=colors,extend='both')
cs1_ = plt.colorbar(cs1,ticks=levels,orientation='horizontal')
cs1_.set_label(r'$\log_{10}$($\tau_\mathrm{S(IV)}^\ast$ [yr])')
plt.xlabel('pH')
plt.ylabel(r'# Earth oceans [$M_{\oplus\mathrm{ocean}}$]')

matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
plt.contour(pHgrid_ex, ocgrid_ex, np.log10(t_SIV_b),levels=[likely],linewidths=3,colors='w')
# remove white lines between contours for vector image
cs1_.solids.set_edgecolor('face')
cs1_.add_lines(zeromark)
interesting_oc3 = Rectangle((6,1e-3),7.9,1.2,fill=False,edgecolor='0.8',lw=3,ls='--',label='\n<10%\n' r'$N^\ast_{\mathrm{S}}/N^\ast_{\mathrm{S,}\oplus}$')
ax1.add_patch(interesting_oc3)
plt.scatter(8.14,1,color='k',s=10,zorder=10)
plt.annotate('modern Earth ocean', (8.3,0.4))
ax2 = ax1.twinx()
oc_aero_b = 3e-9 #[Earth oc]
h_aero_b = oc_aero_b*mass_earth_ocean/rho_h2o/4./np.pi/R_earth**2 # [m]
ax2.set_ylabel('global equivalent ocean layer [m]')
ax2.set_yscale('log')
hgrid_ex = ocgrid_ex*mass_earth_ocean/rho_h2o/4./np.pi/R_earth**2
ax2.contour(pHgrid_ex, hgrid_ex, np.log10(t_SIV_b),levels=[likely],linewidths=3,colors='w')


plt.arrow(0.3, 0.25, 0.65, 0.55, transform=plt.gca().transAxes,
          length_includes_head=True,head_width=0.02, head_length=0.03,fc='k')
plt.annotate(r'observable H$_2$SO$_4$-H$_2$O'
             '\nhaze increasingly unlikely',(0.47,0.47),xycoords='figure fraction')
plt.annotate('observable \n' r'H$_2$SO$_4$-H$_2$O' '\nhaze'
             '\npossible',(0.17,0.33),xycoords='figure fraction',
             color='k')
plt.savefig('figs/fig08.pdf',bbox_inches='tight',transparent=True)
plt.close()



fig, ax1 = plt.subplots(figsize=(4.5,6.5))
plt.yscale('log')
cs1 = plt.contourf(pHgrid_ex2, ocgrid_ex2, np.log10(t_SIV_gas),levels=levels,extend='both',colors=colors)
cs1_ = plt.colorbar(cs1,ticks=levels,orientation='horizontal')
cs1_.set_label(r'$\log_{10}$($\tau_\mathrm{S(IV)}^\ast$ [yr])')
# remove white lines between contours for vector image
cs1_.solids.set_edgecolor('face')
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
plt.contour(pHgrid_ex2, ocgrid_ex2, np.log10(t_SIV_gas),levels=[likely],linewidths=0.5,colors='w')
cs1_.add_lines(zeromark)
plt.xlabel('pH')
plt.ylabel(r'# Earth oceans [$M_{\oplus\mathrm{ocean}}$]')
interesting_oc4 = Rectangle((6,1e-3),7.9,1.2,fill=False,edgecolor='0.8',lw=3,ls='--',label='\n<10%\n' r'$N^\ast_{\mathrm{S}}/N^\ast_{\mathrm{S,}\oplus}$')
ax1.add_patch(interesting_oc4)
plt.scatter(8.14,1,color='k',s=10,zorder=10)
plt.annotate('modern Earth ocean', (8.3,0.4))
plt.tick_params(axis='y', which='minor')
ax2 = ax1.twinx()
plt.tick_params(axis='y', which='minor')
ax2.set_ylabel('global equivalent ocean layer [m]')
ax2.set_yscale('log')
hgrid_ex2 = ocgrid_ex2*mass_earth_ocean/rho_h2o/4./np.pi/R_earth**2
ax2.contour(pHgrid_ex2, hgrid_ex2, np.log10(t_SIV_gas),levels=[likely],linewidths=3,colors='w')

oc_gas_b = 1e-10
h_gas_b = oc_gas_b*mass_earth_ocean/rho_h2o/4./np.pi/R_earth**2

plt.arrow(0.3, 0.27, 0.65, 0.55, transform=plt.gca().transAxes,
          length_includes_head=True,head_width=0.02, head_length=0.03,fc='k')
plt.annotate(r'observable SO$_2$'
             '\n increasingly unlikely',(0.53,0.55),xycoords='figure fraction')
plt.annotate(r'observable SO$_2$'
             '\npossible',(0.17,0.33),xycoords='figure fraction',
             color='k')
plt.savefig('figs/fig07.pdf',bbox_inches='tight',transparent=True)
plt.close()


fig, ax1 = plt.subplots(figsize=(4.5,6.5))
plt.yscale('log')
cs1 = plt.contourf(pHgrid_ex0, ocgrid_ex0, np.log10(t_SIV_gas_lim),levels=levels,extend='both',colors=colors)
cs1_ = plt.colorbar(cs1,ticks=levels,orientation='horizontal')
cs1_.set_label(r'$\log_{10}$($\tau_\mathrm{S(IV)}^\ast$ [yr])')
# remove white lines between contours for vector image
cs1_.solids.set_edgecolor('face')
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
plt.contour(pHgrid_ex0, ocgrid_ex0, np.log10(t_SIV_gas_lim),levels=[likely],linewidths=0.5,colors='w')
cs1_.add_lines(zeromark)
plt.xlabel('pH')
plt.ylabel(r'# Earth oceans [$M_{\oplus\mathrm{ocean}}$]')
interesting_oc5 = Rectangle((6,1e-3),7.9,1.2,fill=False,edgecolor='0.8',lw=3,ls='--',label='\n<10%\n' r'$N^\ast_{\mathrm{S}}/N^\ast_{\mathrm{S,}\oplus}$')
ax1.add_patch(interesting_oc5)
plt.scatter(8.14,1,color='k',s=10,zorder=10)
plt.annotate('modern Earth ocean', (8.3,0.6))
ax2 = ax1.twinx()
ax2.set_ylabel('global equivalent ocean layer [m]')
ax2.set_yscale('log')
hgrid_ex0 = ocgrid_ex0*mass_earth_ocean/rho_h2o/4./np.pi/R_earth**2
ax2.contour(pHgrid_ex0, hgrid_ex0, np.log10(t_SIV_gas_lim),levels=[likely],linewidths=3,colors='w')
oc_gas_l = 2e-5
h_gas_l = oc_gas_l*mass_earth_ocean/rho_h2o/4./np.pi/R_earth**2
plt.arrow(0.35, 0.35, 0.55, 0.50, transform=plt.gca().transAxes,
          length_includes_head=True,head_width=0.02, head_length=0.03,fc='k')
plt.annotate(r'observable SO$_2$'
             '\n increasingly unlikely',(0.53,0.55),xycoords='figure fraction')

plt.annotate(r'observable SO$_2$'
             '\npossible',(0.17,0.35),xycoords='figure fraction',
             color='k')
plt.savefig('figs/fig09.pdf',bbox_inches='tight',transparent=True)
plt.close()
print('Figures 7-10 saved\n')

t = PrettyTable(['obs S','model param','oc mass [Earth oc]','oc GEL [m]'])
t.add_row(['aerosol','best','%1.1E'%oc_aero_b,'%1.3E'%h_aero_b])
t.add_row(['aerosol','limiting M','%1.1E'%oc_aero_lM,'%1.3F'%h_aero_lM])
t.add_row(['aerosol','limiting G','%1.1E'%oc_aero_lG,'%1.3F'%h_aero_lG])
t.add_row(['gas','best','%1.1E'%oc_gas_b,'%1.3E'%h_gas_b])
t.add_row(['gas','limiting','%1.1E'%oc_gas_l,'%1.3F'%h_gas_l])
print('maximum ocean size for t_S(IV) = 0.1 years & ocean pH = 6 \nto have observable atmospheric sulfur')
print(t)
