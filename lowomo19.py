################################################################
# results & plots of Loftus, Wordsworth, & Morley, 2019
# (aka LoWoMo19)
#
# code by Kaitlyn Loftus (2019)
#
# this script will generate all results for reproducing
# LoWoMo19, either via print output or figures
################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import scipy.integrate as integrate
from matplotlib.patches import Rectangle
from prettytable import PrettyTable
import src.atm_pro as atm_pro
import src.sulfur as sulfur
import src.mie as mie
import src.simtransspec as sts
import src.photochem as pc
from src.planet import Planet
from cycler import cycler
import src.h2o as h2o

################################################################
# SETUP
################################################################

print('\nRESULTS FROM LOFTUS, WORDSWORTH, & MORLEY (2019)')
print('figures in paper saved in directory ./figs')
print('additional figures saved in directory ./figs_sup')
print('inputs for transit spectra saved in directory ./spec_inputs')

# create directory to store figures in
os.makedirs('./figs', exist_ok=True)
os.makedirs('./figs_sup', exist_ok=True)
os.makedirs('./spec_inputs', exist_ok=True)

# EARTH SPECIFIC VALUES
M_earth = 5.9721986*10**24 # [kg]
R_earth = 6.371*10**6 # [m]
mass_earth_ocean = 1.4*10**21 # [kg]

mu_air = 0.02896 #[kg/mol]

pH_earth = 8.14
T_earth = 288 # [K]
p_SO2_earth = 1.01325e5*1e-10 # [Pa]

# CONSTANTS
s_in_yr = 365.25*3600.*24. # [s/yr]

mu_n2 = 0.028014 #kg/mol
mu_co2 = 0.04401 #kg/mol
rho_h2o = 1000 #[kg/m3]

R_gas = 8.31446 #[J/mol/K]

# color scheme where 3 colors are necessary
colors3 = ['#002fa7','deepskyblue','#C1DBE6']

# set up dry and normal (wet) Earths
# dry atm composition
f_o2 = 0.2095 # [vmr] O2
f_co2 = 400.e-6 # [vmr] #CO2
# to ensure atm components add up to 1
f_n2 = 1. - f_o2 - f_co2 # [vmr] N2
# composition array X [H2, He, N2, O2, CO2]
X = np.array([0.,0.,f_n2,f_o2,f_co2])
T_strat = 200 # [K]
p_surf_earth = 1.01325e5 # [Pa]
RH_earth = 0.77 # []
# wet Earth
Earth = Planet(1,T_earth,T_strat,p_surf_earth,X,1)
Earth_atm = atm_pro.Atm(Earth,RH_earth)
# integrate to get atmospheric profile
Earth_atm.set_up_atm_pro()
# dry Earth
Earth_dry = Planet(1,T_earth,T_strat,p_surf_earth,X,1)
Earth_atm_dry = atm_pro.Atm(Earth_dry,0.)
# integrate to get atmospheric profile
Earth_atm_dry.set_up_atm_pro()

################################################################
# MIE SCATTERING
# Figure 2 & methods in Section 3.2
################################################################

print('\n-----------------------------------------------\n'
      +'MIE SCATTERING\n-----------------------------------------------')

print('creating Figure 2')
# calculate scattering and extinction efficiencies for Sun-like and M-dwarf light
m_medium = 1. # assume index of refraction of air is 1
lambda_G = 0.556 #[um] wavelength for a Sun-like (G) star
lambda_M = 1. #[um] wavelength for a M-dwarf

# index of refraction of H2SO4-H2O for w=75%
# source:
# https://www.cfa.harvard.edu/HITRAN/HITRAN2012/Aerosols/ascii/single_files/palmer_williams_h2so4.dat
m_G = complex(1.4315,0.)
m_M = complex(1.422,1.53e-6)

# particle radii of interest
r_min = 0.01 #[um]
r_max = 10 #[um]
r = np.logspace(r_min,r_max,500)

# calculate Mie size parameter (x_*), scattering efficiency (Qs_*),
# and extinction efficiency (Qe_*)
x_G, Qs_G, Qe_G = mie.mie_scatter(m_G,
                                        xparams=[r_min,r_max,m_medium,lambda_G],
                                        vary_lambda=False)
x_M, Qs_M, Qe_M = mie.mie_scatter(m_M,
                                  xparams=[r_min,r_max,m_medium,lambda_M],
                                  vary_lambda=False)
# calculate Rayleigh scattering for same size parameters
Qe_G_ray = mie.Rayleigh(x_G,m_G.real)
Qe_M_ray = mie.Rayleigh(x_M,m_M.real)
# translate size parameters back to radii
r_G = x_G/2./np.pi*lambda_G
r_M = x_M/2./np.pi*lambda_M

# FIGURE 2
# extinction efficiency vs particle radius
plt.plot(r_G,Qe_G,c=colors3[0],label='Mie, Sun-like')
plt.plot(r_M,Qe_M,c=colors3[2],label='Mie, M-dwarf')
plt.plot(r_G,Qe_G_ray,c=colors3[0],ls='--',label='Rayleigh, Sun-like')
plt.plot(r_M,Qe_M_ray,c=colors3[2],ls='--',label='Rayleigh, M-dwarf')
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
r_G_sing = 0.1 # [um]
r_M_sing = 0.2 # [um]
Qe_G_sing = mie.mie_scatter(m_G, x0=2.*np.pi*r_G_sing/lambda_G)[2]
r_G_sing2 = 1. # [um]
Qe_G_sing2 = mie.mie_scatter(m_G, x0=2.*np.pi*r_G_sing2/lambda_G)[2]
Qe_M_sing = mie.mie_scatter(m_M, x0=2.*np.pi*r_M_sing/lambda_M)[2]

t = PrettyTable(['star','lambda [um]','r [um]','Qe []'])
t.add_row(['G','0.556','1','%1.3F'%Qe_G_sing2])
t.add_row(['G','0.556','0.1','%1.3F'%Qe_G_sing])
t.add_row(['M','1.0','0.2','%1.3F'%Qe_M_sing])
print(t)

################################################################
# LIMITING PHOTOCHEMICAL TIMESCALE
# methods in Section 3.4
################################################################

print('\n-----------------------------------------------\n'
      +'LIMITING PHOTOCHEMICAL TIMESCALE\n-----------------------------------------------')

# calculate limiting timescale for SO2 to H2SO4 conversion
# for a G star and M star

# set up photochemical calculation by getting stellar spectrum and
# absorption cross sections for H2O, O2, CO2, and SO2
# G star
cross_w_SO2, cross_max, spectrum_photo_G = pc.set_up_photochem()
# M star -- adjust solar spectrum by multiplying XUV flux by 10x
# and UV flux by 0.1x
spectrum_photo_M = pc.set_up_photochem(f_XUV=10.,f_UV=0.1)[2]
# calculate conversion timescale from (13) using Simpson's rule
t_G = (0.25*integrate.simps(cross_max*spectrum_photo_G[:,1],spectrum_photo_G[:,0],even='last'))**(-1) #[s]
t_M = (0.25*integrate.simps(cross_max*spectrum_photo_M[:,1],spectrum_photo_M[:,0],even='last'))**(-1) #[s]

# print results of limiting timescales (without SO2 photodissociation)
t = PrettyTable(['star','t [s]', 't [days]'])
t.add_row(['G','%1.F'%t_G,'%1.2F'%(t_G/3600./24.)])
t.add_row(['M','%1.F'%t_M,'%1.2F'%(t_M/3600./24.)])
print(t)

# create additional figures of interest to photochemical calculation
# plot assumed G star spectrum
print('\ncreating Supplemental Figure stellar_spec_G')
print('to show assumed stellar spectrum for a G star')
pc.plot_stellar_spectrum(spectrum_photo_G)
print('Supplemental Figure stellar_spec_G saved\n')

# plot assumed M star spectrum
print('creating Supplemental Figure stellar_spec_M')
print('to show assumed stellar spectrum for a M star')
pc.plot_stellar_spectrum(spectrum_photo_M,fig_name='stellar_spec_M')
print('Supplemental Figure stellar_spec_M saved\n')

# plot absorption cross sections with and without SO2
print('creating Supplemental Figures abs_x_*')
print('to show absorption cross sections for different molecules of interest')
pc.plot_cross_section(spectrum_photo_G,cross_w_SO2,cross_max)
pc.plot_cross_section(spectrum_photo_G,cross_w_SO2,cross_max,is_SO2=False)
print('Supplemental Figures abs_x_* saved\n')

# establish SO2 is not optically thick and thus should not contribute to
# the absorption cross section
print('creating Supplemental Figure tau_SO2')
print('to establish SO2 is not optically thick')
# calculate SO2 mass column
# (upper estimate of min SO2 given lowering photochemical conversion timescale will drive down u_SO2)
photochem_test = sulfur.Sulfur_Cycle(Earth_atm,'aero',t_convert=max(t_G,t_M))

u_SO2 = photochem_test.calc_uSO2_boundary() # [molecules/cm2]
# plot SO2 opacity given this mass column
pc.plot_SO2_tau(spectrum_photo_G,cross_w_SO2,u_SO2)
print('Supplemental Figure tau_SO2 saved')


################################################################
# SIMULATED TRANSMISSION SPECTRA
# Figure 3 & methods in Section 3.6 & results in Section 4.1
# & discussion in 5.3
# also inputs that generate simulated transmission spectra
################################################################
print('\n-----------------------------------------------\n'
      +'SIMULATED TRANSMISSION SPECTRA\n-----------------------------------------------')
# create inputs for transmission spectra
print('creating inputs for transmission spectra')
# vary vertical tau by orders of magnitude
taus = np.logspace(-4,1,6)
# calc scale height of stratosphere for haze cutoff
H = R_gas*Earth_atm.planet.T_strat/Earth_atm.planet.g/Earth_atm.p2mu(Earth_atm.p_transition_strat)*1e-3 # [km]
for tau in taus:
    sts.input_pro(Earth_atm,1000,tau,1.e-6,0,False)
    print('input for tau_h2so4 = %1.1E, r_h2so4 = 1 um, no haze cutoff saved'%tau)
    sts.input_pro(Earth_atm,1000,tau,1.e-6,0,False,haze_h=2*H)
    print('input for tau_h2so4 = %1.1E, r_h2so4 = 1 um, haze cutoff = %1.2F km (2 scale heights) saved'%(tau,2*H))
# vary haze cutoff height
heights = [1,5,10,20,50,100]
for h in heights:
    sts.input_pro(Earth_atm,1000,tau,1.e-6,0,False,haze_h=h)
    print('input for tau_h2so4 = 0.1, r_h2so4 = 1 um, haze cutoff = %1.0F km saved'%h)
# vary aerosol particle size
rs = np.linspace(1,10,10)*1.e-7 # [m]
for r in rs:
    sts.input_pro(Earth_atm,1000,0.1,r,0,False)
    print('input for tau_h2so4 = 0.1, r_h2so4 = %1.1F um, no haze cutoff saved'%(r*1e6))
#clear
sts.input_pro(Earth_atm,1000,0.1,0,0,False)
print('input for clear sky saved')
# low clouds
sts.input_pro(Earth_atm,1000,0.1,0,5.e-6,False)
print('input for low water clouds saved')
# high clouds
sts.input_pro(Earth_atm,1000,0.1,0,100e-6,True)
print('input for high water clouds saved')

# plot transmission spectra with varying tau
print('\ncreating Figure 3')

# FIGURE 3

# set up color scheme for plot
n = 7
new_colors = [plt.get_cmap('Blues_r')(1. * (n-i-1)/n) for i in range(n)]
plt.rc('axes', prop_cycle=cycler('color', new_colors))

# plot clear spectrum
f = './data/simtransspec/trans_spect_atm_pro_tau_0e+00_r_h2so4h2o_0e+00_r_water_0e+00.txt'
avg_wvlngth, avg_spec = sts.calc_avg_spec(f)
plt.plot(avg_wvlngth, avg_spec,lw='0.9',label='clear')
# plot spectra of various taus considered for r = 1 um
taus = ['0.0001','0.001','0.01','0.1','1', '10']
for i,x in enumerate(['-04','-03','-02','-01','+00', '+01']):
    # f = './data/simtransspec/trans_specs_h_2/trans_spect_atm_pro_tau_1e'+x+'_r_sulfur_1e-06_r_water_0e+00.txt'
    f = './data/simtransspec/trans_spect_atm_pro_tau_1e'+x+'_r_h2so4h2o_1e-06_r_water_0e+00_haze_h_12.txt'
    avg_wvlngth, avg_spec = sts.calc_avg_spec(f)
    plt.plot(avg_wvlngth, avg_spec,lw='0.9',label=r'$\delta$=%s'%(taus[i]))
# plot logistics
plt.xscale('log')
plt.xticks([0.5,1,5,10,50,100],['0.5','1','5','10','50','100'])
plt.xlabel(r'wavelength [$\mu$m]')
plt.ylabel('transit depth [ppm]')
plt.tick_params(axis='both', which='major')
plt.xlim(0.3,50)
# order legend labels to match plot order
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1],bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
plt.savefig('figs/fig03.pdf',bbox_inches='tight',transparent=True)
plt.close()

print('Figure 3 saved')


# show what no haze cutoff looks like with varying tau
print('\ncreating Supplemental Figure spec_no_cutoff')
print('to test effect of no haze cutoff')

# set up color scheme for plot
n = 7
new_colors = [plt.get_cmap('Blues_r')(1. * (n-i-1)/n) for i in range(n)]
plt.rc('axes', prop_cycle=cycler('color', new_colors))

# plot clear spectrum
f = './data/simtransspec/trans_spect_atm_pro_tau_0e+00_r_h2so4h2o_0e+00_r_water_0e+00.txt'
avg_wvlngth, avg_spec = sts.calc_avg_spec(f)
plt.plot(avg_wvlngth, avg_spec,lw='0.9',label='clear')
# plot spectra of various taus considered for r = 1 um
taus = ['0.0001','0.001','0.01','0.1','1']
for i,x in enumerate(['-04','-03','-02','-01','+00']):
    f = './data/simtransspec/trans_spect_atm_pro_tau_1e'+x+'_r_h2so4h2o_1e-06_r_water_0e+00.txt'
    avg_wvlngth, avg_spec = sts.calc_avg_spec(f)
    plt.plot(avg_wvlngth, avg_spec,lw='0.9',label=r'$\delta$=%s'%(taus[i]))
# plot logistics
plt.xscale('log')
plt.xticks([0.5,1,5,10,50,100],['0.5','1','5','10','50','100'])
plt.xlabel(r'wavelength [$\mu$m]')
plt.ylabel('transit depth [ppm]')
plt.tick_params(axis='both', which='major')
plt.xlim(0.3,50)
# order legend labels to match plot order
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1],bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
plt.savefig('figs_sup/spec_no_cutoff.pdf',bbox_inches='tight',transparent=True)
plt.close()

print('Supplemental Figure spec_no_cutoff saved')

print('\ncreating Supplemental Figure spec_vary_cutoff')
print('to test effect of varying haze height cutoff')

# set up color scheme for plot
n = 8
new_colors = [plt.get_cmap('jet_r')(1. * (n-i-1)/n) for i in range(n)]
plt.rc('axes', prop_cycle=cycler('color', new_colors))

# plot clear spectrum
f = './data/simtransspec/trans_spect_atm_pro_tau_0e+00_r_h2so4h2o_0e+00_r_water_0e+00.txt'
avg_wvlngth, avg_spec = sts.calc_avg_spec(f)
plt.plot(avg_wvlngth, avg_spec,lw='0.9',label='clear')
# plot different haze cutoff height spectra
for h in heights:
    f = './data/simtransspec/trans_spect_atm_pro_tau_1e-01_r_h2so4h2o_1e-06_r_water_0e+00_haze_h_%1.f.txt'%h
    avg_wvlngth, avg_spec = sts.calc_avg_spec(f)
    plt.plot(avg_wvlngth, avg_spec,lw='0.9',label=r'$h$=%1.f km'%h)
# plot logistics
plt.xscale('log')
plt.xticks([0.5,1,5,10,50,100],['0.5','1','5','10','50','100'])
plt.xlabel(r'wavelength [$\mu$m]')
plt.ylabel('transit depth [ppm]')
plt.tick_params(axis='both', which='major')
plt.xlim(0.3,50)
# order legend labels to match plot order
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1],bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
plt.savefig('figs_sup/spec_vary_cutoff.pdf',bbox_inches='tight',transparent=True)
plt.close()

print('Supplemental Figure spec_vary_cutoff saved')

print('\ncreating Supplemental Figure spec_smallest_r')
print('to test smallest aerosol particle radius at which Mie vs Rayleigh scattering is distinguishable')

# plot spectra of various H2SO4-H2O aerosol radii considered for tau = 0.1
# set up color scheme for plot
n = 11
new_colors = [plt.get_cmap('jet_r')(1. * (n-i-1)/n) for i in range(n)]
plt.rc('axes', prop_cycle=cycler('color', new_colors))

# plot clear spectrum
f = './data/simtransspec/trans_spect_atm_pro_tau_0e+00_r_h2so4h2o_0e+00_r_water_0e+00.txt'
avg_wvlngth, avg_spec = sts.calc_avg_spec(f)
plt.plot(avg_wvlngth, avg_spec,lw='0.9',label='clear')
# plot different r spectra
for i in range(1,10):
    f = './data/simtransspec/trans_spect_atm_pro_tau_1e-01_r_h2so4h2o_'+str(i)+'e-07_r_water_0e+00.txt'
    avg_wvlngth, avg_spec = sts.calc_avg_spec(f)
    plt.plot(avg_wvlngth, avg_spec,lw='0.9',label=r'$r$ = 0.'+str(i)+r' $\mu$m')
f = './data/simtransspec/trans_spect_atm_pro_tau_1e-01_r_h2so4h2o_1e-06_r_water_0e+00.txt'
avg_wvlngth, avg_spec = sts.calc_avg_spec(f)
plt.plot(avg_wvlngth, avg_spec,lw='0.9',label=r'$r$ = 1 $\mu$m')
# plot logistics
plt.xscale('log')
plt.xticks([0.5,1,5,10,50,100],['0.5','1','5','10','50','100'])
plt.xlabel(r'wavelength [$\mu$m]')
plt.ylabel('transit depth [ppm]')
plt.xlim(0.3,50)
plt.tick_params(axis='both', which='major')
# order legend labels to match plot order
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1],bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)


plt.savefig('figs_sup/spec_smallest_r.pdf',bbox_inches='tight',transparent=True)
plt.close()
print('Supplemental Figure spec_smallest_r saved\n')

# plot transmission spectra with H2SO4-H2O aerosols vs different water clouds
print('creating Supplemental Figure spec_other_scatters')
print('to test spectra distinguishable when high & low water clouds are present vs H2SO4-H2O aerosols')

# plot spectra with various scatters/absorbers present
# set up color scheme for plot
n = 4
new_colors = [plt.get_cmap('Blues_r')(1. * (n-i-1)/n) for i in range(n)]
plt.rc('axes', prop_cycle=cycler('color', new_colors))

# plot clear spectrum
f = './data/simtransspec/trans_spect_atm_pro_tau_0e+00_r_h2so4h2o_0e+00_r_water_0e+00.txt'
avg_wvlngth, avg_spec = sts.calc_avg_spec(f)
plt.plot(avg_wvlngth, avg_spec,lw='0.9',label='clear')

# plot low water clouds
f = './data/simtransspec/trans_spect_atm_pro_tau_1e-01_r_h2so4h2o_0e+00_r_water_5e-06.txt'
avg_wvlngth, avg_spec = sts.calc_avg_spec(f)
plt.plot(avg_wvlngth, avg_spec,lw='0.9',label='low water clouds')

# plot high water clouds
f = './data/simtransspec/trans_spect_atm_pro_tau_1e-01_r_h2so4h2o_0e+00_r_water_1e-04.txt'
avg_wvlngth, avg_spec = sts.calc_avg_spec(f)
plt.plot(avg_wvlngth, avg_spec,lw='0.9',label='high water clouds')

# plot H2SO4-H2O aerosols
f = './data/simtransspec/trans_spect_atm_pro_tau_1e-01_r_h2so4h2o_1e-06_r_water_0e+00_haze_h_12.txt'
avg_wvlngth, avg_spec = sts.calc_avg_spec(f)
plt.plot(avg_wvlngth, avg_spec,lw='0.9',label=r'H$_2$SO$_4$-H$_2$O aerosols')
# plot logistics
plt.xscale('log')
plt.xticks([0.5,1,5,10,50,100],['0.5','1','5','10','50','100'])
plt.xlabel(r'wavelength [$\mu$m]')
plt.ylabel('transit depth [ppm]')
plt.tick_params(axis='both', which='major')
plt.xlim(0.3,50)
# order legend labels to match plot order
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1],bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
plt.savefig('figs_sup/spec_other_scatters.pdf',bbox_inches='tight',transparent=True)
plt.close()
print('Supplemental Figure spec_other_scatters saved')


################################################################
# AQUEOUS S(IV) CHEMISTRY
# Figures 5-6 & methods in Section 3.7 & results in Section 4.3
################################################################

print('\n-----------------------------------------------\n'
      +'AQUEOUS S(IV) CHEMISTRY\n-----------------------------------------------')

# FIGURE 5
# plot distribution of S(IV) among SO2(aq), HSO3-, SO3-- vs pH
# ASSUMING S(IV) saturation
print('creating Figure 5')
pHs = np.linspace(1,14,100)
frac_so2, frac_hso3, frac_so3 = sulfur.S_aq_fractions(pHs,T_earth)
plt.plot(pHs,frac_so2, lw=3.5, label=r'SO$_2$(aq)', c=colors3[0])
plt.plot(pHs, frac_hso3, lw=2, label=r'HSO$_3^-$',c=colors3[1])
plt.plot(pHs,frac_so3, lw=2, label=r'SO$_3^{2-}$', c=colors3[2])
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
    f = sulfur.S_atm_ocean_frac(pHs,m_oc,Earth)
    plt.plot(pHs,f,label=r'M$_\mathrm{ocean}$ = '+str(m_oc)+ ' ' +
             r'M$_{\bigoplus \mathrm{ocean}}$',c=colors3[2-i])
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
print('atmosphere S / ocean S = %1.3E'%sulfur.S_atm_ocean_frac(pH_earth,1,Earth))


################################################################
# SULFUR IN THE ATMOSPHERE
# Figure 4 & results in Section 4.2
################################################################
print('\n-----------------------------------------------\n'
      +'ATMOSPHERIC SULFUR\n-----------------------------------------------')

# *_b => best estimate scenario
# *_lim => physically limiting scenario

# set up baseline Earth-based values
n = 50

# non-standard model parameters (as in Table 1)
r_G_lim = 1.e-7 # [m]
r_M_lim = 2.e-7 # [m]
t_mix = 1.*s_in_yr # [s]
n_outgass_lim = 200. # [modern Earth outgassing]
n_outgass_b = 1. # [modern Earth outgassing]

# critical mass column of SO2 for observation
u_so2_b = 1e-6*p_surf_earth/9.81*sulfur.mu_so2/mu_air # [kg/m3]
u_so2_lim = 1e-2*u_so2_b # [kg/m3]


# set up standard sulfur cycle objects for Earth-baseline conditions
best_aero_Earth = sulfur.Sulfur_Cycle(Earth_atm,'aero')
lim_aero_G_Earth = sulfur.Sulfur_Cycle(Earth_atm_dry,'aero',r=r_G_lim,alpha=1.,t_convert=t_G)
lim_aero_M_Earth = sulfur.Sulfur_Cycle(Earth_atm_dry,'aero',r=r_M_lim,alpha=1.,t_convert=t_M,is_M=True,is_G=False)
best_gas_Earth = sulfur.Sulfur_Cycle(Earth_atm,'gas',u_so2=u_so2_b)
lim_gas_Earth = sulfur.Sulfur_Cycle(Earth_atm_dry,'gas',u_so2=u_so2_lim)

# print critical total atmospheric sulfur to be observable
# for Earth-like planetary conditions
# both best estimate and limiting scenarios
t = PrettyTable(['obs S','model param',' # S atoms','# S moles', 'S kg'])

t.add_row(['aerosol','best','%1.3E'%best_aero_Earth.N_S_atm,
           '%1.3E'%best_aero_Earth.mol_S_atm,
           '%1.3E'%best_aero_Earth.mass_S_atm])

t.add_row(['aerosol','limiting M','%1.3E'%lim_aero_M_Earth.N_S_atm,
           '%1.3E'%lim_aero_M_Earth.mol_S_atm,
           '%1.3E'%lim_aero_M_Earth.mass_S_atm])

t.add_row(['aerosol','limiting G','%1.3E'%lim_aero_G_Earth.N_S_atm,
           '%1.3E'%lim_aero_G_Earth.mol_S_atm,
           '%1.3E'%lim_aero_G_Earth.mass_S_atm])

t.add_row(['gas','best','%1.3E'%best_gas_Earth.N_S_atm,
           '%1.3E'%best_gas_Earth.mol_S_atm,
           '%1.3E'%best_gas_Earth.mass_S_atm])

t.add_row(['gas','limiting','%1.3E'%lim_gas_Earth.N_S_atm,
           '%1.3E'%lim_gas_Earth.mol_S_atm,
           '%1.3E'%lim_gas_Earth.mass_S_atm])

print('critical total atmospheric sulfur to be observable')
print(t)

# FIGURE 4
# sensitivities to planetary parameters for critical sulfur required
# in the atmosphere for observation
print('creating Figure 4')
fig, axarr = plt.subplots(3,2,sharey=True,figsize=(6,8))
print('calculating sensitivities')

# SURFACE TEMPERATURE
print('\t surface temperature')
axarr[0,0].set_title(r'$T_\mathrm{surf}$')
axarr[0,0].set_xlabel(r'$T_\mathrm{surf}$ [K]')
T_surfs = np.linspace(250,400,n)
N_S_T_surfs = np.zeros((n,2))
for i,T in enumerate(T_surfs):
    pl = Planet(1,T,T_strat,p_surf_earth,X,1)
    atm = atm_pro.Atm(pl,RH_earth)
    atm.set_up_atm_pro()
    S_test_aero = sulfur.Sulfur_Cycle(atm,'aero')
    S_test_gas = sulfur.Sulfur_Cycle(atm,'gas')
    N_S_T_surfs[i,0] = S_test_aero.N_S_atm
    N_S_T_surfs[i,1] = S_test_gas.N_S_atm
N_S_base0 = best_aero_Earth.N_S_atm
N_S_base1 = best_gas_Earth.N_S_atm
axarr[0,0].plot(T_surfs,N_S_T_surfs[:,0]/N_S_base0,c='#ff8c00',label='aerosol')
axarr[0,0].plot(T_surfs,N_S_T_surfs[:,1]/N_S_base1,c='#002fa7',label='gas')
axarr[0,0].axvline(T_earth,ls='--',c='0.8')

axarr[0,0].set_ylabel(r'$N^\ast_{\mathrm{S}}/N^\ast_{\mathrm{S,}\oplus}$')
axarr[1,0].set_ylabel(r'$N^\ast_{\mathrm{S}}/N^\ast_{\mathrm{S,}\oplus}$')
axarr[2,0].set_ylabel(r'$N^\ast_{\mathrm{S}}/N^\ast_{\mathrm{S,}\oplus}$')

# STRATOSPHERIC TEMPERATURE
print('\t stratospheric temperature')
axarr[1,0].set_title(r'$T_\mathrm{strat}$')
T_strats = np.linspace(150,215,n)
N_S_T_strats = np.zeros((n,2))
for i,T in enumerate(T_strats):
    pl = Planet(1,T_earth,T,p_surf_earth,X,1)
    atm = atm_pro.Atm(pl,RH_earth)
    atm.set_up_atm_pro()
    S_test_aero = sulfur.Sulfur_Cycle(atm,'aero')
    S_test_gas = sulfur.Sulfur_Cycle(atm,'gas')
    N_S_T_strats[i,0] = S_test_aero.N_S_atm
    N_S_T_strats[i,1] = S_test_gas.N_S_atm
axarr[1,0].plot(T_strats,N_S_T_strats[:,0]/N_S_base0,c='#ff8c00',label='aerosol')
axarr[1,0].plot(T_strats,N_S_T_strats[:,1]/N_S_base1,c='#002fa7',label='gas')
axarr[1,0].axvline(T_strat,ls='--',c='0.8')
axarr[1,0].set_xlabel(r'$T_\mathrm{strat}$ [K]')

# SURFACE PRESSURE
print('\t surface pressure')
axarr[0,1].set_title(r'$p_\mathrm{surf}$')
axarr[0,1].set_xlabel(r'$p_\mathrm{surf}$ [Pa]')
p_surfs = np.logspace(-2,2,n)*1.01325e5
N_S_p_surfs = np.zeros((n,2))
for i,p in enumerate(p_surfs):
    pl = Planet(1,T_earth,T_strat,p,X,1)
    atm = atm_pro.Atm(pl,RH_earth)
    atm.set_up_atm_pro()
    S_test_aero = sulfur.Sulfur_Cycle(atm,'aero')
    S_test_gas = sulfur.Sulfur_Cycle(atm,'gas')
    N_S_p_surfs[i,0] = S_test_aero.N_S_atm
    N_S_p_surfs[i,1] = S_test_gas.N_S_atm
axarr[0,1].plot(p_surfs,N_S_p_surfs[:,0]/N_S_base0,c='#ff8c00',label='aerosol')
axarr[0,1].plot(p_surfs,N_S_p_surfs[:,1]/N_S_base1,c='#002fa7',label='gas')
axarr[0,1].axvline(p_surf_earth,ls='--',c='0.8')
axarr[0,1].set_xscale('log')

# PLANET SIZE
print('\t planet size')
axarr[2,0].set_title(r'$R_\mathrm{P}$')
axarr[2,0].set_xlabel(r'$R_\mathrm{P}$ [$R_\oplus$]')
R_ps = np.linspace(0.25,1.6,n)
N_S_size = np.zeros((n,2))
for i,R in enumerate(R_ps):
    pl = Planet(R,T_earth,T_strat,p_surf_earth,X)
    atm = atm_pro.Atm(pl,RH_earth)
    atm.set_up_atm_pro()
    S_test_aero = sulfur.Sulfur_Cycle(atm,'aero')
    S_test_gas = sulfur.Sulfur_Cycle(atm,'gas')
    N_S_size[i,0] = S_test_aero.N_S_atm/pl.M
    N_S_size[i,1] = S_test_gas.N_S_atm/pl.M
axarr[2,0].plot(R_ps,N_S_size[:,0]/N_S_base0*M_earth,c='#ff8c00',label='aerosol')
axarr[2,0].plot(R_ps,N_S_size[:,1]/N_S_base1*M_earth,c='#002fa7',label='gas')
axarr[2,0].axvline(1,ls='--',c='0.8')

# ATMOSPHERIC COMPOSITION
# vary between all N2 and all CO2
print('\t atmospheric composition')
axarr[1,1].set_title('composition')
axarr[1,1].set_xlabel(r'$\mu$ [g/mol]')
percent_x = np.linspace(0,1,n)
mus = np.zeros(n)
N_S_mus = np.zeros((n,2))
for i,x in enumerate(percent_x):
    mus[i] = ((1-x)*mu_n2 + x*mu_co2)
    X_test = np.zeros(5)
    X_test[2] = 1-x
    X_test[4] = x
    pl = Planet(1,T_earth,T_strat,p_surf_earth,X_test,1)
    atm = atm_pro.Atm(pl,RH_earth)
    atm.set_up_atm_pro()
    S_test_aero = sulfur.Sulfur_Cycle(atm,'aero')
    S_test_gas = sulfur.Sulfur_Cycle(atm,'gas')
    N_S_mus[i,0] = S_test_aero.N_S_atm
    N_S_mus[i,1] = S_test_gas.N_S_atm
axarr[1,1].plot(mus*1e3,N_S_mus[:,0]/N_S_base0,c='#ff8c00',label='aerosol')
axarr[1,1].plot(mus*1e3,N_S_mus[:,1]/N_S_base1,c='#002fa7',label='gas')
axarr[1,1].axvline(Earth.mu_dry*1e3,ls='--',c='0.8')

# WATER CONTENT
# vary RH at surface
print('\t surface relative humidity')
axarr[2,1].set_title(r'$\mathrm{RH}_{\mathrm{surf}}$')
axarr[2,1].set_xlabel(r'$\mathrm{RH}_{\mathrm{surf}}$')
RH_h2o_surfs = np.logspace(-5,0,n)
N_S_f_h2os = np.zeros((n,2))
for i,RH in enumerate(RH_h2o_surfs):
    pl = Planet(1,T_earth,T_strat,p_surf_earth,X,1)
    atm = atm_pro.Atm(pl,RH)
    atm.set_up_atm_pro()
    S_test_aero = sulfur.Sulfur_Cycle(atm,'aero')
    S_test_gas = sulfur.Sulfur_Cycle(atm,'gas')
    N_S_f_h2os[i,0] = S_test_aero.N_S_atm
    N_S_f_h2os[i,1] = S_test_gas.N_S_atm
f_h2o_surfs = RH_h2o_surfs*h2o.p_sat(T_earth)/p_surf_earth # []
f_h2o_surf_b = RH_earth*h2o.p_sat(T_earth)/p_surf_earth # []
axarr[2,1].plot(f_h2o_surfs,N_S_f_h2os[:,0]/N_S_base0,c='#ff8c00',label='aerosol')
axarr[2,1].plot(f_h2o_surfs,N_S_f_h2os[:,1]/N_S_base1,c='#002fa7',label='gas')
axarr[2,1].axvline(f_h2o_surf_b,ls='--',c='0.8',label='Earth-like value')
axarr[2,1].set_xscale('log')

# plot logistics
plt.ylim(1e-2,1e2)
plt.yscale('log')
for i in range(3):
    for j in range(2):
        axarr[i,j].axhspan(0.01,0.1,color='r',alpha=0.85,label='< 10% ' r'$N^\ast_{\mathrm{S}}/N^\ast_{\mathrm{S,}\oplus}$')

fig.subplots_adjust(hspace=0.5,wspace=0.05)
handles, labels = axarr[2,1].get_legend_handles_labels()
fig.legend(handles, labels, ncol=4,loc=8)
plt.savefig('figs/fig04.pdf',bbox_inches='tight',transparent=True)
plt.close()
print('Figure 4 saved')


################################################################
# SULFUR OBSERVABILITY GIVEN OCEAN PARAMETERS
# Figures 7-10 & results in Section 4.4
################################################################

print('\n-----------------------------------------------\n'
      +'SULFUR OBSERVABILITY GIVEN OCEAN PARAMETERS\n-----------------------------------------------')

# various ocean parameters for pH and ocean size
# (diff ocean sizes for optimally framed plots
# compared to likely tSIVs)
pH = np.linspace(1,14,n)
oceans = np.logspace(-3.03,0.1,n)
oceans_ex0 = np.logspace(-6,0.1,n)
oceans_ex = np.logspace(-9,0.1,n)
oceans_ex2 = np.logspace(-10,0.1,n)
ocgrid, pHgrid  = np.meshgrid(oceans,pH)
ocgrid_ex0, pHgrid_ex0  = np.meshgrid(oceans_ex0,pH)
ocgrid_ex, pHgrid_ex  = np.meshgrid(oceans_ex,pH)
ocgrid_ex2, pHgrid_ex2  = np.meshgrid(oceans_ex2,pH)

print('creating Figures 7-10')

# limiting case with M star -- aerosols
lim_aero_M_Earth.calc_oc_S(ocgrid, pHgrid)
t_SIV_M_lim = lim_aero_M_Earth.calc_t_SIV(n_outgass_lim)

# limiting case with G star -- aerosols
lim_aero_G_Earth.calc_oc_S(ocgrid, pHgrid)
t_SIV_G_lim = lim_aero_G_Earth.calc_t_SIV(n_outgass_lim)

# reasonable case -- aerosols
best_aero_Earth.calc_oc_S(ocgrid_ex, pHgrid_ex)
t_SIV_b = best_aero_Earth.calc_t_SIV(n_outgass_b)

# reasonable case -- SO2 (gas)
best_gas_Earth.calc_oc_S(ocgrid_ex2, pHgrid_ex2)
t_SIV_gas = best_gas_Earth.calc_t_SIV(n_outgass_b)

# limiting case -- SO2 (gas)
lim_gas_Earth.calc_oc_S(ocgrid_ex0, pHgrid_ex0)
t_SIV_gas_lim = lim_gas_Earth.calc_t_SIV(n_outgass_lim)

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


# have option to show where t_SIV = 0.1 and pH = 6 is
is_show_maxoc = False

# FIGURE 10
# plot limiting case for aerosols
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

# highlight ocean parameters of interest
interesting_oc = Rectangle((6,1e-3),7.9,1.2,fill=False,edgecolor='0.8',lw=3,ls='--',label='\n<10%\n' r'$N^\ast_{\mathrm{S}}/N^\ast_{\mathrm{S,}\oplus}$')
interesting_oc2 = Rectangle((6,1e-3),7.9,1.2,fill=False,edgecolor='0.8',lw=3,ls='--',label='\n<10%\n' r'$N^\ast_{\mathrm{S}}/N^\ast_{\mathrm{S,}\oplus}$')
ax1.add_patch(interesting_oc)
ax2.add_patch(interesting_oc2)

# ocean size for which pH = 6 and t_SIV_crit = 0.1 yr
oc_aero_lM = 1.3e-3 # [Earth oc]
oc_aero_lG = 1.5e-3 # [Earth oc]
h_aero_lM = oc_aero_lM*mass_earth_ocean/rho_h2o/4./np.pi/R_earth**2 # [m]
h_aero_lG = oc_aero_lG*mass_earth_ocean/rho_h2o/4./np.pi/R_earth**2 # [m]

# highlight on plot where max oc is
if is_show_maxoc:
    ax1.axhline(oc_aero_lM)
    ax2.axhline(oc_aero_lG)
    ax1.axvline(6)
    ax2.axvline(6)

# set labels
ax1.set_title('M-dwarf')
ax2.set_title('Sun-like')
ax2.set_xlabel('pH')
ax1.set_ylabel(r'# Earth oceans [$M_{\oplus\mathrm{ocean}}$]')

# annotate plot for context

# Earth ocean size and pH
ax1.scatter(8.14,1,color='k',s=10,zorder=10)
ax1.annotate('modern Earth ocean', (8.14,0.7))
ax2.scatter(8.14,1,color='k',s=10,zorder=10)
ax2.annotate('modern Earth ocean', (8.14,0.7))

# arrows designating likeliness of haze formation given ocean parameters
ax1.arrow(0.36, 0.35, 0.5, 0.4, transform=ax1.transAxes,
          length_includes_head=True,head_width=0.02, head_length=0.03,fc='k')
ax1.annotate('observable \n' r'H$_2$SO$_4$-H$_2$O haze'
             '\nincreasingly \nunlikely',(0.3,0.425),xycoords='figure fraction')
ax1.annotate('observable \n' r'H$_2$SO$_4$-H$_2$O' '\nhaze'
             '\npossible',(0.09,0.15),xycoords='figure fraction',
             color='k')
ax2.arrow(0.41, 0.4, 0.45, 0.35, transform=ax2.transAxes,
          length_includes_head=True,head_width=0.02, head_length=0.03,fc='k')
ax2.annotate('observable \n' r'H$_2$SO$_4$-H$_2$O haze'
             '\nincreasingly \nunlikely',(0.725,0.375),xycoords='figure fraction')
ax2.annotate('observable \n' r'H$_2$SO$_4$-H$_2$O' '\nhaze'
             '\npossible',(0.525,0.2),xycoords='figure fraction',
             color='k')


plt.savefig('figs/fig10.pdf',bbox_inches='tight',transparent=True)
plt.close()


# FIGURE 8
fig, ax1 = plt.subplots(figsize=(4.5,6.5))

# set up color bar
cs1 = plt.contourf(pHgrid_ex, ocgrid_ex, np.log10(t_SIV_b),levels=levels,colors=colors,extend='both')
cs1_ = plt.colorbar(cs1,ticks=levels,orientation='horizontal')
cs1_.set_label(r'$\log_{10}$($\tau_\mathrm{S(IV)}^\ast$ [yr])')
plt.xlabel('pH')
plt.ylabel(r'# Earth oceans [$M_{\oplus\mathrm{ocean}}$]')
plt.yscale('log')
# don't make negative-valued contours dashed
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
# contour for reasonable t_SIV
plt.contour(pHgrid_ex, ocgrid_ex, np.log10(t_SIV_b),levels=[likely],linewidths=3,colors='w')
# remove white lines between contours for vector image
cs1_.solids.set_edgecolor('face')
cs1_.add_lines(zeromark)

# highlight ocean parameters of interest
interesting_oc3 = Rectangle((6,1e-3),7.9,1.2,fill=False,edgecolor='0.8',lw=3,ls='--',label='\n<10%\n' r'$N^\ast_{\mathrm{S}}/N^\ast_{\mathrm{S,}\oplus}$')
ax1.add_patch(interesting_oc3)
# Earth ocean size and pH
plt.scatter(8.14,1,color='k',s=10,zorder=10)
plt.annotate('modern Earth ocean', (8.3,0.4))

# ocean size for which pH = 6 and t_SIV_crit = 0.1 yr
oc_aero_b = 2.5e-8 #[Earth oc]
h_aero_b = oc_aero_b*mass_earth_ocean/rho_h2o/4./np.pi/R_earth**2 # [m]

# highlight on plot where max oc is
if is_show_maxoc:
    ax1.axhline(oc_aero_b)
    ax1.axvline(6)
# put ocean size in terms of GEL for second y-axis
ax2 = ax1.twinx()
ax2.set_ylabel('global equivalent ocean layer [m]')
ax2.set_yscale('log')
hgrid_ex = ocgrid_ex*mass_earth_ocean/rho_h2o/4./np.pi/R_earth**2
ax2.contour(pHgrid_ex, hgrid_ex, np.log10(t_SIV_b),levels=[likely],linewidths=3,colors='w')

# annotate plot for context
# arrows designating likeliness of haze formation given ocean parameters
plt.arrow(0.34, 0.27, 0.57, 0.55, transform=plt.gca().transAxes,
          length_includes_head=True,head_width=0.02, head_length=0.03,fc='k')
plt.annotate(r'observable H$_2$SO$_4$-H$_2$O'
             '\nhaze increasingly unlikely',(0.47,0.47),xycoords='figure fraction')
plt.annotate('observable \n' r'H$_2$SO$_4$-H$_2$O' '\nhaze'
             '\npossible',(0.17,0.33),xycoords='figure fraction',
             color='k')
plt.savefig('figs/fig08.pdf',bbox_inches='tight',transparent=True)
plt.close()


# FIGURE 7
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

# highlight ocean parameters of interest
interesting_oc4 = Rectangle((6,1e-3),7.9,1.2,fill=False,edgecolor='0.8',lw=3,ls='--',label='\n<10%\n' r'$N^\ast_{\mathrm{S}}/N^\ast_{\mathrm{S,}\oplus}$')
ax1.add_patch(interesting_oc4)
# annotate plot for context

plt.scatter(8.14,1,color='k',s=10,zorder=10)
plt.annotate('modern Earth ocean', (8.3,0.4))
plt.tick_params(axis='y', which='minor')
# put ocean size in terms of GEL for second y-axis
ax2 = ax1.twinx()
plt.tick_params(axis='y', which='minor')
ax2.set_ylabel('global equivalent ocean layer [m]')
ax2.set_yscale('log')
hgrid_ex2 = ocgrid_ex2*mass_earth_ocean/rho_h2o/4./np.pi/R_earth**2
ax2.contour(pHgrid_ex2, hgrid_ex2, np.log10(t_SIV_gas),levels=[likely],linewidths=3,colors='w')

# ocean size for which pH = 6 and t_SIV_crit = 0.1 yr
oc_gas_b = 1e-9 # [Earth oc]
h_gas_b = oc_gas_b*mass_earth_ocean/rho_h2o/4./np.pi/R_earth**2 # [m]

# highlight on plot where max oc is
if is_show_maxoc:
    ax1.axhline(oc_gas_b)
    ax1.axvline(6)

# annotate plot for context
# arrows designating likeliness of haze formation given ocean parameters
plt.arrow(0.31, 0.27, 0.57, 0.55, transform=plt.gca().transAxes,
          length_includes_head=True,head_width=0.02, head_length=0.03,fc='k')
plt.annotate(r'observable SO$_2$'
             '\n increasingly unlikely',(0.53,0.55),xycoords='figure fraction')
plt.annotate(r'observable SO$_2$'
             '\npossible',(0.17,0.33),xycoords='figure fraction',
             color='k')
plt.savefig('figs/fig07.pdf',bbox_inches='tight',transparent=True)
plt.close()


# FIGURE 9
fig, ax1 = plt.subplots(figsize=(4.5,6.5))
cs1 = plt.contourf(pHgrid_ex0, ocgrid_ex0, np.log10(t_SIV_gas_lim),levels=levels,extend='both',colors=colors)
cs1_ = plt.colorbar(cs1,ticks=levels,orientation='horizontal')
cs1_.set_label(r'$\log_{10}$($\tau_\mathrm{S(IV)}^\ast$ [yr])')
# remove white lines between contours for vector image
cs1_.solids.set_edgecolor('face')
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
plt.contour(pHgrid_ex0, ocgrid_ex0, np.log10(t_SIV_gas_lim),levels=[likely],linewidths=0.5,colors='w')
cs1_.add_lines(zeromark)
plt.yscale('log')
plt.xlabel('pH')
plt.ylabel(r'# Earth oceans [$M_{\oplus\mathrm{ocean}}$]')
# highlight ocean parameters of interest
interesting_oc5 = Rectangle((6,1e-3),7.9,1.2,fill=False,edgecolor='0.8',lw=3,ls='--',label='\n<10%\n' r'$N^\ast_{\mathrm{S}}/N^\ast_{\mathrm{S,}\oplus}$')
ax1.add_patch(interesting_oc5)
# Earth ocean size and pH
plt.scatter(8.14,1,color='k',s=10,zorder=10)
plt.annotate('modern Earth ocean', (8.3,0.6))
# put ocean size in terms of GEL for second y-axis
ax2 = ax1.twinx()
ax2.set_ylabel('global equivalent ocean layer [m]')
ax2.set_yscale('log')
hgrid_ex0 = ocgrid_ex0*mass_earth_ocean/rho_h2o/4./np.pi/R_earth**2
ax2.contour(pHgrid_ex0, hgrid_ex0, np.log10(t_SIV_gas_lim),levels=[likely],linewidths=3,colors='w')

# ocean size for which pH = 6 and t_SIV_crit = 0.1 yr
oc_gas_l = 2e-5 # [Earth oc]
h_gas_l = oc_gas_l*mass_earth_ocean/rho_h2o/4./np.pi/R_earth**2 # [m]

# highlight on plot where max oc is
if is_show_maxoc:
    ax1.axhline(oc_gas_l)
    ax1.axvline(6)
# annotate plot for context
# arrows designating likeliness of haze formation given ocean parameters
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

# print results of maximum ocean size for t_S(IV) = 0.1 years & ocean pH = 6
# to have observable atmospheric sulfur
t = PrettyTable(['obs S','model param','oc mass [Earth oc]','oc GEL [m]'])
t.add_row(['aerosol','best','%1.1E'%oc_aero_b,'%1.3E'%h_aero_b])
t.add_row(['aerosol','limiting M','%1.1E'%oc_aero_lM,'%1.3F'%h_aero_lM])
t.add_row(['aerosol','limiting G','%1.1E'%oc_aero_lG,'%1.3F'%h_aero_lG])
t.add_row(['gas','best','%1.1E'%oc_gas_b,'%1.3E'%h_gas_b])
t.add_row(['gas','limiting','%1.1E'%oc_gas_l,'%1.3F'%h_gas_l])
print('maximum ocean size for t_S(IV) = 0.1 years & ocean pH = 6 \nto have observable atmospheric sulfur')
print(t)
