################################################################
# photochemistry of SO2 to H2SO4 conversion
# for lower bound on conversion timescale using (X)UV photon
# delivery as energy limiting factor
################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from cycler import cycler

def set_up_photochem(lambda_max=240,f_XUV=1,f_UV=1,lambda_XUV_UV=91):
    '''
    set up photochemical calculations for limiting photochemical conversion
    timescales of SO2 to H2SO4

    inputs:
        * lambda_max [nm] - maximum wavelength cutoff for photodissociating
        * f_XUV [] - XUV factor to multiply solar spectrum by
        * f_UV [] - UV factor to multiply solar spectrum by
        * lambda_XUV_UV [nm] - wavelength at which XUV transitions to UV

    output:
        * cross_w_SO2 [cm2/particle] - cross sections of SO2, CO2, O2, H2O
        * cross_max [cm2/particle] - maximum cross section for each wavelength (no SO2)
        * spectrum_photo [nm,photons/s/cm^2/nm] - stellar XUV-UV spectrum
    '''
    # read in solar spectrum
    # source: Thuillier et al. (2004)
    spectrum = np.genfromtxt('./data/stellarflux_mean.dat')
    # col 0 - wavelength [nm]
    # col 1 - photon flux per wavelength [photons/s/cm^2/nm]

    # import absorption spectrum data for various molecules
    # SO2
    # sources:
    #   * Hamdy et al. 1991, http://satellite.mpic.de/spectral_atlas/cross_sections/Sulfur%20compounds/Inorganic%20S-compounds/SO2_Hamdy(1991)_298K_14.7-101.7nm.txt
    #   * Manatt & Lane 1993, http://satellite.mpic.de/spectral_atlas/cross_sections/Sulfur%20compounds/Inorganic%20S-compounds/SO2_ManattLane(1993)_293K_106.1-403.7nm.txt
    asSO2 = np.genfromtxt('./data/abs_x/axsSO2.dat')
    # CO2
    # source:
    #   * Huestis & Berkowitz 2010, http://satellite.mpic.de/spectral_atlas/cross_sections/Carbon-oxides/CO2_HuestisBerkowitz(2010)_300K_0.1254-201.6nm(evaluation).txt
    asCO2 = np.genfromtxt('./data/abs_x/axsCO2.dat')
    # O2
    # sources:
    #   * Brion 1979, http://satellite.mpic.de/spectral_atlas/cross_sections/Oxygen/O2_Brion(1979)_298K_4.1-248nm(e,e).txt
    #   * Lu 2010, http://satellite.mpic.de/spectral_atlas/cross_sections/Oxygen/O2_Lu(2010)_303.7K_115-180nm.txt
    #   * Kockarts 1976, http://satellite.mpic.de/spectral_atlas/cross_sections/Oxygen/O2_Kockarts(1976)_300K_176-203nm.txt
    #   * Yoshino et al. 1988, http://satellite.mpic.de/spectral_atlas/cross_sections/Oxygen/O2_JPL-2010(2011)_298K_205-245nm(rec).txt
    asO2 = np.genfromtxt('./data/abs_x/axsO2.dat')
    # H2O
    # sources:
    #   * Chan et al. 1993, http://satellite.mpic.de/spectral_atlas/cross_sections/Hydrogen+water/H2O_Chan(1993)_298K_6.20-206.64nm(e,e).txt
    #   * Mota et al. 2005, http://satellite.mpic.de/spectral_atlas/cross_sections/Hydrogen+water/H2O_Mota(2005)_298K_114.8-193.9nm.txt
    asH2O = np.genfromtxt('./data/abs_x/axsH2O.dat')

    # combine sources of absorption spectrums as needed and interpolate to
    # match 1 nm spaced solar spectrum
    SO2 = np.concatenate((np.array([[0,0]]),asSO2))
    f_as_SO2 = interp1d(SO2[:,0],SO2[:,1])
    CO2 = np.concatenate((np.array([[0,0]]),asCO2,np.array([[300,0]])))
    f_as_CO2 = interp1d(CO2[:,0],CO2[:,1])
    O2 = np.concatenate((np.array([[0,0]]),asO2,np.array([[300,0]])))
    f_as_O2 = interp1d(O2[:,0],O2[:,1])
    H2O = np.concatenate((np.array([[0,0]]),asH2O,np.array([[300,0]])))
    f_as_H2O = interp1d(H2O[:,0],H2O[:,1])

    # take photon flux up to lambda_max
    spectrum_photo = spectrum[np.where(spectrum[:,0]<=lambda_max)]
    # apply UV multiplier
    spectrum_photo[:lambda_XUV_UV,1] = spectrum_photo[:91,1]*f_XUV
    # apply XUV multiplier
    spectrum_photo[lambda_XUV_UV:,1] = spectrum_photo[91:,1]*f_UV

    # set up evenly spaced in wavelength cross sections
    cross_SO2 = f_as_SO2(spectrum_photo[:,0])
    cross_CO2 = f_as_CO2(spectrum_photo[:,0])
    cross_O2 = f_as_O2(spectrum_photo[:,0])
    cross_H2O = f_as_H2O(spectrum_photo[:,0])
    cross_w_SO2 = np.vstack((cross_SO2,cross_CO2,cross_O2,cross_H2O))
    cross = np.vstack((cross_CO2,cross_O2,cross_H2O))
    # maximum cross section across all gasses
    cross_max = np.amax(cross,axis=0)

    return cross_w_SO2, cross_max, spectrum_photo

def plot_stellar_spectrum(spectrum_photo,fig_name='stellar_spec_G',lambda_max=240):
    '''
    plot stellar spectrum

    inputs:
        * spectrum_photo [nm,photons/s/cm^2/nm] - stellar XUV-UV spectrum
        * fig_name [string] - optional, name of figure saved
        * lambda_max [nm] - maximum UV cutoff wavelength
    '''
    # plot spectrum data
    plt.plot(spectrum_photo[:,0],spectrum_photo[:,1],c='k')
    plt.xlabel(r'$\lambda$ [nm]')
    plt.ylabel(r'$\phi_\gamma$ [photons/s/cm$^2$/nm]')
    plt.yscale('log')
    plt.axvline(lambda_max,c='r',ls='--')
    plt.savefig('figs_sup/'+fig_name+'.pdf',bbox_inches='tight',transparent=True)
    plt.close()

def plot_cross_section(spectrum_photo,cross_w_SO2,cross_max,is_SO2=True,lambda_max=240):
    '''
    plot absorption cross sections of different gases

    inputs:
        * spectrum_photo [nm,photons/s/cm^2/nm] - stellar XUV-UV spectrum
        * cross_w_SO2 [cm2/particle] - cross sections of SO2, CO2, O2, H2O
        * cross_max [cm2/particle] - maximum cross section at every wavelength
        * is_SO2 [boolean] - whether to include SO2 cross section
        * lambda_max [nm] - maximum UV cutoff wavelength
    '''
    # gas labels
    labels = [r'SO$_2$',r'CO$_2$',r'O$_2$', r'H$_2$O']
    n = len(labels) + 1
    # set up colors
    new_colors = [plt.get_cmap('Blues')(1. * (n-i-1)/n) for i in range(n)]
    plt.rc('axes', prop_cycle=cycler('color', new_colors))
    # plot cross sections
    for i in range(4):
        # plot SO2 cross section based on boolean
        if is_SO2:
            plt.plot(spectrum_photo[:,0],cross_w_SO2[i,:],label=labels[i])
        else:
            if i!=0:
                plt.plot(spectrum_photo[:,0],cross_w_SO2[i,:],label=labels[i])
    plt.plot(spectrum_photo[:,0],cross_max,c='k',lw=2,ls='--',label='maximum')
    plt.legend()
    plt.yscale('log')
    plt.ylim(1e-23,1e-15)
    plt.xlabel(r'$\lambda$ [nm]')
    plt.ylabel(r'absorption cross section [cm$^2$/molecule]')
    # save figure with name dependent on whether SO2 is included
    if is_SO2:
        plt.savefig('figs_sup/abs_x_w_SO2.pdf',bbox_inches='tight',transparent=True)
    else:
        plt.savefig('figs_sup/abs_x_wo_SO2.pdf',bbox_inches='tight',transparent=True)
    plt.close()

def plot_SO2_tau(spectrum_photo,cross_w_SO2,u_SO2=5E+13):
    '''
    plot opacity (tau) of SO2 with an eye toward whether it's optically thick

    inputs:
        * spectrum_photo [nm,photons/s/cm^2/nm] - stellar XUV-UV spectrum
        * cross_w_SO2 [cm2/particle] - cross sections of SO2, CO2, O2, H2O
        * u_SO2 [molecules/cm2] - SO2 mass column
    '''
    # maximum cross section for each wavelength including SO2
    cross_max_w_SO2 = np.amax(cross_w_SO2,axis=0)
    # plot (max) opacity of SO2, tau
    # tau = cross section x mass column SO2
    plt.plot(spectrum_photo[:,0],cross_max_w_SO2*u_SO2,c='k',lw=2)
    plt.axhline(1,c='0.8',ls='--')
    plt.xlabel(r'$\lambda$ [nm]')
    plt.ylabel(r'$\tau$ []')
    plt.yscale('log')
    plt.savefig('figs_sup/tau_SO2.pdf',bbox_inches='tight',transparent=True)
    plt.close()
