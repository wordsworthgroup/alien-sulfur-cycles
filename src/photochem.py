import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.interpolate import interp1d

def set_up_photochem(lambda_max=240,f_XUV=1,f_UV=1,lambda_XUV_UV=91):
    '''
    inputs:
        *
    output:
        *
    '''
    spectrum = np.genfromtxt('./data/stellarflux_mean.dat')
    # col 0 - wavelength [nm]
    # col 1 - photon flux per wavelength [photons/s/cm^2/nm]

    # import absorbtion spectrum data for various molecules
    asSO2 = np.genfromtxt('./data/abs_x/axsSO2.dat')
    asSO2_2 = np.genfromtxt('./data/abs_x/axsSO2_2.dat')
    asCO2 = np.genfromtxt('./data/abs_x/axsCO2.dat')
    asO2 = np.genfromtxt('./data/abs_x/axsO2.dat')
    asH2O = np.genfromtxt('./data/abs_x/axsH2O.dat')

    # combine sources of absorbtion spectrums as needed and interpolate to
    # match 1 nm spaced solar spectrum
    SO2 = np.concatenate((np.array([[0,0]]),asSO2_2,asSO2))
    f_as_SO2 = interp1d(SO2[:,0],SO2[:,1])
    f_as_CO2 = interp1d(asCO2[:,0],asCO2[:,1])
    O2 = np.concatenate((np.array([[0,0]]),asO2,np.array([[300,0]])))
    f_as_O2 = interp1d(O2[:,0],O2[:,1])
    H2O = np.concatenate((np.array([[0,0]]),asH2O,np.array([[300,0]])))
    f_as_H2O = interp1d(H2O[:,0],H2O[:,1])

    # integrate photon flux up to lambda_max
    spectrum_photo = spectrum[np.where(spectrum[:,0]<=lambda_max)]
    spectrum_photo[:lambda_XUV_UV,1] = spectrum_photo[:91,1]*f_XUV
    spectrum_photo[lambda_XUV_UV:,1] = spectrum_photo[91:,1]*f_UV#*10**(-1.6)


    cross_SO2 = f_as_SO2(spectrum_photo[:,0])
    cross_CO2 = f_as_CO2(spectrum_photo[:,0])
    cross_O2 = f_as_O2(spectrum_photo[:,0])
    cross_H2O = f_as_H2O(spectrum_photo[:,0])
    cross_w_SO2 = np.vstack((cross_SO2,cross_CO2,cross_O2,cross_H2O))
    cross = np.vstack((cross_CO2,cross_O2,cross_H2O))
    cross_max = np.amax(cross,axis=0)
    photon_flux = integrate.simps(cross_max*spectrum_photo[:,1],spectrum_photo[:,0],even='last') #[photons/s/cm2]
    print(photon_flux)
    return cross_w_SO2, cross_max, spectrum_photo, photon_flux

def plot_photochem(cross_w_SO2, cross_max, spectrum_photo):
    # plot solar spectrum data
    # plt.plot(spectrum[:,0],spectrum[:,1],c='k')
    plt.plot(spectrum_photo[:,0],spectrum_photo[:,1],c='k')
    plt.xlabel(r'$\lambda$ [nm]')
    plt.ylabel(r'$\phi_\gamma$ [photons/s/cm$^2$/nm]')
    plt.yscale('log')
    # plt.xlim(0,500)
    plt.ylim(1e5,1e15)
    plt.axvline(240,c='r',ls='--')
    plt.show()
    # plt.savefig()
    # plt.close()

    labels = [r'SO$_2$',r'CO$_2$',r'O$_2$', r'H$_2$O']
    for i in range(4):
        plt.plot(spectrum_photo[:,0],cross_w_SO2[i,:],label=labels[i])
    plt.plot(spectrum_photo[:,0],cross_max,c='k',lw=2,ls='--',label='maximum')
    plt.legend()
    plt.yscale('log')
    plt.ylim(1e-23,1e-15)
    plt.xlabel(r'$\lambda$ [nm]')
    plt.ylabel(r'absorbtion cross section [cm$^2$/molecule]')
    plt.show()

    plt.plot(spectrum_photo[:,0],cross_w_SO2[0,:]*5E+13)
    plt.axhline(1,c='0.8',ls='--')
    plt.xlabel(r'$\lambda$ [nm]')
    plt.ylabel(r'$\tau$ []')
    plt.show()
    # plt.savefig()
    # plt.close()

cross_w_SO2, cross_max, spectrum_photo, photon_flux = set_up_photochem(lambda_max=240,f_XUV=1,f_UV=1,lambda_XUV_UV=91)
plot_photochem(cross_w_SO2, cross_max, spectrum_photo)
