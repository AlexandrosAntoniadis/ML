#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 18:31:41 2018

@author: aantoniadis
"""

import numpy as np
from astropy.io import fits
import matplotlib.pylab as plt
from scipy.interpolate import interp1d


hdulist = fits.open('spectra/FEROS/NewFEROSGJ1061.fits')

hdulist = hdulist[1]

data = hdulist.data

flux = data['FLUX']
wave = data['WAVE']

cdelt1= 0.01

nulos = np.where(~np.isnan(flux))
wave = wave[nulos]
flux = flux[nulos]
f2 = interp1d(wave, flux, kind='linear')
wave_int = np.arange(wave[0], wave[-1], cdelt1)
flux_int = f2(wave_int)


prihdr = fits.Header()
prihdr.set("CDELT1",wave_int[1]-wave_int[0])
prihdr.set("CRVAL1",wave_int[0])
#fits.writeto("spectra/FEROS1D/1DWP_NewFEROSGJ1061.fits", flux_int, prihdr, overwrite=True)

wavemin = 5000
wavemax = 7000

wave_int = np.arange(wavemin, wavemax, cdelt1)
flux_int = f2(wave_int)

highpoints = flux_int > 0.025


print(flux_int)
wave_temp, flux_temp = wave_int[~highpoints], flux_int[~highpoints]
#wave_temp, flux_temp = wave_int[points], flux_int[points]

plt.plot(wave_temp, flux_temp, ".-")

f3 = interp1d(wave_temp, flux_temp, kind='linear')

flux_int = f3(wave_int)

prihdr = fits.Header()
prihdr.set("CDELT1",wave_int[1]-wave_int[0])
prihdr.set("CRVAL1",wave_int[0])
fits.writeto("spectra/FEROS1D/1DNP_NewFEROSGJ1061.fits", flux_int, prihdr, overwrite=True)
