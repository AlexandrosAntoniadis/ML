#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 16:44:33 2018

@author: mac
"""

import numpy as np
from astropy.io import fits
import matplotlib.pylab as plt
from scipy.interpolate import interp1d
from PyAstronomy import pyasl
from astropy.io import fits
from build_pew_verygeneral import pseudo_EW
import numpy as np
import os


# change resolution in the newflux (red number)

def read_data(fname):


    flux = fits.getdata(fname)
    hdr = fits.getheader(fname)
    w0, dw, N = hdr['CRVAL1'], hdr['CDELT1'], hdr['NAXIS1']
    wavelength = w0 + dw * np.arange(N)
    
    newflux = pyasl.instrBroadGaussFast(wavelength, flux, 48000, edgeHandling="firstlast", fullout=False, maxsig=None)
    
    #prihdr = fits.Header()
    #prihdr.set("CDELT1",w0)
    #prihdr.set("CRVAL1",dw)
    #prihdr.set("NAXIS1",N)

    fits.writeto(fname.replace('.fits', '')+'res48.fits', newflux, hdr, overwrite=True)

filepaths = np.loadtxt('HARPSfilelist.dat', dtype=str)
for item in filepaths:
    read_data(item)


# multiple lines
    
filepaths = np.loadtxt('res48HARPSfilelist.dat', dtype=str)

wavelength_range = np.loadtxt('lines.rdb', skiprows=2)
dw = 0.4
plot = False

directory_name = 'res48EWmyresults'

if not os.path.exists(directory_name):
    os.makedirs(directory_name)
    

try:
    size_of_filepaths = len(filepaths)
except TypeError:
    filepaths = [str(filepaths)]
    size_of_filepaths = len(filepaths)

    

for i in np.arange(size_of_filepaths):
    output = np.empty(len(wavelength_range))    
    for j in np.arange(len(wavelength_range)):
        output[j] = pseudo_EW(fname=filepaths[i], w1=wavelength_range[j,0], w2=wavelength_range[j,1], dw=dw, plot=plot)
    np.savetxt('./'+directory_name+'/result_'+filepaths[i].replace('.fits','.dat').replace('spectra/HARPS/',''), output, fmt='%.2f')
    
