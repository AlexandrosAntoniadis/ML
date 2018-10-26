#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 17:22:52 2017

@author: aantoniadis
"""

from build_pew_verygeneral import pseudo_EW
import numpy as np
import os

filepaths = np.loadtxt('FEROS1Dfilelist.dat', dtype=str)

wavelength_range = np.loadtxt('lines.rdb', skiprows=2)
dw = 0.4
plot = False

directory_name = 'FEROSEWmyresults'

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
    np.savetxt('./'+directory_name+'/result_'+filepaths[i].replace('.fits','.dat').replace('spectra/FEROS1D/',''), output, fmt='%.2f')
    

