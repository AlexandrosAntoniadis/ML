#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 13:26:30 2017

@author: aantoniadis
"""

import numpy as np

wavelength_range = np.loadtxt('lines.rdb', skiprows=2)

wcentral = np.empty(len(wavelength_range))

for i in np.arange(len(wavelength_range)):
    winit = wavelength_range[i,0]
    wfin = wavelength_range[i,1]
    wcentral[i] = (winit + wfin)/2
    
np.savetxt('central_lines.dat', wcentral)