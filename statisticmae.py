#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 12:37:11 2018

@author: mac
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

mae = np.loadtxt('maestat.txt', dtype = str)

### T
linearT = mae[1:,0]
linearT = [np.float(item) for item in linearT]

liT = np.mean(linearT)
print ('average linearT MAE is:', liT)

ridgeT = mae[1:,1]
ridgeT = [np.float(item) for item in ridgeT]

rT = np.mean(ridgeT)
print ('average ridgeT MAE is :', rT)

lassoT = mae[1:,2]
lassoT = [np.float(item) for item in lassoT]

laT = np.mean(lassoT)
print ('average lassoT MAE is :',laT)

cvridgeT = mae[1:,6]
cvridgeT = [np.float(item) for item in cvridgeT]

cvrT = np.mean(cvridgeT)
print ('average cvridgeT MAE is :', cvrT)

### FeH

linearFeH = mae[1:,3]
linearFeH = [np.float(item) for item in linearFeH]

liFeH = np.mean(linearFeH)
print ('average linearFeH MAE is:', liFeH)

ridgeFeH = mae[1:,4]
ridgeFeH = [np.float(item) for item in ridgeFeH]

rFeH = np.mean(ridgeFeH)
print ('average ridgeFeH MAE is :', rFeH)

lassoFeH = mae[1:,5]
lassoFeH = [np.float(item) for item in lassoFeH]

laFeH = np.mean(lassoFeH)
print ('average lassoFeH MAE is :',laFeH)

cvridgeFeH = mae[1:,7]
cvridgeFeH = [np.float(item) for item in cvridgeFeH]

cvrFeH = np.mean(cvridgeFeH)
print ('average cvridgeFeH MAE is :', cvrFeH)
