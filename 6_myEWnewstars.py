#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 15:13:44 2018

@author: aantoniadis
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def Load_Data_Pairs(data1):
    
    # Get the data and the names from our results
    our_datanames = []
    our_data = []
    for file in os.listdir(data1):
        # get all .dat files
        if file.endswith(".dat"):
            temp_name = os.path.join(data1, file)
            # remove unwanded elements and append the names
            our_datanames.append(temp_name.split('/')[1].strip( '.dat' ))
            # append data
            our_data.append(np.loadtxt(temp_name))
    
        
    return our_data, our_datanames



def main():
    
    # Define folder names for my data
    myData = "FEROSEWmyresults"
    
    # Load data and return the pairs
    our_data, our_datanames = Load_Data_Pairs(myData)
    
    
    lines = np.loadtxt('central_lines.dat')
    table = np.empty((len(lines), len(our_datanames)+1))
    table[:,0] = lines
    
    
    headers = "newstars"
    for i in range(len(our_datanames)):
        headers = headers + "," + our_datanames[i]
    
    
    
    for i in range(len(our_datanames)):
        table[:, 1+i] = our_data[i]
        print(len(our_data[i]))
        
    np.savetxt("FEROSEW.dat", table, header = headers, delimiter=",")   
    
     # transpose
     
    table = np.loadtxt('FEROSEW.dat', dtype=str, delimiter=',', comments="?")

    table_T = np.transpose(table)
    table_T[0,0] = table_T[0,0].replace('# ','')
    print(table_T[0])
    np.savetxt('FEROSnewstars.csv', table_T, delimiter=',', fmt='%s')

    
    print (table_T[:,0]) 
    
        
    
if __name__ == '__main__':
    main()   