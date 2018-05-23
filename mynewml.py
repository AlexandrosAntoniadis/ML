#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 21:11:41 2018

@author: mac
"""

from __future__ import division
import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import explained_variance_score, r2_score
from sklearn.decomposition import PCA
from sklearn.externals import joblib

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

######### settings ##############
regression = 'linear' #linear, ridge,ridgecv, lassolars
parameter = 'Teff' #Teff, FeH
PCA_used = 'no' #yes, no
percentage_to_keep_PCA = 90
########################

df = pd.read_csv('EW&Par.csv')
df.dropna(axis=1, inplace=True)

newdf = pd.read_csv('EWnewstar.csv') ##
newdf.dropna(axis=1, inplace=True) ##


if regression == 'linear':
    reg = linear_model.LinearRegression()
elif regression == 'ridge':
    reg = linear_model.Ridge(alpha=0.5)
elif regression == 'ridgecv': 
    reg = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0],cv=5,gcv_mode=None)
elif regression == 'lassolars':
    reg = linear_model.LassoLars(alpha=1)


names = ['central_lines']
y = ['FeH','Teff']


labels = df[names]

#newlabels = newdf[names] ##

df_x = df.drop(y,axis=1)

#newdf_x = newdf.drop(y,axis=1) ##



if PCA_used=='yes':
    #I prepare the matrix for passing into PCA: I drop name as it isn't a variale, just a label
    df_x_pca = df_x.drop(names,axis=1)
    #I run the methods with use of sklearn package
    pca = PCA().fit(df_x_pca)
    #I plot the explained_variance_ratio vs the number of components
    #number of components is equal to the number of variables
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()
    
    #I obtain the explained variance ratio for each sum of components:
    #for example
    #var = [73.43, 86.75, 88.21, 89.29, 90.25, 99.99]
    #if we keep 1 component, we will have 73.43 of explained variance
    #if we keep 2 components, we will have 86.75 of explaind variance
    
    var=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
    
    #I check how many components I will keep
    #var<percentage_to_keep_PCA give True to the ones that I want to keep
    #then sum them
    
    n_components_to_keep = np.sum(var<percentage_to_keep_PCA)
    
    print(var)
    
    print("Number of components to keep for PCA : "+ str(n_components_to_keep))
    
    #I tell the algorithm on sklearn that I only want to keep "n" number of components
    
    sklearn_pca = PCA(n_components=n_components_to_keep)
    y_sklearn = sklearn_pca.fit_transform(df_x_pca)
    
    #I concatenate the variables from PCA with the labels that were dropped at the beginning
    df_x = pd.concat([labels,pd.DataFrame(y_sklearn)],axis=1)
    

df_y = df[parameter]

#newdf_y = newdf[parameter] ##


x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.30)

labels_train = x_train[names]
labels_test = x_test[names]

#newlabels_newtest = x_newtest[names] ##

x_train.drop(names,axis=1,inplace=True)
x_test.drop(names,axis=1,inplace=True)

reg.fit(x_train, y_train)

#for saving the model run the next line. For keeping current,comment it out
joblib.dump(reg, 'savedmodel.pkl')

y_pred_test = reg.predict(x_test)
y_pred_train = reg.predict(x_train)


N = len(y_test)
starttime = time()
y_pred_test = reg.predict(x_test)
elapsedtime = time()
t = elapsedtime - starttime



score_test = reg.score(x_test, y_test)

variancescore = explained_variance_score(y_test, y_pred_test[:]) 

r2score = r2_score(y_test, y_pred_test[:])

the_error = mean_absolute_error(y_test, y_pred_test[:])



reg = joblib.load('savedmodel.pkl') #for loading the saved model

newdf = reg.predict(newdf[:]) #applying the saved model to the new star

                  


label = parameter

set_res = 12

plt.figure(figsize=([set_res,set_res]))
plt.ylabel("ML value", fontsize=set_res)
plt.xlabel("AA value", fontsize=set_res)
plt.plot(y_train, y_pred_train[:], 'o')


plt.plot((np.min(y_train),np.max(y_train)),(np.min(y_train),np.max(y_train)),'--b')
#plt.plot((np.min(y_train),np.max(y_train)),(np.min(y_pred_train[:]),np.max(y_pred_train[:])),'--b')
#plt.plot((2000,4000),(2000,4000),'--k') # for Teff
#plt.plot((-1,1),(-1,1),'--b') # for FeH

plt.grid()
plt.title(label+' '+'train'+' '+'comparison')
plt.show()


set_res = 12

plt.figure(figsize=([set_res,set_res]))
plt.ylabel("ML value", fontsize=set_res)
plt.xlabel("AA value", fontsize=set_res)
plt.plot(y_test, y_pred_test[:], 'o')

plt.plot((np.min(y_test),np.max(y_test)),(np.min(y_test),np.max(y_test)),'--b')
#plt.plot((np.min(y_test),np.max(y_test)),(np.min(y_pred[:]),np.max(y_pred[:])),'--b')
#plt.plot((2000,4000),(2000,4000),'--k') # for Teff
#plt.plot((-1,1),(-1,1),'--b') # for FeH

plt.grid()
plt.title(label+' '+'test'+' '+'comparison')
plt.show()


set_res = 12

plt.figure(figsize=([set_res,set_res]))
plt.ylabel("AA-ML Difference", fontsize=set_res)
plt.xlabel("AA value", fontsize=set_res)
plt.plot(y_test, y_test.values - y_pred_test[:], 'o')
plt.grid()
plt.title(label+' '+'difference')
plt.show()


set_res = 12

plt.figure(figsize=([set_res,set_res]))
plt.ylabel("AA-ML % Difference", fontsize=set_res)
plt.xlabel("AA value", fontsize=set_res)
plt.plot(y_test, (y_test.values - y_pred_test[:])/y_test.values*100, 'o')
plt.grid()
plt.title(label+' '+'%'+' '+'difference')
plt.show()


print('Calculated parameters for {} stars in {:.2f}ms'.format(N, t*1e3))
print ('score of test:', score_test)
print ('variance score:', variancescore)
print ('r2score:', r2score)
print('Mean absolute error for {}: {:.2f}'.format(label, the_error))



print('Train :')

mse_train = mean_squared_error(y_train,y_pred_train) 
mae_train = mean_absolute_error(y_train,y_pred_train) 
mape_train = mean_absolute_percentage_error(y_train,y_pred_train) 

print('MSE train : ' + str(mse_train))
print('MAE train : ' + str(mae_train))
print('MAPE train : '+ str(mape_train))

print('Test :')

mse_test = mean_squared_error(y_test,y_pred_test)
mae_test = mean_absolute_error(y_test,y_pred_test)
mape_test = mean_absolute_percentage_error(y_test,y_pred_test)

print('MSE test : ' + str(mse_test))
print('MAE test : ' + str(mae_test))
print('MAPE test : '+ str(mape_test))


print('New Star :')

print ('The new star has'+' '+ parameter+' '+'=', newdf) # printing the new



labels_train['prediction_' + parameter] = y_pred_train
labels_test['prediction_' + parameter] = y_pred_test



result_train = pd.concat([labels_train,y_train],axis=1)
result_test = pd.concat([labels_test,y_test],axis=1)

if PCA_used=='yes':
    result_train.to_csv('results_train_'+regression+'_'+ parameter + '_PCA_'+str(percentage_to_keep_PCA)+'.csv',index=False,encoding='utf-8')
    result_test.to_csv('results_test_'+regression+'_'+ parameter + '_PCA_'+str(percentage_to_keep_PCA)+'.csv',index=False,encoding='utf-8')
else:
    result_train.to_csv('results_train_'+regression+'_'+ parameter + '.csv',index=False,encoding='utf-8')
    result_test.to_csv('results_test_'+regression+'_'+ parameter + '.csv',index=False,encoding='utf-8')






