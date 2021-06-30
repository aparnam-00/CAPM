#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 10:59:54 2021

@author: aparnamarathe
"""

# load modules
import numpy as np
import pandas as pd

# risk-free Treasury rate
R_f = 0.0175 / 252

# read in the market data
data = pd.read_csv('capm_market_data.csv')

data
data.head(10)

Drop the date column

df_no_date = data.drop('date', axis = 1)
df_no_date

percent_change = df_no_date.pct_change(axis=0)
percent_change = percent_change.dropna()
percent_change.head(5)

percent_change.head(5)
spy_returns = percent_change.loc[:, 'spy_adj_close']
spy_returns = spy_returns.to_numpy()
aapl_returns = percent_change.loc[:, 'aapl_adj_close']
aapl_returns = aapl_returns.to_numpy()

print("spy:", spy_returns[:5])
print("aapl:",aapl_returns[:5])

excess_return_aapl = aapl_returns - R_f
excess_return_spy = spy_returns - R_f

print('aapl', excess_return_aapl[-5:])
print('spy', excess_return_spy[-5:])

import matplotlib.pyplot as plt
plt.scatter(excess_return_spy, excess_return_aapl)
plt.xlabel('excess_return_spy')
plt.ylabel('excess_return_aapl')

x_transpose = np.asmatrix(excess_return_spy.T)
x_transpose_x = np.asmatrix(np.matmul(x_transpose,excess_return_spy))
inverse_matrix = np.linalg.inv(x_transpose_x)
xx_inv_x = np.matmul(inverse_matrix, x_transpose)
beta = xx_inv_x@excess_return_aapl
beta = float(beta)
print(beta)

def get_beta(x,y):
    '''
    helper function
    input: 2 ndarrays
    output: returns the beta value
    '''
    x_transpose = np.asmatrix(x.T)
    x_transpose_x = np.asmatrix(np.matmul(x_transpose,x))
    inverse_matrix = np.linalg.inv(x_transpose_x)
    xx_inv_x = np.matmul(inverse_matrix, x_transpose)
    beta = float(xx_inv_x@y)
    return beta

    

def beta_sensitivity(x, y):
    beta_sensitivities= []
    for i in range(len((x))):
        new_x = np.delete(x, i).reshape(-1,1)
        new_y = np.delete(y, i).reshape(-1,1)
        beta = get_beta(new_x, new_y)
        beta_sensitivities.append((i, beta))
    return beta_sensitivities
        

betas = beta_sensitivity(excess_return_spy, excess_return_aapl)
betas[:5]

sensitivities = [b[1] for b in betas]
plt.hist(sensitivities)