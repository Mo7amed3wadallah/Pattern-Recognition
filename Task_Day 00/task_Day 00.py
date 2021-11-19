# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 14:15:20 2021

@author: Client
"""
#%%
# 2. Given the list 𝑑𝑎𝑡𝑎 = [15,18,13,20,20,11,14,99,100,200,201, 0], use 
# NumPy to calculate the mean, median, variance, and standard deviation.


import matplotlib.pyplot as plt

import numpy as np

data = [15,18,13,20,20,11,14,99,100,200,201, 0]

print("\n")

mean = np.mean(data)
print(mean,"\n")

median = np.median(data)
print(median,"\n")

variance = np.var(data)
print(variance,"\n")

std = np.std(data)
print(std,"\n")
#%%
# 4. Using NumPy and matplotlib, plot a figure of a Gaussian distribution of 
# 𝑚𝑒𝑎𝑛 = 6.0, 𝑠𝑡𝑎𝑛𝑑𝑎𝑟𝑑 𝑑𝑒𝑣𝑖𝑎𝑡𝑖𝑜𝑛 = 1.0 with 𝑠𝑖𝑧𝑒 = 100000.

mean = 6.0
standard_deviation = 1.0
size = 100000
Gaussian = np.random.normal(loc = mean, scale = standard_deviation, size = size)
print(Gaussian)
plt.plot(Gaussian)

