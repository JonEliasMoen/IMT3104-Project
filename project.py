# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 11:59:54 2020

@author: jon39
"""

import pandas as pd
import numpy as np

ds = pd.read_csv('C:/Users/jon39/OneDrive/NTNU/3. år/ai/project/items.csv')
cols = ds.columns
print(ds.head())
print(ds.columns)
items = ds[cols[0]]
print(len(items))
print(items)
for i in items:
    print(i)