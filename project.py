# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 11:59:54 2020

@author: jon39
"""
import pandas as pd
import numpy as np

items = pd.read_csv('C:/Users/jon39/OneDrive/NTNU/3. år/ai/project/items.csv')
cols = items.columns
print("items.csv head")
print(items.head())
print("items.csv columns: ", items.columns)
items = items[cols[0]]
print("Number of items: ", len(items))