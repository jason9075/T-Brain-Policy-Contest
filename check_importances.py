#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 19:45:00 2018

@author: jason9075
"""
import pandas as pd


features = pd.read_csv('feature_importance_df.csv')  

features = features[features['importance']==0]

row_count = features.groupby('feature').size()

row_count = row_count[row_count[:]==5]

row_count.index[:50]