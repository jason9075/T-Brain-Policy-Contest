#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 16:15:07 2018

@author: jason9075
"""

import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import warnings
import util
warnings.simplefilter(action='ignore', category=FutureWarning)

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# Preprocess application_train.csv and application_test.csv
def application_train_test():
  df = pd.read_pickle('application_train_test_cached')
  return df
  # Read data and merge
  
# Preprocess bureau.csv and bureau_balance.csv
def claim_read(epison=1):
  cl_agg = pd.read_pickle('claim_cached')
  return cl_agg

def policy_read(num_rows = None, nan_as_category = True, epison=1):
  po_agg = pd.read_pickle('policy_cached')
  return po_agg

def kfold_cat(df, num_folds, output_name):
    # Divide in training/validation and test data
    train_df =df[df['Next_Premium'].notnull()]
    test_df = df[df['Next_Premium'].isnull()]
    print("Starting LGB. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()
    # Cross validation model
    folds = KFold(n_splits= num_folds, shuffle=True, random_state=1001)
    
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feats = [f for f in train_df.columns if f not in ['Policy_Number','Next_Premium','index']]
    
    valid_scores = []
    train_scores = []

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['Next_Premium'])):
      
      # Training data for the fold
      train_features, train_labels = np.array(train_df[feats].iloc[train_idx]), train_df['Next_Premium'].iloc[train_idx]
      # Validation data for the fold
      valid_features, valid_labels = np.array(train_df[feats].iloc[valid_idx]), train_df['Next_Premium'].iloc[valid_idx]
      

      clf = CatBoostRegressor(iterations=800,
                            learning_rate=0.1,
                            depth=6,
                            loss_function='MAE',
                            eval_metric='MAE',
                            random_seed=0,
                            od_type='Iter',
                            od_wait=50)
      
      clf.fit(
          train_features,
          train_labels,
          eval_set=[(train_features, train_labels), (valid_features,valid_labels)],
          use_best_model=True,
          verbose=False
      )

      oof_preds[valid_idx] = clf.predict(valid_features)
      sub_preds += clf.predict(np.array(test_df[feats])) / folds.n_splits
      
      valid_score = mean_absolute_error(valid_labels, clf.predict(valid_features))
      train_score = mean_absolute_error(train_labels, clf.predict(train_features))
      
      valid_scores.append(valid_score)
      train_scores.append(train_score)
      print('Fold %2d MAE : %.6f' % (n_fold + 1, mean_absolute_error(valid_labels, oof_preds[valid_idx])))
      del clf, train_features, train_labels, valid_features, valid_labels
      gc.collect()
      
      
    print('Full MAE score %.6f' % mean_absolute_error(train_df['Next_Premium'], oof_preds))

    print("val score: ", valid_scores, "\n train score: ", train_scores)
    
    sub_df = test_df[['Policy_Number']].copy()
    sub_df['Next_Premium'] = sub_preds
    sub_df['Next_Premium'] = sub_df['Next_Premium'].apply(lambda x: 0 if x < 100 else x)
    sub_df.rename(columns={'Next_Premium': output_name}, inplace=True)
    
    val_df = train_df[['Policy_Number']].copy()
    val_df['Next_Premium'] = oof_preds
    val_df['Next_Premium'] = val_df['Next_Premium'].apply(lambda x: 0 if x < 100 else x)
    val_df.rename(columns={'Next_Premium': output_name}, inplace=True)

    return sub_df, train_df['Next_Premium'], val_df

def main(file_name='default.csv'):
    df = application_train_test()
    df_raw = pd.read_csv('training-set.csv')  
    with timer("Process Policy"):
      po = policy_read()
      print("Policy shape:", po.shape)
      df = df.join(po, how='left', on='Policy_Number')
      del po
      gc.collect()
    with timer("Process Claim"):
      cl = claim_read()
      print("Claim df shape:", cl.shape)
      df = df.join(cl, how='left', on='Policy_Number')
      del cl
      gc.collect()
    with timer("Run train with full data 1"):
      print(df.shape)
      df.drop(columns=util.useless_columns(), inplace=True, errors='ignore')
      print(df.shape)
      sub_df_1, val_label_1, val_train_1 = kfold_cat(df, num_folds= 5, output_name='Next_Premium_1')
      df_raw = df_raw.merge(val_train_1, how='left',  on='Policy_Number')
      gc.collect()
    with timer("Run train with full data 2"):
      df_to_remove = val_train_1[val_train_1['Next_Premium_1']==0]['Policy_Number']
      mask = df['Policy_Number'].isin(df_to_remove.tolist())
      df = df[~mask]
      print(df.shape)
      sub_df_2, _, val_train_2 = kfold_cat(df, num_folds= 5, output_name='Next_Premium_2')
      df_raw = df_raw.merge(val_train_2, how='left',  on='Policy_Number')
      del val_train_1
      gc.collect()
    with timer("Run train with full data 3"):
      df_to_remove = val_train_2[val_train_2['Next_Premium_2']==0]['Policy_Number']
      mask = df['Policy_Number'].isin(df_to_remove.tolist())
      df = df[~mask]
      print(df.shape)
      sub_df_3, _, val_train_3 = kfold_cat(df, num_folds= 5, output_name='Next_Premium_3')
      df_raw = df_raw.merge(val_train_3, how='left',  on='Policy_Number')
      del val_train_2
      gc.collect()

    with timer("blend out put"):
      sub_df = sub_df_1.merge(sub_df_2, how='left',  on='Policy_Number')
      sub_df = sub_df.merge(sub_df_3, how='left',  on='Policy_Number')


      sub_df['Next_Premium'] = 0.33*sub_df['Next_Premium_1'] + \
                        0.33*sub_df['Next_Premium_2'] + \
                        0.34*sub_df['Next_Premium_3']
      sub_df[['Policy_Number', 'Next_Premium']].to_csv('sub_file_cat_blend.csv', index= False)


if __name__ == "__main__":
    with timer("Full model run"):
      main(file_name='submission_log_score5.csv')

