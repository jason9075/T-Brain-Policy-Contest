#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 14:22:52 2018

@author: jason9075
"""


import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
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

def kfold_regession(df, clf, num_folds, output_name):
  
  train_df =df[df['Next_Premium'].notnull()]
  test_df = df[df['Next_Premium'].isnull()]
  print("Starting Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
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
        
    clf.fit(
        train_features,
        train_labels
    )

    oof_preds[valid_idx] = clf.predict(valid_features)
    sub_preds += clf.predict(np.array(test_df[feats])) / folds.n_splits
    
    valid_score = mean_absolute_error(np.expm1(valid_labels), np.expm1(oof_preds[valid_idx]))
    train_score = mean_absolute_error(np.expm1(train_labels), np.expm1(clf.predict(train_features)))
    
    valid_scores.append(valid_score)
    train_scores.append(train_score)
    
    print('Fold %2d MAE : %.6f' % (n_fold + 1, mean_absolute_error(np.expm1(valid_labels), np.expm1(oof_preds[valid_idx]))))
    del train_features, train_labels, valid_features, valid_labels
    gc.collect()
    
    
  print('Full MAE score %.6f' % mean_absolute_error(np.expm1(train_df['Next_Premium']), np.expm1(oof_preds)))
  print("val score: ", valid_scores, "\n train score: ", train_scores)
  
  sub_df = test_df[['Policy_Number']].copy()
  sub_df['Next_Premium'] = np.expm1(sub_preds)
  sub_df.rename(columns={'Next_Premium': output_name}, inplace=True)
  
  val_df = train_df[['Policy_Number']].copy()
  val_df['Next_Premium'] = np.expm1(oof_preds)
  val_df.rename(columns={'Next_Premium': output_name}, inplace=True)
  
  return sub_df, np.expm1(train_df['Next_Premium']), val_df


def main():
    df = application_train_test()
    df_raw = pd.read_csv('training-set.csv')  
    with timer("Process Policy"):
      po = policy_read()
      print("Policy shape:", po.shape)
      df = df.join(po, how='left', on='Policy_Number')
      del po
      gc.collect()
# =============================================================================
#     with timer("Process Claim"):
#       cl = claim_read()
#       print("Claim df shape:", cl.shape)
#       df = df.join(cl, how='left', on='Policy_Number')
#       del cl
#       gc.collect()
# =============================================================================
    with timer("Fill Missing values"):
      df["POLICY_Policy_days_from_ibirth_MEAN"] = df["POLICY_Policy_days_from_ibirth_MEAN"].fillna(0)
      df["POLICY_Policy_days_from_dbirth_MEAN"] = df["POLICY_Policy_days_from_dbirth_MEAN"].fillna(0)
      df["POLICY_EX_04M_Premium_SUM"] = df["POLICY_EX_04M_Premium_SUM"].fillna(0)
      df["POLICY_EX_04M_Premium_MAX"] = df["POLICY_EX_04M_Premium_MAX"].fillna(0)
      df["POLICY_EX_16G_Premium_MAX"] = df["POLICY_EX_16G_Premium_MAX"].fillna(0)
      df["POLICY_EX_16G_Premium_SUM"] = df["POLICY_EX_16G_Premium_SUM"].fillna(0)
      df["POLICY_EX_51O_Premium_MAX"] = df["POLICY_EX_51O_Premium_MAX"].fillna(0)
      df["POLICY_EX_51O_Premium_SUM"] = df["POLICY_EX_51O_Premium_SUM"].fillna(0)
      df["POLICY_EX_55J_Premium_SUM"] = df["POLICY_EX_55J_Premium_SUM"].fillna(0)
      df["POLICY_EX_55J_Premium_MAX"] = df["POLICY_EX_55J_Premium_MAX"].fillna(0)
      df["POLICY_EX_05N_Premium_MAX"] = df["POLICY_EX_05N_Premium_MAX"].fillna(0)
      df["POLICY_EX_05N_Premium_SUM"] = df["POLICY_EX_05N_Premium_SUM"].fillna(0)
      df["POLICY_EX_02K_Premium_MAX"] = df["POLICY_EX_02K_Premium_MAX"].fillna(0)
      df["POLICY_EX_02K_Premium_SUM"] = df["POLICY_EX_02K_Premium_SUM"].fillna(0)
      df["POLICY_EX_16P_Premium_MAX"] = df["POLICY_EX_16P_Premium_MAX"].fillna(0)
      df["POLICY_EX_16P_Premium_SUM"] = df["POLICY_EX_16P_Premium_SUM"].fillna(0)
      df["POLICY_EX_45@_Premium_SUM"] = df["POLICY_EX_45@_Premium_SUM"].fillna(0)
      df["POLICY_EX_45@_Premium_MAX"] = df["POLICY_EX_45@_Premium_MAX"].fillna(0)
    with timer("Drop unrelated columns"):
      print(df.shape)
      df.drop(columns=util.useless_columns(), inplace=True, errors='ignore')
      print(df.shape)
      
    with timer("Run train with Lasso "):
      lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
      sub_df_lasso, _, val_train_lasso = kfold_regession(df, lasso, num_folds= 5, output_name='Next_Premium_Lasso')
      df_raw = df_raw.merge(val_train_lasso, how='left',  on='Policy_Number')
      gc.collect()
    with timer("Run train with ENet "):
      ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
      sub_df_enet, _, val_train_enet = kfold_regession(df, ENet, num_folds= 5, output_name='Next_Premium_Lasso')
      df_raw = df_raw.merge(val_train_enet, how='left',  on='Policy_Number')
      gc.collect()
    with timer("Run train with KRR "):
      KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
      sub_df_krr, _, val_train_krr = kfold_regession(df, KRR, num_folds= 5, output_name='Next_Premium_Lasso')
      df_raw = df_raw.merge(val_train_krr, how='left',  on='Policy_Number')
      gc.collect()
    with timer("Run train with GBoost "):
      GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
      sub_df_gboost, _, val_train_gboost = kfold_regession(df, GBoost, num_folds= 5, output_name='Next_Premium_Lasso')
      df_raw = df_raw.merge(val_train_gboost, how='left',  on='Policy_Number')
      gc.collect()
  
  
