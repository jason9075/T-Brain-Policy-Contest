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
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
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
  df = pd.read_csv('training-set.csv')  
  
  #drop outliers
  df = df.drop(df[100000 < df['Next_Premium']].index)
  df = df.drop(df[df['Policy_Number'].isin([
      '9051e66ecb70104ea8effb673b53417ee0f6d524',
      '39faa452d712749d49b50de487e63d04c31cf974',
      '9dd01ce380772aadcf8e0e1523aeea6099d1bcf6',
      'c8cf6a85e21608951c12d8c99c5f6bfc827c43a7',
      '804bab77dd9a08814aeace950235c8a2aec56a74',
      '03c607373e2ee1771149dc0ce04ed2c7432cf6f4',
      '4397f48dd38f9966e24f6ce30ab0f55fe487a24d',
      '0d92d16671768265fd137e33cf9fc229ab36f2ad',
      'a806a2f2799c2c786ec268d5871ef658cb1b5136',
      'f5800807850a21cecde16efb18ea0277804fed33',
      'e8498ca7b3e614bfd10ebfaecf358b006e93a714',
      'e7790512c2fa56ed905129787df2fab363f74a12',
      '4b1ff4d3d343431ba77ce3a4d96ac3ef01567039',
      'c4710c3209fa166553a3c2ee0be72e00b99dada5',
      '0d1d9034f860861a52b2e7776ec9b8f869d21ac8',
      '09d3ba975198c51d6b79ce49c1cf6da60d234ca0'])].index)
  
  df["Next_Premium"] = np.log1p(df["Next_Premium"])
  test_df = pd.read_csv('testing-set.csv')
  test_df['Next_Premium'] = np.nan
  print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
  df = df.append(test_df).reset_index()

# =============================================================================
#   #cached
#   df.to_pickle('application_train_test_cached')
# 
# =============================================================================
  
  return df

# Preprocess bureau.csv and bureau_balance.csv
def claim_read(epison=1):
  cl_agg = pd.read_pickle('claim_cached')
  return cl_agg

  claim = pd.read_csv('policy_claim/claim_0702.csv')
  claim['Driver\'s_Gender'], _ = pd.factorize(claim['Driver\'s_Gender'])
  claim['Driver_days_from_birth'] = claim['DOB_of_Driver'].apply(lambda x: \
    datetime.strptime('07/2018', "%m/%Y") - datetime.strptime(x, "%m/%Y")).dt.days
  claim['Driver_days_from_accident'] = claim['Accident_Date'].apply(lambda x: \
    datetime.strptime('07/2018', "%m/%Y") - datetime.strptime(x, "%Y/%m")).dt.days

  claim['Driver\'s_Relationship_with_Insured'] = "rls_" + claim['Driver\'s_Relationship_with_Insured'].astype(str)
  claim['Marital_Status_of_Driver'] = "stat_" + claim['Marital_Status_of_Driver'].astype(str)

  claim['Cause_of_Loss'], _ = pd.factorize(claim['Cause_of_Loss'])
  claim['Cause_of_Loss'] = "cause_" + claim['Cause_of_Loss'].astype(str)

  claim['Accident_area'], _ = pd.factorize(claim['Accident_area'])
  claim['Accident_area'] = "area_" + claim['Accident_area'].astype(str)

  claim.drop(columns=['DOB_of_Driver', 'Accident_Date', 
                      'Vehicle_identifier', 'Accident_area',
                      'Claim_Status_(close,_open,_reopen_etc)','Accident_Time'], inplace=True)
  
  claim, claim_cat = util.one_hot_encoder(claim, nan_as_category=False,
                          exclude_list=['Claim_Number','Policy_Number'])

  num_aggregations = {
    'Nature_of_the_claim': ['mean', 'max'],
    'Driver\'s_Gender': ['mean'],
    'Paid_Loss_Amount': ['mean', 'max', 'sum'],
    'paid_Expenses_Amount': ['max', 'mean', 'sum'],
    'Salvage_or_Subrogation?': ['max', 'mean', 'sum'],
    'At_Fault?': ['max', 'mean', 'sum'],
    'Deductible': ['max', 'mean', 'sum'],
    'number_of_claimants': ['max', 'mean', 'sum'],
    'Driver_days_from_birth': ['mean'],
    'Driver_days_from_accident': ['mean']
  }
  
  cat_aggregations = {}
  for cat in claim_cat: cat_aggregations[cat] = ['mean']
  
  claim_agg = claim.groupby('Policy_Number').agg({**num_aggregations, **cat_aggregations})
  claim_agg.columns = pd.Index(['CLAIM_' + e[0] + "_" + e[1].upper() for e in claim_agg.columns.tolist()])
  claim_agg['CLAIM_COUNT'] = claim.groupby('Policy_Number') \
                                    .size()
  claim_agg['CLAIM_NUNIQUE_COUNT'] = claim.groupby('Policy_Number')['Claim_Number'] \
                                    .nunique()
                                    
  del claim
  gc.collect()
    
  #cached
  # claim_agg.to_pickle('claim_cached')

def policy_read(num_rows = None, nan_as_category = True, epison=1):
  po_agg = pd.read_pickle('policy_cached')
  return po_agg
  policy = pd.read_csv('policy_claim/policy_0702.csv')
  
  policy['Flag_first_year_policy'] = policy['lia_class'].apply(lambda x: 1 if x==4 else 0)
  policy['New_Insured_Amount_max'] = policy[['Insured_Amount1', 'Insured_Amount2', 'Insured_Amount3']].max(axis=1)
  policy['Manafactured_Year_and_Month_diff'] = 2018 - policy['Manafactured_Year_and_Month']  
  policy['Cancellation'], _ = pd.factorize(policy['Cancellation'])
  policy['Imported_or_Domestic_Car'] = "index_" + policy['Imported_or_Domestic_Car'].astype(str)
  policy['Policy_days_from_ibirth'] = policy['ibirth'].apply(lambda x: \
    x if pd.isnull(x) else datetime.strptime('07/2018', "%m/%Y") - datetime.strptime(x, "%m/%Y")).dt.days

  res = policy['dbirth'].str.split('/', 1, expand=True)
  policy.loc[2000 < res[1].astype(float),'dbirth'] = np.nan
  policy['Policy_days_from_dbirth'] = policy['dbirth'].apply(lambda x: \
    x if pd.isnull(x) else (datetime.strptime('07/2018', "%m/%Y") - datetime.strptime(x, "%m/%Y")).days)

  policy['fsex'], _ = pd.factorize(policy['fsex'])
  policy['fsex'].replace(-1, np.nan, inplace=True)
  
  policy['fmarriage'], _ = pd.factorize(policy['fsex'])
  policy['fmarriage'].replace(-1, np.nan, inplace=True)
  
  policy['fassured'], _ = pd.factorize(policy['fassured'])
  policy['fassured'] = "cat_" + policy['fassured'].astype(str)
  
  policy['iply_area'], _ = pd.factorize(policy['iply_area'])
  policy['iply_area'] = "area_" + policy['iply_area'].astype(str)
  
  def deductible(df):
    if df['Coverage_Deductible_if_applied'] <0:
      return 0
    if df['Coverage_Deductible_if_applied'] == 1:
      return 5000
    if df['Coverage_Deductible_if_applied'] == 2:
      return 6500
    if df['Coverage_Deductible_if_applied'] == 3:
      return 8000
    if df['Coverage_Deductible_if_applied'] == 10:
      return df['Insured_Amount3'] * 0.1
    if df['Coverage_Deductible_if_applied'] == 20:
      return df['Insured_Amount3'] * 0.2
    if df['Insurance_Coverage'] in ['09I', '10A', '14E', '15F', '15O', '20B', '20K', '29K', '32N', '33F', '33O', '56K', '65K']:
      return 0
    return df['Coverage_Deductible_if_applied']
  
  policy['Deductible_calc'] = policy.apply(deductible, axis=1)

  policy.drop(columns=['Insured\'s_ID', 'Prior_Policy_Number', 'Vehicle_identifier',
                    'Vehicle_Make_and_Model1', 'Vehicle_Make_and_Model2',
                    'Coding_of_Vehicle_Branding_&_Type', 'fpt',
                    'Distribution_Channel',
                    'ibirth', 'dbirth', 'aassured_zip',
                    'fequipment1', 'fequipment2', 'fequipment3',
                    'fequipment4', 'fequipment5', 'fequipment6',
                    'fequipment9', 'nequipment9'], inplace=True)

  policy, polict_cat = util.one_hot_encoder(policy, nan_as_category=False,
                          exclude_list=['Policy_Number'])

  num_aggregations = {
    'Manafactured_Year_and_Month': ['max'], # all same
    'Manafactured_Year_and_Month_diff': ['max'], # all same
    'Engine_Displacement_(Cubic_Centimeter)': ['max'],# all same
    'Insured_Amount1': ['max', 'mean', 'sum'],
    'Insured_Amount2': ['max', 'mean', 'sum'],
    'Insured_Amount3': ['max', 'mean', 'sum'],
    'New_Insured_Amount_max':['max', 'mean', 'sum'],
    'Deductible_calc': ['max', 'mean', 'sum'],
    'qpt':['max', 'mean'],
    'Multiple_Products_with_TmNewa_(Yes_or_No?)': ['max', 'mean', 'sum'],
    'lia_class': ['max', 'mean'],
    'plia_acc': ['max', 'mean'],
    'pdmg_acc': ['max', 'mean'],
    'Coverage_Deductible_if_applied': ['max', 'mean', 'sum'],
    'Premium': ['max', 'mean', 'sum'],
    'Replacement_cost_of_insured_vehicle': ['max', 'mean', 'sum'],
    'Policy_days_from_ibirth': ['mean'],
    'Policy_days_from_dbirth': ['mean']
  }
  
  cat_aggregations = {}
  for cat in [c for c in polict_cat if 'Insurance_Coverage' in c]: \
                                        cat_aggregations[cat] = ['sum']
  for cat in [c for c in polict_cat if 'Insurance_Coverage' not in c]: \
                                          cat_aggregations[cat] = ['mean']
  
  policy_agg = policy.groupby('Policy_Number').agg({**num_aggregations, **cat_aggregations})
  policy_agg.columns = pd.Index(['POLICY_' + e[0] + "_" + e[1].upper() for e in policy_agg.columns.tolist()])
  policy_agg['POLICY_COUNT'] = policy.groupby('Policy_Number') \
                                    .size()       
                                    
  policy_curr = policy[(policy['Coverage_Deductible_if_applied']>=0)&
      (policy['Insured_Amount1']>0) |
      (policy['Insured_Amount2']>0) |
      (policy['Insured_Amount3']>0)]
  
  policy_curr['Insured_to_Premium_ratio'] = policy_curr['Premium'] / policy_curr['New_Insured_Amount_max']

  num_aggregations = {
    'Insured_Amount1': ['max', 'mean', 'sum'],
    'Insured_Amount2': ['max', 'mean', 'sum'],
    'Insured_Amount3': ['mean', 'sum'],
    'New_Insured_Amount_max':['max', 'mean', 'sum'],
    'Insured_to_Premium_ratio':['max', 'mean'],
    'Deductible_calc': ['max', 'mean', 'sum'],
    'Premium': ['max', 'mean', 'sum']
  }
  policy_curr_agg = policy_curr.groupby('Policy_Number').agg(num_aggregations)
  policy_curr_agg.columns = pd.Index(['POLICY_CURR_' + e[0] + "_" + e[1].upper() for e in policy_curr_agg.columns.tolist()])
 
  policy_agg = policy_agg.join(policy_curr_agg, how='left', on='Policy_Number')
  
  for col in policy_curr_agg.columns.tolist():
    policy_agg[col] = policy_agg[col].fillna(0)

  del policy, policy_curr_agg
  gc.collect()
    
  #cached
  # policy_agg.to_pickle('policy_cached')

def kfold_lgb(df, num_folds, output_name):
    # Divide in training/validation and test data
    train_df = df[df['Next_Premium'].notnull()]
    test_df = df[df['Next_Premium'].isnull()]
    print("Starting LGB. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()
    # Cross validation model
    folds = KFold(n_splits= num_folds, shuffle=True, random_state=1001)
    
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['Policy_Number','Next_Premium','index']]
    
    valid_scores = []
    train_scores = []

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['Next_Premium'])):
      dtrain = lgb.Dataset(data=train_df[feats].iloc[train_idx], 
                             label=train_df['Next_Premium'].iloc[train_idx], 
                             free_raw_data=False, silent=True)
      dvalid = lgb.Dataset(data=train_df[feats].iloc[valid_idx], 
                           label=train_df['Next_Premium'].iloc[valid_idx], 
                           free_raw_data=False, silent=True)
      
      # LightGBM parameters found by Bayesian optimization
      params = {
          'objective': 'mae',
          'boosting_type': 'gbdt',
          'nthread': 2,
          'learning_rate': 0.05,  # 02,
          'num_leaves': 64,
          'colsample_bytree': 0.9,
          'subsample': 0.8,
          'subsample_freq': 1,
          'max_depth': 6,
          'reg_alpha': 0.4,
          'reg_lambda': 0.7,
          'min_split_gain': 0.02,
          'min_child_weight': 60, # 39.3259775,
          'seed': 0,
          'verbose': 1,
          'metric': 'l1'
      }
      #evals_result = {} 
      
      clf = lgb.train(
          params=params,
          train_set=dtrain,
          num_boost_round=10000,
          valid_sets=[dtrain, dvalid],
          early_stopping_rounds=50,
          #evals_result=evals_result,
          verbose_eval=False
      )
      
      #print('Plot metrics recorded during training...')
      #lgb.plot_metric(evals_result, metric='l1')

      oof_preds[valid_idx] = clf.predict(dvalid.data)
      sub_preds += clf.predict(test_df[feats]) / folds.n_splits
      
      valid_score = mean_absolute_error(np.expm1(dvalid.label), np.expm1(oof_preds[valid_idx]))
      train_score = mean_absolute_error(np.expm1(dtrain.label), np.expm1(clf.predict(dtrain.data)))
      
      valid_scores.append(valid_score)
      train_scores.append(train_score)
      
      fold_importance_df = pd.DataFrame()
      fold_importance_df["feature"] = feats
      fold_importance_df["importance"] = clf.feature_importance(importance_type='gain')
      fold_importance_df["fold"] = n_fold + 1
      feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
      print('Fold %2d MAE : %.6f' % (n_fold + 1, mean_absolute_error(np.expm1(dvalid.label), np.expm1(oof_preds[valid_idx]))))
      del clf, dtrain, dvalid
      gc.collect()
      
      
    print('Full MAE score %.6f' % mean_absolute_error(np.expm1(train_df['Next_Premium']), np.expm1(oof_preds)))

    feature_importance_df.to_csv('feature_importance_df.csv', index= False)
    display_importances(feature_importance_df)
    print("val score: ", valid_scores, "\n train score: ", train_scores)
    
    sub_df = test_df[['Policy_Number']].copy()
    sub_df['Next_Premium'] = np.expm1(sub_preds)
    sub_df['Next_Premium'] = sub_df['Next_Premium'].apply(lambda x: 0 if x < 100 else x)
    sub_df.rename(columns={'Next_Premium': output_name}, inplace=True)
    
    val_df = train_df[['Policy_Number']].copy()
    val_df['Next_Premium'] = np.expm1(oof_preds)
    val_df['Next_Premium'] = val_df['Next_Premium'].apply(lambda x: 0 if x < 100 else x)
    val_df.rename(columns={'Next_Premium': output_name}, inplace=True)

    return sub_df, np.expm1(train_df['Next_Premium']), val_df
    
# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(12, 8))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('Features (avg over folds)')
    plt.tight_layout
    plt.savefig('importances01.png')
    best_features.to_csv('best_features.csv', index= False)



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
    with timer("Drop unrelated columns"):
      print(df.shape)
      df.drop(columns=util.useless_columns(), inplace=True, errors='ignore')
      print(df.shape)
      
    with timer("Run train with full data "):
      sub_df_1, _, val_train_1 = kfold_lgb(df, num_folds= 5, output_name='Next_Premium_1')
      df_raw = df_raw.merge(val_train_1, how='left',  on='Policy_Number')
      gc.collect()
      
      df_raw['Next_Premium_lgb'] = df_raw['Next_Premium_lgb'].apply(lambda x: 0 if x <0 else x)
      
      not_na = df_raw[df_raw['Next_Premium_1'].notnull()]
      
      df_raw['Next_Premium_boost'] = df_raw['Next_Premium_2'].apply(lambda x: x*1.2 if x>60000 else x)
      mean_absolute_error(df_raw['Next_Premium'], df_raw['Next_Premium_boost'])


      sub_df['Next_Premium'] = sub_df[['Next_Premium_1', 'Next_Premium_2', 'Next_Premium_3', 'Next_Premium_4']].mean(axis=1)

      sub_df[['Policy_Number', 'Next_Premium']].to_csv('sub_file_two_lgbm.csv', index= False)

      
if __name__ == "__main__":
    with timer("Full model run"):
      main(file_name='submission_log_score5.csv')
