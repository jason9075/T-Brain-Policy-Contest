#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 14:13:19 2018

@author: jason9075
"""
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def get_corr(df):
  new_corrs = []

  # Iterate through the columns 
  for col in df.columns:
      # Calculate correlation with the target
      corr = df['TARGET'].corr(df[col])
      
      # Append the list as a tuple
  
      new_corrs.append((col, corr))
  new_corrs = sorted(new_corrs, key = lambda x: abs(x[1]), reverse = True)
  return new_corrs


def missing_values_table(df):
  # Total missing values
  mis_val = df.isnull().sum()
  
  # Percentage of missing values
  mis_val_percent = 100 * df.isnull().sum() / len(df)
  
  # Make a table with the results
  mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
  
  # Rename the columns
  mis_val_table_ren_columns = mis_val_table.rename(
  columns = {0 : 'Missing Values', 1 : '% of Total Values'})
  
  # Sort the table by percentage of missing descending
  mis_val_table_ren_columns = mis_val_table_ren_columns[
      mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
  '% of Total Values', ascending=False).round(1)
  
  # Print some summary information
  print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
      "There are " + str(mis_val_table_ren_columns.shape[0]) +
        " columns that have missing values.")
  
  # Return the dataframe with missing information
  return mis_val_table_ren_columns

def add_score(df, column, low_bound=0, low_bound_weight=0, high_bound=1, high_bound_weight=0, other=0):   
  value = df[column]
  value = value/(value.max() - value.min())
  value = value.apply(lambda x: low_bound_weight*x if x < low_bound else high_bound_weight*x if high_bound < x else other if (np.isnan(x)==False) else 0) 
  return value

def kde_target(var_name, df):
    
    # Calculate the correlation coefficient between the new variable and the target
    corr = df['TARGET'].corr(df[var_name])
    
    # Calculate medians for repaid vs not repaid
    avg_repaid = df.ix[df['TARGET'] == 0, var_name].median()
    avg_not_repaid = df.ix[df['TARGET'] == 1, var_name].median()
    
    plt.figure(figsize = (12, 6))
    
    # Plot the distribution for target == 0 and target == 1
    sns.kdeplot(df.ix[df['TARGET'] == 0, var_name], label = 'TARGET == 0')
    sns.kdeplot(df.ix[df['TARGET'] == 1, var_name], label = 'TARGET == 1')
    
    # label the plot
    plt.xlabel(var_name); plt.ylabel('Density'); plt.title('%s Distribution' % var_name)
    plt.legend();
    
    # print out the correlation
    print('The correlation between %s and the TARGET is %0.4f' % (var_name, corr))
    # Print out average values
    print('Median value for loan that was not repaid = %0.4f' % avg_not_repaid)
    print('Median value for loan that was repaid =     %0.4f' % avg_repaid)
    
def one_hot_encoder(df, nan_as_category = True, exclude_list=[]):
    original_columns = list(df.columns)
    target_columns = df.columns.difference(exclude_list)
    categorical_columns = [col for col in target_columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns
  

def useless_columns():
  return ['CLAIM_Coverage_14N_MEAN',
          'CLAIM_Coverage_15O_MEAN',
          'CLAIM_Coverage_26H_MEAN',
          'CLAIM_Coverage_18I_MEAN',
          'CLAIM_Coverage_36I_MEAN',
          'CLAIM_Coverage_56K_MEAN',
          'CLAIM_Coverage_09@_MEAN',
          'CLAIM_Coverage_18@_MEAN',
          'CLAIM_Coverage_35H_MEAN',
          'CLAIM_Coverage_14E_MEAN',
          'CLAIM_Coverage_70G_MEAN',
          'CLAIM_Coverage_57C_MEAN',
          'CLAIM_Coverage_42F_MEAN',
          'CLAIM_Coverage_06F_MEAN',
          'CLAIM_Coverage_56B_MEAN',
          'CLAIM_Coverage_12L_MEAN',
          'CLAIM_Coverage_10A_MEAN',
          'CLAIM_Coverage_41N_MEAN',
          'CLAIM_Coverage_29K_MEAN',
          'CLAIM_Coverage_08H_MEAN',
          'CLAIM_Coverage_46A_MEAN',
          'CLAIM_Coverage_34P_MEAN',
          'CLAIM_Coverage_29B_MEAN',
          'CLAIM_Coverage_57L_MEAN',
          'CLAIM_Coverage_32N_MEAN',
          'CLAIM_Coverage_01A_MEAN',
          'CLAIM_Coverage_25G_MEAN',
          'CLAIM_Coverage_70P_MEAN',
          'CLAIM_Coverage_40M_MEAN',
          'CLAIM_Coverage_66L_MEAN',
          'CLAIM_Coverage_05N_MEAN',
          'CLAIM_Coverage_33F_MEAN',
          'CLAIM_Coverage_03L_MEAN',
          'CLAIM_Coverage_15F_MEAN',
          'CLAIM_Coverage_33O_MEAN',
          'CLAIM_Marital_Status_of_Driver_stat_1_MEAN'
          'CLAIM_Driver\'s_Relationship_with_Insured_rls_4_MEAN',
          'CLAIM_Driver\'s_Relationship_with_Insured_rls_7_MEAN',
          'CLAIM_Cause_of_Loss_cause_15_MEAN',
          'CLAIM_Cause_of_Loss_cause_14_MEAN',
          'CLAIM_Cause_of_Loss_cause_16_MEAN',
          'CLAIM_Cause_of_Loss_cause_11_MEAN',
          'CLAIM_Cause_of_Loss_cause_6_MEAN',
          'CLAIM_Cause_of_Loss_cause_12_MEAN',
          'CLAIM_Cause_of_Loss_cause_10_MEAN',
          'CLAIM_Cause_of_Loss_cause_3_MEAN',
          'CLAIM_Cause_of_Loss_cause_8_MEAN',
          'CLAIM_Cause_of_Loss_cause_13_MEAN',
          'CLAIM_Cause_of_Loss_cause_7_MEAN',
          'CLAIM_Cause_of_Loss_cause_9_MEAN',
          'CLAIM_Deductible_SUM',
          'CLAIM_Deductible_MEAN',
          'POLICY_fassured_cat_3_MEAN',
          'POLICY_fassured_cat_2_MEAN',
          'POLICY_iply_area_area_3_MEAN',
          'POLICY_iply_area_area_11_MEAN',
          'POLICY_iply_area_area_13_MEAN',
          'POLICY_iply_area_area_6_MEAN',
          'POLICY_iply_area_area_20_MEAN',
          'POLICY_iply_area_area_18_MEAN',
          'POLICY_iply_area_area_17_MEAN',
          'POLICY_iply_area_area_19_MEAN',
          'POLICY_iply_area_area_21_MEAN',
          'POLICY_iply_area_area_7_MEAN',
          'POLICY_iply_area_area_1_MEAN',
          'POLICY_iply_area_area_10_MEAN',
          'POLICY_Imported_or_Domestic_Car_index_22_MEAN',
          'POLICY_Imported_or_Domestic_Car_index_23_MEAN',
          'POLICY_Imported_or_Domestic_Car_index_21_MEAN',
          'POLICY_Imported_or_Domestic_Car_index_24_MEAN']
  
def useless_diff_columns():
  return ['POLICY_Imported_or_Domestic_Car_index_23_MEAN',
          'POLICY_Imported_or_Domestic_Car_index_21_MEAN',
          'POLICY_Imported_or_Domestic_Car_index_22_MEAN',
          'POLICY_iply_area_area_21_MEAN',
          'POLICY_iply_area_area_20_MEAN',
          'POLICY_iply_area_area_19_MEAN',
          'POLICY_iply_area_area_18_MEAN',
          'POLICY_fassured_cat_2_MEAN',
          'POLICY_fassured_cat_3_MEAN',
          'POLICY_CURR_New_Insured_Amount_max_MAX',
          'POLICY_Insurance_Coverage_68E_MEAN',
          'POLICY_Insurance_Coverage_01J_MEAN',
          'POLICY_Insurance_Coverage_66L_MEAN',
          'POLICY_Insurance_Coverage_34P_MEAN',
          'POLICY_Insurance_Coverage_33F_MEAN',
          'POLICY_Insurance_Coverage_72@_MEAN',
          'POLICY_Insurance_Coverage_68N_MEAN',
          'POLICY_Insurance_Coverage_57C_MEAN',
          'POLICY_Insurance_Coverage_36I_MEAN',
          'POLICY_Insurance_Coverage_37J_MEAN',
          'POLICY_Insurance_Coverage_71H_MEAN',
          'POLICY_Insurance_Coverage_35H_MEAN',
          'POLICY_Insurance_Coverage_66C_MEAN',
          'POLICY_Insurance_Coverage_41E_MEAN',
          'POLICY_Insurance_Coverage_67D_MEAN',
          'POLICY_Insurance_Coverage_09I_MEAN',
          'POLICY_Insurance_Coverage_70P_MEAN',
          'POLICY_Insurance_Coverage_70G_MEAN',
          'CLAIM_Coverage_05N_MEAN',
          'CLAIM_Coverage_70P_MEAN',
          'CLAIM_Coverage_41N_MEAN',
          'CLAIM_Coverage_10A_MEAN',
          'CLAIM_Coverage_15O_MEAN',
          'CLAIM_Coverage_32N_MEAN',
          'CLAIM_Coverage_14N_MEAN',
          'CLAIM_Coverage_57C_MEAN',
          'CLAIM_Coverage_18@_MEAN',
          'CLAIM_Coverage_57L_MEAN',
          'CLAIM_Coverage_35H_MEAN',
          'CLAIM_Coverage_18I_MEAN',
          'CLAIM_Coverage_26H_MEAN',
          'CLAIM_Coverage_25G_MEAN',
          'CLAIM_Coverage_14E_MEAN',
          'CLAIM_Coverage_29B_MEAN',
          'CLAIM_Coverage_56K_MEAN',
          'CLAIM_Coverage_66L_MEAN',
          'CLAIM_Coverage_33O_MEAN',
          'CLAIM_Coverage_29K_MEAN',
          'CLAIM_Coverage_33F_MEAN',
          'CLAIM_Coverage_46A_MEAN',
          'CLAIM_Coverage_08H_MEAN',
          'CLAIM_Coverage_36I_MEAN',
          'CLAIM_Coverage_09@_MEAN',
          'CLAIM_Coverage_70G_MEAN',
          'CLAIM_Coverage_42F_MEAN',
          'CLAIM_Coverage_12L_MEAN',
          'CLAIM_Coverage_56B_MEAN',
          'CLAIM_Coverage_34P_MEAN',
          'CLAIM_Coverage_01A_MEAN',
          'CLAIM_Coverage_06F_MEAN',
          'CLAIM_Cause_of_Loss_cause_9_MEAN',
          'CLAIM_Cause_of_Loss_cause_14_MEAN',
          'CLAIM_Cause_of_Loss_cause_3_MEAN',
          'CLAIM_Cause_of_Loss_cause_11_MEAN',
          'CLAIM_Cause_of_Loss_cause_6_MEAN',
          'CLAIM_Cause_of_Loss_cause_10_MEAN',
          'CLAIM_Cause_of_Loss_cause_13_MEAN',
          'CLAIM_Cause_of_Loss_cause_12_MEAN',
          'CLAIM_Cause_of_Loss_cause_15_MEAN',
          'CLAIM_Cause_of_Loss_cause_8_MEAN',
          'CLAIM_Cause_of_Loss_cause_16_MEAN',
          'CLAIM_Cause_of_Loss_cause_7_MEAN',
          'CLAIM_Driver\'s_Relationship_with_Insured_rls_4_MEAN']
  
def useless_pay_flag_columns():
  return ['POLICY_Insurance_Coverage_47B_MEAN',
          'POLICY_Insurance_Coverage_68E_MEAN',
          'POLICY_Insurance_Coverage_33O_MEAN',
          'POLICY_Insurance_Coverage_01J_MEAN',
          'POLICY_Insurance_Coverage_56B_MEAN',
          'POLICY_Insurance_Coverage_70P_MEAN',
          'POLICY_Insurance_Coverage_66L_MEAN',
          'POLICY_Insurance_Coverage_56K_MEAN',
          'POLICY_Insurance_Coverage_70G_MEAN',
          'POLICY_Insurance_Coverage_34P_MEAN',
          'POLICY_Insurance_Coverage_33F_MEAN',
          'POLICY_Insurance_Coverage_32N_MEAN',
          'POLICY_Insurance_Coverage_27I_MEAN',
          'POLICY_Insurance_Coverage_72@_MEAN',
          'POLICY_Insurance_Coverage_68N_MEAN',
          'POLICY_Insurance_Coverage_57C_MEAN',
          'POLICY_Insurance_Coverage_65K_MEAN',
          'POLICY_Insurance_Coverage_36I_MEAN',
          'POLICY_Insurance_Coverage_37J_MEAN',
          'POLICY_Insurance_Coverage_71H_MEAN',
          'POLICY_Insurance_Coverage_35H_MEAN',
          'POLICY_Insurance_Coverage_66C_MEAN',
          'POLICY_Insurance_Coverage_41E_MEAN',
          'POLICY_Insurance_Coverage_67D_MEAN',
          'POLICY_Insurance_Coverage_18I_MEAN',
          'POLICY_Insurance_Coverage_25G_MEAN',
          'POLICY_Insurance_Coverage_09I_MEAN',
          'POLICY_CURR_Insured_Amount3_MAX',
          'POLICY_CURR_Insured_Amount3_SUM',
          'POLICY_CURR_New_Insured_Amount_max_MAX',
          'POLICY_Imported_or_Domestic_Car_index_21_MEAN',
          'POLICY_Imported_or_Domestic_Car_index_22_MEAN',
          'POLICY_Imported_or_Domestic_Car_index_23_MEAN',
          'POLICY_iply_area_area_20_MEAN',
          'POLICY_iply_area_area_18_MEAN',
          'POLICY_iply_area_area_16_MEAN',
          'POLICY_iply_area_area_19_MEAN',
          'POLICY_iply_area_area_21_MEAN',
          'POLICY_iply_area_area_17_MEAN',
          'POLICY_fassured_cat_2_MEAN',
          'POLICY_fassured_cat_3_MEAN',
          'CLAIM_Coverage_70P_MEAN',
          'CLAIM_Coverage_41N_MEAN',
          'CLAIM_Coverage_03L_MEAN',
          'CLAIM_Coverage_10A_MEAN',
          'CLAIM_Coverage_15O_MEAN',
          'CLAIM_Coverage_15F_MEAN',
          'CLAIM_Coverage_00I_MEAN',
          'CLAIM_Coverage_32N_MEAN',
          'CLAIM_Coverage_14N_MEAN',
          'CLAIM_Coverage_57C_MEAN',
          'CLAIM_Coverage_18@_MEAN',
          'CLAIM_Coverage_42F_MEAN',
          'CLAIM_Coverage_05N_MEAN',
          'CLAIM_Coverage_12L_MEAN',
          'CLAIM_Coverage_56B_MEAN',
          'CLAIM_Coverage_41E_MEAN',
          'CLAIM_Coverage_34P_MEAN',
          'CLAIM_Coverage_45@_MEAN',
          'CLAIM_Coverage_01A_MEAN',
          'CLAIM_Coverage_40M_MEAN',
          'CLAIM_Coverage_06F_MEAN',
          'CLAIM_Coverage_57L_MEAN',
          'CLAIM_Coverage_35H_MEAN',
          'CLAIM_Coverage_18I_MEAN',
          'CLAIM_Coverage_26H_MEAN',
          'CLAIM_Coverage_25G_MEAN',
          'CLAIM_Coverage_14E_MEAN',
          'CLAIM_Coverage_07P_MEAN',
          'CLAIM_Coverage_29B_MEAN',
          'CLAIM_Coverage_56K_MEAN',
          'CLAIM_Coverage_66L_MEAN',
          'CLAIM_Coverage_33O_MEAN',
          'CLAIM_Coverage_29K_MEAN',
          'CLAIM_Coverage_33F_MEAN',
          'CLAIM_Coverage_46A_MEAN',
          'CLAIM_Coverage_05E_MEAN',
          'CLAIM_Coverage_08H_MEAN',
          'CLAIM_Coverage_36I_MEAN',
          'CLAIM_Coverage_09@_MEAN',
          'CLAIM_Coverage_70G_MEAN',
          'CLAIM_Nature_of_the_claim_MAX',
          'CLAIM_Deductible_MAX',
          'CLAIM_Deductible_MEAN',
          'CLAIM_Deductible_SUM',
          'CLAIM_Salvage_or_Subrogation?_MEAN',
          'CLAIM_Cause_of_Loss_cause_7_MEAN',
          'CLAIM_Cause_of_Loss_cause_12_MEAN',
          'CLAIM_Cause_of_Loss_cause_15_MEAN',
          'CLAIM_Cause_of_Loss_cause_8_MEAN',
          'CLAIM_Cause_of_Loss_cause_16_MEAN',
          'CLAIM_Cause_of_Loss_cause_13_MEAN',
          'CLAIM_Cause_of_Loss_cause_10_MEAN',
          'CLAIM_Cause_of_Loss_cause_3_MEAN',
          'CLAIM_Cause_of_Loss_cause_14_MEAN',
          'CLAIM_Cause_of_Loss_cause_9_MEAN',
          'CLAIM_Cause_of_Loss_cause_11_MEAN',
          'CLAIM_Cause_of_Loss_cause_6_MEAN',
          'CLAIM_Driver\'s_Relationship_with_Insured_rls_4_MEAN'
          'CLAIM_Driver\'s_Relationship_with_Insured_rls_7_MEAN']
  
def useless_pay_more_columns():
  return ['CLAIM_Cause_of_Loss_cause_10_MEAN',
       'CLAIM_Cause_of_Loss_cause_11_MEAN',
       'CLAIM_Cause_of_Loss_cause_12_MEAN',
       'CLAIM_Cause_of_Loss_cause_13_MEAN',
       'CLAIM_Cause_of_Loss_cause_14_MEAN',
       'CLAIM_Cause_of_Loss_cause_15_MEAN',
       'CLAIM_Cause_of_Loss_cause_16_MEAN', 'CLAIM_Cause_of_Loss_cause_3_MEAN',
       'CLAIM_Cause_of_Loss_cause_6_MEAN', 'CLAIM_Cause_of_Loss_cause_7_MEAN',
       'CLAIM_Cause_of_Loss_cause_8_MEAN', 'CLAIM_Cause_of_Loss_cause_9_MEAN',
       'CLAIM_Coverage_00I_MEAN', 'CLAIM_Coverage_01A_MEAN',
       'CLAIM_Coverage_03L_MEAN', 'CLAIM_Coverage_05E_MEAN',
       'CLAIM_Coverage_05N_MEAN', 'CLAIM_Coverage_06F_MEAN',
       'CLAIM_Coverage_07P_MEAN', 'CLAIM_Coverage_08H_MEAN',
       'CLAIM_Coverage_09@_MEAN', 'CLAIM_Coverage_10A_MEAN',
       'CLAIM_Coverage_12L_MEAN', 'CLAIM_Coverage_14E_MEAN',
       'CLAIM_Coverage_14N_MEAN', 'CLAIM_Coverage_15F_MEAN',
       'CLAIM_Coverage_15O_MEAN', 'CLAIM_Coverage_18@_MEAN',
       'CLAIM_Coverage_18I_MEAN', 'CLAIM_Coverage_25G_MEAN',
       'CLAIM_Coverage_26H_MEAN', 'CLAIM_Coverage_29B_MEAN',
       'CLAIM_Coverage_29K_MEAN', 'CLAIM_Coverage_32N_MEAN',
       'CLAIM_Coverage_33F_MEAN', 'CLAIM_Coverage_33O_MEAN',
       'CLAIM_Coverage_34P_MEAN', 'CLAIM_Coverage_35H_MEAN',
       'CLAIM_Coverage_36I_MEAN', 'CLAIM_Coverage_40M_MEAN',
       'CLAIM_Coverage_41E_MEAN', 'CLAIM_Coverage_41N_MEAN',
       'CLAIM_Coverage_42F_MEAN', 'CLAIM_Coverage_45@_MEAN',
       'CLAIM_Coverage_46A_MEAN', 'CLAIM_Coverage_56B_MEAN',
       'CLAIM_Coverage_56K_MEAN', 'CLAIM_Coverage_57C_MEAN',
       'CLAIM_Coverage_57L_MEAN', 'CLAIM_Coverage_66L_MEAN',
       'CLAIM_Coverage_70G_MEAN', 'CLAIM_Coverage_70P_MEAN',
       'CLAIM_Deductible_MAX', 'CLAIM_Deductible_MEAN', 'CLAIM_Deductible_SUM',
       'CLAIM_Driver\'s_Relationship_with_Insured_rls_4_MEAN',
       'CLAIM_Nature_of_the_claim_MAX',
       'POLICY_CURR_New_Insured_Amount_max_MAX',
       'POLICY_Imported_or_Domestic_Car_index_21_MEAN',
       'POLICY_Imported_or_Domestic_Car_index_22_MEAN',
       'POLICY_Imported_or_Domestic_Car_index_23_MEAN',
       'POLICY_Insurance_Coverage_01J_MEAN',
       'POLICY_Insurance_Coverage_09I_MEAN',
       'POLICY_Insurance_Coverage_18I_MEAN',
       'POLICY_Insurance_Coverage_25G_MEAN',
       'POLICY_Insurance_Coverage_27I_MEAN',
       'POLICY_Insurance_Coverage_32N_MEAN',
       'POLICY_Insurance_Coverage_33F_MEAN',
       'POLICY_Insurance_Coverage_33O_MEAN',
       'POLICY_Insurance_Coverage_34P_MEAN',
       'POLICY_Insurance_Coverage_35H_MEAN',
       'POLICY_Insurance_Coverage_36I_MEAN',
       'POLICY_Insurance_Coverage_37J_MEAN',
       'POLICY_Insurance_Coverage_41E_MEAN',
       'POLICY_Insurance_Coverage_45@_MEAN',
       'POLICY_Insurance_Coverage_47B_MEAN',
       'POLICY_Insurance_Coverage_56B_MEAN',
       'POLICY_Insurance_Coverage_56K_MEAN',
       'POLICY_Insurance_Coverage_57C_MEAN',
       'POLICY_Insurance_Coverage_65K_MEAN',
       'POLICY_Insurance_Coverage_66C_MEAN',
       'POLICY_Insurance_Coverage_66L_MEAN',
       'POLICY_Insurance_Coverage_67D_MEAN',
       'POLICY_Insurance_Coverage_68E_MEAN',
       'POLICY_Insurance_Coverage_68N_MEAN',
       'POLICY_Insurance_Coverage_70G_MEAN',
       'POLICY_Insurance_Coverage_70P_MEAN',
       'POLICY_Insurance_Coverage_71H_MEAN',
       'POLICY_Insurance_Coverage_72@_MEAN', 'POLICY_fassured_cat_2_MEAN',
       'POLICY_fassured_cat_3_MEAN', 'POLICY_iply_area_area_11_MEAN',
       'POLICY_iply_area_area_13_MEAN', 'POLICY_iply_area_area_16_MEAN',
       'POLICY_iply_area_area_17_MEAN', 'POLICY_iply_area_area_18_MEAN',
       'POLICY_iply_area_area_19_MEAN', 'POLICY_iply_area_area_20_MEAN',
       'POLICY_iply_area_area_21_MEAN']
  
def useless_pay_less_columns():
  return ['CLAIM_Cause_of_Loss_cause_10_MEAN',
       'CLAIM_Cause_of_Loss_cause_11_MEAN',
       'CLAIM_Cause_of_Loss_cause_12_MEAN',
       'CLAIM_Cause_of_Loss_cause_13_MEAN',
       'CLAIM_Cause_of_Loss_cause_14_MEAN',
       'CLAIM_Cause_of_Loss_cause_15_MEAN',
       'CLAIM_Cause_of_Loss_cause_16_MEAN', 'CLAIM_Cause_of_Loss_cause_2_MEAN',
       'CLAIM_Cause_of_Loss_cause_3_MEAN', 'CLAIM_Cause_of_Loss_cause_6_MEAN',
       'CLAIM_Cause_of_Loss_cause_7_MEAN', 'CLAIM_Cause_of_Loss_cause_8_MEAN',
       'CLAIM_Cause_of_Loss_cause_9_MEAN', 'CLAIM_Coverage_00I_MEAN',
       'CLAIM_Coverage_01A_MEAN', 'CLAIM_Coverage_03L_MEAN',
       'CLAIM_Coverage_05E_MEAN', 'CLAIM_Coverage_05N_MEAN',
       'CLAIM_Coverage_06F_MEAN', 'CLAIM_Coverage_07P_MEAN',
       'CLAIM_Coverage_08H_MEAN', 'CLAIM_Coverage_09@_MEAN',
       'CLAIM_Coverage_10A_MEAN', 'CLAIM_Coverage_12L_MEAN',
       'CLAIM_Coverage_14E_MEAN', 'CLAIM_Coverage_14N_MEAN',
       'CLAIM_Coverage_15F_MEAN', 'CLAIM_Coverage_15O_MEAN',
       'CLAIM_Coverage_18@_MEAN', 'CLAIM_Coverage_18I_MEAN',
       'CLAIM_Coverage_25G_MEAN', 'CLAIM_Coverage_26H_MEAN',
       'CLAIM_Coverage_29B_MEAN', 'CLAIM_Coverage_29K_MEAN',
       'CLAIM_Coverage_32N_MEAN', 'CLAIM_Coverage_33F_MEAN',
       'CLAIM_Coverage_33O_MEAN', 'CLAIM_Coverage_34P_MEAN',
       'CLAIM_Coverage_35H_MEAN', 'CLAIM_Coverage_36I_MEAN',
       'CLAIM_Coverage_40M_MEAN', 'CLAIM_Coverage_41E_MEAN',
       'CLAIM_Coverage_41N_MEAN', 'CLAIM_Coverage_42F_MEAN',
       'CLAIM_Coverage_45@_MEAN', 'CLAIM_Coverage_46A_MEAN',
       'CLAIM_Coverage_56B_MEAN', 'CLAIM_Coverage_56K_MEAN',
       'CLAIM_Coverage_57C_MEAN', 'CLAIM_Coverage_57L_MEAN',
       'CLAIM_Coverage_66L_MEAN', 'CLAIM_Coverage_70G_MEAN',
       'CLAIM_Coverage_70P_MEAN', 'CLAIM_Deductible_MAX',
       'CLAIM_Deductible_MEAN', 'CLAIM_Deductible_SUM',
       'CLAIM_Driver\'s_Relationship_with_Insured_rls_4_MEAN',
       'CLAIM_Driver\'s_Relationship_with_Insured_rls_7_MEAN',
       'CLAIM_Nature_of_the_claim_MAX',
       'POLICY_CURR_New_Insured_Amount_max_MAX',
       'POLICY_CURR_New_Insured_Amount_max_SUM',
       'POLICY_Imported_or_Domestic_Car_index_21_MEAN',
       'POLICY_Imported_or_Domestic_Car_index_22_MEAN',
       'POLICY_Imported_or_Domestic_Car_index_23_MEAN',
       'POLICY_Insurance_Coverage_01J_MEAN',
       'POLICY_Insurance_Coverage_09I_MEAN',
       'POLICY_Insurance_Coverage_14N_MEAN',
       'POLICY_Insurance_Coverage_18I_MEAN',
       'POLICY_Insurance_Coverage_25G_MEAN',
       'POLICY_Insurance_Coverage_27I_MEAN',
       'POLICY_Insurance_Coverage_32N_MEAN',
       'POLICY_Insurance_Coverage_33F_MEAN',
       'POLICY_Insurance_Coverage_33O_MEAN',
       'POLICY_Insurance_Coverage_34P_MEAN',
       'POLICY_Insurance_Coverage_35H_MEAN',
       'POLICY_Insurance_Coverage_36I_MEAN',
       'POLICY_Insurance_Coverage_37J_MEAN',
       'POLICY_Insurance_Coverage_41E_MEAN',
       'POLICY_Insurance_Coverage_45@_MEAN',
       'POLICY_Insurance_Coverage_46A_MEAN',
       'POLICY_Insurance_Coverage_47B_MEAN',
       'POLICY_Insurance_Coverage_56B_MEAN',
       'POLICY_Insurance_Coverage_56K_MEAN',
       'POLICY_Insurance_Coverage_57C_MEAN',
       'POLICY_Insurance_Coverage_65K_MEAN',
       'POLICY_Insurance_Coverage_66C_MEAN',
       'POLICY_Insurance_Coverage_66L_MEAN',
       'POLICY_Insurance_Coverage_67D_MEAN',
       'POLICY_Insurance_Coverage_68E_MEAN',
       'POLICY_Insurance_Coverage_68N_MEAN',
       'POLICY_Insurance_Coverage_70G_MEAN',
       'POLICY_Insurance_Coverage_70P_MEAN',
       'POLICY_Insurance_Coverage_71H_MEAN',
       'POLICY_Insurance_Coverage_72@_MEAN', 'POLICY_fassured_cat_2_MEAN',
       'POLICY_fassured_cat_3_MEAN', 'POLICY_iply_area_area_16_MEAN',
       'POLICY_iply_area_area_17_MEAN', 'POLICY_iply_area_area_18_MEAN',
       'POLICY_iply_area_area_19_MEAN', 'POLICY_iply_area_area_20_MEAN',
       'POLICY_iply_area_area_21_MEAN']
  
def useless_pay_not_much_diff_columns():
  return ['CLAIM_Cause_of_Loss_cause_10_MEAN',
       'CLAIM_Cause_of_Loss_cause_11_MEAN',
       'CLAIM_Cause_of_Loss_cause_12_MEAN',
       'CLAIM_Cause_of_Loss_cause_13_MEAN',
       'CLAIM_Cause_of_Loss_cause_14_MEAN',
       'CLAIM_Cause_of_Loss_cause_15_MEAN',
       'CLAIM_Cause_of_Loss_cause_16_MEAN', 'CLAIM_Cause_of_Loss_cause_3_MEAN',
       'CLAIM_Cause_of_Loss_cause_4_MEAN', 'CLAIM_Cause_of_Loss_cause_6_MEAN',
       'CLAIM_Cause_of_Loss_cause_7_MEAN', 'CLAIM_Cause_of_Loss_cause_8_MEAN',
       'CLAIM_Cause_of_Loss_cause_9_MEAN', 'CLAIM_Coverage_00I_MEAN',
       'CLAIM_Coverage_01A_MEAN', 'CLAIM_Coverage_02K_MEAN',
       'CLAIM_Coverage_03L_MEAN', 'CLAIM_Coverage_05E_MEAN',
       'CLAIM_Coverage_05N_MEAN', 'CLAIM_Coverage_06F_MEAN',
       'CLAIM_Coverage_07P_MEAN', 'CLAIM_Coverage_08H_MEAN',
       'CLAIM_Coverage_09@_MEAN', 'CLAIM_Coverage_10A_MEAN',
       'CLAIM_Coverage_12L_MEAN', 'CLAIM_Coverage_14E_MEAN',
       'CLAIM_Coverage_14N_MEAN', 'CLAIM_Coverage_15F_MEAN',
       'CLAIM_Coverage_15O_MEAN', 'CLAIM_Coverage_18@_MEAN',
       'CLAIM_Coverage_18I_MEAN', 'CLAIM_Coverage_25G_MEAN',
       'CLAIM_Coverage_29B_MEAN', 'CLAIM_Coverage_29K_MEAN',
       'CLAIM_Coverage_32N_MEAN', 'CLAIM_Coverage_33F_MEAN',
       'CLAIM_Coverage_33O_MEAN', 'CLAIM_Coverage_34P_MEAN',
       'CLAIM_Coverage_35H_MEAN', 'CLAIM_Coverage_36I_MEAN',
       'CLAIM_Coverage_40M_MEAN', 'CLAIM_Coverage_41E_MEAN',
       'CLAIM_Coverage_41N_MEAN', 'CLAIM_Coverage_42F_MEAN',
       'CLAIM_Coverage_45@_MEAN', 'CLAIM_Coverage_46A_MEAN',
       'CLAIM_Coverage_56B_MEAN', 'CLAIM_Coverage_56K_MEAN',
       'CLAIM_Coverage_57C_MEAN', 'CLAIM_Coverage_57L_MEAN',
       'CLAIM_Coverage_66L_MEAN', 'CLAIM_Coverage_70G_MEAN',
       'CLAIM_Coverage_70P_MEAN', 'CLAIM_Deductible_MAX',
       'CLAIM_Deductible_MEAN', 'CLAIM_Deductible_SUM',
       'CLAIM_Driver\'s_Relationship_with_Insured_rls_4_MEAN',
       'CLAIM_Driver\'s_Relationship_with_Insured_rls_7_MEAN',
       'CLAIM_Nature_of_the_claim_MAX',
       'POLICY_CURR_New_Insured_Amount_max_MAX',
       'POLICY_Imported_or_Domestic_Car_index_21_MEAN',
       'POLICY_Imported_or_Domestic_Car_index_22_MEAN',
       'POLICY_Imported_or_Domestic_Car_index_23_MEAN',
       'POLICY_Insurance_Coverage_01J_MEAN',
       'POLICY_Insurance_Coverage_09I_MEAN',
       'POLICY_Insurance_Coverage_14N_MEAN',
       'POLICY_Insurance_Coverage_18I_MEAN',
       'POLICY_Insurance_Coverage_25G_MEAN',
       'POLICY_Insurance_Coverage_27I_MEAN',
       'POLICY_Insurance_Coverage_32N_MEAN',
       'POLICY_Insurance_Coverage_33F_MEAN',
       'POLICY_Insurance_Coverage_33O_MEAN',
       'POLICY_Insurance_Coverage_34P_MEAN',
       'POLICY_Insurance_Coverage_35H_MEAN',
       'POLICY_Insurance_Coverage_36I_MEAN',
       'POLICY_Insurance_Coverage_37J_MEAN',
       'POLICY_Insurance_Coverage_41E_MEAN',
       'POLICY_Insurance_Coverage_45@_MEAN',
       'POLICY_Insurance_Coverage_47B_MEAN',
       'POLICY_Insurance_Coverage_56B_MEAN',
       'POLICY_Insurance_Coverage_56K_MEAN',
       'POLICY_Insurance_Coverage_57C_MEAN',
       'POLICY_Insurance_Coverage_65K_MEAN',
       'POLICY_Insurance_Coverage_66C_MEAN',
       'POLICY_Insurance_Coverage_66L_MEAN',
       'POLICY_Insurance_Coverage_67D_MEAN',
       'POLICY_Insurance_Coverage_68E_MEAN',
       'POLICY_Insurance_Coverage_68N_MEAN',
       'POLICY_Insurance_Coverage_70G_MEAN',
       'POLICY_Insurance_Coverage_70P_MEAN',
       'POLICY_Insurance_Coverage_71H_MEAN',
       'POLICY_Insurance_Coverage_72@_MEAN', 'POLICY_fassured_cat_2_MEAN',
       'POLICY_fassured_cat_3_MEAN', 'POLICY_iply_area_area_16_MEAN',
       'POLICY_iply_area_area_17_MEAN', 'POLICY_iply_area_area_18_MEAN',
       'POLICY_iply_area_area_19_MEAN', 'POLICY_iply_area_area_20_MEAN',
       'POLICY_iply_area_area_21_MEAN']
  