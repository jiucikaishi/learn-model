#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 13:13:14 2019

@author: lx
"""


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import metric,cross_validation
import xgboost as xgb
import lightgbm as lgb
from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np

import sys

def cac_deep(data_p):
    n_df=data_p.shape[0]
    data_p['deep']=1
    p_list=[0,0.001,0.002,0.003,0.004,0.005,0.01,0.05,0.1, 0.15, 0.2,0.25, 0.3, 0.35, 0.4, 0.45,0.5,0.55,\
            0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.96,0.97,0.98,0.99,1]
    for i in range(n_df):
        for j in range(len(p_list)):
            if i>p_list[j]*n_df and i<=p_list[j+1]*n_df:
                data_p.loc[i,'deep']=p_list[j+1]*100
    return data_p
    
    

def cac_p_r(data_y_pre):
    static_all=[]
    n_df=np.shape(data_y_pre)[0]
    range_list=[0.001,0.002,0.003,0.004,0.005,0.01,0.05,0.1, 0.15, 0.2,0.25, 0.3, 0.35, 0.4, 0.45,0.5,0.55,\
                0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.96,0.97,0.98,0.99,1]
    for i in range_list:
        static=[0.0,0,0,0,0.0000,0.0000000000000,0.0000]
        static[0]=i
        a=1.0
        for j in range(len(data_y_pre)):
            if j<=n_df*i:
                if data_y_pre[j][1]<a:a=data_y_pre[j][1]
                static[1]+=1
                if data_y_pre[j][0]==1:
                    static[2]+=1
                else:
                    static[3]+=1
        static[4]=round(static[2]/static[1],4)
        static[5]=round(static[2]/sum(x[0] for x in data_y_pre),4)
        static[6]=a
        static_all.append(static)
    static_all=pd.DataFrame(static_all,columns=['thred','num','num_1','num_0','hit','recall','p_min'])
    return static_all
    

def con_list(data_y,data_pre):
    conlist=[]
    for i in range(len(data_y)):
        conlist.append([data_y[i],data_pre[i]])
    conlist=sorted(conlist,key=lambda x:x[1],reverse=True)
    return conlist


path=sys.argv[1]+'/'

df_valid_1=pd.read_csv(path+'valid_19_csjtest_1015.csv',encoding='gbk')
df_valid=pd.read_csv(path+'valid19_trans_csjtest_1015',encoding='gbk')
df_valid=pd.concat([df_valid_1,df_valid],axis=1)
df_valid_new=df_valid

df_valid_new.fillna(-100000000000,inplace=True)

target='is_y2'
predictors=['ins_bmi','app_day','num_sex_flag']
print(predictors)

csj_Lgbmodel_new=lgb.Booster(model_file=path+'csj_1106_new.txt')
csj_Lgbmodel_old=lgb.Booster(model_file=path+'csj_1106_old.txt')


prediction_lgb_new=csj_Lgbmodel_new.predict(df_valid_new[predictors])
prediction_lgb_old=csj_Lgbmodel_old.predict(df_valid_new[predictors])
weight=0.04
dvalid_predictions=prediction_lgb_old*weight+prediction_lgb_new(1-weight)

print("\n origion Model Report")


valid_y=df_valid_new['is_y2'].values
valid_y_pre=con_list(valid_y,dvalid_predictions)
static_all_test=cac_p_r(valid_y_pre)

print('valid orgion\n')
print(static_all_test)


