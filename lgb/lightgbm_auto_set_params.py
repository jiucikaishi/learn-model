#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 16:38:28 2019

@author: lx
"""

import hyperopt

import lightgbm as lgb
import pandas as pd
import numpy as np
from xgboost.sklearn import XGBClassifier
from sklearn import metric
from scripy.stats import ks_2samp
import os

from hyperopt import fmin, tpe, hp, partial
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, zero_one_loss
import sys
import threading
from sklearn.feature_selection import SelectFromModel
from sklearn.externals import joblib



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
        static=[0.0,0,0,0,0.0000]
        static[0]=i
        for j in range(len(data_y_pre)):
            if j<=n_df*i:
                static[1]+=1
                if data_y_pre[j][0]==1:
                    static[2]+=1
                else:
                    static[3]+=1
        static[4]=round(static[2]/static[1],4)
        static_all.append(static)
    static_all=pd.DataFrame(static_all,columns=['thred','num','num_1','num_0','hit'])
    return static_all
    

def con_list(data_y,data_pre):
    conlist=[]
    for i in range(len(data_y)):
        conlist.append([data_y[i],data_pre[i]])
    conlist=sorted(conlist,key=lambda x:x[1],reverse=True)
    return conlist

    

    
def eval_error1(preds,valid_data):
    labels=valid_data.get_label()
    valid_pre_y=con_list(labels,preds)
    static_all_test=cac_p_r(valid_pre_y)
    return 'hit_rate',-static_all_test.loc[0,'hit'],False





#读取数据
path=sys.argv[1]+'/'
df_train_new=pd.read_csv(path+'train_trans_'+sys.argv[2]+'.csv')
df_train_add=df_train_new[df_train_new['is_y2']==0].sample(n=123456,random_state=13,replace=True)#抽样
df_train_new=pd.concat([df_train_new,df_train_add],axis=0)#按行拼接

df_test_new=pd.read_csv(path+'test_trans_'+sys.argv[2]+'.csv')

df_valid_1=pd.read_csv(path+'valid_'+sys.argv[2]+'.csv',encoding='gbk')
df_valid=pd.read_csv(path+'valid_trans'+sys.argv[2]+'.csv',encoding='gbk')
df_valid=pd.concat([df_valid_1,df_valid],axis=1)
#df_valid=df_valid[df_valid['app_date']>='2019-05-01']
df_valid=df_valid[df_valid['app_date'].apply(lambda x:x[0:7])!='2019-05'] #验证集只取相应的月份

df_valid_new=df_valid
        
#筛选变量
target='is_y2'
#predictos=df_train_new.columns.tolist()
#predictos.remove('is_y2')
#predictos.remove('appno')
#predictos.remove('Unnamed:0')
predictos=['ins_bmi','app_day','num_sex_flag']

#筛选变量后
print(predictos)    



# 自定义hyperopt的参数空间
space = {#"max_depth": hp.randint("max_depth", 15),
         "num_boost_round": hp.randint("num_boost_round", 900),
         "num_trees": hp.randint("num_trees", 300),
         'learning_rate': hp.uniform('learning_rate', 0.01, 0.5),
         "bagging_fraction": hp.uniform("bagging_fraction", 0.8,1),
         "num_leaves": hp.randint("num_leaves", 60),
         "bin_construct_sample_cnt": hp.randint("bin_construct_sample_cnt", 30000),
         "min_data_in_bin": hp.randint("min_data_in_bin", 200),
         "lambda_11": hp.uniform("lambda_11", 0,2),
         "lambda_12": hp.uniform("lambda_12", 0,2),
         "feature_fraction": hp.uniform("feature_fraction", 0.4,1),
         "min_gain_to_split": hp.uniform("min_gain_to_split", 0,0.2),
         "seed": hp.randint("seed", 1000),
         "max_bin": hp.randint("max_bin", 150),


         }

def argsDict_tranform(argsDict, isPrint=False):
    argsDict["seed"] = argsDict["seed"] + 1
    argsDict['min_gain_to_split'] = argsDict['min_gain_to_split'] + 150
    argsDict["learning_rate"] = argsDict["learning_rate"]
    argsDict["bagging_fraction"] = argsDict["bagging_fraction"]
    argsDict["num_boost_round"] = argsDict["num_boost_round"] +10
    argsDict["num_leaves"] = argsDict["num_leaves"] +2
    argsDict["max_bin"] = argsDict["max_bin"]+4
    argsDict["bin_construct_sample_cnt"] = argsDict["bin_construct_sample_cnt"] + 10
    argsDict["min_data_in_bin"] = argsDict["min_data_in_bin"] +2
    argsDict["lambda_11"] = argsDict["lambda_11"] 
    argsDict["lambda_12"] = argsDict["lambda_12"]

    if isPrint:
        print(argsDict)
    else:
        pass

    return argsDict


def lightgbm_factory(argsDict):
    argsDict = argsDict_tranform(argsDict)

    params = {'nthread': 8,  # 进程数
              'boosting_type':'gbdt',
              'max_bin': argsDict['max_bin'],  # 最大深度
              'bin_construct_sample_cnt': argsDict['bin_construct_sample_cnt'],  # 树的数量
              'learning_rate': argsDict['learning_rate'],  # 学习率
              'bagging_fraction': argsDict['bagging_fraction'],  # bagging采样数
              'num_leaves': argsDict['num_leaves'],  # 终点节点最小样本占比的和
              'objective': 'binary',
              'feature_fraction': argsDict['feature_fraction'],  # 样本列采样
              'lambda_l1': argsDict['lambda_l1'] ,  # L1 正则化
              'lambda_l2': argsDict['lambda_l2'] ,  # L2 正则化
              'seed': argsDict['seed'] ,  # 随机种子,light中默认为100
              'min_data_in_bin': argsDict['min_data_in_bin'] ,
              'min_gain_to_split': argsDict['min_gain_to_split'] ,
              'task':'train' ,
              'min_sum_hessian_in_leaf': 0.0 ,
              'metric':{'binary_logloss'}
              }

    lgbtrain=lgb.Dataset(df_train_new[predictos].values,label=df_train_new['is_y2'].values)
    lgbtest=lgb.Dataset(df_test_new[predictos].values,label=df_test_new['is_y2'].values)
    
    #model_lgb = lgb.train(params, lgbtrain, num_boost_round=argsDict['num_boost_round'], valid_sets=[lgbtest],early_stopping_rounds=100)
    model_lgb = lgb.train(params, lgbtrain, num_boost_round=argsDict['num_boost_round'])

    return get_tranformer_score(model_lgb,params,argsDict)
    

def get_tranformer_score(tranformer,params,argsDict):

    model = tranformer
    prediction = model.predict(df_valid_new[predictos])
    labels=df_valid_new['is_y2'].values
    valid_y_pre=con_list(labels,prediction)
    static_all_valid=cac_p_r(valid_y_pre)
    print(params,argsDict['num_boost_round'],static_all_valid)
  
    return -static_all_valid.loc[0,'hit']


# 开始使用hyperopt进行自动调参,同时通过返回值获取最佳模型的结果
algo = partial(tpe.suggest, n_startup_jobs=8)
best = fmin(lightgbm_factory, space, algo=algo, max_evals=10000, pass_expr_memo_ctrl=None)
print(best)  
    
  
#展示结果
params={'n_thread':8,
        'boosting_type':'gbdt',
        'seed':100,
        'min_sum_hessian_in_leaf': 0.0,
        'task':'train',
        'metric':{'binary_logloss'}
        }  
best['num_leaves']=best['num_leaves']+2
best['num_boost_round']=best['num_boost_round']+10
params.update(best)
print(params)





























