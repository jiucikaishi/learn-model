#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 20:06:33 2019

@author: lx
"""

import hyperopt

import xgboost as xgb
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
df_test_new=pd.read_csv(path+'test_trans_'+sys.argv[2]+'.csv')

df_valid_1=pd.read_csv(path+'valid_'+sys.argv[2]+'.csv',encoding='gbk')
df_valid=pd.read_csv(path+'valid_trans'+sys.argv[2]+'.csv',encoding='gbk')
df_valid=pd.concat([df_valid_1,df_valid],axis=1)
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
space = {"max_depth": hp.randint("max_depth", 15),
         "n_estimators": hp.randint("n_estimators", 300),
         'learning_rate': hp.uniform('learning_rate', 1e-3, 5e-1),
         "subsample": hp.randint("subsample", 5),
         "min_child_weight": hp.randint("min_child_weight", 6),
         }

def argsDict_tranform(argsDict, isPrint=True):
    argsDict["max_depth"] = argsDict["max_depth"] + 5
    argsDict['n_estimators'] = argsDict['n_estimators'] + 150
    argsDict["learning_rate"] = argsDict["learning_rate"] * 0.02 + 0.05
    argsDict["subsample"] = argsDict["subsample"] * 0.1 + 0.5
    argsDict["min_child_weight"] = argsDict["min_child_weight"] + 1
    if isPrint:
        print(argsDict)
    else:
        pass

    return argsDict

    
def xgboost_factory(argsDict):
    argsDict = argsDict_tranform(argsDict)
    
    params = {'nthread': 8,  # 进程数
              'max_depth': argsDict['max_depth'],  # 最大深度
              'n_estimators': argsDict['n_estimators'],  # 树的数量
              'eta': argsDict['learning_rate'],  # 学习率
              'subsample': argsDict['subsample'],  # 采样数
              'min_child_weight': argsDict['min_child_weight'],  # 终点节点最小样本占比的和
              'objective': 'binary:logistic',
              'silent': 1,  # 是否显示
              'gamma': 0,  # 是否后剪枝
              'colsample_bytree': 0.7,  # 样本列采样
              'alpha': 0,  # L1 正则化
              'lambda': 0,  # L2 正则化
              #'scale_pos_weight': 0,  # 取值>0时,在数据不平衡时有助于收敛
              'seed': 100,  # 随机种子
              #'missing': -999,  # 填充缺失值
              }
    #params['eval_metric'] = ['rmse']
    dtrain = xgb.DMatrix(df_train_new[predictos],label=df_train_new[target].values)
    dpredict = xgb.DMatrix(df_valid_new[predictos],label=df_valid_new[target].values)

    xrf = xgb.train(params, dtrain, params['n_estimators'])

    return get_tranformer_score(xrf,params,dpredict)

def get_tranformer_score(tranformer,params,dpredict):
    
    model = tranformer
    prediction=model.predict(dpredict,ntree_limit=model.best_iteration)
    labels=df_valid_new['is_y2'].values
    valid_y_pre=con_list(labels,prediction)
    static_all_valid=cac_p_r(valid_y_pre)
    print(params,static_all_valid)
  
    return -static_all_valid.loc[0,'hit']

    
    
# 开始使用hyperopt进行自动调参,同时通过返回值获取最佳模型的结果
algo = partial(tpe.suggest, n_startup_jobs=8)
best = fmin(xgboost_factory, space, algo=algo, max_evals=20, pass_expr_memo_ctrl=None)
print(best)  
    
  
#展示结果
params={'n_thread':8,
        'seed':100,
        'objective': 'binary:logistic',
        'eval_metric':{'auc'}
        }  
best['max_depth']=best['max_depth']+5
best['n_estimators']=best['n_estimators']+150
params.update(best)
print(params)


#==============================================================================
# RMSE = xgboost_factory(best)
# print('best :', best)
# print('best param after transform :')
# argsDict_tranform(best,isPrint=True)
# print('rmse of the best xgboost:', np.sqrt(RMSE))
#==============================================================================

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    