#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 16:10:19 2019

@author: lx
"""

import pandas as pd
import lightgbm as lgb
import os
import json
import operator
import datetime
import xgboost as xgb 
import sys


if __name__=='__main__':
    

    path='/home/data/c'
    df_train=pd.read_csv(path+'jianmoAll_trans_'+sys.argv[2]+'.csv',encoding='gbk')
    target='is_y2'
    df_train.fillna(-1000000000,inpace=True)
    predictors=df_train.cloumns.tolist()
    predictors.remove('appno')
    predictors.remove('is_y2')
    predictors.remove('Unnamed: 0')
    params={'nthread': 8,
            'booster':'gbtree',
            'silent':True,
            'eval_metric':['logloss','aucpr','auc'],
            'tree_method':'auto',
            'max_depth': 6,
            'n_estimators': 237,
            'eta': 0.02,
            'subsample': 0.8,
            'min_child_weight': 1,
            'min_split_loss': 4,
            'grow_policy':'depthwise',
            'max_leaves':32,
            'max_bin':255,
            'num_round':1100,
            'early_stopping_rounds':40,
            'feval':None,
            'maximize':True,
            'objective': 'binary:logistic',
            'gamma': 0, 
            'colsample_bytree': 0.6, 
            'colsample_bylevel': 1, 
            'alpha': 0,
            'lambda': 10,
            'scale_pos_weight': 1, 
            'seed': 0}
        
    dtrain=xgb.DMatrix(df_train[predictors],label=df_train[target].values)
    model=xgb.train(params=params,dtrain=dtrain,num_boost_round=1100)
    imp_type='gain'
    importance=model.get_score(fmap='',importance_type=imp_type)
    importance=sorted(importance.item(),key=operator.itemgetter(1),reversed=True)
    df=pd.DataFrame(importance,columns=['feature','fscore'])
    df.set_index('feature',inplace=True)
    df['fscore']=df['fscore']/df['fscore'].sum()
    df.to_csv(path+'/imp_feature_gain.csv')
        
        
            
            
            
            
            
            
            
            
            
            
            