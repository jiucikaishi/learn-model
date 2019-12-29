#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 12:23:15 2019

@author: lx
"""

import pandas as pd
import lightgbm as lgb
import sys

path='/home/data/c'
csjLgbModel=lgb.Booster(model_flie='/home/data/csj_model.txt')
df_train_1=pd.read_csv(path+'jianmo'+sys.argv[2]+'.csv',encoding='gbk')
df_train=pd.read_csv(path+'jianmoAll_trans_'+sys.argv[2]+'.csv',encoding='gbk')
df_train=pd.concat([df_train_1,df_train],axis=1)
df_train_new=df_train
predictors=['ins_bmi','app_day','num_sex_flag']
data=df_train_new[predictors]
labels=df_train_new['is_y2'].values
calc_permutation_importance(csjLgbModel,data,labels,1,3,path+'/shai_feature.csv')

def calc_permutation_importance(model,data,labels,baseline,folds=3,file=None):
    from sklearn.utils import shuffle
    from sklearn.metrics import precision_recall_curve
    import lightgbm as lgb
    import xgboost as xgb
    from itertools import product
    import numpy as np
    print('permutation-importance'.center(50,'-'))
    print('folds={},baseline={}'.format(folds,baseline))
    
    
    def clac_pr(y_true,probs):
        precisions,recall,thrs=precision_recall_curve(y_true,probs)
        mean_precisions=0.5*(precisions[:-1]+precisions[1:])
        intervals=recall[:-1]-recall[1:]
        auc_pr=np.dot(mean_precisions,intervals)
        
        return auc_pr
        
    feats=data.columns.tolist()
    ret={}
    if isinstance(model,lgb.Booster):
        for col in feats:
            print('features= {}'.format(col).center(30,'-'))
            mdata=data.copy()
            
            for _ in range(folds):
                mdata[col]=shuffle(data[col]).tolist()
                y_prob=model.predict(mdata,num_iteration=model.best_iteration)
                aucpr=clac_pr(labels,y_prob)
                ret.setdefault(col,[]).append(aucpr)
    
    elif isinstance(model,xgb.Booster):
        for col in feats:
            mdata=data.copy()
            
            for _ in range(folds):
                mdata[col]=shuffle(data[col]).tolist()
                dtest=xgb.DMatrix(mdata.values,label=None,weight=None)
                y_prob=model.predict(dtest,ntree_limit=model.best_ntree_limit)
                aucpr=clac_pr(labels,y_prob)
                ret.setdefault(col,[]).append(aucpr)    
    
    else:
        print('permutation-importane igored')
        
    
    df=pd.DataFrame(ret)
    df=baseline-df #the more falling the more important
    mean=round(df.mean(),4)
    std=round(df.std(),4)
    view=mean.astype(str)+'(+-'+std.astype(str)+')'
    df=pd.concat([mean,view],axis=1)
    
    df.sort_values(by=0,ascending=False,inplace=True)
    df.drop(columns=[0],inplace=True)
    
    if file:
        df.to_csv(file,index=True)
    
    else:
        print(df)

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            