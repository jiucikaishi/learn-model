#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 21:02:47 2019

@author: lx
"""


import xgboost as xgb
import pandas as pd
import numpy as np
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from scripy.stats import ks_2samp
import os

import sys
import threading
from sklearn.feature_selection import SelectFromModel
from sklearn.externals import joblib



def modelfit(alg, dtrain,dtest,dvalid, predictors,target,useTrainCV=False, eval_metric='auc',cv_folds=5, early_stopping_rounds=50,feature_thresh=0.01):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics=eval_metric, early_stopping_rounds=early_stopping_rounds, show_progress=False)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target],eval_metric=eval_metric)
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    dtest_predictions = alg.predict(dtest[predictors])
    dtest_predprob = alg.predict_proba(dtest[predictors])[:,1]
    dvalid_predictions = alg.predict(dvalid[predictors])
    dvalid_predprob = alg.predict_proba(dvalid[predictors])[:,1]   

     
    #Print model report:
    print ("\nModel Report")
    print( "Accuracy train: %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
    print( "Accuracy test: %.4g" % metrics.accuracy_score(dtest[target].values, dtest_predictions))
    print( "Accuracy valid: %.4g" % metrics.accuracy_score(dvalid[target].values, dvalid_predictions))

    print( "Prediction  train: %.4g" % metrics.precision_score(dtrain[target].values, dtrain_predictions))
    print( "Prediction test: %.4g" % metrics.precision_score(dtest[target].values, dtest_predictions))
    print( "Prediction valid: %.4g" % metrics.precision_score(dvalid[target].values, dvalid_predictions))

    print( "Recall  train: %.4g" % metrics.recall_score(dtrain[target].values, dtrain_predictions))
    print( "Recall test: %.4g" % metrics.recall_score(dtest[target].values, dtest_predictions))
    print( "Recall valid: %.4g" % metrics.recall_score(dvalid[target].values, dvalid_predictions))
               

    print ("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob))
    print ("AUC Score (Test): %f" % metrics.roc_auc_score(dtest[target], dtest_predprob))
    print ("AUC Score (Valid): %f" % metrics.roc_auc_score(dvalid[target], dvalid_predprob))

    #plot train ROC curve
    fpr,tpr,thresholds=metrics.roc_curve(dtrain[target].values, dtrain_predprob)
    print('KS-NEW-TRAIN')
    print(abs(fpr-tpr).max())
    roc_auc=metrics.auc(fpr,tpr)
    print(roc_auc)

    #plot test ROC curve
    fpr,tpr,thresholds=metrics.roc_curve(dtest[target].values, dtest_predprob)
    print('KS-NEW-TEST')
    print(abs(fpr-tpr).max())
    roc_auc=metrics.auc(fpr,tpr)
    print(roc_auc)
    

    #plot valid  ROC curve
    fpr,tpr,thresholds=metrics.roc_curve(dvalid[target].values, dvalid_predprob)
    print('KS-NEW-VALID')
    print(abs(fpr-tpr).max())
    roc_auc=metrics.auc(fpr,tpr)
    print(roc_auc)

    y_true=dvalid[target].values
    get_ks=lambda dvalid_predprob,y_true:ks_2samp(dvalid_predprob[y_true==1],dvalid_predprob[y_true !=1 ].statistic)
    
    print(get_ks(dvalid_predprob,y_true))
    
    path=sys.argv[1]+'/'
    
    #feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    #feat_imp.to_csv(path+'feature_imp_new_'+sys.argv[2]+'_all.csv',index=True)
    for importance_type in ('weight','gain','cover','total_gain','total_cover'):
        pd.Series(alg.booster().get_fscore(importance_type=importance_type)).sort_values(ascending=False).to_csv(path+'feature_imp_new_'+sys.argv[2]+importance_type+'_all.csv',index=True)
    #feat_imp.plot(kind='bar', title='Feature Importances')
    #plt.ylabel('Feature Importance Score')
    
    return dtrain_predprob,dtest_predprob,dvalid_predprob
    

    
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


xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 n_jobs=8,
 eval_metric='auc',
 seed=27,
 reg_alpha=0,
 reg_lambda=0)


#alg=xgb1.fit(df_train_new[predictos],df_train_new[target])
#==============================================================================
# ##save_model##########################
# joblib.dump(alg,path+'xgb_model_00.m')
# clf=joblib.load(path+'xgb_model_00.m')
# df_valid_09_pre=clf.predict_proba(df_valid_09[predictos])[:,1]
# valid_09_y_pre=con_list(df_valid_09_pre,dvalid_09_predprob)
# static_all_valid_09=cac_p_r(valid_09_y_pre)
# print('valid_09\n')
# print(static_all_valid_09)
# static_all_valid_09.to_csv(path+'/valid_09_p_rlgb.csv')
########################################
#==============================================================================



dtrain_predprob,dtest_predprob,dvalid_predprob=modelfit(xgb1,df_train_new,df_test_new,df_valid_new,predictos,target)





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


train_y=df_train_new['is_y2'].values
test_y=df_test_new['is_y2'].values
valid_y=df_valid_new['is_y2'].values


train_y_pre=con_list(train_y,dtrain_predprob)
static_all_train=cac_p_r(train_y_pre)
print('train\n')
print(static_all_train)
static_all_train.to_csv(path+'/train_p_rlgb.csv')




test_y_pre=con_list(test_y,dtest_predprob)
static_all_test=cac_p_r(test_y_pre)
print('test\n')
print(static_all_test)
static_all_test.to_csv(path+'/test_p_rlgb.csv')



valid_y_pre=con_list(valid_y,dvalid_predprob)
static_all_valid=cac_p_r(valid_y_pre)
print('valid\n')
print(static_all_valid)
static_all_valid.to_csv(path+'/valid_p_rlgb.csv')