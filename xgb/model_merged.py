#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 08:04:37 2017

@author: lx
"""

import hyperopt

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import metric
import xgboost as xgb
import lightgbm as lgb
import sys
import numpy as np


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
    
    
class BasicModel(object):
    def __init__(self):
        self.model_name=''
    
    def train(self,x_train,y_train,x_val,y_val):
        pass
    
    def predict(self,model,x_test):
        pass
    
    def get_oof(self,x_train,y_train,x_test,n_folds=5):
        num_train,num_test=x_train.shape[0],x_test.shape[0]
        print(num_train,y_train.shape[0],num_test)
        oof_train=np.zeros((num_train,))
        oof_test=np.zeros((num_test,))
        oof_test_all_fold=np.zeros((num_test,n_folds))
        KF=KFold(n_splits=n_folds,shuffle=True,random_state=10)
        for i,(train_index,val_index) in enumerate(KF.split(x_train)):
            print('{0} fold,train {1},val {2}'.format(i,len(train_index),len(val_index)))
            x_tra,y_tra=x_train.iloc[train_index],y_train.iloc[train_index]
            x_val,y_val=x_train.iloc[val_index],y_train.iloc[val_index]
            model=self.train(x_tra,y_tra,x_val,y_val)
            print(model)
            oof_train[val_index]=self.predict(model,x_val)
            oof_test_all_fold[:,i]=self.predict(model,x_test)
        oof_test=np.mean(oof_test_all_fold,axis=1)
        return oof_train,oof_test
        
        

class LGBClassifier(BasicModel):
    def __init__(self):
        BasicModel.__init__(self)
        self.num_boot_round=465
        self.early_stopping_rounds=15
        self.params={'nthread':8,'boosting_type':'gbdt','learning_rate':0.04,'bagging_fraction':0.88,'num_leaves':56,'max_bin':52,'bin_construct_sample_cnt':2345,'objective': 'binary','feature_fraction': 0.7,'seed': 100,'lambda_l1': 0,'lambda_l2': 0,'min_data_in_bin':34,'min_sum_hessian_in_leaf':0.0,'task':'train','metric':{'binary_logloss'},'min_gain_to_split':0.07}

    def train(self,x_train,y_train,x_val,y_val):
        lgbtrain=lgb.Dataset(x_train,y_train)
        lgbval=lgb.Dataset(x_val,y_val)
        model=lgb.train(self.params,lgbtrain,valid_sets=lgbval,verbose_eval=self.num_boot_round,\
                        num_boost_round=self.num_boot_round,early_stoping_rounds=self.early_stopping_rounds)
        return model
    
    def predict(self,model,x_test):
        return model.predict(x_test,num_iteration=model.best_iteration)
        


class XGBClassifier(BasicModel):
    def __init__(self):
        self.num_rounds=15
        self.early_stopping_rounds=15
        self.params={'nthread': 8, 'max_depth': 15,'n_estimators': 237,'eta': 0.05,'subsample': 0.6,'min_child_weight': 2,'objective': 'binary:logistic','gamma': 0, 'colsample_bytree': 0.7, 'alpha': 0,'lambda': 0,'scale_pos_weight': 0, 'seed': 100}

    def train(self,x_train,y_train,x_val,y_val):
        xgbtrain=xgb.DMatrix(x_train,y_train)
        xgbval=xgb.DMatrix(x_val,y_val)
        watchlist=[(xgbtrain,'train'),(xgbval,'val')]
        model=xgb.train(self.params,xgbtrain,self.num_rounds,watchlist,early_stoping_rounds=self.early_stopping_rounds)
        
        score=float(model.eval(xgbval).split()[1].split(':')[1])
        return model
        
    
        def predict(self,model,x_test):
            xgbtest=xgb.DMatrix(x_test)
            
            return model.predict(xgbtest)
            
            

if __name__=='main':
    #读取数据
    path='/home/luoxue001/data_52'+'/'
    df_train_new=pd.read_csv(path+'train_trans_'+'csjtest_1129'+'.csv')
    df_test_new=pd.read_csv(path+'test_trans_'+'csjtest_1129'+'.csv')
    
    df_valid_1=pd.read_csv(path+'valid_'+'csjtest_1129'+'.csv',encoding='gbk')
    df_valid=pd.read_csv(path+'valid_trans'+'csjtest_1129'+'.csv',encoding='gbk')
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

    x_train=df_train_new[predictos]
    y_train=df_train_new['is_y2']
    x_test=df_valid_new[predictos]


    xgb_classifier=XGBClassifier()
    xgb_oof_train,xgb_oof_test=XGBClassifier.get_oof(xgb_classifier,x_train,y_train,x_test,5)
    print(xgb_oof_train.shape,xgb_oof_test.shape)
    
    
    lgb_classifier=LGBClassifier()
    lgb_oof_train,lgb_test_oof=LGBClassifier.get_oof(lgb_classifier,x_train,y_train,x_test,5)
    print(lgb_oof_train.shape,lgb_test_oof.shape)
    
    
    input_train=[xgb_oof_train,lgb_oof_train]
    input_test=[xgb_oof_test,lgb_test_oof]


    stacked_train=np.concatenate([f.reshape(-1,1)  for f in input_train],axis=1)
    stacked_test=np.concatenate([f.reshape(-1,1)  for f in input_test],axis=1)
    
    
    final_model=LinearRegression()
    final_model.fit(stacked_train,y_train)
    test_prediction=final_model.predict(stacked_test)
    
    prediction=final_model.predict(stacked_test)
    labels=df_valid_new['is_y2'].values
    valid_y_pre=con_list(labels,prediction)
    static_all_valid=cac_p_r(valid_y_pre)
    
    print(static_all_valid.loc[0,'hit'],static_all_valid)
    
    

                  











        