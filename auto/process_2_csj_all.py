# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 17:38:37 2019

@author: Administrator
"""

import pandas as pd
import numpy as np
from sklearn import cross_validation,metrics
from sklearn.model_selection import train_test_split
from cont_feature_discretization import feature_preprocessing
import threading
import math
import sys
import time

def var_to_str(data):
    data['import_disease_group_id']=data['import_disease_group_id'].astype(object)
    data['education']=data['education'].astype(object)
    data['ins_mari_sts']=data['ins_mari_sts'].astype(object)
    data['has_history_disease']=data['has_history_disease'].astype(object)
    return data

def float_to_str(ser):
    return ser.astype(str)

def str_to_float(ser):
    return ser.astype(float)

def char_process(data1,data2,char_vars):
    cat_vars=[]
    var_name=[]
    var_all=pd.DataFrame([])
    n_df=data1.shape[0]
    for col in char_vars:
        if data1[col].dtype=="object":
            cat_vars.append(col)
            print(col)
            data1[col].fillna("missing",inplace=True)
            data1[col]=float_to_str(data1[col])
            data2[col]=float_to_str(data2[col])
            values=data1[col].unique().to_list()
            if len(values)>50:
                if col !='appno':data1.drop(columns=[col],inplace=True)
                continue
            value_cnts=data1[col].value_counts()
            for value in values:
                if(value_cnts[value]<=50):
                    data1[col]=data1[col].apply(lambda x:'other' if x==value else x)
                    data2[col]=data2[col].apply(lambda x:'other' if x==value else x)
            values=data1[col].unique().tolist()
            value_cnts=data1[col].value_counts()
            var_statis=[]
            tmp=data1.groupby([col,'is_y2'])[['appno']].count()
            tmp=tmp.reset_index()
            for value in values:
                statis=[col,value,0, 0, 0,0]
                statis[2]=value_cnts[value]
                if len(tmp[(tmp[col]==value)&(tmp['is_y2']==1)].appno)==0:statis[3]==0
                else:statis[3]=tmp[(tmp[col]==value)&(tmp['is_y2']==1)].appno.values[0]
                if len(tmp[(tmp[col]==value)&(tmp['is_y2']==0)].appno)==0:statis[4]==0
                else:statis[4]=tmp[(tmp[col]==value)&(tmp['is_y2']==0)].appno.values[0]
                statis[5]=float(statis[3]/value_cnts[value])
                var_statis.append(statis)
            var_statis=sorted(var_statis,key=lambda x:x[5],reversed=True)
            num_sum=[]
            num_sum.insert(0,var_statis[0][2])
            for i in range(len(var_statis)-1):
                num_sum.insert(i+1,var_statis[i+1][2]+num_sum[i])
            
            var_statis=pd.DataFrame(var_statis,columns=['col_name','value','cnt','bad_num','good_num'\
                                                        ,'badrate'])
            num_sum=pd.DataFrame(num_sum,columns=['num_sum'])
            var_statis=pd.concat([var_statis,num_sum],axis=1)
            var_statis['sum']=n_df
            if len(values)<10:
                for i in range(len(values)):
                    var_statis.loc[i,'clus']=i+1
            else:
                var_statis['clus']=np.floor(var_statis['num_sum']/(var_statis['sum']/10+1)+1)
                
            data1=pd.merge(data1,var_statis,left_on=col,right_on='values',how='left')
            data1.drop([col,'col_name','values','cnt','badrate','bad_num','good_num','num_sum','sum'],axis=1,inplace=True)
            data1.rename(coulmns={'clus':'num_'+col},inplace=True)
            print(col+'\t is over')
            var_all=var_all.append(var_statis)
    return var_all,cat_vars,data1

def char_clus_1(data,char_vars,var_all):
    for col in char_vars:
        print(col)
        var_statis=var_all[var_all['col_name']==col]
        data[col]=data[col].fillna("missing")
        if len (var_statis[var_statis['values']=='other'].clus.values)>0:
            clus_n=var_statis[var_statis['values']=='other'].clus.values[0]
        else:
            clus_n=1
        data=pd.merge(data,var_statis,left_on=col,right_on='values',how='left')
        data.drop([col,'col_name','values','cnt','badrate','bad_num','good_num','num_sum','sum'],axis=1,inplace=True)
        data['clus'].fillna(clus_n,inplace=True)
        data['clus'].fillna(1,inplace=True)
        data.rename(coulmns={'clus':'num_'+col},inplace=True)
        print(col+'\t is over')
    return data

def null_padding(data):
    data['num_inso_age_group']=data['num_inso_age_group'].fillna(5)
    data['num_dept_grade']=data['num_dept_grade'].fillna(5)
    data['num_inso_age_range']=data['num_inso_age_range'].fillna(4)
    data['num_fmaily_is_black']=data['num_fmaily_is_black'].fillna(1)
    return data

def thread_padding(cat_vars,var_all):
    myThread=[]
    time1=time.time()
    num=len(cat_vars)
    print(num)
    global target_a
    target_a={}*num
    
    def char_clus(col,var_all):
        print(col)
        var_statis=var_all[var_all['col_name']==col]
        valid_bak=valid[[col]]
        valid_bak[col]=valid_bak[col].fillna("missing")
        if len(var_statis[var_statis['values']=='other'].clus.cnt)>0:
            clus_n=var_statis[var_statis['values']=='other'].clus.values[0]
        else:
            clus_n=1
        valid_bak=pd.merge(valid_bak,var_statis,left_on=col,right_on='values',how='left')
        valid_bak.drop([col,'col_name','values','cnt','badrate','bad_num','good_num','num_sum','sum'],axis=1,inplace=True)
        valid_bak['clus'].fillna(clus_n,inplace=True)
        valid_bak['clus'].fillna(1,inplace=True)
        valid_bak.rename(coulmns={'clus':'num_'+col},inplace=True)
        print('start')
        data_arr.append(valid_bak)
        print("end")
        
    for i in range(num)：:
        col=cat_vars[i]
        t=threading.Thread(target=char_clus,args=(col,val_all))
        myThread.append(t)
    for in range(num):
        myThread[i].start()
    for i in range(num):
        myThread[i].join()
    
    a=pd.DataFrame([])
    
    for i in range(num):
        a1=pd.DataFrame(data_arr[i])
        a=pd.concat([a,a1],axis=1)
    
    valid_new=a
    print("padding completed！\n")
    
    interval1=time.time()-time1
    
    print('time:',interval1)
    
    return valid_new

def thread_padding_train(cat_vars1):
    myThread=[]
    time1=time.time()
    num=len(cat_vars1)
    print(num)
    global target_a
    target_a={}*num
    
    n_df=train.shape[0]
    def char_process_1(col):
        print(col)
        train_bak=train[['is_y2','appno',col]]
        train_bak[col].fillna('missing',inplace=True)
        train_bak[col]=float_to_str(train[col])
        values=train_bak[col].unique().to_list()
        value_cnts=train_bak[col].value_counts()
        for value in values:
            if(value_cnts[value]<=50):
                train_bak[col]=train_bak[col].apply(lambda x:'other' if x==value else x)            
        values=train_bak[col].unique().tolist()
        value_cnts=train_bak[col].value_counts()
        tmp=train_bak.groupby([col,'is_y2'])[['appno']].count()
        tmp=tmp.reset_index()
        for value in values:
            statis=[col,value,0, 0, 0,0]
            statis[2]=value_cnts[value]
            if len(tmp[(tmp[col]==value)&(tmp['is_y2']==1)].appno)==0:statis[3]==0
            else:statis[3]=tmp[(tmp[col]==value)&(tmp['is_y2']==1)].appno.values[0]
            if len(tmp[(tmp[col]==value)&(tmp['is_y2']==0)].appno)==0:statis[4]==0
            else:statis[4]=tmp[(tmp[col]==value)&(tmp['is_y2']==0)].appno.values[0]
            statis[5]=float(statis[3]/value_cnts[value])
            var_statis.append(statis)
        var_statis=sorted(var_statis,key=lambda x:x[5],reversed=True)
        num_sum=[]
        num_sum.insert(0,var_statis[0][2])
        for i in range(len(var_statis)-1):
            num_sum.insert(i+1,var_statis[i+1][2]+num_sum[i])
            
        var_statis=pd.DataFrame(var_statis,columns=['col_name','value','cnt','bad_num','good_num'\
                                                        ,'badrate'])
        num_sum=pd.DataFrame(num_sum,columns=['num_sum'])
        var_statis=pd.concat([var_statis,num_sum],axis=1)
        var_statis['sum']=n_df
        if len(values)<10:
            for i in range(len(values)):
                var_statis.loc[i,'clus']=i+1
        else:
            var_statis['clus']=np.floor(var_statis['num_sum']/(var_statis['sum']/10+1)+1)
        
        var_statis['woe']=0
        var_statis['iv']=0
        for i in range(len(var_statis)):
            if var_statis.loc[i,'bad_num']==0:
                var_statis.loc[i,'woe']=math.log((0.0001/train_bak['is_y2'].sum())/(var_statis.loc[i,'good_num']/(train_bak.shape[0]-train_bak['is_y2'].sum())))
                var_statis.loc[i,'iv']=((var_statis.loc[i,'bad_num']/train_bak['is_y2'].sum())-(var_statis.loc[i,'good_num']/(train_bak.shape[0]-train_bak['is_y2'].sum())))*var_statis[i,'woe']
            else:
                var_statis.loc[i,'woe']=math.log((var_statis.loc[i,'bad_num']/train_bak['is_y2'].sum())/(var_statis.loc[i,'good_num']/(train_bak.shape[0]-train_bak['is_y2'].sum())))
                var_statis.loc[i,'iv']=((var_statis.loc[i,'bad_num']/train_bak['is_y2'].sum())-(var_statis.loc[i,'good_num']/(train_bak.shape[0]-train_bak['is_y2'].sum())))*var_statis[i,'woe']
        train_bak.drop(columns=['is_y2'],inplace=True)
        train_bak=pd.merge(train_bak,var_statis,left_on=col,right_on='values',how='left')
        train_bak.drop([col,'col_name','values','cnt','badrate','bad_num','good_num','num_sum','sum'],axis=1,inplace=True)
        train_bak.rename(coulmns={'clus':'num_'+col},inplace=True)
        print(col+'\t is over')
        data_arr_1.append(train_bak)
    for i in range(num)：:
        col=cat_vars[i]
        t=threading.Thread(target=char_process_1,args=(col,))
        myThread.append(t)
    for in range(num):
        myThread[i].start()
    for i in range(num):
        myThread[i].join()
    
    a=pd.DataFrame([])
    b=pd.DataFrame([])
    for i in range(num):
        a1=pd.DataFrame(data_arr_1[i])
        a=pd.concat([a,a1],axis=1)
        b1=pd.DataFrame(var_all[i])
        b=pd.concat([b,b1],axis=1)
    
    train_char=a
    val_char_all=b
    print("padding completed!\n")
    
    interval1=time.time()-time1
    
    print('time:',interval1)
    
    return train_char,val_char_all

def ger_stats(group):return {'n':group.count(),'badrate':group.mean(),'n1':group.sum()}

def q_cut(num_vars_valid,data,y,n,g):
    for col in num_vars_valid:
        print(col)
        valuecnt=data[col].value_counts()
        if valuecnt.shape[0]==1:
            grouped2=data[y].groupby(pd.qcut(data[col],1,duplicates='drop').replace(np.nan,'(-10000000000000001.0,-10000000000000001.0]'))
        else:
            grouped2=data[y].groupby(pd.qcut(data[col],n,duplicates='drop').replace(np.nan,'(-10000000000000001.0,-10000000000000001.0]'))#等深分组
                
        g2=grouped2.apply(ger_stats).unstack().reset_index()
        g2['column']=col
        g2.rename(columns={col:'test_name'},inplace=True)
        g2=g2.sort_values(by=['badrate'],ascending=False)
        g2=g2.reset_index(drop=True)
        g2['percent']=g2['n']/data.shape[0]
        g2['clus']=0
        g2['woe']=0
        g2['iv']=0
        g2['n0']=g2['n']-g2['n1']
        for i in range(len(g2)):
            if g2.loc[i,'n1']==0:
                g2.loc[i,'woe']=math.log((0.0001/data[y].sum())/(g2.loc[i,'n0']/(data.shape[0]-data[y].sum())))
                g2.loc[i,'iv']=((g2.loc[i,'n1']/data[y].sum())-(g2.loc[i,'n0']/(data.shape[0]-data[y].sum())))*g2.loc[i,'woe']
            else:
                g2.loc[i,'woe']=math.log((g2.loc[i,'n1']/data[y].sum())/(g2.loc[i,'n0']/(data.shape[0]-data[y].sum())))
                g2.loc[i,'iv']=((g2.loc[i,'n1']/data[y].sum())-(g2.loc[i,'n0']/(data.shape[0]-data[y].sum())))*g2.loc[i,'woe']
    
        for i in g2['badrate'].index:
            g2.loc[i,'clus']=i+1
        
        g=g.append(g2)
    
    return g


if __name__=='__main__':
    path=sys.argv[1]+'/'
    col_path=path+'column_name.xlxs'
    cols=pd.read_excel(col_path,header=None)
    cl=cols[cols[:,1]=='string'].iloc[ : ,0:1]
    dtypes={}
    for x in cl[0]:
        dtypes[x]=str
    #dtypes={'appno':str,'polno':str,'insno_age_group':str}
    
    base=pd.read_csv(path+'jianmo_'+sys.argv[2]+'.csv',encoding='gbk',dtype=dtypes)
    base['app_education']=base['app_education'].replace({'1':'01','2':'02','3':'03','4':'04'})
    base['ins_education']=base['ins_education'].replace({'1':'01','2':'02','3':'03','4':'04'})
    base.drop(columns=['app_date'],inplace=True)
    train,test=train_test_split(base,test_size=0.2,random_state=1)
    train=train.reset_index(drop=True)
    test=test.reset_index(drop=True)
    
    valid=pd.read_csv(path+'valid_'+sys.argv[2]+'.csv',encoding='gbk',dtype=dtypes)
    data_arr=[]
    data_arr_1=[]
    var_all=[]
    
    FP=feature_preprocessing(train,'is_y2')
    num_vars,char_vars=FP.get_vars()
    var_all_char,cat_vars,train=char_process(train,test,char_vars)
    var_all_char.to_excel(path+'class_bin_'+sys.argv[2]+'.xlsx',encoding='gbk',index=False)
    char_vars=var_all_char['col_name'].unique().to_list()
    train.to_csv(path+'train_trans'+sys.argv[2]+'.csv',index=False)
    test=char_clus_1(test,char_vars,var_all_char)
    test.to_csv(path+'test_trans'+sys.argv[2]+'.csv',index=False)
    valid_new=thread_padding(char_vars,var_all_char)
    valid_new.to_csv(path+'valid_trans'+sys.argv[2]+'.csv',index=False)
    
    




    
                