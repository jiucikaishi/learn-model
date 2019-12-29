# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 12:43:07 2019

@author: Administrator
"""

import pandas as pd
import numpy as np
import math
from datetime import datetime
from sklearn import preprocessing

#1.变量预处理
#将描述变量存储到cont_vars 这个list中去

class feature_preprocessing(object):
    def __init__(self,data,y,features_path):
        self.n_df=data.shape[0]
        self.n1_df=data[y].sum()
        self.n0_df=self.n_df-self.n1_df
        self.features_columns_types=pd.read_excel(features_path,header=None)
        self.features_columns_types.columns=['features','types']
        print("共有{row}行{col}列数据,{n1}个坏客户，{n0}个好客户".format(
                row=data.shpae[0],
                col=data.shape[1],
                n1=self.n1_df,
                n0=self.n0_df
                ))
        self.cols=data.columns.to_list()
        self.cols.remove(y)
        del self.cols[0]
        self.num_vars_valid=[]
        self.char_vars_valid=[]
        self.num_vars_novalid=[]
        self.char_vars_novalid=[]
        self.null_count=[]
        self.sim_count=[]
        print("\n单一变量和缺失值分析：")
        for col in self.cols:
            if (self.features_columns_types[self.features_columns_types['features']==col].types).any()!="string":
            #if data[col].dtype!="object":
                missing=self.n_df-np.count_nonzero(data[col].isnull().values)
                mis_perc=100-float(missing)/self.n_df*100
                self.null_count.append([col,mis_perc])
                value_cnt=data[col].value_counts()
                sim_perc=float(max(value_cnt,default=100))/self.n_df*100
                self.sim_count.append([col,sim_perc])
                if mis_perc<99 and sim_perc<99:self.num_vars_valid.append(col)
            else:
                print("{col}的缺失值比例是{miss}%,单一值比例是{simple}%".format(col=col,miss=mis_perc,simple=sim_perc))
                self.num_vars_novalid.append(col)
        
        print ("\n连续有效变量有\n:")
        print(len(self.num_vars_valid))
        print(self.num_vars_valid)
        print ("\n连续无效变量有\n:")
        print(len(self.num_vars_novalid))
        print(self.num_vars_novalid)
        for col in self.cols:
            if (self.features_columns_types[self.features_columns_types['features']==col].types).any()=="string":
            #if data[col].dtype=="object":
                missing=self.n_df-np.count_nonzero(data[col].isnull().values)
                mis_perc=100-float(missing)/self.n_df*100
                self.null_count.append([col,mis_perc])
                value_cnt=data[col].value_counts()
                sim_perc=float(max(value_cnt,default=100))/self.n_df*100
                self.sim_count.append([col,sim_perc])                
                if mis_perc<99 and sim_perc<99:self.char_vars_valid.append(col)
            else:
                print("{col}的缺失值比例是{miss}%,单一值比例是{simple}%".format(col=col,miss=mis_perc,simple=sim_perc))
                self.char_vars_novalid.append(col)
        print ("\n分类有效变量有\n:")
        print(len(self.char_vars_valid))
        print(self.char_vars_valid)
        print ("\n分类无效变量有\n:")
        print(len(self.char_vars_novalid))
        print(self.char_vars_novalid)
        print(self.null_count)
        print(self.sim_count)
        data.drop(self.num_vars_novalid,axis=1,inplace=True)
        data.drop(self.char_vars_novalid,axis=1,inplace=True)
    #获取有效的数值变量和文本变量
    def get_vars(self):
        return self.num_vars_valid,self.char_vars_valid
