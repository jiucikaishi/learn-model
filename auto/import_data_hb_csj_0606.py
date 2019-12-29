# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 11:46:19 2019

@author: Administrator
"""

import os
import pandas as pd
import time
import threading
import numpy as np
import sys

def thread_file(file_packg,col_path):
    myThread=[]
    time1=time.time()
    filelist=os.listdir(file_packg)
    filelist=[os.path.join(file_packg,x) for x in filelist]
    num=len(filelist)
    print(num)
    global target
    target=[0]*num
    def df_file_read(filepath,i):
        raw_data=pd.read_csv(filepath,header=None,seq='\u0001')
        raw_data=raw_data.replace('\\N',np.nan)
        target[i]=raw_data.values
    
    for i in range(num):
        t=threading.Thread(target=df_file_read,args=(filelist[i],i))
        myThread.append(t)
        
    for i in range(num):
        t=myThread[i].start()
    
    for i in range(num):
        myThread[i].join()
    
    
    print("reading completed!\n")
    
    res=np.vstack[target]
    del target
    
    interval1=time.time()-time1
    print('time:',interval1)
    cols=pd.read_excel(col_path,header=None)
    raw_data=pd.DataFrame(res,columns=cols.iloc[:,0])
    print(raw_data.shape)
    interval2=time.time()-time1
    print('time:',interval2)
    return raw_data


if __name__=='__main__':
    file=sys.argv[1]+'/'
    file_pack_train1=file+sys.argv[2]
    file_pack_valid1=file+sys.argv[3]
    #file_pack_train2=file+'/hb_csj_0828_jianmo_1_4'
    col_path=file+'column_name.xlxs'
    train1=thread_file(file_pack_train1,col_path)
    train1.to_csv(file+'jiamo_'+sys.argv[4]+'.csv',encoding='gbk')
    valid1=thread_file(file_pack_valid1,col_path)
    valid1.to_csv(file+'valid_'+sys.argv[4]+'.csv',encoding='gbk')    
    
    #随机抽样
    #train2=pd.read_csv(file+'train1.csv',encoding='gbk')
    #smaple2=train2.smaple(n=100000,frac=None,replace=False,weights=None,random_state=777,axis=0)
    #smaple2.to_csv(file+'train1_smaple.csv',encoding='gbk')
    