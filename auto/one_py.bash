#!/bin/sh
source ExitCodeCheck.sh

opts=$@
getparam(){
arg=$1
echo $opts |xargs -n1 |cut -b 2- |awk -F'=' '{if($1=="'"$arg"'") print $2}'
}
file=`getparam file`
valid_table=`getparam valid_table`
jianmo_table=`getparam jianmo_table`
c_time=`getparam c_time`
current_path=`echo $(cd "$(dirname "${BASH_SOURCE[0]}")/" && pwd)`
cd $current_path

[ ! -d $file ]&&mkdir -p ${file}
cd $file
WORK_PATH=`echo $(cd "$(dirname "${BASH_SOURCE[0]}")/" && pwd)`
echo $WORK_PATH

cp -f ${current_path}/column_name.xlxs  .
cp -f ${current_path}/import_data_hb_csj_0606.py  .
cp -f ${current_path}/cont_feature_discretization.py  .
cp -f ${current_path}/process_2_csj_all.py     .
cp -f ${current_path}/lightgbm_new_csj_0606.py   .
cp -f ${current_path}/xgboost_train_csj_0606.py   .
cp -f ${current_path}/best_lgb_csj_0606.py    .
 
hdfs dfs -get /apps-data/hduser1510/AI-export/$valid_table
hdfs dfs -get /apps-data/hduser1510/AI-export/$jianmo_table

python import_data_hb_csj_0606.py $WORK_PATH $jianmo_table $valid_table $c_time
python process_2_csj_all.py $WORK_PATH  $c_time
python lightgbm_new_csj_0606.py $WORK_PATH  $c_time
python xgboost_train_csj_0606.py $WORK_PATH  $c_time
python best_lgb_csj_0606.py $WORK_PATH  $c_time


