#!/bin/sh
source ExitCodeCheck.sh
 
opts=$@
 
getparam(){
arg=$1
echo $opts |xargs -n1 |cut -b 2- |awk -F'=' '{if($1=="'"$arg"'") print $2}'
}
 
 
IncStart=`getparam inc_start`
IncEnd=`getparam inc_end`
oracle_connection=`getparam jdbc_str`
oracle_username=`getparam db_user`
oracle_password=`getparam db_psw`
dataName=`getparam db_sid`
queueName=`getparam hdp_queue`
hdfshostname=`getparam hdfs_host`
 
 
IncStartYear=`echo ${IncStart:0:4}`;
IncStartMonth=`echo ${IncStart:4:2}`;
IncStartDay=`echo ${IncStart:6:2}`;
IncStartAll=${IncStartYear}"-"${IncStartMonth}"-"${IncStartDay}" 00:00:00.0";
 
 
IncStartAllFormat=${IncStartYear}"-"${IncStartMonth}"-"${IncStartDay};
 
 
IncEndYear=`echo ${IncEnd:0:4}`;
IncEndMonth=`echo ${IncEnd:4:2}`;
IncEndDay=`echo ${IncEnd:6:2}`;
IncEndAll=${IncEndYear}"-"${IncEndMonth}"-"${IncEndDay}" 00:00:00.0";
 
 
twoDayAgo=`date -d "$IncStart 2 days ago  " +%Y%m%d  `;
twoDayAgoYear=`echo ${twoDayAgo:0:4}`;
twoDayAgoMonth=`echo ${twoDayAgo:4:2}`;
twoDayAgoDay=`echo ${twoDayAgo:6:2}`;
twoDayAgoAll=${twoDayAgoYear}"-"${twoDayAgoMonth}"-"${twoDayAgoDay}" 00:00:00.0";
twoDayAgoAllFormat=${twoDayAgoYear}"-"${twoDayAgoMonth}"-"${twoDayAgoDay};
 
 
job_name=$0
 
 
#需要导出的数据oracle表名
export_table_name=NCHRMS_ORGANIZATION_INTF;
 
 
#需要导出到oracle的数据的临时文件名
sqoop_export_data_filename=${export_table_name};
 
 
#需要导出的数据oracle列名
export_table_columns=ORG_ID,PARENT_ORG_ID,ORG_CODE,ORG_EN_NAME,ORG_CH_NAME,ORG_TAG,EFFECTIVE_DATE,LAPSED_DATE,PLACE_CODE,ORG_BIZ_CODE,IS_ACTIVE,ORG_LEVEL,ORG_SERIES,CREATED_BY,CREATED_DATE,UPDATED_BY,UPDATED_DATE
 
 
#需要导出到oracle的数据的临时文件目录
sqoop_export_data_dir=/apps-data/hduser0101/sx_360_safe/export/${sqoop_export_data_filename};
 
 
 
 
 
hadoop dfs -rmr ${sqoop_export_data_dir};
 
 
#创建用于导出到oracle的临时数据
hive -v -e "set mapred.job.queue.name=${queueName}; 
set mapred.job.name=${job_name}_1;
use an_pafc_safe;
insert overwrite directory '${sqoop_export_data_dir}' 
select 
ORG_ID,
PARENT_ORG_ID,
ORG_CODE,
ORG_EN_NAME,
ORG_CH_NAME,
ORG_TAG,
EFFECTIVE_DATE,
LAPSED_DATE,
PLACE_CODE,
ORG_BIZ_CODE,
IS_ACTIVE,
ORG_LEVEL,
ORG_SERIES,
CREATED_BY,
CREATED_DATE,
UPDATED_BY,
UPDATED_DATE
from lnc_cris_safe.nchrms_organization_intf ;";
exitCodeCheck $?
 
 
#先删除目的数据库的数据2天前数
sqoop eval -Dmapred.job.queue.name=${queueName} \
--connect ${oracle_connection} \
--username ${oracle_username} \
--password ${oracle_password} \
--verbose \
--query  "delete from ${export_table_name}";
exitCodeCheck $?
 
 
#先删除目的数据库的数据，支持二次运行
sqoop eval -Dmapred.job.queue.name=${queueName} \
--connect ${oracle_connection} \
--username ${oracle_username} \
--password ${oracle_password} \
--verbose \
--query  "delete from ${export_table_name}";
exitCodeCheck $?
 
 
#再导出数据
sqoop export -D mapred.job.name=${job_name}_2 -D sqoop.export.statements.per.transaction=4500 -D mapreduce.map.tasks=1 -D mapred.map.max.attempts=1 -D mapred.reduce.max.attempts=1 -D mapreduce.map.maxattempts=1 -D mapreduce.reduce.maxattempts=1 -D mapred.job.queue.name=${queueName} \
--connect ${oracle_connection} \
--username ${oracle_username} \
--password ${oracle_password} \
--export-dir ${sqoop_export_data_dir} \
--verbose \
--num-mappers 1 \
--table ${export_table_name} \
--columns ${export_table_columns} \
--input-fields-terminated-by '\001'  \
--input-lines-terminated-by '\n'  \
--input-null-string '\\N'  \
--input-null-non-string '\\N';
exitCodeCheck $?
