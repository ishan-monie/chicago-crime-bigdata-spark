#!/bin/bash
# connect to wolf
ssh eil8095@wolf.mlds.private

# copy crime data to hdfs (run once)
hdfs dfs -mkdir -p /user/eil8095/crime
hdfs dfs -copyFromLocal /home/public/crime/Crimes_-_2001_to_present.csv /user/eil8095/crime/
hdfs dfs -ls /user/eil8095/crime/

# upload all scripts and exogenous datasets from local machine (run in a separate terminal)
scp code/monie_1.py code/monie_2.py code/monie_3.py code/monie_4.py bias/monie_bias.py eil8095@wolf.mlds.private:~/
scp Census_Data.csv ACS_5_Year_Data.csv eil8095@wolf.mlds.private:~/

# copy exogenous datasets to hdfs
hdfs dfs -copyFromLocal Census_Data.csv /user/eil8095/crime/
hdfs dfs -copyFromLocal ACS_5_Year_Data.csv /user/eil8095/crime/

# run tasks on wolf
spark-submit monie_1.py
spark-submit monie_2.py
spark-submit monie_3.py
spark-submit monie_4.py
spark-submit monie_bias.py

# download output files
scp eil8095@wolf.mlds.private:~/monie_*.txt ./output/
