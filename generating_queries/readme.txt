test_queries_baseline,training_queries_baseline
数据格式：{"query":query（被查询文件的地址）,"positives":positives（正样本序号数组）,"negatives":negatives（负样本序号数组）}

oxford_inference_database
{'query':row['file'],'northing':row['northing'],'easting':row['easting']} 文件名，和坐标（单位m）

oxford_evaluation_database

{'query':row['file'],'northing':row['northing'],'easting':row['easting']} 整个范围内的文件和坐标 
oxford_evaluation_evaluation_query

{'query':row['file'],'northing':row['northing'],'easting':row['easting']， 0:[], 1:[]} 测试范围内文件坐标，以及其他路径的相似点云
