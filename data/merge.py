import pandas as pd
import os

root_path = os.getcwd()
inputfile_dir = root_path
outputfile = 'all_tf.csv'
list = ['1_tf.csv', '2_tf.csv', '3_tf.csv', '4_tf.csv', '5_tf.csv']

for inputfile in list:
     k = pd.read_csv(inputfile, header=None) #header=None表示原始文件数据没有列索引，这样的话read_csv会自动加上列索引
     k.to_csv(outputfile, mode='a', index=False, header=False) #header=0表示不保留列名，index=False表示不保留行索引，mode='a'表示附加方式写入，文件原有内容不会被清除