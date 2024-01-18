import pandas as pd
import numpy as np

def data_process(array_size, file_path):
    df = pd.read_csv(file_path)
    # 获取数据的行数和列数
    num_rows, num_cols = df.shape
    # 创建一个空的 Numpy 数组来存储结果
    result_array = np.zeros((num_rows - array_size + 1, array_size, num_cols), dtype=object)
    j = 0
    i = 0
    # 从 DataFrame 中提取子数组
    while i + array_size + 1 < len(df.values) :
            time_step = df.iloc[i : i+1, 1].values
            a = time_step[0]
            time_step_last = df.iloc[i+array_size - 1 : i+array_size, 1].values
            b = time_step_last[0]
            if round(b - a , 1) == round((array_size-1) * 0.1,1): ###当且仅当取值区间连续，才提取
                result_array[j, :, :] = df.iloc[i: i+array_size, :].values
                j += 1
                i += 1
            else:
                i += 1
    # 打印结果数组
    result_array = result_array[0:j,:,:]
    return result_array



