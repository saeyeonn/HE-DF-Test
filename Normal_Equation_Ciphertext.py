'''
'''


import os
os.environ["OMP_NUM_THREADS"] = "32"  # set the number of CPU threads to use for parallel regions

from pathlib import Path
import numpy as np
import pandas as pd
import time
import heaan_sdk as heaan
import math

# set key_dir_path
key_dir_path = Path('./keys')

# set parameter
params = heaan.HEParameter.from_preset("FGb")

# init context and load all keys
# if set generate_keys=True, then make key
# if set generate_keys=False, then not make key. just use existing key.

context = heaan.Context(
    params,
    key_dir_path=key_dir_path,
    load_keys="all",
    generate_keys=False,
)


num_slot = context.num_slots
print("num_slot : ", num_slot)

log_num_slot = context.log_slots
log_num_slot



def rotate_sum(input_ctxt):
#    input_ctxt=input.copy()
    for i in range(int(np.log2(num_slot))):
        tmp_ctxt = input_ctxt.__lshift__(2 ** i)
        input_ctxt = input_ctxt + tmp_ctxt
    return input_ctxt



def matrix_X_calculation(data_X, n, y):
    data_ctxt = heaan.Block(context, encrypted = False, data = data_X)
    data_ctxt.encrypt() 

    x_sqr_ctxt = data_ctxt * data_ctxt
    
    x_sum_slot = rotate_sum(data_ctxt) # sigma x
    x_sqr_sum_slot = rotate_sum(x_sqr_ctxt) # sigma (x^2)
    
    n_slot = heaan.Block(context, encrypted=False, data = [n for i in range(num_slot)])
    n_slot.encrypt()

    det_slot = n_slot * x_sqr_sum_slot
    det_slot = det_slot - x_sum_slot * x_sum_slot
    
    det_inverse = det_slot.inverse()

    det_inverse.bootstrap()
 
    result_row1 = data_ctxt * x_sum_slot

    result_row1 = result_row1 * (-1) # checked
    result_row1 = result_row1 + x_sqr_sum_slot


    result_row1 = det_inverse * result_row1

    result_row2 = x_sum_slot * (-1)
    result_row2 = result_row2 + n_slot * data_ctxt 
    result_row2 = det_inverse * result_row2

    row1=result_row1.copy()
    row2=result_row2.copy()
    row1.decrypt()
    row2.decrypt()

    for i in range(4):
         print("row1\n", row1[i])
         print("row2\n", row2[i])


data_x = [4,9]
y = [3, 4]
data_x_ctxt = matrix_X_calculation(data_x, 2, y)


'''
    # (X^T * X )! ^ (-1) * X^T = [m1 + m2 * x1, m1 + m2 * x2, ..., m1 + m2 * xn, m3 + m4 * x1, m3 + m4 * x2, ..., m3 + m4 * xn]
    temp = [row]
 
    for i in range(row): # make [m1, m1, ..., m1], [m2, m2, ..., m2], [m3, m3, ..., m3], [m4, m4, ..., m4]
        for j in range(row):
            temp[i] = 0

        temp[i] = 1
        temp_ctxt = heaan.Block(context, encrypted=False, data = temp)
        temp_ctxt = temp_ctxt.encrypt()

        for i in range(row):
            m1_slot += temp_ctxt * reverse_ctxt.__lshift__(i)
            m2_slot += temp_ctxt * reverse_ctxt.__lshift__(i + 1)
            m3_slot += temp_ctxt * reverse_ctxt.__lshift__(i + 2)
            m4_slot += temp_ctxt * reverse_ctxt.__lshift__(i + 3)
    
        res_row1 = m1_slot + m2_slot * data_ctxt # [m1 + m2 * x1, m1 + m2 * x2, ..., m1 + m2 * xn]    
        res_row2 = m3_slot + m4_slot * data_ctxt # [m3 + m4 * x1, m3 + m4 * x2, ..., m3 + m4 * xn]    

    matrix_result = res_row1, res_row2 # res_row1 와 res_row2 합치는 법 알아내서 result에 담아야함 (해당 코드는 임시 코드)

    return matrix_result



def normal_equation(matrix_result, y):

    # y = [?], y 값을 좀 더 확실하게 알아야함...
    y_ctxt = heaan.Block(context, encrypted=False, data = y)
    y_ctxt = y_ctxt.encrypt()
    w = matrix_result * y_ctxt
    
    return w



def dickey_fuller_with_constant(a, data_x_previous, data_x):

    # data_x should be array -> [[1, 1], [1, 2], [1, 3]]
    row = len(data_x_previous)
    n = heaan.Block(context,encrypted = False, data = row)
    n_ctxt = n.encrypt()

    data_x_ctxt = data_x_encryption(data_x, row)
    data_x_previous_ctxt = data_x_encryption(data_x_previous)
    matrix_X_calculation_value = matrix_X_calculation(data_x_previous_ctxt, n_ctxt, row, data_x_ctxt)

    # Delta X_t = a0 + delta * X_(t - 1) + W_t

    a = heaan.Block(context,encrypted = False, data = a)
    a_ctxt = a.encrypt()

    i = heaan.Block(context, enrcrypt = False, data = 1)
    i_ctxt = i.encrypt()

    big_delta_X = a_ctxt + (matrix_X_calculation - i_ctxt) * data_x_ctxt.__lshift__(time - 2)
    
    return big_delta_X.decrypt()
'''
