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
log_num_slot = context.log_slots
log_num_slot



def rotate_sum(input_ctxt):
    for i in range(int(np.log2(num_slot))):
        tmp_ctxt = input_ctxt.__lshift__(2 ** i)
        input_ctxt = input_ctxt + tmp_ctxt
    return input_ctxt



def matrix_X_calculation(data_x, n, data_y):
    data_x_ctxt = heaan.Block(context, encrypted = False, data = data_x)
    data_x_ctxt.encrypt() 

    x_sqr_ctxt = data_x_ctxt * data_x_ctxt
    x_sum_slot = rotate_sum(data_x_ctxt) # sigma x
    x_sqr_sum_slot = rotate_sum(x_sqr_ctxt) # sigma (x^2)
    
    n_slot = heaan.Block(context, encrypted=False, data = [n for i in range(num_slot)])
    n_slot.encrypt()

    det_slot = n_slot * x_sqr_sum_slot
    det_slot = det_slot - x_sum_slot * x_sum_slot
    
    det_inverse = det_slot.inverse()
    det_inverse.bootstrap()
 
    result_row1 = data_x_ctxt * x_sum_slot
    result_row1 = result_row1 * (-1) 
    result_row1 = result_row1 + x_sqr_sum_slot
    result_row1 = det_inverse * result_row1

    result_row2 = x_sum_slot * (-1)
    result_row2 = result_row2 + n_slot * data_x_ctxt 
    result_row2 = det_inverse * result_row2


    v1_cp = result_row1.copy()
    v1_cp.decrypt()
    for i in range(3):
        print("v1 : ", v1_cp[i])
    v2_cp = result_row2.copy()
    v2_cp.decrypt()
    for i in range(3):
        print("v2 : ", v2_cp[i])


    # checked
        
        
    data_y_ctxt = heaan.Block(context, encrypted = False, data = data_y)
    data_y_ctxt.encrypt() 

    vector1 = result_row1 * data_y_ctxt
    vector2 = result_row2 * data_y_ctxt

    v1_cp = vector1.copy()
    v1_cp.decrypt()
    for i in range(3):
        print("v1 : ", v1_cp[i])
    v2_cp = vector2.copy()
    v2_cp.decrypt()
    for i in range(3):
        print("v2 : ", v2_cp[i])


# dickey fuller Delta_X(t) = a + delta * X(t-1)
        
    a = rotate_sum(vector1 * vector1)
    small_delta = rotate_sum(vector2 * vector2)
    small_delta_x = small_delta * vector2
    big_delta_X = a + small_delta_x    # ...........? 여기 어떻게 하지..
    return big_delta_X





# test
            
x = [2,4,5,6,7]
y = [4,5,6,7,8]

matrix_X_calculation(x, 2, y)

