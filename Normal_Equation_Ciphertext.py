import os
os.environ["OMP_NUM_THREADS"] = "8"

from pathlib import Path
import numpy as np
import pandas as pd
import time
import math
import heaan_sdk as heaan

key_dir_path = Path('./BinaryDT/keys')

params = heaan.HEParameter.from_preset("FGb")
heaan_sdk.
context = heaan.Context(
    params,
    key_dir_path = key_dir_path,
    load_keys = "all",
    generate_keys = False,
)

num_slot = context.num_slots
num_slot

log_num_slot = context.log_slots

v1 = [0, 0, 0, 0]
v2 = [1, 1, 1, 1]

v1 = heaan.Block[context, encrypted = False, data = a]
v2 = heaan.Block[context, encrypted = False, data = b]

ctxt1 = v1.encrypt()
ctxt2 = v2.encrypt()

ctxt1.save('root/tutorial/python/x1.ctxt')
ctxt2.save('root/tutorial/python/x2.ctxt')

empty_msg = heaan.Block(context, encrypted = False)

x1 = load_ctxt1.decrypt()
x2 = load_ctxt2.decrypt()

block = heaan.Block(context, encrypted = False, data = [0] * num_slot)
ctxt = block.encrypt()



def calculate_between_matrices(n, ctxt):
    add(n, X_transpose_X_ctxt)

    for i in range(n):
        sum_ctxt += ctxt.__lshift__(i)
        sum_power_ctxt += (ctxt.__lshift__(i))^2
    
    add(sum_ctxt, X_transpose_X_ctxt)
    add(sum_power_ctxt, X_transpose_X_ctxt)

    det = res_ctxt * res_ctxt.__lshift__(3) - res_ctxt.__lshift__(1) * res_ctxt.__lshift__(2)
    
    add(1 / det * inversed_element.__lshift__(3), X_transpose_X_inversed_ctxt)
    add(-1 / det * inversed_element.__lshift__(1), X_transpose_X_inversed_ctxt)
    add(-1 / det * inversed_element.__lshift__(2), X_transpose_X_inversed_ctxt)
    add(1 / det * inversed_element, X_transpose_X_inversed_ctxt)
    
    for i in range(n):
        res_element = X_transpose_X_inversed_ctxt * ctxt.__lshift__(i)
                    + X_transpose_X_inversed_ctxt.__lshift__(1)
        add(res_element, res_ctxt)

    for i in range(n):
        res_element = X_transpose_X_inversed_ctxt.__lshift__(2) * ctxt.__lshift__(i)
                    + X_transpose_X_inversed_ctxt.__lshift__(3)
        add(res_element, res_ctxt)
        
return res_ctxt



def get_weight(X, y):
    X = X.encrypt()
    # n = count of X

    for i in range(n):
        add(X.__lshift__(i), matrix_cal_res)
    matrix_cal_res = calculate_between_matrices(X)
    for i in range():
        
