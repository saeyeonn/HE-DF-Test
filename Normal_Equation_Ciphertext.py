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


def transpose(Ciphertext ctxt):
    rows, cols = 2
    
    for i in range(4)
        rot_idx = 0

        if i % 2 == 1:
            rot_idx = 2

        if i == 2:
            rot_idx = 3

        add(ctxt.__lshift__(rot_idx), result) 

    return result


def matrix_multiplication(Ciphertext &ctxt1, Ciphertext &ctxt2):

    Ciphertext ctxt3 = (ctxt1 * ctxt2).encrypt()
    prev_rot_idx, lat_rot_idx = 0

    for i in range(4):
        
        for j in range(2):

            if j == 1:
                prev_rot_idx = 1    
                lat_rot_idx = 2

            else if i % 2 == 1:
                prev_rot_idx, lat_rot_idx = 3
            
            else if i == 2:
                prev_rot_idx = 2
                lat_rot_idx = 1

            prev_ctxt = ctxt1.__lshift__(prev_rot_idx)
            lat_ctxt = ctxt2.__rshift__(lat_rot_idx)

            temp += ctxt1 * ctxt2

        add(temp, result)

    return result


def matrix_inverse(Ciphertext &ctxt):
    rot_idx = 3
    det = ctxt.decrypt() * ctxt.__lshift__(rot_idx).decrypt() 
        - ctxt.__lshift__(--rot_idx).decrypt * ctxt.__lshift__(--rot_idx).decrypt

    if det == 0:
        raise ValueError("Matrix is not invertible")

    inv_det = 1 / det

    for i in ragne(4):
        
        if i % 2 == 0:
            rot_idx = 1
            temp = ctxt.__lshift__(rot_idx)

        else
            rot_idx = 2
            if i == 1:
                ++rot_idx
            temp = ctxt.__lshift__(rot_idx).__neg__()

        add(temp * inv_det, result)
        
    return result


    def linear_regression(X, y):

        X = X.encrypt()

        X_transpose = transpose(X)
        X_transpose_X = matrix_multiplication(X_transpose, X)
        X_transpose_X_inverse = matrix_inverse(X_transpose_X)
        X_transpose_y = matrix_multiplication(X_transpose, [val for val in y])
        theta = matrix_multiplication(X_transpose_X_inverse, X_transpose_y)

        return theta
