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



def normal_equation(data_x_ctxt, n, data_y_ctxt):
    
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
        
    vector1 = result_row1 * data_y_ctxt
    vector2 = result_row2 * data_y_ctxt
        
    a = rotate_sum(vector1)
    delta = rotate_sum(vector2)

    a_cp = a.copy()
    d_cp = delta.copy()
    a_cp.decrypt()
    d_cp.decrypt()
    
    return a,delta, n_slot, x_sqr_sum_slot


# 차분 계산 : delta_X = X[1:] = np.diff(x_arr, n = 1)
# lagged 시계열 데이터 (t-1 시점) : X_lagged = x_arr[:-1]
# 회귀 모델에 상수항과 lagged 시계열을 포함

def dickey_fuller_test(x_arr, n, y_arr):
    
    data_x_ctxt = heaan.Block(context, encrypted = False, data = x_arr)
    data_x_ctxt.encrypt() 
    data_y_ctxt = heaan.Block(context, encrypted = False, data = y_arr)
    data_y_ctxt.encrypt() 

    beta = normal_equation(data_x_ctxt, n, data_y_ctxt)

    # 회귀 계수
    # 정규방정식에 들어가는 X값[:-1] :첫번째 시점부터 n-1 시점까지 , Y 값 = X[1:]-X[:-1] t 시점에서 t-1 시점을  뺀값 
    zero = [0]
    zero_ctxt = heaan.Block(context, encrypted = False, data = zero)

    a0, delta, n_ctxt, x_sqr_sum_ctxt = beta
    a0_ctxt = zero_ctxt + a0
    a0_ctxt = rotate_sum(a0_ctxt)

    g_ctxt = zero_ctxt + delta 
    g_ctxt = rotate_sum(g_ctxt)
    g_ctxt = 1- g_ctxt

    g_ctxt = g_ctxt * data_x_ctxt

    # 계수의 표준 오차 계산
    delta_X_hat = g_ctxt + a0_ctxt  # 회귀 예측값 -> 수식으로 풀어서 계산해서 이부분이 필요없을 것 같긴함
    residuals = g_ctxt + a0_ctxt # 잔차=에러 값들 

    residuals_sqr = residuals * residuals
    residuals_avg = rotate_sum(residuals) / n
    residuals_sum_sqr = residuals_avg * residuals_avg
    sigma2 = residuals_sqr / n
    sigma2 = sigma2 - residuals_sum_sqr # 잔차의 분산 = sigma2 = np.var(residuals, ddof = 2)  
    

    # XTX = X_design.T @ X_design, XTX_inv = np.linalg.inv(XTX)

  
    # 계수의 분산 추정치
    # diag 는 행렬에 대각값만 가져온 것 [1], [3] -> var_beta = sigma2 * np.diag(XTX_inv)
    var_beta0 = sigma2 * x_sqr_sum_ctxt
    var_beta1 = sigma2 * n_ctxt

    # delta 계수의 표준 오차 -> 암호문으로 구현해야함
    s_delta = np.sqrt(var_beta1)

    # Dickey-Fuller 검정 통계량 계산
    df_statistic = (delta) / s_delta

    # 검정 통계량 반환
    return df_statistic, a0, delta


# 시계열 데이터셋에 대한 Dickey-Fuller 검정 수행
# 결과는 검정 통계량, 회귀 계수 a0, delta를 포함
# df_statistic, a0, delta = dickey_fuller_test(data_column)