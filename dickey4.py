import os
os.environ["OMP_NUM_THREADS"] = "32"  # set the number of CPU threads to use for parallel regions

from pathlib import Path
import numpy as np
import pandas as pd
import time
import heaan_sdk as heaan
import math
import datetime

# set key_dir_path
key_dir_path = Path('./DF_test/keys')

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

def rotate_sum(input_ctxt):
    for i in range(int(np.log2(num_slot))):
        tmp_ctxt = input_ctxt.__lshift__(2 ** i)
        input_ctxt = input_ctxt + tmp_ctxt
    return input_ctxt

# 암호문 배열의 값의 합을 계산하기 위한 rotate_sum 함수.

# 디키-풀러 검정의 식과 데이터를 기반으로 선형 회귀 모델의 계수를 예측하는 정규방정식에 대한 이해 필요


def normal_equation(data_x_ctxt, n, data_y_ctxt):
    # 정규방정식의 파라미터는 일정 기간의 시계열, 데이터의 개수 n-1(차분을 고려해야 함), 기존 데이터로 부터 1단위 시점 이후의 시계열의 각 3가지
    # y_i = x_i+1
    # 연산에 필요한 암호문 slot의 개수는 n-1 -> 나중에 input으로는 데이터의 개수 - 1이 들어감
    
    x_sqr_ctxt = data_x_ctxt * data_x_ctxt
    x_sum_slot = rotate_sum(data_x_ctxt) # sigma x
    x_sqr_sum_slot = rotate_sum(x_sqr_ctxt) # sigma (x^2)
    
    n_slot = heaan.Block(context, encrypted=False, data = [n for i in range(num_slot)])
    n_slot.encrypt()
    
    # 이하는 2*2 행렬의 역행렬에 대한 determinant를 구하기 위한 코드 구현
    # determinant = (1 / ad - bc)

    # ad - bc
    det_slot = n_slot * x_sqr_sum_slot
    # n * sum_x_sqr = (데이터 개수) * (데이터 값의 제곱의 합) = ad
    det_slot = det_slot - x_sum_slot * x_sum_slot
    # 각각 b, c에 해당.
    
    det_inverse = det_slot.inverse(greater_than_one=1)
    # inverse() method를 사용하여, 해당 수를 역수로 만듦.

    det_inverse.bootstrap()

    
    # 역행렬을 구현한 이후의 2*n 행렬 계산 결과 = (x_T * x)^-1 * x_T 
    
    # 결과의 1st_row
    result_row1 = data_x_ctxt * x_sum_slot # 개별 데이터 * 데이터 값들의 합 -> x * sum(x)
    result_row1 = result_row1 * (-1) # 데이터셋에 대해 절댓값이 같고, 부호가 다른 음수로 데이터를 수정함. -> -1 * x * sum(x)
    result_row1 = result_row1 + x_sqr_sum_slot # 위에서 계산된 결괏값을 기준으로, 데이터 제곱의 합을 더해줌

    result_row1 = det_inverse * result_row1 # 이전에 구했던 스칼라값(det의 역수)를 곱해줌
    
    # 결과의 2nd_row
    result_row2 = x_sum_slot * (-1) # 각 데이터값의 합에 대해 부호를 변경함 -> 2*2 행렬의 역함수 구현
    result_row2 = result_row2 + n_slot * data_x_ctxt # 이전 값의 합에 대해서 n * 각 데이터 값을 더함. -> -sum_x + n * x_i
    result_row2 = det_inverse * result_row2 # 이전에 구했던 스칼라값(det의 역수)를 곱해줌
    
    
    # y벡터와의 벡터곱을 진행 - 결과값을 2개의 값으로 분리하여 파악
    vector1 = result_row1 * data_y_ctxt # 상수항의 계수(= b0)
    vector2 = result_row2 * data_y_ctxt # 회귀 직선의 계수(= b1)
        
    # vector의 n개 slot에 저장되어 있는 값들을 모두 합쳐야 각각 회귀 직선의 추정 상수항과 계수의 값
    a = rotate_sum(vector1) # 전체 상수항 slot의 합을 더한다. -> 추정 상수항
    delta = rotate_sum(vector2) # 전체 계수 slot 값의 합을 더한다. -> 추정 계수

    return a, delta, n_slot, x_sqr_sum_slot , det_inverse



# 차분 계산 : delta_X = X[1:] = np.diff(x_arr, n = 1)
# lagged 시계열 데이터 (t-1 시점) : X_lagged = x_arr[:-1]
# 회귀 모델에 상수항과 lagged 시계열 포함

# 이하는 디키-풀러 검정 통계량 계산 과정의 구현
def dickey_fuller_test( x_arr, n, y_arr):
    # 파라미터 : 시계열 데이터 x, 데이터셋의 크기 n, 차분값 배열 y_arr
    # 데이터의 차분은 main에서 실행해 주는 부분 존재 -> 전처리
    
    # p-value = 0.05(유의수준 5%, 디키-풀러 검정의 가설에 따라 단측 검정)에 해당하는 t-stat의 비교 위한 bound 값 설정
    # 이후 dickey-fuller t-table값과 비교
    if n <= 25:
        bound = -3
    elif 25 <n <= 50:
        bound = -2.93
    elif 50 <= n < 100:
        bound = -2.89
    elif 100 <= n < 250:
        bound =- 2.88
    elif 250 <= n < 500:
        bound = -2.97
    else:
        bound = -2.86
                
    
    data_x_ctxt = heaan.Block(context, encrypted = False, data = x_arr) # 1 ~ n-1
    data_x_ctxt.encrypt()
    data_y_ctxt = heaan.Block(context, encrypted = False, data = y_arr) # 데이터의 차분값
    data_y_ctxt.encrypt()
    
    a0, delta, n_ctxt, x_sqr_sum_ctxt, det_inverse =  normal_equation(data_x_ctxt, n, data_y_ctxt)
    # 상수항, 회귀 직선의 계수, n, 데이터 값의 제곱의 합, (1 / det) 
    # 5개의 배열의 n개의 slot에 해당 값이 저장되어있음
    
    mask = heaan.Block(context, encrypted = False, data = [1 for __ in range(len(x_arr))])
    
    
    # 데이터셋과 회귀 직선을 기반으로 하는 t-stat을 계산
    # t-stat은 계수 추정치를 계수 추정치의 표준 오차로 나눈 값 
    # 계수 추정치는 parameter에서 delta라는 블록으로 받아왔음 -> 표준 오차는 아래에서 계산


    # 잔차(residual) = (참값(수집한 데이터의 차분값) - 회귀 직선을 통해 추정된 값)
    residuals = a0 + delta * data_x_ctxt # 추정 회귀값
    residuals = data_y_ctxt - residuals # 회귀 모델을 구축하기 위해 사용하는 데이터도 y차분값
    
    
    residuals = residuals * mask 
    residuals_sqr = residuals * residuals # 각 잔차 값의 제곱 값
    residuals_sqr = rotate_sum(residuals_sqr) # 잔차의 제곱 값의 총합
    
    residuals_avg = rotate_sum(residuals) # 기존 잔차 값들의 합을 저장
    residuals_avg = residuals_avg * (1 / (n - 2)) # 잔차 값의 평균 (표본 자유도 = n-2)
    
    residuals_sum_sqr = residuals_avg * residuals_avg # 잔차의 평균의 제곱 
    sigma2 = residuals_sqr * (1 / (n - 2)) # 잔차 제곱의 평균
    sigma2 = sigma2 - residuals_sum_sqr    # 잔차의 분산 = sigma2 = np.var(residuals, ddof = 2)
    
    tmp = n_ctxt * det_inverse # (n / det)

    var_beta1 = sigma2 * tmp # 분산 * (n / det)
    var_beta1.bootstrap()
    
    var_beta1 = var_beta1 * mask 
    var_beta1.bootstrap() 

    s_delta = var_beta1.sqrt_inv(greater_than_one = 0)
    s_delta.bootstrap()         
    
    # Dickey-Fuller 검정 통계량 계산
    df_statistic = (delta) * s_delta # 계수 추정치를 표준 오차로 나눈 것에 해당
    dickey_result = df_statistic - bound  # 통계량에서 바운드 뺴서 음수면 정상성 있음, 양수면 없음 
    dickey_result = dickey_result.sign(inplace = True,log_range = 10)
    
    # 검정 통계량 반환
    return df_statistic, a0, delta, dickey_result # 통계량, 상수항, 회귀 직선의 계수, 디키-풀러 검정 결과



# test

drink = pd.read_csv('Bimonthly cigarette consumption per adult in dollars and real price in.csv', encoding='utf-8')
res = drink.iloc[:,1]*0.01

res = np.array(res)

X = np.array(res[:-1]) # 데이터셋의 마지막 항을 제외한 데이터 배열(t = 0 ... n-1)
y = np.array(res[1:]) # 데이터셋의 첫번째 항을 제외한 데이터 배열(t = 1 ... n)

# 차분을 구하기 위한 과정 (시간 단위 하나의 차이를 가지는 차분값)
# difference of y_t = y - x = x_t+1 - x
y = y - X 

start = time.time()

stat,a0,delta,dicky_result = dickey_fuller_test(X,len(res) - 1,y) 
stat.decrypt()


print("stat")
for i in range(5):
    print(stat[i]) # 검정 통계량 t-stat
    
    
a0.decrypt()

print("a0")
for i in range(5):
    print(a0[i]) # 회귀 모델의 상수항(b0)

delta.decrypt()


print("delta")
for i in range(5):
    print(delta[i]) # 회귀 직선 모델의 계수(b1)


dicky_result.decrypt()

print("dicky_result")
for i in range(5):
    print(dicky_result[i]) # 유의수준 5%를 만족하는가
    
time = time.time() - start 
    
times = str(datetime.timedelta(seconds = time))
short = times.split(".")[0]
print(f"{times} sec") 
    
# 시계열 데이터셋에 대한 Dickey-Fuller 검정 수행
# 결과는 검정 통계량, 회귀 계수 a0, delta를 포함
# df_statistic, a0, delta = dickey_fuller_test(data_column)
