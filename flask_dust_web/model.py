import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
import os
import warnings
warnings.filterwarnings('ignore')

from keras.models import Sequential, Model, load_model
from keras.layers import LSTM, Dropout, Dense, Activation, Conv1D, GlobalMaxPooling1D, MaxPooling1D, concatenate, Flatten, Reshape
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras import Input

np.random.seed(0)
tf.random.set_seed(0)

## 하이퍼 파라메터

delta = 1e-7
seq_len = 30
test_date = 257
factor_num = 4

level_1 = 50
level_2 = 100
level_3 = 150

num_level = 4

pm_target = level_3 * 2
temp_target = 24 * 2
humidity_target = 100
wind_speed_target = 14 # 강한 바람
wind_direction_target = 360*2 # 16방위

year = 365
train_cut = year * 6 + 2

test_cut = -test_date

## 데이터 로드
data = pd.read_csv('raw_dataset(W_direction).csv', index_col=0)
data.head()

## 초기 데이터 입력
pm_seoul_data = data['Seoul'].values
pm_tianjin_data = data['Tianjin'].values
pm_weihai_data = data['Weihai'].values
temp_data = data['Temperature'].values
humidity_data = data['Humidity'].values
wind_direction_data = data['W_direction'].values
wind_speed_data = data['W_speed'].values

## 시계열 함수
def make_sequential(data):
    for i in range(len(data)):
        if data[i] == 0:
            data[i] = data[i]+delta
        
    sequence_length = seq_len + 1
    
    temp_data = []
    for index in range((len(data) - sequence_length)+1):
        temp_data.append(data[index: index + sequence_length])
        
    return np.array(temp_data)

def pm_norm_window(data):
    norm_data = []
    
    for i in range(len(data)):
        norm_data.append(data[i] / pm_target)
        
    return norm_data

def temp_norm_window(data):
    norm_data = []
    
    for i in range(len(data)):
        norm_data.append((data[i] / temp_target)**2)
        
    return norm_data

def humidity_norm_window(data):
    norm_data = []
    
    for i in range(len(data)):
        norm_data.append((data[i] / humidity_target)**2)
        
    return norm_data

def humidity_norm_window(data):
    norm_data = []
    
    for i in range(len(data)):
        norm_data.append((data[i] / humidity_target)**2)
        
    return norm_data

def wind_speed_norm_window(data):
    norm_data = []
    
    for i in range(len(data)):
        norm_data.append(data[i] / wind_speed_target)
        
    return norm_data

def wind_direction_norm_window(data):
    norm_data = []
    
    for i in range(len(data)):
        norm_data.append((data[i] / wind_direction_target)**2)
        
    return norm_data

norm_pm_seoul = make_sequential(pm_norm_window(pm_seoul_data)) #a
norm_pm_tianjin = make_sequential(pm_norm_window(pm_tianjin_data)) #b
norm_pm_weihai = make_sequential(pm_norm_window(pm_weihai_data)) #c
norm_temp = make_sequential(temp_norm_window(temp_data)) #d
norm_humidity = make_sequential(humidity_norm_window(humidity_data)) #e
norm_wind_speed = make_sequential(wind_speed_norm_window(wind_speed_data)) #f
norm_wind_direction = make_sequential(wind_direction_norm_window(wind_direction_data)) #g

## 병합함수
def marge_data(a, b, c, d, e, f, g):
    marged_data = []
    marge = []
    
    for a_index, b_index, c_index, d_index, e_index, f_index, g_index in zip(a, b, c, d, e, f, g):
        for i in range(len(a_index)):
            marge.append(a_index[i])
            # marge.append(b_index[i])
            marge.append(c_index[i])
            marge.append(d_index[i])
            marge.append(e_index[i])
            # marge.append(f_index[i])
            # marge.append(g_index[i])
            # marge.append(h_index[i])
            
        for i in range(factor_num-1):
            marge.pop()
        
        marged_data.append(marge)
        marge = []
    
    return np.array(marged_data)

norm_result = marge_data(norm_pm_seoul, norm_pm_tianjin, norm_pm_weihai, norm_temp, norm_humidity, norm_wind_speed, norm_wind_direction)

train = norm_result[:train_cut, :]
np.random.shuffle(train)

valid = norm_result[train_cut:test_cut, :]
#np.random.shuffle(valid)

test = norm_result[test_cut:, :]

x_train = train[:, :-1]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
y_train = train[:,-1]
y_train = np.reshape(y_train, (y_train.shape[0], 1))

x_valid = valid[:, :-1]
x_valid = np.reshape(x_valid, (x_valid.shape[0], x_valid.shape[1], 1))
y_valid = valid[:,-1]
y_valid = np.reshape(y_valid, (y_valid.shape[0], 1))

x_test = test[:, :-1]
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
y_test = test[:,-1]
y_test = np.reshape(y_test, (y_test.shape[0], 1))

# print(x_train.shape, x_valid.shape, x_test.shape)
# print(y_train.shape, y_valid.shape, y_test.shape)

### LCRNN 모델 설계
model = Sequential()

model.add(Conv1D(32, 2, activation='linear',strides=2, input_shape=(seq_len*factor_num,1)))
model.add(Conv1D(64, 2, activation='linear',strides=2))
model.add(Conv1D(128, 1, activation='linear',strides=1))

# model.add(Conv1D(60, 3, activation='relu',strides=1, padding="same"))
for i in range (2):
    model.add(LSTM(128, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))


model.add(LSTM(64, return_sequences=False))

model.add(Dense(1, activation='linear'))

# model.summary()

### 모델 학습
start_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

model.compile(loss='mse', optimizer='adam')

hist = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs = 20, batch_size = 50)
#hist = model.fit(x_train_dict, y_train, validation_data=(x_valid_dict, y_valid), epochs=20, batch_size=100)

# ## 손실함수 변화 측정
# fig = plt.figure(facecolor='white', figsize=(5, 3))
# loss_ax = fig.add_subplot(111)

# loss_ax.plot(hist.history['loss'], 'b', label='train loss')
# loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
# loss_ax.set_ylim([0.1, 1.0])

# loss_ax.set_xlabel('epoch')
# loss_ax.set_ylabel('loss')

# loss_ax.legend(loc='upper left')

# plt.title('Loss for PM Prediction using LCRNN with four factors', fontsize=10)

# plt.xlim([0, 50])
# plt.ylim([0.001, 0.01])

# plt.show()

## 결과 실제화
y_true = pm_seoul_data[test_cut:]

pred = model.predict(x_test)

y_pred = pred * pm_target

# fig = plt.figure(facecolor='white', figsize=(20, 10))

# plt.title('PM Prediction using LCRNN with four factors', fontsize=20)

# ax = fig.add_subplot(111)
# ax.plot(y_true, label='True', marker='.')
# ax.plot(y_pred, label='Prediction', marker='^')

# plt.grid(color='gray', dashes=(5,5))

# plt.axhline(y=1, color='blue', linewidth=1, label='Good')
# plt.axhline(y=level_1, color='green', linewidth=1, label='Normal')
# plt.axhline(y=level_2, color='yellow', linewidth=1, label='Bad')
# plt.axhline(y=level_3, color='red', linewidth=1, label='Very Bad')

# plt.xlabel('day', fontsize=13)
# plt.ylabel('PM 10', fontsize=13)

# plt.xlim([0, test_date])
# plt.ylim([0, 180])

# ax.legend()
# plt.show()

## 단계별 정확도 및 총 정확도 계산 함수
def pm_level(pm):
    
    level_temp = []
    
    for i in range(len(pm)):
        if pm[i] <= level_1:
            level_temp.append(1)
        elif pm[i] <= level_2:
            level_temp.append(2)
        elif pm[i] <= level_3:
            level_temp.append(3)
        else:
            level_temp.append(4)
            
    return level_temp

def error_check(true, pred):
    y_true_lv = pm_level(true)
    y_pred_lv = pm_level(pred)
    
    level_1_acc = 0
    level_2_acc = 0
    level_3_acc = 0
    level_4_acc = 0

    level_1_count = y_true_lv.count(1)
    level_2_count = y_true_lv.count(2)
    level_3_count = y_true_lv.count(3)
    level_4_count = y_true_lv.count(4)
            
    
    error_rate_temp = []
    
    for i in range(len(y_pred_lv)):
        if y_pred_lv[i] == y_true_lv[i]:
            error_rate_temp.append(1)
            
            if y_pred_lv[i] == 1:
                level_1_acc += 1
            elif y_pred_lv[i] == 2:
                level_2_acc += 1
            elif y_pred_lv[i] == 3:
                level_3_acc += 1
            else:
                level_4_acc += 1
                    
        else:
            error_rate_temp.append(0)
    
    total_acc = sum(error_rate_temp) / len(error_rate_temp)
    
    
    level_1_accuracy = float(level_1_acc / level_1_count)
    level_2_accuracy = float(level_2_acc / level_2_count)
    level_3_accuracy = float(level_3_acc / level_3_count)
    level_4_accuracy = float(level_4_acc / level_4_count)
    
    level_accuracy = [level_1_accuracy,level_2_accuracy, level_3_accuracy, level_4_accuracy]
    
    return total_acc, level_accuracy

## 정확도 결과
total_acc, level_acc = error_check(y_true, y_pred)

model_acc = total_acc
good_level = level_acc[0]
normal_level = level_acc[1]
bad_level = level_acc[2]
very_bad_level = level_acc[3]
    