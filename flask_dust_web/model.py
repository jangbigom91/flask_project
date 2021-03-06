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
test_date = 365
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
    model.add(LSTM(128, return_sequences=True, dropout=0.4, recurrent_dropout=0.4))


model.add(LSTM(64, return_sequences=False))

model.add(Dense(1, activation='linear'))

# model.summary()

### 모델 학습
start_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

model.compile(loss='mse', optimizer='adam')

hist = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs = 30, batch_size = 120)
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

# model_acc = total_acc
# good_level = level_acc[0]
# normal_level = level_acc[1]
# bad_level = level_acc[2]
# very_bad_level = level_acc[3]

date = [
        20210901, 20210902, 20210903, 20210904, 20210905, 20210906, 20210907, 20210908, 20210909, 20210910,
        20210911, 20210912, 20210913, 20210914, 20210915, 20210916, 20210917, 20210918, 20210919, 20210920,
        20210921, 20210922, 20210923, 20210924, 20210925, 20210926, 20210927, 20210928, 2021929, 20210930, 
        20211001, 20211002, 20211003, 20211004, 20211005, 20211006, 20211007, 20211008, 20211009, 20211010,
        20211011, 20211012, 20211013, 20211014, 20211015, 20211016, 20211017, 20211018, 20211019, 20211020,
        20211021, 20211022, 20211023, 20211024, 20211025, 20211026, 20211027, 20211028, 20211029, 20211030, 20211031,
        20211101, 20211102, 20211103, 20211104, 20211105, 20211106, 20211107, 20211108, 20211109, 20211110,
        20211111, 20211112, 20211113, 20211114, 20211115, 20211116, 20211117, 20211118, 20211119, 20211120,
        20211121, 20211122, 20211123, 20211124, 20211125, 20211126, 20211127, 20211128, 20211129, 20211130,
        20211201, 20211202, 20211203, 20211204, 20211205, 20211206, 20211207, 20211208, 20211209, 20211210,
        20211211, 20211212, 20211213, 20211214, 20211215, 20211216, 20211217, 20211218, 20211219, 20211220,
        20211221, 20211222, 20211223, 20211224, 20211225, 20211226, 20211227, 20211228, 20211229, 20211230, 20211231]

def date_indexing(date) :
    if date == '20210901' :
        return y_pred[243]
    elif date == '20210902' :
        return y_pred[244]
    elif date == '20210903' :
        return y_pred[245]
    elif date == '20210904' :
        return y_pred[246]
    elif date == '20210905' :
        return y_pred[247]
    elif date == '20210906' :
        return y_pred[248]
    elif date == '20210907' :
        return y_pred[249]
    elif date == '20210908' :
        return y_pred[250]
    elif date == '20210909' :
        return y_pred[251]
    elif date == '20210910' :
        return y_pred[252]
    elif date == '20210911' :
        return y_pred[253]
    elif date == '20210912' :
        return y_pred[254]
    elif date == '20210913' :
        return y_pred[255]
    elif date == '20210914' :
        return y_pred[256]
    elif date == '20210915' :
        return y_pred[257]
    elif date == '20210916' :
        return y_pred[258]
    elif date == '20210917' :
        return y_pred[259]
    elif date == '20210918' :
        return y_pred[260]
    elif date == '20210919' :
        return y_pred[261]
    elif date == '20210920' :
        return y_pred[262]
    elif date == '20210921' :
        return y_pred[263]
    elif date == '20210922' :
        return y_pred[264]
    elif date == '20210923' :
        return y_pred[265]
    elif date == '20210924' :
        return y_pred[266]
    elif date == '20210925' :
        return y_pred[267]
    elif date == '20210926' :
        return y_pred[268]
    elif date == '20210927' :
        return y_pred[269]
    elif date == '20210928' :
        return y_pred[270]
    elif date == '20210929' :
        return y_pred[271]
    elif date == '20210930' :
        return y_pred[272]
    elif date == '20211001' :
        return y_pred[273]
    elif date == '20211002' :
        return y_pred[274]
    elif date == '20211003' :
        return y_pred[275]
    elif date == '20211004' :
        return y_pred[276]
    elif date == '20211005' :
        return y_pred[277]
    elif date == '20211006' :
        return y_pred[278]
    elif date == '20211007' :
        return y_pred[279]
    elif date == '20211008' :
        return y_pred[280]
    elif date == '20211009' :
        return y_pred[281]
    elif date == '20211010' :
        return y_pred[282]
    elif date == '20211011' :
        return y_pred[283]
    elif date == '20211012' :
        return y_pred[284]
    elif date == '20211013' :
        return y_pred[285]
    elif date == '20211014' :
        return y_pred[286]
    elif date == '20211015' :
        return y_pred[287]
    elif date == '20211016' :
        return y_pred[288]
    elif date == '20211017' :
        return y_pred[289]
    elif date == '20211018' :
        return y_pred[290]
    elif date == '20211019' :
        return y_pred[291]
    elif date == '20211020' :
        return y_pred[292]
    elif date == '20211021' :
        return y_pred[293]
    elif date == '20211022' :
        return y_pred[294]
    elif date == '20211023' :
        return y_pred[295]
    elif date == '20211024' :
        return y_pred[296]
    elif date == '20211025' :
        return y_pred[297]
    elif date == '20211026' :
        return y_pred[298]
    elif date == '20211027' :
        return y_pred[299]
    elif date == '20211028' :
        return y_pred[300]
    elif date == '20211029' :
        return y_pred[301]
    elif date == '20211030' :
        return y_pred[302]
    elif date == '20211031' :
        return y_pred[303]
    elif date == '20211101' :
        return y_pred[304]
    elif date == '20211102' :
        return y_pred[305]
    elif date == '20211103' :
        return y_pred[306]
    elif date == '20211104' :
        return y_pred[307]
    elif date == '20211105' :
        return y_pred[308]
    elif date == '20211106' :
        return y_pred[309]
    elif date == '20211107' :
        return y_pred[310]
    elif date == '20211108' :
        return y_pred[311]
    elif date == '20211109' :
        return y_pred[312]
    elif date == '20211110' :
        return y_pred[313]
    elif date == '20211111' :
        return y_pred[314]
    elif date == '20211112' :
        return y_pred[315]
    elif date == '20211113' :
        return y_pred[316]
    elif date == '20211114' :
        return y_pred[317]
    elif date == '20211115' :
        return y_pred[318]
    elif date == '20211116' :
        return y_pred[319]
    elif date == '20211117' :
        return y_pred[320]
    elif date == '20211118' :
        return y_pred[321]
    elif date == '20211119' :
        return y_pred[322]
    elif date == '20211120' :
        return y_pred[323]
    elif date == '20211121' :
        return y_pred[324]
    elif date == '20211122' :
        return y_pred[325]
    elif date == '20211123' :
        return y_pred[326]
    elif date == '20211124' :
        return y_pred[327]
    elif date == '20211125' :
        return y_pred[328]
    elif date == '20211126' :
        return y_pred[329]
    elif date == '20211127' :
        return y_pred[330]
    elif date == '20211128' :
        return y_pred[331]
    elif date == '20211129' :
        return y_pred[332]
    elif date == '20211130' :
        return y_pred[333]
    elif date == '20211201' :
        return y_pred[334]
    elif date == '20211202' :
        return y_pred[335]
    elif date == '20211203' :
        return y_pred[336]
    elif date == '20211204' :
        return y_pred[337]
    elif date == '20211205' :
        return y_pred[338]
    elif date == '20211206' :
        return y_pred[339]
    elif date == '20211207' :
        return y_pred[340]
    elif date == '20211208' :
        return y_pred[341]
    elif date == '20211209' :
        return y_pred[342]
    elif date == '20211210' :
        return y_pred[343]
    elif date == '20211211' :
        return y_pred[344]
    elif date == '20211212' :
        return y_pred[345]
    elif date == '20211213' :
        return y_pred[346]
    elif date == '20211214' :
        return y_pred[347]
    elif date == '20211215' :
        return y_pred[348]
    elif date == '20211216' :
        return y_pred[349]
    elif date == '20211217' :
        return y_pred[350]
    elif date == '20211218' :
        return y_pred[351]
    elif date == '20211219' :
        return y_pred[352]
    elif date == '20211220' :
        return y_pred[353]
    elif date == '20211221' :
        return y_pred[354]
    elif date == '20211222' :
        return y_pred[355]
    elif date == '20211223' :
        return y_pred[356]
    elif date == '20211224' :
        return y_pred[357]
    elif date == '20211225' :
        return y_pred[358]
    elif date == '20211226' :
        return y_pred[359]
    elif date == '20211227' :
        return y_pred[360]
    elif date == '20211228' :
        return y_pred[361]
    elif date == '20211229' :
        return y_pred[362]
    elif date == '20211230' :
        return y_pred[363]
    elif date == '20211231' :
        return y_pred[364]

tomorrow_dust = round(float(date_indexing('20210928')))
after_tomorrow_dust = round(float(date_indexing('20210929')))

