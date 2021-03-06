from flask import Flask, render_template, request, redirect, session
from bs4 import BeautifulSoup
from datetime import timedelta
import datetime
# import pymysql
import urllib.request
import csv
import pandas as pd

app = Flask(__name__)

# Database 연결
# db = pymysql.connect(
#     host = 'mydatabase.cr7yob8emqao.us-east-2.rds.amazonaws.com',
#     port = 3306,
#     user = 'admin',
#     passwd = 'altpaltp12!',
#     db = 'preprocessed_data'
# )

# index page
@app.route('/', methods = ['GET', 'POST'])
def index():
    ## 네이버 날씨 실시간 크롤링
    url = 'https://search.naver.com/search.naver?sm=tab_hty.top&where=nexearch&query=%EC%98%A4%EB%8A%98+%EB%AF%B8%EC%84%B8%EB%A8%BC%EC%A7%80' # 오늘 미세먼지(서울) url
    url_tomorrow = 'https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=1&ie=utf8&query=%EB%82%B4%EC%9D%BC+%EB%AF%B8%EC%84%B8%EB%A8%BC%EC%A7%80' # 내일 미세먼지(서울) url

    html = urllib.request.urlopen(url).read()
    html_tomorrow = urllib.request.urlopen(url_tomorrow).read()
    
    soup = BeautifulSoup(html, 'html.parser')
    soup_tomorrow = BeautifulSoup(html_tomorrow, 'html.parser')

    mydata = [] # mydata 안에 오늘 미세먼지(서울) 저장
    mydata_tomorrow_am = [] # mydata_tomorrow_am 안에 내일 오전 미세먼지(서울) 등급 저장
    mydata_tomorrow_pm = [] # mydata_tomorrow_pm 안에 내일 오후 미세먼지(서울) 등급 저장

    # 네이버 오늘 미세먼지(서울) 데이터 크롤링
    for i in soup.select('#main_pack > section.sc_new._atmospheric_environment > div > div.api_cs_wrap > div > div:nth-child(3) > div.main_box > div.detail_box > div.tb_scroll > table > tbody > tr:nth-child(1) > td:nth-child(2)'):
        mydata.append(i.find("span").text)
        # print(i.find("span").text)

    # 네이버 내일 오전 미세먼지(서울) 등급 크롤링
    for k in soup_tomorrow.select('#main_pack > section.sc_new._atmospheric_environment > div > div.api_cs_wrap > div > div:nth-child(4) > div.main_box > div.detail_box.list3 > div.tb_scroll > table > tbody > tr:nth-child(1) > td:nth-child(2)'):
        mydata_tomorrow_am.append(k.find("span").text)
    
    # 네이버 내일 오후 미세먼지(서울) 등급 크롤링
    for j in soup_tomorrow.select('#main_pack > section.sc_new._atmospheric_environment > div > div.api_cs_wrap > div > div:nth-child(4) > div.main_box > div.detail_box.list3 > div.tb_scroll > table > tbody > tr:nth-child(1) > td:nth-child(3)'):
        mydata_tomorrow_pm.append(j.find("span").text)

    # 현재 시간과 내일 시간 표시
    from datetime import timedelta
    import datetime

    now = datetime.datetime.now().strftime('%Y-%m-%d') # 오늘 날짜 표시
    tomorrow = datetime.date.today() + datetime.timedelta(days=1) # 내일 날짜 표시
    tomorrow_after = datetime.date.today() + datetime.timedelta(days=2) # 모레 날짜 표시

    # 모델 라이브러리 import
    import pandas as pd
    import numpy as np
    # import matplotlib.pyplot as plt
    import tensorflow as tf
    import datetime
    import os
    import warnings
    # import keras
    warnings.filterwarnings('ignore')

    from keras.models import Sequential, Model, load_model
    from keras.layers import LSTM, Dropout, Dense, Activation, Conv1D, GlobalMaxPooling1D, MaxPooling1D, concatenate, Flatten, Reshape
    from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
    from keras import Input
    from tensorflow import keras

    # random seed 설정
    np.random.seed(0)
    # tf.random.set_seed(0)

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
    # model = Sequential()

    # model.add(Conv1D(32, 2, activation='linear',strides=2, input_shape=(seq_len*factor_num,1)))
    # model.add(Conv1D(64, 2, activation='linear',strides=2))
    # model.add(Conv1D(128, 1, activation='linear',strides=1))

    # # model.add(Conv1D(60, 3, activation='relu',strides=1, padding="same"))
    # for i in range (2):
    #     model.add(LSTM(128, return_sequences=True, dropout=0.4, recurrent_dropout=0.4))


    # model.add(LSTM(64, return_sequences=False))

    # model.add(Dense(1, activation='linear'))

    # # model.summary()

    # ### 모델 학습
    # start_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    # model.compile(loss='mse', optimizer='adam')

    # hist = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs = 30, batch_size = 120)
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
    
    ### 저장한 모델 불러오기
    model = keras.models.load_model('model\LCNN_four(tem,hum,WH).h5')
    
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

    today = [
            20211001, 20211002, 20211003, 20211004, 20211005, 20211006, 20211007, 20211008, 20211009, 20211010,
            20211011, 20211012, 20211013, 20211014, 20211015, 20211016, 20211017, 20211018, 20211019, 20211020,
            20211021, 20211022, 20211023, 20211024, 20211025, 20211026, 20211027, 20211028, 20211029, 20211030, 20211031]

    def date_indexing(today) :
        if today == '20211001' :
            return y_pred[273]
        elif today == '20211002' :
            return y_pred[274]
        elif today == '20211003' :
            return y_pred[275]
        elif today == '20211004' :
            return y_pred[276]
        elif today == '20211005' :
            return y_pred[277]
        elif today == '20211006' :
            return y_pred[278]
        elif today == '20211007' :
            return y_pred[279]
        elif today == '20211008' :
            return y_pred[280]
        elif today == '20211009' :
            return y_pred[281]
        elif today == '20211010' :
            return y_pred[282]
        elif today == '20211011' :
            return y_pred[283]
        elif today == '20211012' :
            return y_pred[284]
        elif today == '20211013' :
            return y_pred[285]
        elif today == '20211014' :
            return y_pred[286]
        elif today == '20211015' :
            return y_pred[287]
        elif today == '20211016' :
            return y_pred[288]
        elif today == '20211017' :
            return y_pred[289]
        elif today == '20211018' :
            return y_pred[290]
        elif today == '20211019' :
            return y_pred[291]
        elif today == '20211020' :
            return y_pred[292]
        elif today == '20211021' :
            return y_pred[293]
        elif today == '20211022' :
            return y_pred[294]
        elif today == '20211023' :
            return y_pred[295]
        elif today == '20211024' :
            return y_pred[296]
        elif today == '20211025' :
            return y_pred[297]
        elif today == '20211026' :
            return y_pred[298]
        elif today == '20211027' :
            return y_pred[299]
        elif today == '20211028' :
            return y_pred[300]
        elif today == '20211029' :
            return y_pred[301]
        elif today == '20211030' :
            return y_pred[302]
        elif today == '20211031' :
            return y_pred[303]

    # 오늘 날짜 불러오기
    import datetime
    today = datetime.datetime.now().strftime('%Y%m%d')
    
    # 내일 날짜 지정하기, strftime을 이용해서 날짜타입을 str로 변환 -> 모델 예측 시, float에 들어오는 date_indexing 타입이 int나 string 형태여야 오류가 발생하지 않음.
    tomorrow_date = datetime.date.today() + datetime.timedelta(days=1)
    tomorrow_date_strftime = tomorrow_date.strftime('%Y%m%d')
    
    # 모레 날짜 지정하기, strftime을 이용해서 날짜타입을 str로 변환 -> 모델 예측 시, float에 들어오는 date_indexing 타입이 int나 string 형태여야 오류가 발생하지 않음.
    after_tomorrow_date = datetime.date.today() + datetime.timedelta(days=2)
    after_tomorrow_date_strftime = after_tomorrow_date.strftime('%Y%m%d')

    # 모델에 예측하는 날짜를 넣기
    tomorrow_dust = round(float(date_indexing(tomorrow_date_strftime)))
    after_tomorrow_dust = round(float(date_indexing(after_tomorrow_date_strftime)))
    
    return render_template("index.html", data = mydata, time = now, next_time = tomorrow, next_time_after = tomorrow_after, data_tomorrow_am = mydata_tomorrow_am, data_tomorrow_pm = mydata_tomorrow_pm,
                           tomorrow_dust = tomorrow_dust, after_tomorrow_dust = after_tomorrow_dust)

# process page
@app.route('/process', methods = ['GET', 'POST'])
def process():
    return render_template("process.html")

# # data page
# @app.route('/data', methods = ['GET', 'POST'])
# def data():
#     # cursor = db.cursor()
    
#     # sql = 'SELECT * FROM preprocessed_dataset;'

#     # cursor.execute(sql)
#     # dust = cursor.fetchall()

#     # return render_template("data.html", data=dust)
#     return render_template('data.html')

# data_result page
# @app.route('/data_result', methods = ['GET', 'POST'])
# def data_result():
#     # csv파일을 DataFrame으로 구현
#     if request.method == 'POST':
#         f = request.form['csvfile']
#         data = []

#         with open(f) as file:
#             csvfile = csv.reader(file)
#             for row in csvfile:
#                 data.append(row)
        
#         data = pd.DataFrame(data)
        
#         return render_template('data_result.html', data=data.to_html(header=False, index=False))

# predict page
# @app.route('/predict', methods = ['GET', 'POST'])
# def predict():
#     import model

#     return render_template("predict.html", data=model.model_acc, data_1=model.good_level, data_2=model.normal_level, data_3=model.bad_level, data4=model.very_bad_level)

# graph page
# @app.route('/graph', methods = ['GET', 'POST'])
# def graph():
#     return render_template('graph.html')

# # seoul pm10 graph page
# @app.route('/seoul_graph', methods = ['GET', 'POST'])
# def seoul_graph():
#     import pandas as pd
#     import numpy as np
#     import seaborn as sns
#     import matplotlib as mpl
#     import matplotlib
#     matplotlib.use('TkAgg')
#     import matplotlib.pyplot as plt
#     from dateutil.parser import parse

#     df = pd.read_csv('raw_dataset_plus.csv', parse_dates=['Date'], index_col='Date')

#     def plot_df(df, x, y, title="", xlabel='Date', ylabel='Seoul', dpi=100):
#         plt.figure(figsize=(16, 5), dpi=dpi)
#         plt.plot(x, y, color='tab:red')
#         plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
#         plt.show()

#         plt.close()

#     graph_seoul = plot_df(df, x=df.index, y=df.Seoul, title='PM10 of Seoul from 2014 to 2021.')
    
#     # 인덱스 컬럼(Date)을 데이터 컬럼으로 바꾸고 순번으로 인덱스 변경
#     df.reset_index(inplace=True)

#     # Date컬럼에서 년도, 월을 분리
#     df['year'] = [d.year for d in df.Date]
#     df['month'] = [d.strftime('%b') for d in df.Date]

#     # 도표 작성
#     fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=100)
#     sns.boxplot(x='year', y='Seoul', data=df, ax=axes[0])
#     sns.boxplot(x='month', y='Seoul', data=df, ax=axes[1])

#     # 제목 설정
#     axes[0].set_title('PM10 of Seoul(2014~2021 year)', fontsize=18)
#     axes[1].set_title('PM10 of Seoul(2014~2021 month)', fontsize=18)

#     plt.show()
#     plt.close()

#     return render_template('graph.html')

# # tianjin pm10 graph page
# @app.route('/tianjin_graph', methods = ['GET', 'POST'])
# def tianjin_graph():
#     import pandas as pd
#     import numpy as np
#     import seaborn as sns
#     import matplotlib as mpl
#     import matplotlib
#     matplotlib.use('TkAgg')
#     import matplotlib.pyplot as plt
#     from dateutil.parser import parse

#     df = pd.read_csv('raw_dataset_plus.csv', parse_dates=['Date'], index_col='Date')

#     def plot_df(df, x, y, title="", xlabel='Date', ylabel='Tianjin', dpi=100):
#         plt.figure(figsize=(16, 5), dpi=dpi)
#         plt.plot(x, y, color='tab:red')
#         plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
#         plt.show()

#         plt.close()

#     graph_Tianjin = plot_df(df, x=df.index, y=df.Tianjin, title='PM10 of Tianjin from 2014 to 2021.')
    
#     df.reset_index(inplace=True)

#     df['year'] = [d.year for d in df.Date]
#     df['month'] = [d.strftime('%b') for d in df.Date]

#     fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=100)
#     sns.boxplot(x='year', y='Tianjin', data=df, ax=axes[0])
#     sns.boxplot(x='month', y='Tianjin', data=df, ax=axes[1])

#     axes[0].set_title('PM10 of Tianjin(2014~2021 year)', fontsize=18)
#     axes[1].set_title('PM10 of Tianjin(2014~2021 month)', fontsize=18)

#     plt.show()
#     plt.close()

#     return render_template('graph.html')

# # weihai pm10 graph page
# @app.route('/weihai_graph', methods = ['GET', 'POST'])
# def weihai_graph():
#     import pandas as pd
#     import numpy as np
#     import seaborn as sns
#     import matplotlib as mpl
#     import matplotlib
#     matplotlib.use('TkAgg')
#     import matplotlib.pyplot as plt
#     from dateutil.parser import parse

#     df = pd.read_csv('raw_dataset_plus.csv', parse_dates=['Date'], index_col='Date')

#     def plot_df(df, x, y, title="", xlabel='Date', ylabel='Weihai', dpi=100):
#         plt.figure(figsize=(16, 5), dpi=dpi)
#         plt.plot(x, y, color='tab:red')
#         plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
#         plt.show()

#         plt.close()

#     graph_Weihai = plot_df(df, x=df.index, y=df.Weihai, title='PM10 of Weihai from 2014 to 2021.')
    
#     df.reset_index(inplace=True)

#     df['year'] = [d.year for d in df.Date]
#     df['month'] = [d.strftime('%b') for d in df.Date]

#     fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=100)
#     sns.boxplot(x='year', y='Weihai', data=df, ax=axes[0])
#     sns.boxplot(x='month', y='Weihai', data=df, ax=axes[1])

#     axes[0].set_title('PM10 of Weihai(2014~2021 year)', fontsize=18)
#     axes[1].set_title('PM10 of Weihai(2014~2021 month)', fontsize=18)

#     plt.show()
#     plt.close()

#     return render_template('graph.html')

# reference page
@app.route('/reference', methods = ['GET', 'POST'])
def reference():
    return render_template('reference.html')

# dust_predict page
@app.route('/dust_predict', methods = ['GET', 'POST'])
def dust_predict():
    return render_template('dust_predict.html')

# dust_predict_result page
@app.route('/dust_predict_result', methods = ['GET', 'POST'])
def dust_predict_result():
    # 모델 라이브러리 import
    import pandas as pd
    import numpy as np
    # import matplotlib.pyplot as plt
    import tensorflow as tf
    import datetime
    import os
    import warnings
    # import keras
    warnings.filterwarnings('ignore')

    
    from keras.models import Sequential, Model, load_model
    from keras.layers import LSTM, Dropout, Dense, Activation, Conv1D, GlobalMaxPooling1D, MaxPooling1D, concatenate, Flatten, Reshape
    from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
    from keras import Input
    from tensorflow import keras

    np.random.seed(0)
    # tf.random.set_seed(0)

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
    # model = Sequential()

    # model.add(Conv1D(32, 2, activation='linear',strides=2, input_shape=(seq_len*factor_num,1)))
    # model.add(Conv1D(64, 2, activation='linear',strides=2))
    # model.add(Conv1D(128, 1, activation='linear',strides=1))

    # # model.add(Conv1D(60, 3, activation='relu',strides=1, padding="same"))
    # for i in range (2):
    #     model.add(LSTM(128, return_sequences=True, dropout=0.4, recurrent_dropout=0.4))


    # model.add(LSTM(64, return_sequences=False))

    # model.add(Dense(1, activation='linear'))

    # # model.summary()

    # ### 모델 학습
    # start_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    # model.compile(loss='mse', optimizer='adam')

    # hist = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs = 30, batch_size = 120)
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
    
    ### 저장한 모델 불러오기
    model = keras.models.load_model('model\LCNN_four(tem,hum,WH).h5')
    
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

    today = [
            20211001, 20211002, 20211003, 20211004, 20211005, 20211006, 20211007, 20211008, 20211009, 20211010,
            20211011, 20211012, 20211013, 20211014, 20211015, 20211016, 20211017, 20211018, 20211019, 20211020,
            20211021, 20211022, 20211023, 20211024, 20211025, 20211026, 20211027, 20211028, 20211029, 20211030, 20211031]

    def date_indexing(today) :
        if today == '20211001' :
            return y_pred[273]
        elif today == '20211002' :
            return y_pred[274]
        elif today == '20211003' :
            return y_pred[275]
        elif today == '20211004' :
            return y_pred[276]
        elif today == '20211005' :
            return y_pred[277]
        elif today == '20211006' :
            return y_pred[278]
        elif today == '20211007' :
            return y_pred[279]
        elif today == '20211008' :
            return y_pred[280]
        elif today == '20211009' :
            return y_pred[281]
        elif today == '20211010' :
            return y_pred[282]
        elif today == '20211011' :
            return y_pred[283]
        elif today == '20211012' :
            return y_pred[284]
        elif today == '20211013' :
            return y_pred[285]
        elif today == '20211014' :
            return y_pred[286]
        elif today == '20211015' :
            return y_pred[287]
        elif today == '20211016' :
            return y_pred[288]
        elif today == '20211017' :
            return y_pred[289]
        elif today == '20211018' :
            return y_pred[290]
        elif today == '20211019' :
            return y_pred[291]
        elif today == '20211020' :
            return y_pred[292]
        elif today == '20211021' :
            return y_pred[293]
        elif today == '20211022' :
            return y_pred[294]
        elif today == '20211023' :
            return y_pred[295]
        elif today == '20211024' :
            return y_pred[296]
        elif today == '20211025' :
            return y_pred[297]
        elif today == '20211026' :
            return y_pred[298]
        elif today == '20211027' :
            return y_pred[299]
        elif today == '20211028' :
            return y_pred[300]
        elif today == '20211029' :
            return y_pred[301]
        elif today == '20211030' :
            return y_pred[302]
        elif today == '20211031' :
            return y_pred[303]
    
    # GET방식으로 dust_predict 페이지의 input 데이터 받아오기
    if request.method == 'POST':
        pass
    elif request.method == 'GET':
        date = request.args.get('date')  # dust_predict 페이지에서 날짜를 GET방식으로 받아오기
    
    result = round(float(date_indexing(date)))  # GET방식으로 날짜를 입력받고 result라는 변수에 모델을 실행
    
    return render_template('dust_predict_result.html', date=date, result = result)

# 오류 표시, 나중에 배포할 때는 app.debug 지우거나 False로 고쳐주기
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)