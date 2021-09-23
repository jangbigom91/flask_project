from flask import Flask, render_template, request, redirect, session
from bs4 import BeautifulSoup
from datetime import timedelta
import datetime
import pymysql
import urllib.request
import csv
import pandas as pd

app = Flask(__name__)

# Database 연결
db = pymysql.connect(
    host = 'mydatabase.cr7yob8emqao.us-east-2.rds.amazonaws.com',
    port = 3306,
    user = 'admin',
    passwd = 'altpaltp12!',
    db = 'preprocessed_data'
)

# index page
@app.route('/', methods = ['GET', 'POST'])
def index():
    ## 네이버 날씨 실시간 크롤링
    url = 'https://search.naver.com/search.naver?sm=tab_hty.top&where=nexearch&query=%EC%98%A4%EB%8A%98+%EB%AF%B8%EC%84%B8%EB%A8%BC%EC%A7%80'
    url_tomorrow = 'https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=1&ie=utf8&query=%EB%82%B4%EC%9D%BC+%EB%AF%B8%EC%84%B8%EB%A8%BC%EC%A7%80'

    html = urllib.request.urlopen(url).read()
    html_tomorrow = urllib.request.urlopen(url_tomorrow).read()
    
    soup = BeautifulSoup(html, 'html.parser')
    soup_tomorrow = BeautifulSoup(html_tomorrow, 'html.parser')

    mydata = []
    mydata_tomorrow_am = []
    mydata_tomorrow_pm = []

    for i in soup.select('#main_pack > section.sc_new._atmospheric_environment > div > div.api_cs_wrap > div > div:nth-child(3) > div.main_box > div.detail_box > div.tb_scroll > table > tbody > tr:nth-child(1) > td:nth-child(2)'):
        mydata.append(i.find("span").text)
        # print(i.find("span").text)

    for k in soup_tomorrow.select('#main_pack > section.sc_new._atmospheric_environment > div > div.api_cs_wrap > div > div:nth-child(4) > div.main_box > div.detail_box.list3 > div.tb_scroll > table > tbody > tr:nth-child(1) > td:nth-child(2)'):
        mydata_tomorrow_am.append(k.find("span").text)
    
    for j in soup_tomorrow.select('#main_pack > section.sc_new._atmospheric_environment > div > div.api_cs_wrap > div > div:nth-child(4) > div.main_box > div.detail_box.list3 > div.tb_scroll > table > tbody > tr:nth-child(1) > td:nth-child(3)'):
        mydata_tomorrow_pm.append(j.find("span").text)

    # 현재 시간과 내일 시간 표시
    now = datetime.datetime.now().strftime('%Y-%m-%d %p%H')
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    tomorrow_after = datetime.date.today() + datetime.timedelta(days=2)

    import model

    return render_template("index.html", data = mydata, time = now, next_time = tomorrow, next_time_after = tomorrow_after, data_tomorrow_am = mydata_tomorrow_am, data_tomorrow_pm = mydata_tomorrow_pm, tomorrow_dust = model.tomorrow_dust)

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

@app.route('/graph', methods = ['GET', 'POST'])
def graph():
    return render_template('graph.html')

@app.route('/seoul_graph', methods = ['GET', 'POST'])
def seoul_graph():
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib as mpl
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from dateutil.parser import parse

    df = pd.read_csv('raw_dataset_plus.csv', parse_dates=['Date'], index_col='Date')

    def plot_df(df, x, y, title="", xlabel='Date', ylabel='Seoul', dpi=100):
        plt.figure(figsize=(16, 5), dpi=dpi)
        plt.plot(x, y, color='tab:red')
        plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
        plt.show()

        plt.close()

    graph_seoul = plot_df(df, x=df.index, y=df.Seoul, title='PM10 of Seoul from 2014 to 2021.')
    
    # 인덱스 컬럼(Date)을 데이터 컬럼으로 바꾸고 순번으로 인덱스 변경
    df.reset_index(inplace=True)

    # Date컬럼에서 년도, 월을 분리
    df['year'] = [d.year for d in df.Date]
    df['month'] = [d.strftime('%b') for d in df.Date]

    # 도표 작성
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=100)
    sns.boxplot(x='year', y='Seoul', data=df, ax=axes[0])
    sns.boxplot(x='month', y='Seoul', data=df, ax=axes[1])

    # 제목 설정
    axes[0].set_title('PM10 of Seoul(2014~2021 year)', fontsize=18)
    axes[1].set_title('PM10 of Seoul(2014~2021 month)', fontsize=18)

    plt.show()
    plt.close()

    return render_template('graph.html')

@app.route('/tianjin_graph', methods = ['GET', 'POST'])
def tianjin_graph():
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib as mpl
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from dateutil.parser import parse

    df = pd.read_csv('raw_dataset_plus.csv', parse_dates=['Date'], index_col='Date')

    def plot_df(df, x, y, title="", xlabel='Date', ylabel='Tianjin', dpi=100):
        plt.figure(figsize=(16, 5), dpi=dpi)
        plt.plot(x, y, color='tab:red')
        plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
        plt.show()

        plt.close()

    graph_Tianjin = plot_df(df, x=df.index, y=df.Tianjin, title='PM10 of Tianjin from 2014 to 2021.')
    
    df.reset_index(inplace=True)

    df['year'] = [d.year for d in df.Date]
    df['month'] = [d.strftime('%b') for d in df.Date]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=100)
    sns.boxplot(x='year', y='Tianjin', data=df, ax=axes[0])
    sns.boxplot(x='month', y='Tianjin', data=df, ax=axes[1])

    axes[0].set_title('PM10 of Tianjin(2014~2021 year)', fontsize=18)
    axes[1].set_title('PM10 of Tianjin(2014~2021 month)', fontsize=18)

    plt.show()
    plt.close()

    return render_template('graph.html')

@app.route('/weihai_graph', methods = ['GET', 'POST'])
def weihai_graph():
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib as mpl
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from dateutil.parser import parse

    df = pd.read_csv('raw_dataset_plus.csv', parse_dates=['Date'], index_col='Date')

    def plot_df(df, x, y, title="", xlabel='Date', ylabel='Weihai', dpi=100):
        plt.figure(figsize=(16, 5), dpi=dpi)
        plt.plot(x, y, color='tab:red')
        plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
        plt.show()

        plt.close()

    graph_Weihai = plot_df(df, x=df.index, y=df.Weihai, title='PM10 of Weihai from 2014 to 2021.')
    
    df.reset_index(inplace=True)

    df['year'] = [d.year for d in df.Date]
    df['month'] = [d.strftime('%b') for d in df.Date]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=100)
    sns.boxplot(x='year', y='Weihai', data=df, ax=axes[0])
    sns.boxplot(x='month', y='Weihai', data=df, ax=axes[1])

    axes[0].set_title('PM10 of Weihai(2014~2021 year)', fontsize=18)
    axes[1].set_title('PM10 of Weihai(2014~2021 month)', fontsize=18)

    plt.show()
    plt.close()

    return render_template('graph.html')

@app.route('/reference', methods = ['GET', 'POST'])
def reference():
    return render_template('reference.html')

# 오류 표시, 나중에 배포할 때는 app.debug 지우거나 False로 고쳐주기
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')