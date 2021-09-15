from logging import debug
from flask import Flask, render_template, request
from datetime import timedelta
import datetime
import pymysql

app = Flask(__name__)

# Database 연결
db = pymysql.connect(
    host='mydatabase.cr7yob8emqao.us-east-2.rds.amazonaws.com',
    port=3306,
    user='admin',
    passwd='altpaltp12!',
    db = 'subject'
)

@app.route('/', methods=['GET', 'POST'])
def index():
    now = datetime.datetime.now().strftime('%Y-%m-%d %p%H')
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    
    return render_template('index.html', data=now, data_1=tomorrow)

@app.route('/subject', methods=['GET', 'POST'])
def subject():
    cursor = db.cursor()
    sql = 'SELECT * FROM programming;'

    cursor.execute(sql)

    subjects = cursor.fetchall()

    return render_template('subject.html', data=subjects)

import csv
@app.route('/pandas', methods = ['GET', 'POST'])
def pandas():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == 'POST':
        results = []

        user_csv = request.form.get('practice.csv').split('\n')
        reader = csv.DictReader(user_csv)

        for row in reader:
            results.append(dict(row))
        print(results)

        return 'post'

@app.route('/plot', methods = ['GET', 'POST'])
def plot():
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    from dateutil.parser import parse

    df = pd.read_csv('raw_dataset_plus.csv', parse_dates=['Date'], index_col='Date')

    def plot_df(df, x, y, title="", xlabel='Date', ylabel='Seoul', dpi=100):
        plt.figure(figsize=(16, 5), dpi=dpi)
        plt.plot(x, y, color='tab:red')
        plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
        plt.show()

    a = plot_df(df, x=df.index, y=df.Seoul, title='PM10 of Seoul from 2014 to 2021.')

    return render_template('plot.html')


if __name__ == '__main__':
    app.run(debug=True)