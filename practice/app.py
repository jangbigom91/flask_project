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
    # import pandas as pd
    # import plotly.express as px

    # df = pd.read_csv('PM10_seoul.csv')

    # fig = px.line(df, x='Date', y='Seoul', title='PM10 of Seoul(2014~2021)')

    # fig_seoul = fig.show()

    # return render_template('plot.html', data=fig_seoul)

    import plotly.graph_objects as go
    import pandas as pd
    
    df = pd.read_csv('PM10_seoul.csv')

    fig = go.Figure(go.Scatter(x = df['Date'], y = df['Seoul'],
                            name='PM10'))

    fig.update_layout(title='PM10 of Seoul',
                    plot_bgcolor='rgb(230, 230, 230)',
                    showlegend=True)

    fig_seoul = fig.show()

    return render_template('plot.html', data=fig_seoul)

if __name__ == '__main__':
    app.run(debug=True)