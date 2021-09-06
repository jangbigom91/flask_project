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

if __name__ == '__main__':
    app.run(debug=True)