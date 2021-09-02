from logging import debug
from flask import Flask, render_template
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

if __name__ == '__main__':
    app.run(debug=True)