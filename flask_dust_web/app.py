from flask import Flask, render_template, request, redirect, session
from bs4 import BeautifulSoup
from datetime import timedelta
import datetime
import pymysql
import urllib.request

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
    url = 'https://search.naver.com/search.naver?sm=tab_hty.top&where=nexearch&query=%EC%98%A4%EB%8A%98+%EB%AF%B8%EC%84%B8%EB%A8%BC%EC%A7%80'

    html = urllib.request.urlopen(url).read()
    soup = BeautifulSoup(html, 'html.parser')

    mydata = []

    for i in soup.select('#main_pack > section.sc_new._atmospheric_environment > div > div.api_cs_wrap > div > div:nth-child(3) > div.main_box > div.detail_box > div.tb_scroll > table > tbody > tr:nth-child(1) > td:nth-child(2)'):
        mydata.append(i.find("span").text)
        # print(i.find("span").text)

    now = datetime.datetime.now().strftime('%Y-%m-%d %p%H')
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)

    return render_template("index.html", data = mydata, time = now, next_time = tomorrow)

# show page
@app.route('/show', methods = ['GET', 'POST'])
def show():
    cursor = db.cursor()

    sql = 'SELECT * FROM preprocessed_dataset;'

    cursor.execute(sql)
    dust = cursor.fetchall()

    return render_template("show.html", data=dust)

# process page
@app.route('/process', methods = ['GET', 'POST'])
def process():
    return render_template("process.html")

# 오류 표시, 나중에 배포할 때는 app.debug 지우거나 False로 고쳐주기
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')