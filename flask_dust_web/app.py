from flask import Flask, render_template, request, redirect, session
from bs4 import BeautifulSoup
from datetime import datetime
import pymysql
import urllib.request

app = Flask(__name__)

# Database 연결
# db = pymysql.connect(
#     host = 'mydatabase.cr7yob8emqao.us-east-2.rds.amazonaws.com',
#     port = 3306,
#     user = 'admi',
#     passwd = 'altpaltp12!',
#     db = 'dust'
# )

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

    today_time = datetime.today().strftime("%Y/%m/%d %H")

    return render_template("index.html", data = mydata, time = today_time)

# news page
@app.route('/news', methods = ['GET', 'POST'])
def news():
    return render_template("news.html")

# info page
@app.route('/info', methods = ['GET', 'POST'])
def info():
    return render_template("info.html")


# 오류 표시, 나중에 배포할 때는 app.debug 지우거나 False로 고쳐주기
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')