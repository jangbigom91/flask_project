from flask import Flask, render_template, request, redirect, session
from bs4 import BeautifulSoup
from passlib.hash import sha256_crypt
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
@app.route('/', methods = ['GET'])
def index():
    url = 'https://search.naver.com/search.naver?sm=tab_hty.top&where=nexearch&query=%EC%98%A4%EB%8A%98+%EB%AF%B8%EC%84%B8%EB%A8%BC%EC%A7%80'

    html = urllib.request.urlopen(url).read()
    soup = BeautifulSoup(html, 'html.parser')

    mydata = []

    for i in soup.select('#main_pack > section.sc_new._atmospheric_environment > div > div.api_cs_wrap > div > div:nth-child(3) > div.main_box > div.detail_box > div.tb_scroll > table > tbody > tr:nth-child(1) > td:nth-child(2)'):
        mydata.append(i.find("span").text)
        # print(i.find("span").text)

    return render_template("index.html", data = mydata)

# news page
@app.route('/news', methods = ['GET', 'POST'])
def news():
    return render_template("news.html")

# info page
@app.route('/info', methods = ['GET', 'POST'])
def info():
    return render_template("info.html")

# login page
# @app.route('/login', methods = ['GET', 'POST'])
# def login():
#     cursor = db.cursor()
    
#     if request.method == "POST":
#         userid = request.form['userid']
#         password = request.form['password']

#         sql = "SELECT * FROM `users` WHERE userid = %s;"
#         input_data = [userid]

#         cursor.execute(sql, input_data)
#         user = cursor.fetchone()

#         if user == None:
#             print(user)
#             return redirect('/register')
#         else:
#             return redirect('/')
#     else:
#         return render_template('login.html')
    
# # register page
# @app.route('/register', methods = ['GET', 'POST'])
# def register():
#     cursor = db.cursor()
    
#     if request.method == "POST":
#         userid = request.form['userid']
#         password = sha256_crypt.encrypt(request.form['password'])

#         sql = "INSERT INTO `users` (`userid`, `password`) VALUES (%s, %s);"
#         input_data = [userid, password]

#         cursor.execute(sql, input_data)
#         db.commit()

#         # db.close()

#         return redirect("/")
#     else:
#         return render_template("register.html")

# @app.route('/logout', methods = ['GET'])
# def logout():
#     session.pop('userid', None)
#     return redirect("/")

# 오류 표시, 나중에 배포할 때는 app.debug 지우거나 False로 고쳐주기
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')