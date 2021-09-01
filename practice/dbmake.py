import pymysql

db = pymysql.connect(
    host = 'mydatabase.cr7yob8emqao.us-east-2.rds.amazonaws.com',
    port = 3306,
    user = 'admin',
    passwd = 'altpaltp12!',
    db = 'subject'
)

sql = '''
    CREATE TABLE `programming` (
        `name` VARCHAR(25) NOT NULL
    ) ENGINE=innoDB DEFAULT CHARSET=utf8;
'''

sql_1 = "INSERT INTO `programming` (`name`) VALUES ('C++');"

cursor = db.cursor()

cursor.execute(sql_1)

db.commit()
db.close()