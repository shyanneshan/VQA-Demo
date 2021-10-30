import mysql.connector
import pymysql
import pandas

def insert_ct_info(patient_id, sym, photo_id, dia_list, annotation, dataset):
    mydb = mysql.connector.connect(user='root', password='3822186',
                                   host='127.0.0.1',
                                   database='vqademo')
    # cursor.execute("CREATE TABLE sites (name VARCHAR(255), url VARCHAR(255))")
    mycursor = mydb.cursor()

    sql = "INSERT INTO ct_information (id, patient_id, sym, photo_id, dia_list, annotation, status, dataset) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
    val = (0, patient_id, sym, photo_id, dia_list,annotation, "0", dataset,)
    mycursor.execute(sql, val)

    mydb.commit()  # 数据表内容有更新，必须使用到该语句

    print(mycursor.rowcount, "success")

def getPro(datasetname):
    mydb = mysql.connector.connect(user='root', password='3822186',
                                   host='127.0.0.1',
                                   database='vqademo')
    # cursor.execute("CREATE TABLE sites (name VARCHAR(255), url VARCHAR(255))")
    mycursor = mydb.cursor()
    sql = "SELECT train,valid,test FROM dataset WHERE name = %s"
    na = (datasetname,)
    mycursor.execute(sql, na)

    myresult = mycursor.fetchall()
    print( myresult )
    return myresult[0][0],myresult[0][1],myresult[0][2]


def setDatasetStatus(dataset,path):
    mydb = mysql.connector.connect(user='root', password='3822186',
                                   host='127.0.0.1',
                                   database='vqademo')
    # cursor.execute("CREATE TABLE sites (name VARCHAR(255), url VARCHAR(255))")
    mycursor = mydb.cursor()

    sql = "UPDATE dataset SET status = '1' where name = %s"
    sql2 =  "UPDATE dataset SET link = %s where name = %s"
    val = (dataset,)
    val2 = (path,dataset,)
    mycursor.execute(sql, val)
    mycursor.execute(sql2, val2)

    mydb.commit()  # 数据表内容有更新，必须使用到该语句


class TestMysql(object):
    #运行数据库和建立游标对象
    def __init__(self):
        self.connect = pymysql.connect(host="127.0.0.1", port=3306, user="root", password="3822186", database="vqademo",
                                  charset="utf8")
        # 返回一个cursor对象,也就是游标对象
        self.cursor = self.connect.cursor(cursor=pymysql.cursors.DictCursor)
    #关闭数据库和游标对象
    # def __del__(self):
    #     self.connect.close()
    #     self.cursor.close()
    def write(self, csv_path, dataset):
        #将数据转化成DataFrame数据格式
        data = pandas.DataFrame(self.read())

        data = data[data['dataset']==dataset]
        #把id设置成行索引
        #写写入数据数据
        pandas.DataFrame.to_csv(data[1:],csv_path+".csv",header=None,index=False,encoding="utf_8_sig")
        print("写入成功")
        return csv_path+".csv"
    def read(self):
        #读取数据库的所有数据
        data = self.cursor.execute("""select * from ct_validation;""")
        field_2 = self.cursor.fetchall()
        # pprint(field_2)
        return field_2









