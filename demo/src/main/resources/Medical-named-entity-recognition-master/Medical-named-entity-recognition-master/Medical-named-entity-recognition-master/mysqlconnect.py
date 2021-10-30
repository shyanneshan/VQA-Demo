#DataSourceSettings#
#LocalDataSource: root@localhost  3306
#BEGIN#
#<data-source source="LOCAL" name="root@localhost" uuid="76382d5a-2588-45f3-9e31-bbbd0aedce2f"><database-info product="MySQL" version="8.0.26-0ubuntu0.20.04.2" jdbc-version="4.2" driver-name="MySQL Connector/J" driver-version="mysql-connector-java-8.0.21 (Revision: 33f65445a1bcc544eb0120491926484da168f199)" dbms="MYSQL" exact-version="8.0.26" exact-driver-version="8.0"><extra-name-characters>#@</extra-name-characters><identifier-quote-string>`</identifier-quote-string></database-info><case-sensitivity plain-identifiers="exact" quoted-identifiers="exact"/><driver-ref>mysql.8</driver-ref><synchronize>true</synchronize><jdbc-driver>com.mysql.cj.jdbc.Driver</jdbc-driver><jdbc-url>jdbc:mysql://localhost:3306</jdbc-url><secret-storage>master_key</secret-storage><user-name>root</user-name><schema-mapping><introspection-scope><node kind="schema" qname="vqademo"/></introspection-scope></schema-mapping><working-dir>$ProjectFileDir$</working-dir></data-source>
#END#

import mysql.connector

def insert_ct_info(patient_id, sym, photo_id, dia_list, annotation, dataset):
    mydb = mysql.connector.connect(user='root', password='123456',
                                   host='127.0.0.1',
                                   database='vqademo')
    # cursor.execute("CREATE TABLE sites (name VARCHAR(255), url VARCHAR(255))")
    mycursor = mydb.cursor()

    sql = "INSERT INTO ct_information (patient_id, sym, photo_id, dia_list, annotation, status, dataset) VALUES (%s, %s, %s, %s, %s, %s, %s)"
    val = (patient_id, sym, photo_id, dia_list,annotation, "0", dataset,)
    mycursor.execute(sql, val)

    mydb.commit()  # 数据表内容有更新，必须使用到该语句

    print(mycursor.rowcount, "success")