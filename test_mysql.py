import pymysql

try:
    conn = pymysql.connect(
        host="127.0.0.1",
        user="root",
        password="Password@123",
        database="brandusers"
    )

    print("Connected successfully")

except Exception as e:
    print("Error:", e)