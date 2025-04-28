from flask import Flask
from flask import *
from flaskext.mysql import MySQL
import os

app = Flask(__name__)

app.secret_key = "attentrack"
mysql = MySQL()
app.config['MYSQL_DATABASE_USER'] = 'root'  # Database user
app.config['MYSQL_DATABASE_PASSWORD'] = ''  # Database password
app.config['MYSQL_DATABASE_DB'] = 'attentrack'  # Name of database
app.config['MYSQL_DATABASE_HOST'] = 'localhost'  # Hosting site
app.config['UPLOAD_FOLDER'] = 'static/files'

mysql.init_app(app)


app=Flask(__name__)

#routes
#landing
@app.route('/')
def index():
    return render_template("index.html")

#teacher
@app.route('/user_home', methods=["POST", "GET"])
def user_home():
    return render_template("user/index.html")

#admin
@app.route('/admin_home', methods=["POST", "GET"])
def admin_home():
    return render_template("admin/index.html")

if __name__ == '__main__':
    app.run(debug=True)
    