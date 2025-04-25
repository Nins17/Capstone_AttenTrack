from flask import Flask
from flask import *
from flaskext.mysql import MySQL
import os


app=Flask(__name__)

@app.route('/')
def index():
    return render_template("pages/index.html")

if __name__ == '__main__':
    app.run(debug=True)
    