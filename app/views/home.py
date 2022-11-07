import os

import psycopg2
from flask import render_template

from app import app, User, db


@app.route('/')
def hello_world():
    return 'Hello World'

@app.route('/hey')
def hello_world2():

    users = User.query.all()
    print(users)

    res = db.engine.execute("SELECT * FROM user;")
    names = [row[0] for row in res]
    print(names)

    return render_template("hello.html")
