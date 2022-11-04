from flask import render_template
from app import app

@app.route('/')
def hello_world():
    return 'Hello World'

@app.route('/hey')
def hello_world2():
    print("3dsdsdhj")
    return render_template("hello.html")
