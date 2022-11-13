from flask import render_template
from flask_sqlalchemy import SQLAlchemy
from app import app

if __name__ == '__main__':
    app.run(debug=True)
