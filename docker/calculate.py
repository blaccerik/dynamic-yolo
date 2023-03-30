import json
import time
import os

from sqlalchemy import create_engine, Table, Integer, String, MetaData, Column, text, BigInteger, LargeBinary
from sqlalchemy.orm import sessionmaker, declarative_base

# create metadata
metadata = MetaData()

# define table
subset_table = Table(
    'image', metadata,
    Column("id", BigInteger, primary_key=True),
    # Column("image", LargeBinary, nullable=False),
)

Base = declarative_base()
class Subset(Base):
    __tablename__ = 'subset'

    id = Column(Integer, primary_key=True)
    name = Column(String(128), nullable=False)

def request():
    # create a SQLAlchemy engine
    # engine = create_engine('postgresql://erik:erik@yolo_postgres/yolo_db')
    engine = create_engine('postgresql://erik:erik@localhost/yolo_db')

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # # create a connection and execute a SELECT statement
    # with engine.connect() as conn:
    #     select_stmt = subset_table.select().execution_options(stream_results=True)
    #     cursor = conn.execution_options(stream_results=True).execute(select_stmt)
    #     while True:
    #         rows = cursor.fetchmany(10)
    #         if not rows:
    #             break
    #         for row in rows:
    #             print(row)

    # # Create a new session
    # db = SessionLocal()
    #
    # # Create a new subset object
    # new_subset = Subset(name='New Subset')
    #
    # # Add the subset object to the session
    # db.add(new_subset)
    #
    # # Commit the transaction to the database
    # db.commit()
    #
    # # Close the session
    # db.close()


    # db = SessionLocal()
    #
    # images = Subset.query
    #
    # for image in images.yield_per(10):
    #     print(image)

if __name__ == '__main__':
    print("start")
    time.sleep(0.5)
    print("end")
    result = 42
    request()
