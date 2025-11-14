from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base , sessionmaker
import os
from dotenv import load_dotenv


load_dotenv()

DATABASE_URL = f'postgresql+psycopg2://{os.getenv("user")}:{os.getenv("password")}@{os.getenv("host")}:{os.getenv("port")}/{os.getenv("database")}'



engin=create_engine(DATABASE_URL)


SessionLocal=sessionmaker(autocommit=False,autoflush=False,bind=engin)

Base=declarative_base()

def get_db():
    db=SessionLocal()
    try:
        yield db
    finally:
        db.close()
        