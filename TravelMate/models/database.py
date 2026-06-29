from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

DATABASE_URL = "sqlite:///travelmate.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Trip(Base):
    __tablename__ = "trips"
    id = Column(String, primary_key=True)
    destination = Column(String)
    days = Column(Integer)
    budget = Column(Float)
    style = Column(String)
    plan_json = Column(Text)
    status = Column(String, default="draft")  # draft / confirmed / completed
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Expense(Base):
    __tablename__ = "expenses"
    id = Column(String, primary_key=True)
    trip_id = Column(String)
    payer = Column(String)
    item = Column(String)
    amount = Column(Float)
    currency = Column(String, default="CNY")
    category = Column(String, default="other")
    note = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)


class UserPreference(Base):
    __tablename__ = "user_preferences"
    id = Column(String, primary_key=True)
    key = Column(String, unique=True)
    value = Column(Text)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SavedItinerary(Base):
    __tablename__ = "saved_itineraries"
    id = Column(String, primary_key=True)
    trip_id = Column(String)
    title = Column(String)
    content_json = Column(Text)
    exported_as = Column(String, default="")
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)
