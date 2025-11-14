from sqlalchemy import Column , Integer , String , Float, DateTime
from src.database import Base
from datetime import datetime


class EmotionRecord(Base):
    __tablename__ = "emotion_records"

    id = Column(Integer, primary_key=True, index=True)
    emotion = Column(String)
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

