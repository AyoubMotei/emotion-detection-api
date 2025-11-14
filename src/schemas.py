from pydantic import BaseModel
from datetime import datetime



class Emotion(BaseModel):
    emotion : str
    confidence : float
    created_at : datetime
    
    class Config:
        orm_mode = True
        
    
class EmotionResponse(Emotion):
    id : int
