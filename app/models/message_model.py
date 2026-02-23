# app/models/message_model.py

from pydantic import BaseModel


class MessageModel(BaseModel):
    message: str
