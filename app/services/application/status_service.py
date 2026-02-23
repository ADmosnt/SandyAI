# app/services/application/status_service.py

from app.domain.messages import WELCOME_MESSAGE


def get_root_message():
    return WELCOME_MESSAGE
