import requests

class Telegram:
    def __init__(self, token: str, chat_id: str):
        self.base = f"https://api.telegram.org/bot{token}/sendMessage"
        self.chat_id = chat_id

    def send(self, text: str):
        try:
            requests.post(self.base, json={"chat_id": self.chat_id, "text": text, "parse_mode":"HTML"}, timeout=8)
        except Exception:
            pass
