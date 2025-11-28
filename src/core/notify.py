import requests
import aiohttp
import asyncio
import logging

logger = logging.getLogger(__name__)

class Telegram:
    def __init__(self, token: str, chat_id: str):
        self.base = f"https://api.telegram.org/bot{token}/sendMessage"
        self.chat_id = chat_id

    def send(self, text: str):
        """Synchronous send using requests (blocking)"""
        try:
            requests.post(self.base, json={"chat_id": self.chat_id, "text": text, "parse_mode":"HTML"}, timeout=10)
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")

    async def send_async(self, text: str):
        """Asynchronous send using aiohttp (non-blocking)"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"}
                async with session.post(self.base, json=payload, timeout=10) as resp:
                    if resp.status != 200:
                        logger.error(f"Telegram send_async failed: {resp.status} {await resp.text()}")
        except Exception as e:
            logger.error(f"Telegram send_async error: {e}")
