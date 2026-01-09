import requests
import aiohttp
import asyncio
import logging

logger = logging.getLogger(__name__)

class Telegram:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.base = f"https://api.telegram.org/bot{token}/sendMessage"
        self.base_api = f"https://api.telegram.org/bot{token}"
        self.chat_id = chat_id

    def send(self, text: str) -> bool:
        """Synchronous send using requests (blocking)"""
        try:
            resp = requests.post(
                self.base,
                json={
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
                timeout=10,
            )
            if resp.status_code != 200:
                logger.error(
                    "Telegram send failed: status=%s body=%s",
                    resp.status_code,
                    resp.text[:500],
                )
                return False
            return True
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False

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

    def get_me(self) -> dict:
        """Best-effort auth check against Telegram Bot API (no message send)."""
        try:
            resp = requests.get(f"{self.base_api}/getMe", timeout=10)
            if resp.status_code != 200:
                logger.error(
                    "Telegram getMe failed: status=%s body=%s",
                    resp.status_code,
                    resp.text[:500],
                )
                return {"ok": False, "status": resp.status_code, "body": resp.text}
            return resp.json()
        except Exception as exc:
            logger.error("Telegram getMe error: %s", exc)
            return {"ok": False, "error": str(exc)}
