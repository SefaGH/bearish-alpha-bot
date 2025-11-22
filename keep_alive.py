import os
from flask import Flask
from threading import Thread
import logging

# Flask loglarını sustur
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask('')

@app.route('/')
def home():
    return "Bearish Alpha Bot is Running! 🚀", 200

@app.route('/health')
def health():
    return "OK", 200

def run():
    # Azure portu environment variable'dan okur, yoksa 8000 kullanır
    port = int(os.getenv("PORT", os.getenv("WEBSITES_PORT", "8000")))
    app.run(host='0.0.0.0', port=port)

def start_health_server():
    t = Thread(target=run)
    t.daemon = True
    t.start()