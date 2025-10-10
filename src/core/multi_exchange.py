import os
from .ccxt_client import CcxtClient

def build_clients_from_env():
    ex_names = [e.strip() for e in os.getenv('EXCHANGES','').split(',') if e.strip()]
    clients = {}
    for name in ex_names:
        creds = {}
        up = name.upper()
        key = os.getenv(f'{up}_KEY')
        sec = os.getenv(f'{up}_SECRET')
        pwd = os.getenv(f'{up}_PASSWORD')
        if key and sec:
            creds |= {'apiKey': key, 'secret': sec}
        if pwd:
            creds |= {'password': pwd}
        clients[name] = CcxtClient(name, creds)
    return clients
