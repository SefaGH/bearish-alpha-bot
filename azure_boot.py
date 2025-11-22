import os
import sys
import subprocess
import time
import logging
from keep_alive import start_health_server

# Azure SDK (Hata almamak için try-except bloğu ile import edelim)
try:
    from azure.identity import DefaultAzureCredential
    from azure.keyvault.secrets import SecretClient
    from azure.core.exceptions import AzureError
    AZURE_SDK_AVAILABLE = True
except ImportError:
    AZURE_SDK_AVAILABLE = False

# Loglama Ayarları
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("azure_boot")

# 1. Sağlık Sunucusunu Başlat
log.info("🟢 Azure Health Check Sunucusu Başlatılıyor...")
start_health_server()

# 2. Key Vault Entegrasyonu (Plan A)
def load_secrets_from_keyvault(vault_name, secret_names):
    if not AZURE_SDK_AVAILABLE:
        log.warning("Azure SDK yüklü değil, Key Vault atlanıyor.")
        return

    if not vault_name:
        log.info("ℹ️ KEYVAULT_NAME tanımlı değil; .env veya App Settings kullanılacak.")
        return

    kv_uri = f"https://{vault_name}.vault.azure.net"
    try:
        # App Service Managed Identity'sini otomatik kullanır
        credential = DefaultAzureCredential()
        client = SecretClient(vault_url=kv_uri, credential=credential)
        log.info(f"🔐 Key Vault Bağlantısı Başarılı: {vault_name}")
    except Exception as e:
        log.error(f"❌ Key Vault bağlantı hatası: {e}")
        return

    for s in secret_names:
        # Eğer sistemde bu değişken zaten yoksa Key Vault'tan çekmeye çalış
        if os.getenv(s) is None:
            try:
                secret = client.get_secret(s)
                os.environ[s] = secret.value
                log.info(f"✅ Secret başarıyla yüklendi: {s}")
            except AzureError as ae:
                log.warning(f"⚠️ Secret okunamadı {s}: {ae}")

# Key Vault Konfigürasyonu
# Azure'da Environment Variable olarak KEYVAULT_NAME verirsek devreye girer.
KV_NAME = os.getenv("KEYVAULT_NAME")
# Çekilecek secret listesi (Virgülle ayrılmış)
SECRETS_TO_LOAD = os.getenv("KV_SECRETS", "KUCOIN_API_KEY,KUCOIN_API_SECRET").split(",")

if AZURE_SDK_AVAILABLE:
    load_secrets_from_keyvault(KV_NAME, SECRETS_TO_LOAD)

# 3. Ortam Değişkenleri ve Bot Ayarları
MODE = os.getenv("BOT_MODE", "paper")
DURATION = os.getenv("BOT_DURATION", "0")
ENABLE_ML = str(os.getenv("ENABLE_ML", "true")).lower() == "true"
DEBUG_MODE = str(os.getenv("DEBUG_MODE", "false")).lower() == "true"

# Başlatılacak asıl scriptin yolu
launcher_path = os.getenv("LAUNCHER_PATH", "scripts/live_trading_launcher.py")

# Komut setini hazırla
command_base = [sys.executable, launcher_path, "--mode", MODE, "--duration", DURATION]
if ENABLE_ML:
    command_base.append("--enable_ml")
if DEBUG_MODE:
    command_base.append("--debug_mode")

# 4. Akıllı Yeniden Başlatma Döngüsü
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
RETRY_BASE_SECONDS = int(os.getenv("RETRY_BASE_SECONDS", "5"))
attempt = 0

while True:
    attempt += 1
    log.info(f"🤖 Bot başlatılıyor (Deneme {attempt}/{MAX_RETRIES}) | Mod: {MODE}")
    
    try:
        proc = subprocess.Popen(
            command_base, 
            stdout=sys.stdout, 
            stderr=sys.stderr, 
            universal_newlines=True, 
            bufsize=1
        )
        exit_code = proc.wait()
        
        if exit_code == 0:
            log.info("✅ Bot başarıyla görevini tamamladı ve kapandı.")
            break
        
        log.error(f"⚠️ Bot kapandı! Hata Kodu: {exit_code}")

    except KeyboardInterrupt:
        log.info("🛑 Manuel durdurma.")
        try:
            proc.terminate()
        except:
            pass
        sys.exit(0)
    except Exception as e:
        log.exception(f"❌ Kritik Hata: {e}")

    if attempt >= MAX_RETRIES:
        log.critical("❌ Maksimum deneme sayısı aşıldı.")
        break

    backoff = RETRY_BASE_SECONDS * attempt
    log.info(f"⏳ {backoff} saniye bekleniyor...")
    time.sleep(backoff)