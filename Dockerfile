# Base Image - Python 3.11'e güncellendi
FROM python:3.11-slim

# Çalışma dizini
WORKDIR /app

# Sistem bağımlılıklarını kur (ML kütüphaneleri için libgomp1 şart)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libc-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python bağımlılıklarını kur
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Proje dosyalarını kopyala
COPY . .

# Azure'a portu bildir
EXPOSE 8000

# Performans ve Log Ayarları
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Başlatıcıyı çalıştır
CMD ["python", "azure_boot.py"]