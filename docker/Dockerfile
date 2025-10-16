FROM python:3.11-slim  # 3.12 yerine 3.11
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "-u", "src/main.py"]
