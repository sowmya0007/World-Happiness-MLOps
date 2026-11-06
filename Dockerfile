# Base image ga Python 3.12 ni theesukuntunnam
FROM python:3.12-slim

# Container lo mana project kosam /app ani oka folder create chesthunnam
WORKDIR /app

# Mana laptop nunchi requirements.txt ni container loki copy chesthunnam
COPY requirements.txt .

# aa file lo unna libraries anni container lo install chesthunnam
RUN pip install --no-cache-dir -r requirements.txt

# Migilina files annitini container loki copy chesthunnam
COPY . .

# API 8000 port lo run avuthundi ani chepthunnam
EXPOSE 8000

# Container start ayinappudu API ni run chese command
CMD ["uvicorn", "api.lab11_main:app", "--host", "0.0.0.0", "--port", "8000"]