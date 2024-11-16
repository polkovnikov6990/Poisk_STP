# Используем базовый образ Python
FROM python:3.11-slim

# Установка необходимых пакетов
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Установка зависимостей Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода в контейнер
COPY . /app
WORKDIR /app

# Запуск приложения
CMD ["python", "app.py"]
