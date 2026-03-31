# Stage 1: Builder
# Встановлюємо важкі залежності та збираємо wheels
FROM apache/airflow:2.9.2-python3.11 AS builder

USER root
# Встановлення build-essential для компіляції деяких бібліотек
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

USER airflow
COPY requirements.txt /tmp/requirements.txt
# Встановлюємо Python пакети локально (в ~/.local)
RUN pip install --user --no-cache-dir -r /tmp/requirements.txt


# Stage 2: Final
# Отримуємо легковаговий фінальний образ
FROM apache/airflow:2.9.2-python3.11

# Копіюємо встановлені пакети з Stage 1
COPY --from=builder /home/airflow/.local /home/airflow/.local

# Переконуємось, що ~/.local/bin у PATH
ENV PATH=/home/airflow/.local/bin:$PATH

# Робоча директорія (для Airflow)
WORKDIR /opt/airflow

# Вказуємо змінні середовища
ENV PYTHONPATH="/opt/airflow/src:${PYTHONPATH}"
