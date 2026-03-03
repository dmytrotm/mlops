# MLOps Lab — Cyberbullying Tweet Detection

✔ Лабораторна робота №1: Організація робочого простору MLOps та автоматизація відстеження експериментів.

Лабораторна робота №2: Версіонування даних та побудова пайплайнів (DVC)

Лабораторна робота №3: Гіперпараметрична оптимізація та оркестрація ML-пайплайнів з Optuna

Лабораторна робота №4: CI/CD для ML-проєктів: автоматизація тестування та звітності з GitHub Actions та CML

Лабораторна робота №5: Оркестрація ML-пайплайнів: Від CI/CD до Continuous Training

## Опис

Проєкт виявлення кібербулінгу у Twitter-повідомленнях за допомогою RandomForest + TF-IDF. Використовується MLflow для відстеження експериментів.

**Датасет:** [Twitter Sentiment Analysis (Hate Speech)](https://www.kaggle.com/datasets/arkhoshghalb/twitter-sentiment-analysis-hatred-speech) — 31962 твіти, бінарна класифікація (93%/7% дисбаланс).

## Структура проєкту

```
mlops_lab_1/
├── .gitignore
├── requirements.txt
├── README.md
├── data/
│   └── raw/            # Сирі дані (не в Git)
│       ├── train.csv
│       └── test.csv
├── notebooks/
│   └── 01_eda.ipynb    # EDA ноутбук
├── src/
│   └── train.py        # Скрипт навчання
├── models/             # Збережені моделі (не в Git)
└── mlruns/             # MLflow логи (не в Git)
```

## Встановлення

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Використання

```bash
# Запуск навчання (базовий)
python src/train.py --max_depth 10

# З SMOTE для балансування класів
python src/train.py --max_depth 10 --smote

# З іншими гіперпараметрами
python src/train.py --n_estimators 200 --max_depth 20

# MLflow UI
mlflow ui
```
