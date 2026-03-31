# MLOps Lab — Cyberbullying Tweet Detection

Повний цикл MLOps проєкту для виявлення кібербулінгу у Twitter-повідомленнях за допомогою `RandomForest` + `TF-IDF`.

В рамках проєкту були успішно реалізовані наступні етапи:

- ✔ **Лабораторна робота №1:** Організація робочого простору MLOps та автоматизація відстеження експериментів (MLflow).
- ✔ **Лабораторна робота №2:** Версіонування даних та побудова пайплайнів відтворюваності (DVC).
- ✔ **Лабораторна робота №3:** Гіперпараметрична оптимізація та оркестрація ML-пайплайнів з Optuna та конфігураціями Hydra.
- ✔ **Лабораторна робота №4:** CI/CD для ML-проєктів: автоматизація тестування коду, даних, та генерація звітності (CML) через GitHub Actions.
- ✔ **Лабораторна робота №5:** Оркестрація ML-пайплайнів та Continuous Training за допомогою Apache Airflow і багатоетапних Docker-збірок.

## Опис

**Завдання:** Бінарна класифікація (є/немає hate speech).
**Датасет:** [Twitter Sentiment Analysis (Hate Speech)](https://www.kaggle.com/datasets/arkhoshghalb/twitter-sentiment-analysis-hatred-speech) — 31962 твіти (сильний дисбаланс класів: 93%/7%). В рамках пайплайну застосовується NLTK для очищення та SMOTE для балансування мінорного класу.

## Структура репозиторію

```text
mlops_lab_1/
├── .github/workflows/  # CI/CD пайплайни (GitHub Actions, CML)
├── config/             # Hydra конфігурації (hpo, model, data)
├── dags/               # Apache Airflow DAGs
├── data/               # Сирі (raw) та підготовлені (prepared) дані
├── notebooks/          # EDA та тестування гіпотез
├── reports/            # Звіти по лабораторним та json/csv артефакти
├── src/                # Основний код (train, optimize, feature extraction)
├── tests/              # Pytest: data validation, model Quality Gate, DAG tests
├── Dockerfile          # Multi-stage образ для Airflow та ML
├── docker-compose.yaml # Налаштування оркестратора
├── dvc.yaml            # Data Version Control pipeline
├── requirements.txt    # Залежності проєкту
├── README.md           # Опис проєкту
└── RUN.md              # ⚡ ІНСТРУКЦІЇ З ЗАПУСКУ ЛР 3-5
```

## Інструкції з тестування та запуску

Всі необхідні деталі для розгортання та перевірки фінальних етапів (Optuna, Airflow під Docker, тести Pytest) знаходяться у спеціальному файлі:

👉 **[Див. RUN.md](./RUN.md)**

## Короткий довідник (Legacy)

**Ініціалізація оточення:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Регістр експериментів (MLflow UI):**
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

**Швидкий запуск пайплайну (DVC):**
```bash
dvc repro
```
