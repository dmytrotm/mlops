"""
src/optimize.py
Гіперпараметрична оптимізація (ЛР3).
Використовує Optuna + MLflow (nested runs) + Hydra конфігурацію.
Датасет: Twitter Hate Speech (підготовлений у data/prepared/).
"""

import json
import logging
import os
import random
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import hydra
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from scipy.sparse import hstack, csr_matrix
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Заглушуємо зайві логи Optuna і MLflow
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger("mlflow").setLevel(logging.WARNING)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Seed / відтворюваність
# ---------------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Завантаження та побудова feature matrix
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = ["num_hashtags", "num_mentions", "tweet_len", "sentiment_score"]


def load_data(cfg: DictConfig) -> Tuple[Any, Any, np.ndarray, np.ndarray, TfidfVectorizer]:
    """
    Завантажує data/prepared/train.csv, будує feature matrix (numeric + TF-IDF),
    повертає X_train, X_test, y_train, y_test, tfidf.
    """
    path = cfg.data.processed_path
    df = pd.read_csv(path)
    log.info(f"Завантажено {len(df)} рядків із {path}")

    label_col: str = cfg.data.label_col
    text_col: str = cfg.data.text_col
    numeric_features: List[str] = list(cfg.data.numeric_features)

    df[text_col] = df[text_col].fillna("")
    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    X_numeric = csr_matrix(df[numeric_features].values.astype(float))

    # TF-IDF з біграмами для кращого вловлювання контексту
    ngram_max = cfg.data.get("tfidf_ngram_max", 2)
    tfidf = TfidfVectorizer(
        max_features=cfg.data.tfidf_max_features,
        stop_words="english",
        ngram_range=(1, ngram_max),
        sublinear_tf=True,           # log(1+tf) — кращий для тексту
        min_df=2,                     # ігнорувати дуже рідкісні терми
    )
    X_tfidf = tfidf.fit_transform(df[text_col])

    X = hstack([X_numeric, X_tfidf])
    y = df[label_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.data.test_size,
        random_state=cfg.seed,
        stratify=y,
    )
    log.info(f"Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}  |  Features: {X.shape[1]}")

    # --- SMOTE для балансування класів ---
    if cfg.data.get("use_smote", False):
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=cfg.seed)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        log.info(f"Після SMOTE: Train: {X_train.shape[0]}  |  "
                 f"Class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")

    return X_train, X_test, y_train, y_test, tfidf


# ---------------------------------------------------------------------------
# Побудова моделі
# ---------------------------------------------------------------------------

def build_model(model_type: str, params: Dict[str, Any], seed: int) -> Any:
    if model_type == "random_forest":
        return RandomForestClassifier(
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced",
            **params,
        )
    if model_type == "logistic_regression":
        # Перевіримо сумісність solver/penalty
        solver = params.get("solver", "lbfgs")
        penalty = params.get("penalty", "l2")
        if penalty == "l1" and solver not in ("liblinear", "saga"):
            solver = "liblinear"
        clf = LogisticRegression(
            random_state=seed,
            max_iter=1000,
            solver=solver,
            penalty=penalty,
            C=params.get("C", 1.0),
            class_weight="balanced",
        )
        return Pipeline([("scaler", StandardScaler(with_mean=False)), ("clf", clf)])
    if model_type == "gradient_boosting":
        return GradientBoostingClassifier(
            random_state=seed,
            **params,
        )
    raise ValueError(f"Невідомий model.type='{model_type}'.")


# ---------------------------------------------------------------------------
# Оцінка
# ---------------------------------------------------------------------------

def _score(model, X_tr, y_tr, X_te, y_te, metric: str) -> float:
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    if metric == "f1":
        return float(f1_score(y_te, y_pred, average="binary", zero_division=0))
    if metric == "roc_auc":
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_te)[:, 1]
        else:
            y_score = model.decision_function(X_te)
        return float(roc_auc_score(y_te, y_score))
    raise ValueError(f"Непідтримувана метрика: {metric}")


def evaluate(model, X_train, y_train, X_test, y_test, metric: str) -> float:
    return _score(clone(model), X_train, y_train, X_test, y_test, metric)


def evaluate_cv(model, X, y, metric: str, seed: int, n_splits: int = 5) -> float:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for tr_idx, te_idx in cv.split(X, y):
        m = clone(model)
        scores.append(_score(m, X[tr_idx], y[tr_idx], X[te_idx], y[te_idx], metric))
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

def make_sampler(sampler_name: str, seed: int, grid_space: Optional[Dict] = None):
    name = sampler_name.lower()
    if name == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if name == "grid":
        if not grid_space:
            raise ValueError("Для sampler='grid' потрібно вказати grid_space.")
        return optuna.samplers.GridSampler(search_space=grid_space)
    raise ValueError(f"Невідомий sampler: {sampler_name}. Підтримуються: tpe, random, grid.")


# ---------------------------------------------------------------------------
# Простір пошуку (suggest_params) — розширений
# ---------------------------------------------------------------------------

def suggest_params(trial: optuna.Trial, model_type: str, cfg: DictConfig) -> Dict[str, Any]:
    if model_type == "random_forest":
        space = cfg.hpo.random_forest
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", space.n_estimators.low, space.n_estimators.high),
            "max_depth":         trial.suggest_int("max_depth", space.max_depth.low, space.max_depth.high),
            "min_samples_split": trial.suggest_int("min_samples_split", space.min_samples_split.low, space.min_samples_split.high),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", space.min_samples_leaf.low, space.min_samples_leaf.high),
        }
        # Додаткові параметри (якщо задані в конфігурації)
        if "max_features" in space:
            params["max_features"] = trial.suggest_categorical("max_features", list(space.max_features))
        if "criterion" in space:
            params["criterion"] = trial.suggest_categorical("criterion", list(space.criterion))
        return params

    if model_type == "logistic_regression":
        space = cfg.hpo.logistic_regression
        return {
            "C":       trial.suggest_float("C", space.C.low, space.C.high, log=True),
            "solver":  trial.suggest_categorical("solver", list(space.solver)),
            "penalty": trial.suggest_categorical("penalty", list(space.penalty)),
        }

    if model_type == "gradient_boosting":
        space = cfg.hpo.gradient_boosting
        return {
            "n_estimators":    trial.suggest_int("n_estimators", space.n_estimators.low, space.n_estimators.high),
            "max_depth":       trial.suggest_int("max_depth", space.max_depth.low, space.max_depth.high),
            "learning_rate":   trial.suggest_float("learning_rate", space.learning_rate.low, space.learning_rate.high, log=True),
            "subsample":       trial.suggest_float("subsample", space.subsample.low, space.subsample.high),
            "min_samples_split": trial.suggest_int("min_samples_split", space.min_samples_split.low, space.min_samples_split.high),
        }

    raise ValueError(f"Невідомий model.type='{model_type}'.")


# ---------------------------------------------------------------------------
# Objective factory
# ---------------------------------------------------------------------------

def objective_factory(cfg: DictConfig, X_train, X_test, y_train, y_test):
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, cfg.model.type, cfg)

        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number:03d}"):
            mlflow.set_tag("trial_number", str(trial.number))
            mlflow.set_tag("model_type", cfg.model.type)
            mlflow.set_tag("sampler", cfg.hpo.sampler)
            mlflow.set_tag("seed", str(cfg.seed))
            mlflow.set_tag("use_smote", str(cfg.data.get("use_smote", False)))
            mlflow.log_params(params)

            model = build_model(cfg.model.type, params=params, seed=cfg.seed)

            if cfg.hpo.use_cv:
                from scipy.sparse import vstack as sp_vstack
                X_all = sp_vstack([X_train, X_test])
                y_all = np.concatenate([y_train, y_test])
                score = evaluate_cv(model, X_all, y_all,
                                    metric=cfg.hpo.metric,
                                    seed=cfg.seed,
                                    n_splits=cfg.hpo.cv_folds)
            else:
                score = evaluate(model, X_train, y_train, X_test, y_test,
                                 metric=cfg.hpo.metric)

            mlflow.log_metric(cfg.hpo.metric, score)
            log.info(f"Trial {trial.number}: {cfg.hpo.metric}={score:.4f}  params={params}")
        return score

    return objective


# ---------------------------------------------------------------------------
# Реєстрація моделі в MLflow Registry (опційно)
# ---------------------------------------------------------------------------

def register_model_if_enabled(model_uri: str, cfg: DictConfig) -> None:
    if not cfg.mlflow.register_model:
        return
    client = mlflow.tracking.MlflowClient()
    mv = mlflow.register_model(model_uri, cfg.mlflow.model_name)
    client.transition_model_version_stage(
        name=cfg.mlflow.model_name,
        version=mv.version,
        stage=cfg.mlflow.stage,
    )
    client.set_model_version_tag(cfg.mlflow.model_name, mv.version, "registered_by", "lab3")
    log.info(f"Модель зареєстрована: {cfg.mlflow.model_name} v{mv.version} → {cfg.mlflow.stage}")


# ---------------------------------------------------------------------------
# Допоміжна: отримати git commit hash
# ---------------------------------------------------------------------------

def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main(cfg: DictConfig) -> None:
    set_global_seed(cfg.seed)
    log.info(f"Конфігурація:\n{OmegaConf.to_yaml(cfg)}")

    # --- MLflow setup ---
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    # --- Дані ---
    X_train, X_test, y_train, y_test, tfidf = load_data(cfg)

    # --- Grid space (тільки для GridSampler) ---
    grid_space = None
    if cfg.hpo.sampler.lower() == "grid":
        if cfg.model.type == "random_forest":
            g = cfg.hpo.grid.random_forest
            grid_space = {
                "n_estimators":      list(g.n_estimators),
                "max_depth":         list(g.max_depth),
                "min_samples_split": list(g.min_samples_split),
                "min_samples_leaf":  list(g.min_samples_leaf),
            }
        elif cfg.model.type == "logistic_regression":
            g = cfg.hpo.grid.logistic_regression
            grid_space = {
                "C":       list(g.C),
                "solver":  list(g.solver),
                "penalty": list(g.penalty),
            }

    sampler = make_sampler(cfg.hpo.sampler, seed=cfg.seed, grid_space=grid_space)
    git_commit = get_git_commit()

    # --- Parent MLflow run ---
    parent_run_name = f"hpo_{cfg.hpo.sampler}_{cfg.model.type}"
    with mlflow.start_run(run_name=parent_run_name) as parent_run:
        mlflow.set_tag("model_type", cfg.model.type)
        mlflow.set_tag("sampler", cfg.hpo.sampler)
        mlflow.set_tag("seed", str(cfg.seed))
        mlflow.set_tag("git_commit", git_commit)
        mlflow.set_tag("n_trials", str(cfg.hpo.n_trials))
        mlflow.set_tag("use_cv", str(cfg.hpo.use_cv))
        mlflow.set_tag("use_smote", str(cfg.data.get("use_smote", False)))
        mlflow.set_tag("tfidf_max_features", str(cfg.data.tfidf_max_features))

        # Зберігаємо конфігурацію як артефакт
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        mlflow.log_dict(cfg_dict, "config_resolved.json")

        # --- Optuna Study ---
        study = optuna.create_study(
            direction=cfg.hpo.direction,
            sampler=sampler,
            study_name=parent_run_name,
        )
        objective = objective_factory(cfg, X_train, X_test, y_train, y_test)
        study.optimize(objective, n_trials=cfg.hpo.n_trials, show_progress_bar=False)

        # --- Результати study ---
        best_trial = study.best_trial
        best_value = float(best_trial.value)
        best_params = dict(best_trial.params)

        log.info(f"Найкращий trial #{best_trial.number}: {cfg.hpo.metric}={best_value:.4f}")
        log.info(f"Найкращі параметри: {best_params}")

        mlflow.log_metric(f"best_{cfg.hpo.metric}", best_value)
        mlflow.log_dict(best_params, "best_params.json")

        # Зберегти trials як CSV артефакт
        trials_df = study.trials_dataframe()
        os.makedirs("reports", exist_ok=True)
        trials_csv = f"reports/trials_{cfg.hpo.sampler}_{cfg.model.type}.csv"
        trials_df.to_csv(trials_csv, index=False)
        mlflow.log_artifact(trials_csv)

        # --- Фінальна модель (retrain на повних даних) ---
        best_model = build_model(cfg.model.type, params=best_params, seed=cfg.seed)
        best_model.fit(X_train, y_train)

        final_score = evaluate(best_model, X_train, y_train, X_test, y_test,
                               metric=cfg.hpo.metric)
        mlflow.log_metric(f"final_{cfg.hpo.metric}", final_score)
        log.info(f"Фінальна модель: {cfg.hpo.metric}={final_score:.4f}")

        # Зберегти модель локально і в MLflow
        os.makedirs("models", exist_ok=True)
        model_path = "models/best_model.pkl"
        joblib.dump(best_model, model_path)
        mlflow.log_artifact(model_path)

        if cfg.mlflow.log_model:
            mlflow.sklearn.log_model(best_model, artifact_path="model")

        # Зберегти загальний summary в reports/
        summary = {
            "sampler": cfg.hpo.sampler,
            "model_type": cfg.model.type,
            "n_trials": cfg.hpo.n_trials,
            "metric": cfg.hpo.metric,
            "best_trial_number": best_trial.number,
            "best_value": best_value,
            "final_value": final_score,
            "best_params": best_params,
            "seed": cfg.seed,
            "git_commit": git_commit,
            "use_smote": cfg.data.get("use_smote", False),
            "tfidf_max_features": cfg.data.tfidf_max_features,
        }
        summary_path = f"reports/summary_{cfg.hpo.sampler}_{cfg.model.type}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        mlflow.log_artifact(summary_path)

        # Реєстрація в MLflow Registry (якщо увімкнено)
        if cfg.mlflow.log_model and cfg.mlflow.register_model:
            model_uri = f"runs:/{parent_run.info.run_id}/model"
            register_model_if_enabled(model_uri, cfg)

    print(f"\n{'='*60}")
    print(f"HPO завершено: {cfg.hpo.sampler.upper()} | {cfg.model.type}")
    print(f"  Кількість trials:   {cfg.hpo.n_trials}")
    print(f"  Метрика:            {cfg.hpo.metric}")
    print(f"  Найкраще значення:  {best_value:.4f}  (trial #{best_trial.number})")
    print(f"  Фінальна метрика:   {final_score:.4f}")
    print(f"  Найкращі параметри: {best_params}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../config", config_name="config")
def hydra_entry(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    hydra_entry()
