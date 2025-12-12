import os, gc
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score

import lightgbm as lgb
from catboost import CatBoostClassifier

from config import CONFIG
from data_utils import (
    set_seed,
    load_train_log_and_update_config,
    make_folds,
    load_all_train_lightcurves,
    build_basic_agg_feature,
)

def prepare_train_features(config: Dict) -> Tuple[pd.DataFrame, Dict, Dict]:
    set_seed(config["general"]["seed"])
    df_log, cls2idx, idx2cls, config = load_train_log_and_update_config(config)
    df_log = make_folds(config, df_log)
    train_lc = load_all_train_lightcurves(config)
    train_feats = build_basic_agg_feature(config, train_lc, df_log, is_train=True)
    return train_feats, cls2idx, idx2cls

def get_feature_target_fold_arrays(config: Dict, train_feats: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    id_col = config["train"]["id_col"]
    target_col = "target_idx"
    fold_col = "fold"

    used_cols = [c for c in train_feats.columns if c not in [id_col, target_col, fold_col]]
    X = train_feats[used_cols].values.astype(np.float32)
    y = train_feats[target_col].values.astype(int)
    folds = train_feats[fold_col].values.astype(int)

    return X, y, folds, np.array(used_cols)

def train_lgbm_cv(config: Dict, X: np.ndarray, y: np.ndarray, folds: np.ndarray, feature_names: np.ndarray,) -> Tuple[np.ndarray, Dict[int, lgb.Booster]]:
    model_conf = config["models"]["lgbm"]
    params = model_conf["params"].copy()
    n_splits = config["cv"]["n_splits"]

    num_class = params["num_class"]
    assert num_class is not None, "num_class is None, config update."

    oof_pred = np.zeros((len(y), num_class), dtype=np.float32)
    models = {}

    for fold in range(n_splits):
        print(f"\n========== LGBM FOLD {fold} ==========")
        trn_idx = np.where(folds != fold)[0]
        val_idx = np.where(folds == fold)[0]

        X_trn, y_trn = X[trn_idx], y[trn_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        lgb_train = lgb.Dataset(X_trn, label=y_trn)
        lgb_valid = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

        n_estimators = params.pop("n_estimators")
        early_stopping_rounds = params.pop("early_stopping_rounds", 100)

        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=n_estimators,
            valid_sets=[lgb_train,lgb_valid],
            valid_names=["train","valid"],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=100,
        )

        oof_pred[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        models[fold] = model
        fold_logloss = log_loss(y_val, oof_pred[val_idx])
        fold_acc = accuracy_score(y_val, oof_pred[val_idx].argmax(axis=1))
        print(f"[LGBM][FOLD {fold}] logloss: {fold_logloss:.6f}, acc: {fold_acc:.6f}")

        params["n_estimators"] = n_estimators
        params["early_stopping_rounds"] = early_stopping_rounds

        gc.collect()
    
    cv_logloss = log_loss(y, oof_pred)
    cv_acc = accuracy_score(y, oof_pred.argmax(axis=1))
    print(f"\n[LGBM][CV] logloss: {cv_logloss:.6f}, acc: {cv_acc:.6f}")

    return oof_pred, models

def train_catboost_cv(config: Dict, X: np.ndarray, y: np.ndarray, folds: np.ndarray,) -> Tuple[np.ndarray, Dict[int, CatBoostClassifier]]:
    model_conf = config["models"]["catboost"]
    params = model_conf["params"].copy()
    n_splits = config["cv"]["n_splits"]
    num_class = params.get("classes_count", None)
    oof_pred = np.zeros((len(y), len(np.unique(y))), dtype=np.float32)
    models = {}

    for fold in range(n_splits):
        print(f"\n========== CatBoost FOLD {fold} ==========")
        trn_idx = np.where(folds != fold)[0]
        val_idx = np.where(folds == fold)[0]

        X_trn, y_trn = X[trn_idx], y[trn_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model = CatBoostClassifier(**params)
        model.fit(X_trn, y_trn, eval_set=(X_val, y_val), use_best_model=True, verbose=100)
        proba_val = model.predict_proba(X_val)
        oof_pred[val_idx] = proba_val

        models[fold] = model
        fold_logloss = log_loss(y_val, proba_val)
        fold_acc = accuracy_score(y_val, proba_val.argmax(axis=1))
        print(f"[CAT][FOLD {fold}] logloss: {fold_logloss:.6f}, acc:{fold_acc:.6f}")

        gc.collect()
    
    cv_logloss = log_loss(y, oof_pred)
    cv_acc = accuracy_score(y, oof_pred.argmax(axis=1))
    print(f"\n[CAT][CV] logloss: {cv_logloss:.6f}, acc: {cv_acc:.6f}")

    return oof_pred, models

def save_oof_and_models(config: Dict, oof_lgbm: np.ndarray, oof_cat: np.ndarray, lgbm_models: Dict[int, lgb.Booster], cat_models: Dict[int, CatBoostClassifier]) -> None:
    out_dir = config["logging"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    if config["logging"].get("save_oof", True):
        np.save(os.path.join(out_dir, "oof_lgbm.npy"), oof_lgbm)
        np.save(os.path.join(out_dir, "oof_cat.npy"), oof_cat)
        print("[INFO] Saved OOF predictions.")
    
    for fold, model in lgbm_models.items():
        model_path = os.path.join(out_dir, f"lgbm_fold{fold}.txt")
        model.save_model(model_path)
    print("[INFO] Saved CatBoost models.")

def main():
    train_feats, cls2idx, idx2cls = prepare_train_features(CONFIG)
    X, y, folds, feature_names = get_feature_target_fold_arrays(CONFIG, train_feats)

    if CONFIG["models"]["lgbm"]["use"]:
        oof_lgbm, lgbm_models = train_lgbm_cv(CONFIG, X, y, folds, feature_names)
    else:
        oof_lgbm, lgbm_models = None, {}
    
    if CONFIG["models"]["catboost"]["use"]:
        oof_cat, cat_models = train_catboost_cv(CONFIG, X, y, folds)
    else:
        oof_cat, cat_models = None, {}
    
    save_oof_and_models(CONFIG, oof_lgbm, oof_cat, lgbm_models, cat_models)

if __name__ == "__main__":
    main()
