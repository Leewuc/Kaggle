import os
from typing import Dict, Tuple
import random
import torch
import numpy as np
import pandas as pd 
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

from config import CONFIG

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def load_train_log_and_update_config(config: Dict) -> Tuple[pd.DataFrame, Dict, Dict, Dict]:
    data_conf = config["data"]
    train_log_path = os.path.join(
        data_conf["input_dir"],
        data_conf["train_log_filename"]
    )
    df_log = pd.read_csv(train_log_path)
    id_col = data_conf["id_col"]
    target_col = data_conf["target_col"]

    class_labels = sorted(df_log[target_col].unique())
    num_class = len(class_labels)

    cls2idx = {label: idx for idx, label in enumerate(class_labels)}
    idx2cls = {idx: label for label, idx in cls2idx.items()}

    df_log["target_idx"] = df_log[target_col].map(cls2idx)

    config["models"]["lgbm"]["params"]["num_class"] = num_class
    config["models"]["seq_cnn_lstm"]["model"]["num_classes"] = num_class

    print(f"[INFO] num_class: {num_class}")
    print(f"[INFO] class labels: {class_labels}")

    return df_log, cls2idx, idx2cls, config

def load_all_train_lightcurves(config: Dict) -> pd.DataFrame:
    data_conf = config["data"]
    root = data_conf["input_dir"]
    split_names = data_conf["split_names"]
    lc_filename = data_conf["train_lc_filename"]
    split_col = data_conf["split_col"]

    dfs = []
    for split_name in split_names:
        path = os.path.join(root, split_name, lc_filename)
        if not os.path.exists(path):
            print(f"[WARNING] {path} does not exist. Skipping.")
            continue

        df_split = pd.read_csv(path)

        if split_col not in df_split.columns:
            df_split[split_col] = split_name
        
        dfs.append(df_split)
    
    train_lc = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] Loaded train lightcurves: {train_lc.shape}")
    return train_lc

def load_all_test_lightcurves(config: Dict) -> pd.DataFrame:
    data_conf = config["data"]
    root = data_conf["input_dir"]
    split_names = data_conf["split_names"]
    lc_filename = data_conf["test_lc_filename"]
    split_col = data_conf["split_col"]

    dfs = []
    for split_name in split_names:
        path = os.path.join(root, split_name, lc_filename)
        if not os.path.exists(path):
            print(f"[WARN] file not found: {path}")
            continue

        df_split = pd.read_csv(path)

        if split_col not in df_split.columns:
            df_split[split_col] = split_name
        
        dfs.append(df_split)
    
    test_lc = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] test lightcurves shape: {test_lc.shape}")
    return test_lc

def make_folds(config: Dict, df_log: pd.DataFrame) -> pd.DataFrame:
    cv_conf = config["cv"]
    data_conf = config["data"]

    n_splits = cv_conf["n_splits"]
    strategy = cv_conf["strategy"]
    shuffle = cv_conf["shuffle"]
    random_state = cv_conf["random_state"]
    group_col = cv_conf["group_col"]

    df_log = df_log.copy()
    df_log["fold"] = -1

    y = df_log["target_idx"].values

    if strategy == "stratified":
        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
        )
        for fold, (trn_idx, val_idx) in enumerate(skf.split(df_log, y)):
            df_log.loc[val_idx, "fold"] = fold
    
    elif strategy == "group":
        if group_col is None:
            group_col = data_conf["split_col"]
        groups = df_log[group_col].values

        gkf = GroupKFold(n_splits=n_splits)
        for fold, (trn_idx, val_idx) in enumerate(gkf.split(df_log, y, groups)):
            df_log.loc[val_idx, "fold"] = fold
    
    elif strategy == "kfold":
        kf = KFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
        )
        for fold, (trn_idx, val_idx) in enumerate(kf.split(df_log)):
            df_log.loc[val_idx, "fold"] = fold
    
    else:
        raise ValueError(f"Unknown CV strategy: {strategy}")
    
    print("[INFO] fold distribution:")
    print(df_log["fold"].value_counts())

    return df_log

def build_basic_agg_feature(
        config: Dict,
        train_lc: pd.DataFrame,
        df_log: pd.DataFrame,
        is_train: bool = True,
) -> pd.DataFrame:
    data_conf = config["data"]
    feat_conf = config["features"]

    id_col = data_conf["id_col"]
    time_col = data_conf["time_col"]
    flux_col = data_conf["flux_col"]
    flux_err_col = data_conf["flux_err_col"]
    band_col = data_conf["band_col"]

    lc = train_lc.copy()
    lc["_t_min"] = lc.groupby(id_col)[time_col].transform("min")
    lc["t_rel"] = lc[time_col] - lc["_t_min"]
    lc.drop(columns=["_t_min"], inplace=True)

    lc["snr"] = lc[flux_col] / (lc[flux_err_col] + 1e-6)

    agg_dict = {}

    agg_dict["n_obs"] = (time_col, "count")
    agg_dict["t_rel_max"] = ("t_rel", "max")
    agg_dict["t_rel_min"] = ("t_rel", "min")

    if feat_conf.get("use_flux_stats", True):
        for stat in feat_conf.get("flux_stats", ["mean", "std", "min", "max", "median"]):
            agg_dict[f"flux_{stat}"] = (flux_col, stat)
    
    agg_dict["flux_err_mean"] = (flux_err_col, "mean")
    agg_dict["flux_err_std"] = (flux_err_col, "std")

    if feat_conf.get("use_snr_features", True):
        agg_dict["snr_mean"] = ("snr", "mean")
        agg_dict["snr_std"] = ("snr", "std")
    
    agg_items = {k: v for k, v in agg_dict.items()}
    grouped = lc.groupby(id_col).agg(**agg_items).reset_index()

    grouped["t_span"] = grouped["t_rel_max"] - grouped["t_rel_min"]

    if feat_conf.get("use_bandwise_stats", True):
        band_stats = lc.pivot_table(
            index=id_col,
            columns=band_col,
            values=flux_col,
            aggfunc=["mean", "std"],
        )

        band_stats.columns = [
            f"flux_{stat}_{band}"
            for stat, band in band_stats.columns
        ]

        band_stats = band_stats.reset_index()
        grouped = grouped.merge(band_stats, on=id_col, how="left")

    if feat_conf.get("use_meta_features", True):
        meta_cols = [
            id_col,
            data_conf["z_col"],
            data_conf["z_err_col"],
            data_conf["ebv_col"],
            data_conf["spec_type_col"],
            data_conf["split_col"],
        ]
        meta_cols = [c for c in meta_cols if c in df_log.columns]
        meta = df_log[meta_cols].drop_duplicates(id_col)
        grouped = grouped.merge(meta, on=id_col, how="left")

    if is_train:
        add_cols = [id_col, "target_idx", config["data"]["split_col"], "fold"]
        add_cols = [c for c in add_cols if c in df_log.columns]
        df_target = df_log[add_cols].drop_duplicates(id_col)
        grouped = grouped.merge(df_target, on=id_col, how="left")
    
    print(f"[INFO] agg features shape: {grouped.shape}")
    return grouped

if __name__ == "__main__":
    set_seed(CONFIG["general"]["seed"])

    df_log, cls2idx, idx2cls, CONFIG = load_train_log_and_update_config(CONFIG)
    df_log = make_folds(CONFIG, df_log)
    train_lc = load_all_train_lightcurves(CONFIG)
    train_feats = build_basic_agg_feature(CONFIG, train_lc, df_log, is_train=True)
