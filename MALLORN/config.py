# config.py

CONFIG = {
    "general": {
        "seed": 42,
        "n_jobs": -1,
        "device": "cuda",   # or "cpu"
        "exp_name": "mallorn_v1",
        "debug": False,
    },

    "data": {
        "input_dir": "",

        "split_names": [f"split_{i:02d}" for i in range(1, 21)],  # split_01 ~ split_20
        "train_lc_filename": "train_full_lightcurves.csv",
        "test_lc_filename": "test_full_lightcurves.csv",

        # 로그/메타 정보 파일 (split 폴더 바깥)
        "train_log_filename": "train_log.csv",
        "test_log_filename": "test_log.csv",
        "sample_submission_filename": "sample_submission.csv",

        # 공통 key & 컬럼명들
        "id_col": "object_id",
        "target_col": "target",

        # light curve 테이블 컬럼
        "time_col": "Time (MJD)",
        "flux_col": "Flux",
        "flux_err_col": "Flux_err",
        "band_col": "Filter",

        # 로그/메타 테이블 컬럼
        "z_col": "Z",
        "z_err_col": "Z_err",
        "ebv_col": "EBV",
        "spec_type_col": "SpecType",
        "name_translation_col": "English Translation",
        "split_col": "split",

        # 캐시/세이브 경로
        "feature_dir": "./features",
        "lgbm_oof_path": "./oof_lgbm.npy",
        "cat_oof_path": "./oof_cat.npy",
        "seq_oof_path": "./oof_seq.npy",
        "lgbm_test_pred_path": "./pred_lgbm.npy",
        "cat_test_pred_path": "./pred_cat.npy",
        "seq_test_pred_path": "./pred_seq.npy",
    },

    "cv": {
        "n_splits": 5,
        "shuffle": True,
        "random_state": 42,
        "strategy": "stratified",   # ["stratified", "kfold", "group"]
        "group_col": None,
    },

    "features": {
        "use_time_stats": True,
        "time_stats": [
            "min", "max", "mean", "std", "median",
            "skew", "kurt",
        ],
        "use_flux_stats": True,
        "flux_stats": [
            "min", "max", "mean", "std", "median",
            "skew", "kurt",
        ],
        "use_bandwise_stats": True,    # Filter별 통계
        "use_color_features": True,    # 밴드 간 flux ratio / diff
        "use_duration_features": True, # 첫/마지막 관측, rise/fall 등
        "use_snr_features": True,      # Flux / Flux_err 기반
        "use_meta_features": True,     # Z, Z_err, EBV, SpecType 등

        "use_feature_selection": False,
        "max_feature_num": 300,
    },

    "models": {
        "lgbm": {
            "use": True,
            "params": {
                "objective": "multiclass",
                "num_class": None,
                "metric": "multi_logloss",
                "boosting_type": "gbdt",
                "learning_rate": 0.05,
                "num_leaves": 64,
                "max_depth": -1,
                "min_data_in_leaf": 40,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 1,
                "lambda_l1": 0.0,
                "lambda_l2": 1.0,
                "verbosity": -1,
                "n_estimators": 2000,
                "early_stopping_rounds": 100,
            },
        },

        "catboost": {
            "use": True,
            "params": {
                "loss_function": "MultiClass",
                "eval_metric": "MultiClass",
                "learning_rate": 0.05,
                "depth": 8,
                "l2_leaf_reg": 3.0,
                "random_strength": 1.0,
                "bagging_temperature": 1.0,
                "border_count": 128,
                "iterations": 2000,
                "early_stopping_rounds": 100,
                "task_type": "GPU", 
                "thread_count": -1,
            },
        },

        "seq_cnn_lstm": {
            "use": True,
            "input": {
                "max_len": 120,
                "feature_dim": 12,
                "pad_value": 0.0,
            },
            "model": {
                "d_model": 64,
                "cnn_channels": 64,
                "cnn_kernel_size": 3,
                "cnn_layers": 2,
                "lstm_hidden_size": 128,
                "lstm_num_layers": 1,
                "bidirectional": True,
                "fc_hidden": 256,
                "dropout": 0.3,
                "num_classes": None,  
            },
            "train": {
                "batch_size": 64,
                "num_epochs": 30,
                "optimizer": "adam",
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "scheduler": "cosine",
                "warmup_epochs": 2,
                "gradient_clip_val": 1.0,
                "label_smoothing": 0.0,
            },
        },
    },

    "ensemble": {
        "use": True,
        "weights": {
            "lgbm": 0.5,
            "catboost": 0.3,
            "seq_cnn_lstm": 0.2,
        },
        "stacking": {
            "use": False,
            "meta_model": "logreg",
            "meta_params": {},
        },
    },

    "logging": {
        "use_mlflow": False,
        "output_dir": "./outputs",
        "save_oof": True,
        "save_test_pred": True,
    },
}
