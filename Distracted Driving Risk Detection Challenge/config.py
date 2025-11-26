import os, gc, warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import xgboost as xgb
import lightgbm as lgb
try:
    from catboost import CatBoostClassifier
    HAVE_CAT = True
except Exception:
    HAVE_CAT = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

from sklearn.model_selection import StratifiedKFold, GroupKFold, TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = int(os.environ.get("SEED", "42"))
np.random.seed(RANDOM_STATE)

DATA_DIR = os.environ.get(
    "DATA_DIR",
    "/kaggle/input/distracted-driving-risk-detection-challenge"
)

AGGRESSIVE = True

if AGGRESSIVE:
    os.environ["DOMAIN_W_MIN"] = "0.3"
    os.environ["DOMAIN_W_MAX"] = "1.8"

    os.environ["PL_THRESH"] = "0.90"
    os.environ["PL_WEIGHT"] = "0.80"
    os.environ["PL_MAX_PER_CLASS"] = "60000"
    os.environ["PL_TTA_K"] = "5"
    os.environ["PL_STAB_MIN"] = "0.65"
    os.environ["PL_ALPHA"] = "2.0"
    os.environ["PL_BETA"] = "1.0"