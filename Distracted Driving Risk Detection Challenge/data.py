from config import *
from features import create_features

def load_raw_data():
    train_df = pd.read_csv(f"{DATA_DIR}/kaggle_train.csv")
    test_df = pd.read_csv(f"{DATA_DIR}/kaggle_test.csv")

    print("Shapes:", train_df.shape, test_df.shape)

    drop_meta_cols = [c for c in ['label_source'] if c in train_df.columns]
    if drop_meta_cols:
        train_df = train_df.drop(columns=drop_meta_cols)
        test_df = test_df.drop(columns=[c for c in drop_meta_cols if c in test_df.columns])

    assert 'risk_level' in train_df.columns, "train_df에 risk_level 컬럼이 필요합니다."
    return train_df, test_df

def build_features_and_scale(train_df, test_df):
    train_feat = create_features(train_df)
    test_feat = create_features(test_df)

    feature_cols = [c for c in train_feat.columns if c != 'risk_level']

    X = train_feat[feature_cols].replace([np.inf, -np.inf], np.nan)
    X_test = test_feat[feature_cols].replace([np.inf, -np.inf], np.nan)

    X = X.fillna(X.median(numeric_only=True))
    X_test = X_test.fillna(X.median(numeric_only=True))

    y = (train_feat['risk_level'].astype(int) - 1)
    num_class = 4

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = scaler.fit_transform(X.values)
    Xtest_scaled = scaler.transform(X_test.values)

    print("Num features:", X.shape[1])

    return X, X_test, X_scaled, Xtest_scaled, y.values, num_class, feature_cols, scaler

def compute_sample_weights(X_scaled, Xtest_scaled, y):
    adv_y = np.concatenate([
        np.zeros(len(X_scaled), dtype=int),
        np.ones(len(Xtest_scaled), dtype=int)
    ])
    adv_X = np.vstack([X_scaled, Xtest_scaled])

    adv_clf = LogisticRegression(max_iter=200, n_jobs=None, random_state=RANDOM_STATE)
    adv_clf.fit(adv_X, adv_y)
    p_test_like = adv_clf.predict_proba(X_scaled)[:,1]

    DOMAIN_W_MIN = float(os.environ.get("DOMAIN_W_MIN", "0.5"))
    DOMAIN_W_MAX = float(os.environ.get("DOMAIN_W_MAX", "1.5"))
    w_domain = DOMAIN_W_MIN + (DOMAIN_W_MAX - DOMAIN_W_MIN) * p_test_like

    w_class = compute_sample_weight(class_weight='balanced', y=y)
    sample_w = w_class * w_domain

    print("Domain weight range:", w_domain.min(), "->", w_domain.max())

    return sample_w

def get_cv_splitter(train_df, y):
    CV_MODE = os.environ.get("CV_MODE", "stratified").lower()
    N_SPLITS = int(os.environ.get("N_SPLITS", "5"))

    groups = None
    if CV_MODE == "group":
        if "driver_id" in train_df.columns:
            groups = train_df["driver_id"].values
        else:
            print("CV_MODE=group 이지만 driver_id 없음 -> stratified로 fallback.")
            CV_MODE = "stratified"
    
    def make_splitter():
        if CV_MODE == "time":
            return TimeSeriesSplit(n_splits=N_SPLITS), None
        elif CV_MODE == "group":
            return GroupKFold(n_splits=N_SPLITS), groups
        else:
            return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE), None
    
    splitter, groups = make_splitter()
    return splitter, groups, CV_MODE, N_SPLITS