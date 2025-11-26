from config import *
from data import (
    load_raw_data,
    build_features_and_scale,
    compute_sample_weights,
    get_cv_splitter,
)
from models import (
    train_oof_models,
    temp_and_stacking,
    train_full_models,
    final_submissions,
)

from ssl_pl import run_ssl_mean_teacher, run_pseudo_label
from config import DATA_DIR

def main():
    train_df, test_df = load_raw_data()
    X, X_test, X_scaled, Xtest_scaled, y_np, num_class, feature_cols, scaler = build_features_and_scale(train_df, test_df)
    sample_w = compute_sample_weights(X_scaled, Xtest_scaled, y_np)
    splitter, groups, CV_MODE, N_SPLITS = get_cv_splitter(train_df, y_np)
    (oof_xgb, test_xgb,
     oof_lgb, test_lgb,
     oof_cat, test_cat,
     xgb_best, lgb_best, cat_best,
     xgb_params, lgb_params, cat_params) = train_oof_models(X_scaled, y_np, Xtest_scaled, sample_w, splitter, groups, CV_MODE, N_SPLITS, num_class)
    
    T_global, oof_best, test_best, proba_test_cal = temp_and_stacking(oof_xgb, oof_lgb, oof_cat, test_xgb, test_lgb, test_cat, y_np, num_class)

    xgb_full, lgb_full, cat_full = train_full_models(
        X_scaled, y_np, sample_w,
        xgb_best, lgb_best, cat_best,
        xgb_params, lgb_params, cat_params
    )

    train_median = X.median(numeric_only=True)
    ssl = run_ssl_mean_teacher(
        DATA_DIR = DATA_DIR,
        scaler=scaler,
        feature_cols = feature_cols,
        train_median=train_median,
        num_class=num_class,
        y_np=y_np,
        Xtest_np=Xtest_scaled,
        proba_test_cal=proba_test_cal,
        T_global=T_global,
    )

    pred_test_pl = run_pseudo_label(
        DATA_DIR = DATA_DIR,
        scaler= scaler,
        feature_cols=feature_cols,
        train_median=train_median,
        num_class=num_class,
        y_np=y_np,
        X_np=X_scaled,
        Xtest_np=Xtest_scaled,
        xgb_full=xgb_full,
        lgb_full=lgb_full,
        cat_full=cat_full,
        ssl=ssl,
        T_global=T_global,
        xgb_params=xgb_params,
        lgb_params=lgb_params,
        cat_params=cat_params,
        xgb_best=xgb_best,
        lgb_best=lgb_best,
        cat_best=cat_best,
    )

    if pred_test_pl is not None and not AGGRESSIVE:
        sub_pl = pd.DataFrame({'id': np.arange(len(X_test)), 'risk_level': pred_test_pl.astype(int)})
        sub_pl.to_csv('/kaggle/working/sub_pseudolabel_consensus_stable.csv', index=False)
        sub_pl.to_csv('/kaggle/working/submission.csv', index=False)

    final_submissions(X_test, Xtest_scaled, feature_cols, test_best, xgb_full, lgb_full, cat_full, T_global)

    if AGGRESSIVE:
        sub_stack = pd.read_csv('sub_stack.csv')
        sub_stack.to_csv('/kaggle/working/submission.csv',index=False)

if __name__ == "__main__":
    main()