from config import *

def train_oof_models(X_np, y_np, Xtest_np, sample_w, splitter, groups, CV_MODE, N_SPLITS, num_class):
    xgb_params = dict(
        objective='multi:softprob',
        num_class=num_class,
        eval_metric='mlogloss',
        n_estimators=20000,
        learning_rate=0.05,
        max_depth=10,
        subsample=0.95,
        colsample_bytree=0.95,
        gamma=0.0,
        min_child_weight=1.0,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method='hist'
    )

    lgb_params=dict(
        objective='multiclass',
        num_class=num_class,
        metric='multi_logloss',
        n_estimators=30000,
        learning_rate=0.05,
        num_leaves=127,
        feature_fraction=0.95,
        bagging_fraction=0.95,
        bagging_freq=1,
        min_data_in_leaf=20,
        lambda_l1=0.0,
        lambda_l2=0.0,
        min_gain_to_split=0.0,
        max_depth=-1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1
    )

    cat_params=None
    if HAVE_CAT:
        cat_params=dict(
            loss_function='MultiClass',
            iterations=30000,
            learning_rate=0.05,
            depth=10,
            l2_leaf_reg=2.0,
            random_strength=0.5,
            random_seed=RANDOM_STATE,
            od_type='Iter',
            od_wait=600,
            verbose=False,
            allow_writing_files=False
        )
    
    oof_xgb = np.zeros((len(X_np), num_class))
    test_xgb = np.zeros((Xtest_np.shape[0], num_class))
    oof_lgb = np.zeros_like(oof_xgb)
    test_lgb = np.zeros_like(test_xgb)

    if HAVE_CAT and cat_params is not None:
        oof_cat = np.zeros_like(oof_xgb)
        test_cat = np.zeros_like(test_xgb)
    else:
        oof_cat = None
        test_cat = None
    
    xgb_best, lgb_best, cat_best = [], [], []

    if CV_MODE == "group" and groups is not None:
        folds = splitter.split(X_np, y_np, groups=groups)
    elif CV_MODE == "time":
        folds = splitter.split(X_np)
    else:
        folds = splitter.split(X_np, y_np)
    
    for fold, (tr, va) in enumerate(folds, 1):
        print(f"\nFOLD {fold}")
        Xtr, Xva = X_np[tr], X_np[va]
        ytr, yva = y_np[tr], y_np[va]
        wtr = sample_w[tr]

        xgbc = xgb.XGBClassifier(**xgb_params)
        xgbc.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xva,yva)], early_stopping_rounds=400, verbose=False)
        oof_xgb[va] = xgbc.predict_proba(Xva)
        test_xgb += xgbc.predict_proba(Xtest_np) / N_SPLITS
        xgb_best.append(getattr(xgbc, "best_iteration", xgb_params['n_estimators']))

        lgbc = lgb.LGBMClassifier(**lgb_params)
        lgbc.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xva,yva)], callbacks=[lgb.early_stopping(stopping_rounds=600, verbose=False), lgb.log_evaluation(period=0)])
        oof_lgb[va] = lgbc.predict_proba(Xva)
        test_lgb += lgbc.predict_proba(Xtest_np) / N_SPLITS
        lgb_best.append(getattr(lgbc, "best_iteration_", lgb_params['n_estimators']))

        if HAVE_CAT and cat_params is not None:
            cat = CatBoostClassifier(**cat_params)
            cat.fit(Xtr, ytr, sample_weight=wtr, eval_set=(Xva, yva), verbose=False)
            oof_cat[va] = cat.predict_proba(Xva)
            test_cat += cat.predict_proba(Xtest_np) / N_SPLITS
            cat_best.append(int(cat.tree_count_))
    
    def acc_of(p): return accuracy_score(y_np, p.argmax(1))
    print("\nOOF Acc XGB:", round(acc_of(oof_xgb), 4))
    print("OOF Acc LGB:", round(acc_of(oof_lgb), 4))
    if oof_cat is not None:
        print("OOF Acc CAT:", round(acc_of(oof_cat), 4))
    
    return (oof_xgb, test_xgb, oof_lgb, test_lgb, oof_cat, test_cat, xgb_best, lgb_best, cat_best, xgb_params, lgb_params, cat_params)

def logits_of(proba, eps=1e-9):
    return np.log(np.clip(proba, eps, 1.0))

def temp_and_stacking(oof_xgb, oof_lgb, oof_cat,
                      test_xgb, test_lgb, test_cat,
                      y_np, num_class):
    base_logits_oof = logits_of(oof_xgb) + logits_of(oof_lgb)
    base_logits_test = logits_of(test_xgb) + logits_of(test_lgb)
    if oof_cat is not None:
        base_logits_oof += logits_of(oof_cat)
        base_logits_test += logits_of(test_cat)

    T = 1.0
    for _ in range(50):
        cands = np.clip(np.array([T*0.8, T*0.9, T, T*1.1, T*1.25]),
                        0.05, 10.0)
        losses = []
        for t in cands:
            p = base_logits_oof / t
            p = np.exp(p - p.max(axis=1, keepdims=True))
            p = p / p.sum(axis=1, keepdims=True)
            losses.append(log_loss(y_np, p, labels=np.arange(num_class)))
        T = cands[np.argmin(losses)]
    T_global = float(T)

    def apply_temp(logits, T):
        p = logits / T
        p = np.exp(p - p.max(axis=1, keepdims=True))
        return p / p.sum(axis=1, keepdims=True)

    proba_oof_cal = apply_temp(base_logits_oof, T_global)
    proba_test_cal = apply_temp(base_logits_test, T_global)
    oof_ens_acc = accuracy_score(y_np, proba_oof_cal.argmax(1))
    print(f"OOF Acc (Ensemble + TempCal, T={T_global:.2f}): {oof_ens_acc:.4f}")

    def add_meta_features(logits, proba):
        max_p = proba.max(axis=1, keepdims=True)
        entropy = -(proba * np.log(np.clip(proba, 1e-9, 1))).sum(axis=1, keepdims=True)
        margin = np.sort(proba, axis=1)[:, [-1, -2]]  # top1, top2
        margin_diff = (margin[:, [0]] - margin[:, [1]])  # top1 - top2
        return np.hstack([logits, max_p, entropy, margin_diff])

    meta_feat_oof = add_meta_features(base_logits_oof, proba_oof_cal)
    meta_feat_test = add_meta_features(base_logits_test, proba_test_cal)

    meta = LogisticRegression(
        multi_class="multinomial",
        max_iter=500,
        C=5.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    meta.fit(meta_feat_oof, y_np)
    oof_meta = meta.predict_proba(meta_feat_oof)
    test_meta = meta.predict_proba(meta_feat_test)

    oof_meta_acc = accuracy_score(y_np, oof_meta.argmax(1))
    print("OOF Acc (Stacking Meta):", round(oof_meta_acc, 4))

    oof_best = oof_meta if oof_meta_acc >= oof_ens_acc else proba_oof_cal
    test_best = test_meta if oof_meta_acc >= oof_ens_acc else proba_test_cal

    return T_global, oof_best, test_best, proba_test_cal

def train_full_models(X_np, y_np, sample_w, xgb_best, lgb_best, cat_best, xgb_params, lgb_params, cat_params):
    xgb_full = xgb.XGBClassifier(**{**xgb_params, "n_estimators": int(np.mean(xgb_best))})
    lgb_full = lgb.LGBMClassifier(**{**lgb_params, "n_estimators": int(np.mean(lgb_best))})
    xgb_full.fit(X_np, y_np, sample_weight=sample_w, verbose=False)
    lgb_full.fit(X_np, y_np, sample_weight=sample_w, callbacks=[lgb.log_evaluation(period=0)])

    cat_full =None
    if HAVE_CAT and cat_params is not None and len(cat_best) >0:
        cat_full = CatBoostClassifier(**{**cat_params, "iterations": int(np.mean(cat_best))})
        cat_full.fit(X_np, y_np, sample_weight=sample_w, verbose=False)

    return xgb_full, lgb_full, cat_full

def final_submissions(X_test, Xtest_np, feature_cols, test_best, xgb_full, lgb_full, cat_full, T_global):
    full_logits_test = logits_of(xgb_full.predict_proba(Xtest_np)) + logits_of(lgb_full.predict_proba(Xtest_np))
    if cat_full is not None:
        full_logits_test += logits_of(cat_full.predict_proba(Xtest_np))
    
    def apply_temp(logits, T):
        p = logits / T
        p = np.exp(p - p.max(axis=1, keepdims=True))
        return p / p.sum(axis=1, keepdims=True)

    full_proba_test_cal = apply_temp(full_logits_test, T_global)
    pred_test_full_ens = full_proba_test_cal.argmax(1) + 1
    
    sub_stack = pd.DataFrame({
        'id': np.arange(len(X_test)),
        'risk_level': (test_best.argmax(1) + 1).astype(int)
    })

    sub_full = pd.DataFrame({
        'id': np.arange(len(X_test)),
        'risk_level': pred_test_full_ens.astype(int)
    })

    sub_stack.to_csv('sub_stack.csv', index=False)
    sub_full.to_csv('sub_full_ens_cal.csv', index=False)

    print("\nsub_stack class dist (%):")
    print((sub_stack['risk_level'].value_counts(normalize=True).sort_index() * 100).round(2))
    print("\nsub_full_ens_cal class dist (%):")
    print((sub_full['risk_level'].value_counts(normalize=True).sort_index() * 100).round(2))

    try:
        importances = xgb_full.feature_importances_
        fi = (
            pd.DataFrame({'feature': feature_cols, 'importance': importances})
            .sort_values('importance', ascending=False)
            .head(30)
        )
        print("\nTop-30 XGB features:")
        print(fi)
    except Exception as e:
        print("Feature importance error:", e)
    
