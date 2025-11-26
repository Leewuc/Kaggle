from config import *
from features import create_features

def run_ssl_mean_teacher(
        DATA_DIR,
        scaler,
        feature_cols,
        train_median,
        num_class,
        y_np,
        Xtest_np,
        proba_test_cal,
        T_global,
):
    ssl = {"enabled": False}
    if not HAVE_TORCH:
        print("Torch not available; skip Mean-Teacher SSL")
        return ssl
    
    unlabeled_path = f"{DATA_DIR}/kaggle_full_unlabeled_data.csv"
    if not os.path.exists(unlabeled_path):
        print("Unlabeled file not found for SSL; skip Mean-Teacher.")
        return ssl
    
    unlabeled_df = pd.read_csv(unlabeled_path)
    unl_base = unlabeled_df.drop(columns=[c for c in ['label_source'] if c in unlabeled_df.columns])
    unl_feat = create_features(unl_base).reindex(columns=feature_cols)

    Xu_df = unl_feat.replace([np.inf, -np.inf], np.nan).fillna(train_median)
    Xu_np = scaler.transform(Xu_df.values).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_in, d_out = Xu_np.shape[1], num_class

    class TabMLP(nn.Module):
        def __init__(self, d_in, d_out):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_in, 256), nn.BatchNorm1d(256), nn.ReLU(),
                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
                nn.Linear(128, d_out)
            )
        def forward(self, x):
            return self.net(x)
    
    def ema_update(teacher, student, decay=0.999):
        with torch.no_grad():
            for tp, sp in zip(teacher.parameters(), student.parameters()):
                tp.data.mul_(decay).add_(sp.data * (1.0 - decay))
    
    def aug_weak(x, noise_std = 0.01):
        return x + np.random.normal(0, noise_std, size=x.shape).astype(np.float32)

    def aug_strong(x, drop_prob=0.05, noise_std = 0.05):
        mask = (np.random.rand(*x.shape) > drop_prob).astype(np.float32)
        return (x * mask + np.random.normal(0, noise_std, size=x.shape)).astype(np.float32)
    
    def distribution_align(p, prior_src, prior_tgt, eps=1e-8, strength=1.0):
        adj = p * (prior_tgt[None, :] + eps) / (prior_src[None, :] + eps)
        adj = adj / np.clip(adj.sum(axis=1, keepdims=True), eps, None)
        return strength * adj + (1 - strength) * p
    
    student = TabMLP(d_in, d_out).to(device)
    teacher = TabMLP(d_in, d_out).to(device)
    teacher.load_state_dict(student.state_dict())
    opt = torch.optim.AdamW(student.parameters(), lr=2e-3, weight_decay=1e-5)

    prior_src = np.bincount(y_np, minlength=num_class).astype(np.float64)
    prior_src = prior_src / prior_src.sum()
    target_prior = proba_test_cal.mean(axis=0)

    dl_u = DataLoader(TensorDataset(torch.from_numpy(Xu_np)), batch_size=1024, shuffle=True, drop_last=True)

    THR = float(os.environ.get("SSL_THR", "0.90"))
    UNSUP_W = float(os.environ.get("SSL_UNSUP_W", "3.0"))
    EPOCHS = int(os.environ.get("SSL_EPOCHS", "30"))
    STEPS = int(os.environ.get("SSL_STEPS", "300"))

    student.train()
    teacher.eval()
    for ep in range(EPOCHS):
        it = 0
        for (xb_u,) in dl_u:
            it += 1
            if it > STEPS:
                break

            xb_u_np = xb_u.numpy()
            xw = torch.from_numpy(aug_weak(xb_u_np)).to(device)
            xs = torch.from_numpy(aug_strong(xb_u_np)).to(device)

            with torch.no_grad():
                pw = F.softmax(teacher(xw), dim=1).cpu().numpy()
            pw = distribution_align(pw, prior_src, target_prior, strength=0.7)
            pw = torch.from_numpy(pw).to(device)

            logits_s = student(xs)
            conf_w, hard_w = pw.max(dim=1)
            mask = (conf_w >= THR).float()
            unsup_loss = (F.cross_entropy(logits_s, hard_w, reduction='none') * mask).mean()

            loss = UNSUP_W * unsup_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            ema_update(teacher, student, decay=0.999)
    
    teacher.eval()
    with torch.no_grad():
        p_unl = F.softmax(teacher(torch.from_numpy(Xu_np).to(device)), dim=1).cpu().numpy()
        p_tst = F.softmax(teacher(torch.from_numpy(Xtest_np.astype(np.float32)).to(device)), dim=1).cpu().numpy()
    
    def apply_temp_np(p, T):
        lg = np.log(np.clip(p, 1e-9, 1.0)) / T
        e = np.exp(lg - lg.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
    
    p_unl = apply_temp_np(p_unl, T_global)
    p_tst = apply_temp_np(p_tst, T_global)

    p_unl = distribution_align(p_unl, p_unl.mean(axis=0), target_prior, strength=0.7)
    p_tst = distribution_align(p_tst, p_tst.mean(axis=0), target_prior, strength=0.7)

    ssl = {"enabled": True, "Xu_np": Xu_np, "p_unl": p_unl, "p_tst": p_tst}
    return ssl

def run_pseudo_label(
        DATA_DIR, scaler, feature_cols, train_median, num_class, y_np, X_np, Xtest_np, xgb_full, lgb_full, cat_full, ssl,
        T_global, xgb_params, lgb_params, cat_params, xgb_best, lgb_best, cat_best,
):
    USE_PSEUDO = os.environ.get("USE_PSEUDO", "true").lower() == "true"
    if not USE_PSEUDO:
        print("Pseudo-Label disabled by USE_PSEUDO")
        return None
    
    unlabeled_path = f"{DATA_DIR}/kaggle_full_unlabeled_data.csv"
    if not os.path.exists(unlabeled_path):
        print("Unlabeled file not found, Skipping Pseudo-Labeling")
        return None
    
    if ssl.get("enabled", False) and "Xu_np" in ssl:
        Xu_np = ssl["Xu_np"]
    else:
        unlabeled_df = pd.read_csv(unlabeled_path)
        unl_base = unlabeled_df.drop(columns=[c for c in ['label_source'] if c in unlabeled_df.columns])
        unl_feat = create_features(unl_base).reindex(columns=feature_cols)
        Xu_df = unl_feat.replace([np.inf, -np.inf], np.nan).fillna(train_median)
        Xu_np = scaler.transform(Xu_df.values)

    def logits_of(proba, eps=1e-9):
        return np.log(np.clip(proba, eps, 1.0))
    
    def apply_temp(logits, T):
        p = logits / T
        p = np.exp(p - p.max(axis=1, keepdims=True))
        return p / p.sum(axis=1, keepdims=True)
    
    px = xgb_full.predict_proba(Xu_np)
    pl = lgb_full.predict_proba(Xu_np)
    logits_all = logits_of(px) + logits_of(pl)
    pred_each = [px.argmax(1), pl.argmax(1)]

    if cat_full is not None:
        pc = cat_full.predict_proba(Xu_np)
        logits_all += logits_of(pc)
        pred_each.append(pc.argmax(1))

    if ssl.get("enabled", False) and "p_unl" in ssl:
        p_ssl_u = ssl["p_unl"]
        logits_all += logits_of(p_ssl_u)
        pred_each.append(p_ssl_u.argmax(1))
    
    proba_ens = apply_temp(logits_all, T_global)
    pred_each = np.stack(pred_each, axis=0)
    pred_ens = proba_ens.argmax(1)
    conf_ens = proba_ens.max(1)

    agree_counts = (pred_each == pred_ens).sum(axis=0)
    need_votes = 2 if pred_each.shape[0] >= 2 else 1
    consensus_mask = agree_counts >= need_votes

    PSEUDO_THRESH = float(os.environ.get("PL_THRESH", "0.950"))
    PSEUDO_WEIGHT = float(os.environ.get("PL_WEIGHT", "0.60"))
    PSEUDO_MAX_PER_CLASS = int(os.environ.get("PL_MAX_PER_CLASS", "40000"))
    TTA_K = int(os.environ.get("PL_TTA_K", "3"))
    TTA_SIGMA = float(os.environ.get("PL_TTA_SIGMA", "0.012"))
    STAB_MIN = float(os.environ.get("PL_STAB_MIN", "0.70"))
    ALPHA = float(os.environ.get("PL_ALPHA", "2.0"))
    BETA = float(os.environ.get("PL_BETA", "1.0"))

    def predict_calibrated_proba(models, Xs):
        px_ = models['xgb'].predict_proba(Xs)
        pl_ = models['lgb'].predict_proba(Xs)
        logits = logits_of(px_) + logits_of(pl_)
        out = {
            'xgb': apply_temp(logits_of(px_), T_global),
            'lgb': apply_temp(logits_of(pl_), T_global),
        }
        if 'cat' in models:
            pc_ = models['cat'].predict_proba(Xs)
            out['cat'] = apply_temp(logits_of(pc_), T_global)
            logits += logits_of(pc_)
        out['ens'] = apply_temp(logits, T_global)
        return out
    
    models = {'xgb': xgb_full, 'lgb': lgb_full}
    if cat_full is not None:
        models['cat'] = cat_full
    
    TTA_votes = np.zeros((Xu_np.shape[0],), dtype=int)
    TTA_conf_sum = np.zeros((Xu_np.shape[0],), dtype=float)
    TTA_margin_sum = np.zeros((Xu_np.shape[0],), dtype=float)

    for k in range(TTA_K):
        noise = np.random.normal(0.0, TTA_SIGMA, size=Xu_np.shape)
        Xu_noisy = Xu_np + noise
        proba_k = predict_calibrated_proba(models, Xu_noisy)['ens']
        pred_k = proba_k.argmax(1)
        conf_k = proba_k.max(1)
        sorted_p = np.sort(proba_k, axis=1)
        margin_k = sorted_p[:, -1] - sorted_p[:, -2]
        TTA_votes += (pred_k == pred_ens).astype(int)
        TTA_conf_sum += conf_k
        TTA_margin_sum += margin_k

    stab_ratio = TTA_votes / float(TTA_K)
    conf_tta = TTA_conf_sum / float(TTA_K)
    margin_tta = TTA_margin_sum / float(TTA_K)

    class_counts = np.bincount(y_np, minlength=num_class).astype(float)
    mean_count = class_counts.mean()
    thr_per_class = np.array([
        PSEUDO_THRESH + (0.01 if class_counts[c] < mean_count else 0.0)
        for c in range(num_class)
    ], dtype=float)

    for c in range(num_class):
        env_key = f"PL_THR_C{c}"
        if env_key in os.environ:
            thr_per_class[c] = float(os.environ[env_key])
    
    print("Per-class PL thresholds:", np.round(thr_per_class, 4))

    u_lab = pred_ens
    per_class = {c: 0 for c in range(num_class)}
    selected = []

    idx_order = np.argsort(-conf_tta)
    for i in idx_order:
        c = int(u_lab[i])
        if not consensus_mask[i]:
            continue
        if stab_ratio[i] < STAB_MIN:
            continue
        if conf_tta[i] < thr_per_class[c]:
            continue
        if per_class[c] >= PSEUDO_MAX_PER_CLASS:
            continue
        selected.append(i)
        per_class[c] += 1
    
    selected = np.array(selected, dtype=int)
    print(f"Pseudo selected (consensus+stab): {len(selected)} | per-class:", per_class)

    if len(selected) == 0:
        print("No pseudo samples selected.")
        return None

    conf_sel = conf_tta[selected]
    marg_sel = margin_tta[selected]
    w_pl_soft = PSEUDO_WEIGHT * np.power(np.clip(conf_sel, 1e-6, 1.0), ALPHA) * np.power(np.clip(marg_sel, 1e-6, 1.0), BETA)
    w_pl_soft = np.clip(w_pl_soft, 0.05, 2.0)

    X_aug = np.vstack([X_np, Xu_np[selected]])
    y_aug = np.concatenate([y_np, u_lab[selected]])
    w_class = compute_sample_weight(class_weight='balanced', y=y_aug)
    w_aug = np.concatenate([w_class[:len(y_np)], w_pl_soft])

    xgb_pl = xgb.XGBClassifier(**{**xgb_params, "n_estimators": int(np.mean(xgb_best))})
    lgb_pl = lgb.LGBMClassifier(**{**lgb_params, "n_estimators": int(np.mean(lgb_best))})
    xgb_pl.fit(X_aug, y_aug, sample_weight=w_aug, verbose=False)
    lgb_pl.fit(X_aug, y_aug, sample_weight=w_aug, callbacks=[lgb.log_evaluation(period=0)])

    logits_pl = logits_of(xgb_pl.predict_proba(Xtest_np)) + logits_of(lgb_pl.predict_proba(Xtest_np))
    if HAVE_CAT and cat_params is not None and len(cat_best) > 0:
        cat_pl = CatBoostClassifier(**{**cat_params, "iterations": int(np.mean(cat_best))})
        cat_pl.fit(X_aug, y_aug, sample_weight=w_aug, verbose=False)
        logits_pl += logits_of(cat_pl.predict_proba(Xtest_np))
    
    if ssl.get("enabled", False) and "p_tst" in ssl:
        logits_pl += logits_of(ssl["p_tst"])
    
    proba_pl = apply_temp(logits_pl, T_global)
    pred_test_pl = proba_pl.argmax(1) + 1

    return pred_test_pl