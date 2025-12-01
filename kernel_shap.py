import itertools
import random
import numpy as np
import scipy

def predict_proba(x_vec, clf):
    """Ambil seluruh probabilitas untuk semua label (multilabel & binary)."""
    if hasattr(x_vec, "shape") and len(x_vec.shape) == 1:
        x_vec = x_vec.reshape(1, -1)

    proba = clf.predict_proba(x_vec)

    # ✅ Case 1: OneVsRestClassifier -> list of (n_samples, 2)
    if isinstance(proba, list):
        proba = np.array([p[0, 1] for p in proba])

    # ✅ Case 2: model biasa -> array (1, n_labels)
    elif isinstance(proba, np.ndarray):
        if proba.ndim == 2 and proba.shape[0] == 1:
            proba = proba[0]
        elif proba.ndim == 1:
            proba = proba
        else:
            proba = proba.flatten()

    # pastikan bentuk akhir (n_labels,)
    proba = np.atleast_1d(proba)
    return proba


def predict_on_subset(subset, x_instance, clf):
    """Hitung probabilitas multilabel untuk subset fitur tertentu."""
    x_masked = x_instance.copy().toarray().flatten()
    keep = set(subset)
    for j in range(len(x_masked)):
        if j not in keep:
            x_masked[j] = 0.0

    proba = clf.predict_proba(x_masked.reshape(1, -1))

    # ✅ Case 1: OneVsRestClassifier
    if isinstance(proba, list):
        proba = np.array([p[0, 1] for p in proba])

    # ✅ Case 2: model biasa
    elif isinstance(proba, np.ndarray):
        if proba.ndim == 2 and proba.shape[0] == 1:
            proba = proba[0]
        elif proba.ndim == 1:
            proba = proba
        else:
            proba = proba.flatten()

    return np.atleast_1d(proba)

def kernel_weight(M, k):
    if k == 0 or k == M:
        return 1e-6
    else:
        return (M-1)/scipy.special.comb(M,k)*k*(M-k)

def shap_kernel_instance(sentence, vectorizer, clf, num_samples=500, label_names=None):
    """Hitung Kernel SHAP manual untuk semua label dalam model multilabel.
       label_names: daftar nama label (opsional)."""
    # --- terima input DataFrame/Series juga ---
    if not isinstance(sentence, str):
        if hasattr(sentence, "values"):
            sentence = sentence.values[0]
        elif hasattr(sentence, "iloc"):
            sentence = sentence.iloc[0]

    x_instance = vectorizer.transform([sentence])
    idx_features = x_instance.nonzero()[1]
    features = vectorizer.get_feature_names_out()
    fitur_aktif = [features[i] for i in idx_features]
    print(f"\nKalimat:  {sentence}")
    print(f"Fitur aktif ({len(fitur_aktif)}): {fitur_aktif}")

    M = len(idx_features)
    total_subsets = 2**M
    print(f"Total fitur aktif = {M}, total subset = {total_subsets}")

    if num_samples >= total_subsets:
        subsets = [list(s) for k in range(M+1) for s in itertools.combinations(idx_features, k)]
    else:
        subsets = []
        for _ in range(num_samples):
            k = np.random.randint(0, M+1)
            s = np.random.choice(idx_features, size=k, replace=False)
            subsets.append(list(s))
    print(f"Jumlah subset yang dipakai: {len(subsets)}\n")

    sample_pred = predict_proba(x_instance, clf)
    n_labels = len(sample_pred)
    print(f"Model memiliki {n_labels} label.\n")

    # kalau label_names disediakan, gunakan itu
    if label_names is not None and len(label_names) == n_labels:
        label_ids = label_names
    else:
        label_ids = [f"Label {i}" for i in range(n_labels)]

    all_results = {}

    for label_idx, label_name in enumerate(label_ids):
        X_mask, y, weights = [], [], []
        for subset in subsets:
            pred = predict_on_subset(subset, x_instance, clf)
            y.append(pred[label_idx])

            mask = np.zeros(len(idx_features))
            for s in subset:
                idx = list(idx_features).index(s)
                mask[idx] = 1

            w = kernel_weight(M, len(subset))
            weights.append(w)
            X_mask.append(np.concatenate(([1], mask)))

        X_mask = np.array(X_mask)
        y = np.array(y)
        W = np.diag(weights)

        beta = np.linalg.pinv(X_mask.T @ W @ X_mask) @ (X_mask.T @ W @ y)
        base_value = np.ravel(beta[0])[0]
        shap_values = beta[1:].astype(float).flatten()

        all_results[label_name] = {
            "base_value": base_value,
            "shap_values": dict(zip(fitur_aktif, shap_values))
        }

        print(f"=== {label_name} ===")
        for f, val in zip(fitur_aktif, shap_values):
            print(f"{f}: {float(val):.4f}")
        print(f"Baseline: {base_value:.4f}")
        print(f"Prediksi asli {label_name}: {np.ravel(sample_pred[label_idx])[0]:.4f}\n")

    return all_results