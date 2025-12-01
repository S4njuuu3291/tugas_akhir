import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import itertools
import time

# ============================================================
# LOAD ARTIFACT
# ============================================================
@st.cache_resource
def load_artifact(path):
    with open(path, "rb") as f:
        return pickle.load(f)

artifact = load_artifact("../saved_models/hate_speech_models_20251201_065246.pkl")

vectorizer = artifact["vectorizer"]
labels = artifact["label_columns"]
features_list = vectorizer.get_feature_names_out()

# Load models
arovr_lr  = artifact["ovr_lr"]
arovr_svm = artifact["ovr_svm"]
arcc_lr   = artifact["cc_lr"]
arcc_svm  = artifact["cc_svm"]

# Load thresholds
th_ovr_lr  = np.array(artifact["thresh"]["ovr_lr"])
th_ovr_svm = np.array(artifact["thresh"]["ovr_svm"])
th_cc_lr   = np.array(artifact["thresh"]["cc_lr"])
th_cc_svm  = np.array(artifact["thresh"]["cc_svm"])

MODEL_MAP = {
    "OVR Logistic Regression": (arovr_lr,  th_ovr_lr),
    "OVR Calibrated SVM":      (arovr_svm, th_ovr_svm),
    "CC Logistic Regression":  (arcc_lr,   th_cc_lr),
    "CC Calibrated SVM":       (arcc_svm,  th_cc_svm),
}

# ============================================================
# PREDICT FUNCTION
# ============================================================
def predict_text(text, model_name):
    clf, thresholds = MODEL_MAP[model_name]
    vec = vectorizer.transform([text])
    probs = clf.predict_proba(vec)
    pred = (probs >= thresholds).astype(int)[0]
    return dict(zip(labels, pred)), probs[0]


# ============================================================
# SHAP KERNEL FUNCTIONS
# ============================================================
def predict_proba_single(vec, clf, target_idx):
    return clf.predict_proba(vec)[:, target_idx][0]


def predict_on_subset(subset, x_instance, clf, target_idx):
    x_masked = x_instance.copy().toarray().flatten()
    keep = set(subset)
    
    for j in range(len(x_masked)):
        if j not in keep:
            x_masked[j] = 0.0
    
    return predict_proba_single(x_masked.reshape(1, -1), clf, target_idx)


def kernel_weight(M, k):
    if k == 0 or k == M:
        return 1e-6
    return (M - 1) / (scipy.special.comb(M, k) * k * (M - k))


def shap_kernel_instance(sentence, vectorizer, clf, features_list, labels, target_label, num_samples=250):
    target_idx = labels.index(target_label)
    x_instance = vectorizer.transform([sentence])
    idx_features = x_instance.nonzero()[1]

    fitur_aktif = [features_list[i] for i in idx_features]
    M = len(idx_features)

    if M == 0:
        return 0, np.array([]), []

    # Subset sampling
    if num_samples >= 2**M:
        subsets = [list(s) for k in range(M+1) for s in itertools.combinations(idx_features, k)]
    else:
        subsets = [[], list(idx_features)]
        while len(subsets) < num_samples:
            k = np.random.randint(0, M+1)
            s = np.random.choice(idx_features, k, replace=False)
            if list(s) not in subsets:
                subsets.append(list(s))

    X_mask, y, weights = [], [], []
    idx_list = idx_features.tolist()

    for subset in subsets:
        pred = predict_on_subset(subset, x_instance, clf, target_idx)
        y.append(pred)

        mask = np.zeros(M)
        for s_idx in subset:
            mask[idx_list.index(s_idx)] = 1

        X_mask.append(np.concatenate(([1], mask)))
        weights.append(kernel_weight(M, len(subset)))

    X_mask = np.array(X_mask)
    W = np.diag(weights)
    y = np.array(y)

    beta = np.linalg.pinv(X_mask.T @ W @ X_mask) @ (X_mask.T @ W @ y)

    base_value = beta[0]
    shap_values = beta[1:]

    return base_value, shap_values, fitur_aktif


def plot_shap(base_value, shap_values, fitur_aktif, label):
    df = pd.DataFrame({"Feature": fitur_aktif, "SHAP": shap_values})
    df["abs"] = df["SHAP"].abs()
    df = df.sort_values("abs", ascending=True)

    fig, ax = plt.subplots(figsize=(8, len(df)*0.4 + 1))
    ax.barh(df["Feature"], df["SHAP"], color=["red" if x>0 else "blue" for x in df["SHAP"]], edgecolor="black")
    ax.set_title(f"SHAP Contribution — {label}")
    ax.set_xlabel("SHAP Value")
    ax.axvline(base_value, linestyle="--", color="gray")
    plt.tight_layout()
    return fig


# ============================================================
# HIGHLIGHT TEXT FUNCTION
# ============================================================
def highlight_text(sentence, fitur_aktif, shap_values):
    shap_map = dict(zip(fitur_aktif, shap_values))
    words = sentence.split()

    html = ""
    for w in words:
        score = 0

        # unigram
        if w in shap_map:
            score = shap_map[w]

        # bigram
        for f in shap_map:
            if " " in f and f in sentence and w in f.split():
                score += shap_map[f]

        intensity = min(1, abs(score) * 3)  
        if score > 0:
            color = f"rgba(255,0,0,{intensity})"
        elif score < 0:
            color = f"rgba(0,0,255,{intensity})"
        else:
            color = "rgba(255,255,255,0)"

        html += f"<span style='background-color:{color}; padding:3px; border-radius:4px; margin-right:4px'>{w}</span>"

    return html


# ============================================================
# STREAMLIT UI
# ============================================================

st.title("🔥 Hate Speech Multilabel Classification — SHAP Full Explanation + Highlight")
st.write("Skripsi Interface — Model, Prediksi, SHAP, dan Highlight Kata")

text = st.text_area("Masukkan kalimat untuk dianalisis:", height=150)
model_name = st.selectbox("Pilih Model:", list(MODEL_MAP.keys()))

if st.button("Prediksi"):
    if text.strip() == "":
        st.error("Input kalimat tidak boleh kosong.")
    else:

        # =============================
        # PREDIKSI
        # =============================
        pred, probs = predict_text(text, model_name)

        st.subheader("📌 Hasil Prediksi")
        st.success(" | ".join([f"{k}={v}" for k,v in pred.items()]))

        df_prob = pd.DataFrame({
            "Label": labels,
            "Prob": probs,
            "Threshold": MODEL_MAP[model_name][1]
        })
        st.dataframe(df_prob)

        # =============================
        # HIGHLIGHT
        # =============================
        st.subheader("✨ Highlight Kata Berdasarkan SHAP (Semua Label)")

        for label in labels:
            st.write(f"### 🔹 {label}")
            clf = MODEL_MAP[model_name][0]

            base_value, shap_vals, akt = shap_kernel_instance(
                text, vectorizer, clf, features_list, labels, label
            )

            if len(akt) == 0:
                st.warning("Tidak ada fitur aktif untuk label ini.")
                continue

            html = highlight_text(text, akt, shap_vals)
            st.markdown(html, unsafe_allow_html=True)
            st.markdown("---")

        # =============================
        # SHAP TABS
        # =============================
        st.subheader("📊 Grafik SHAP Per Label")

        tabs = st.tabs(labels)

        for i, label in enumerate(labels):
            with tabs[i]:
                clf = MODEL_MAP[model_name][0]

                base_value, shap_vals, akt = shap_kernel_instance(
                    text, vectorizer, clf, features_list, labels, label
                )

                if len(akt) == 0:
                    st.warning("Tidak ada fitur aktif.")
                    continue

                fig = plot_shap(base_value, shap_vals, akt, label)
                st.pyplot(fig)
