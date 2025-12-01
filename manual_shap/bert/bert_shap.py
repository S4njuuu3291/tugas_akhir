import pandas as pd
import pickle
import scipy.special
import numpy as np
import random
import itertools
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# ASUMSI: Objek-objek ini sudah dimuat dan didefinisikan di lingkungan Anda
# TARGET_LABEL = 'HS_Individual' 
# TARGET_INDEX = labels.index(TARGET_LABEL)
# tokenizer = AutoTokenizer.from_pretrained(SAVE_PATH)
# clf_bert = TFAutoModelForSequenceClassification.from_pretrained(SAVE_PATH)
# labels = [...]

# --- FUNGSI PREDIKSI BERT UNTUK KELAS TUNGGAL ---

def predict_proba_bert(sentence, clf_bert, tokenizer, target_index):
    inputs = tokenizer(
        sentence, 
        return_tensors="tf", 
        padding=True, 
        truncation=True
    )
    outputs = clf_bert(inputs)
    logits = outputs.logits
    probabilities = tf.sigmoid(logits).numpy()
    return probabilities[0, target_index]


def predict_on_subset_bert(subset_token_indices, original_tokens, clf_bert, tokenizer, target_index):
    masked_tokens = original_tokens.copy()
    num_tokens = len(original_tokens)
    
    for i in range(1, num_tokens - 1): 
        if i not in subset_token_indices:
            masked_tokens[i] = tokenizer.mask_token

    masked_sentence = tokenizer.convert_tokens_to_string(masked_tokens[1:-1])
    
    return predict_proba_bert(masked_sentence, clf_bert, tokenizer, target_index)


# --- FUNGSI INTI KERNEL SHAP MANUAL ---

def kernel_weight(M, k):
    if k == 0 or k == M:
        return 1e-6
    else:
        return (M - 1) / (scipy.special.comb(M, k) * k * (M - k))


def shap_kernel_instance(sentence, tokenizer, clf_bert, labels, target_label, num_samples=500):
    start_time = time.time()
    target_index = labels.index(target_label)
    
    # 1. Persiapan Token dan Fitur Aktif
    tokens = tokenizer.tokenize(sentence)
    original_tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
    
    # Indeks fitur yang bisa di-mask (token kalimat, tanpa [CLS]/[SEP])
    idx_features = list(range(1, len(original_tokens) - 1))
    fitur_aktif = tokens
    
    print(f"Fitur aktif ({len(fitur_aktif)}): {fitur_aktif}")
    print(f"Menjelaskan label: {target_label} (Index {target_index})")

    X_mask = []
    y = []
    weights = []

    M = len(idx_features)
    total_subsets = 2**M

    # 2. Sampling Subset
    if num_samples >= total_subsets:
        subsets = [list(s) for k in range(M + 1) for s in itertools.combinations(idx_features, k)]
    else:
        subsets = []
        subsets.append([])
        subsets.append(list(idx_features))
        
        while len(subsets) < num_samples:
            k = np.random.randint(0, M + 1)
            s = np.random.choice(idx_features, size=k, replace=False)
            subset = list(s)
            if subset not in subsets:
                subsets.append(subset)

    print(f"Jumlah subset yang dipakai: {len(subsets)}")

    # 3. Iterasi dan WLS Data Collection
    for subset in subsets:
        pred = predict_on_subset_bert(subset, original_tokens, clf_bert, tokenizer, target_index)
        y.append(pred)

        mask = np.zeros(M)
        for s_idx in subset:
            position = idx_features.index(s_idx)
            mask[position] = 1

        w = kernel_weight(M, len(subset))
        weights.append(w)

        X_mask.append(np.concatenate(([1], mask)))
        
    X_mask = np.array(X_mask)
    y = np.array(y)
    W = np.diag(weights)

    # 4. Weighted Least Squares
    beta = np.linalg.pinv(X_mask.T @ W @ X_mask) @ (X_mask.T @ W @ y)
    
    base_value = beta[0]
    shap_values = beta[1:]

    shap_results = list(zip(fitur_aktif, shap_values))
    
    sorted_shap_results = sorted(
        shap_results, 
        key=lambda item: abs(item[1]), 
        reverse=True
    )
    
    
    # 5. Output
    print("-" * 50)
    print(f"Kalimat: {sentence}")
    print(f"Penjelasan untuk: {target_label}")
    print("-" * 50)

    for f, val in sorted_shap_results:
        print(f"{f}: {val:.4f}")

    original_proba = predict_proba_bert(sentence, clf_bert, tokenizer, target_index)
    
    print("\nBaseline (E[f(x)]):", base_value)
    print("Baseline + ΣKernelSHAP:", base_value + shap_values.sum())
    print("Prediksi Asli f(x):", original_proba)
    
    end_time = time.time()
    print(f"Total waktu SHAP: {end_time - start_time:.4f} detik")
    
    return base_value, shap_values, fitur_aktif, sentence, target_label

def plot_shap_values(base_value, shap_values, fitur_aktif, sentence, target_label, top_n=10):
    
    df_shap = pd.DataFrame({
        'Fitur': fitur_aktif,
        'SHAP Value': shap_values
    })
    
    df_shap['Abs SHAP'] = df_shap['SHAP Value'].abs()
    df_shap = df_shap.sort_values(by='Abs SHAP', ascending=False)
    
    df_plot = df_shap.head(top_n)
    colors = ['red' if val > 0 else 'blue' for val in df_plot['SHAP Value']]
    
    df_plot = df_plot.iloc[::-1]
    colors = colors[::-1]
    
    plt.figure(figsize=(10, len(df_plot) * 0.6 + 1.5))
    
    plt.barh(df_plot['Fitur'], df_plot['SHAP Value'], color=colors, edgecolor='black')
    
    final_prediction = base_value + shap_values.sum()
    
    plt.axvline(x=base_value, color='gray', linestyle='--', linewidth=1.5, label='Baseline E[f(x)]')
    
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    
    plt.title(f'SHAP Feature Contribution untuk Kelas: {target_label}', fontsize=14)
    plt.xlabel('SHAP Value (Kontribusi terhadap Probabilitas Target)')
    plt.ylabel('Fitur Aktif (Token)')
    
    plt.text(
        0.98, 
        0.98, 
        f'Prediksi Akhir: {final_prediction:.3f}\nBaseline: {base_value:.3f}', 
        transform=plt.gca().transAxes, 
        ha='right', 
        va='top',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )
    
    plt.legend()
    plt.grid(axis='x', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()
    