# 📊 Evaluation Metrics for Multilabel Classification

## 1. **Micro-F1**

- **Definisi**: Menghitung TP, FP, FN secara **global** (gabungan semua label).
- **Kegunaan**: Bagus kalau dataset **imbalanced**, karena label dominan punya pengaruh lebih besar.
- **Formula**:

  $$
  \text{Micro-F1} = \frac{2 \times \sum TP}{2 \times \sum TP + \sum FP + \sum FN}
  $$

- **Contoh**:

  - Label dominan (HS, Abusive) diprediksi bagus → Micro-F1 tinggi.
  - Label jarang (Religion) diprediksi jelek → pengaruh kecil ke skor.

---

## 2. **Macro-F1**

- **Definisi**: Hitung F1-score **per label**, lalu ambil rata-ratanya.
- **Kegunaan**: Adil ke semua label (label jarang tetap dihitung).
- **Formula**:

  $$
  \text{Macro-F1} = \frac{1}{L}\sum_{i=1}^L F1_i
  $$

- **Contoh**:

  - Kalau model gagal total di label minoritas → Macro-F1 jatuh.
  - Memberi gambaran apakah model “fair” ke semua label.

---

## 3. **Hamming Loss**

- **Definisi**: Proporsi label yang salah diprediksi (FP + FN), dihitung per label per sample.
- **Formula**:

  $$
  \text{Hamming Loss} = \frac{\text{Jumlah salah keseluruhan}}{N \times L}
  $$

- **Interpretasi**:

  - Nilai 0 = sempurna.
  - Semakin kecil semakin baik.

- **Contoh**:

  - 4 sampel × 3 label = 12 total prediksi.
  - 3 salah → Hamming Loss = 3/12 = **0.25**.

---

## 4. **Subset Accuracy (Exact Match Ratio)**

- **Definisi**: Persentase sampel di mana **semua label** diprediksi benar.
- **Formula**:

  $$
  \text{Subset Accuracy} = \frac{1}{N} \sum_{i=1}^N \mathbf{1}(y_i = \hat{y}_i)
  $$

- **Interpretasi**:

  - Sangat ketat → kalau ada 1 label salah, seluruh sample dihitung salah.

- **Contoh**:

  - Dari 4 tweet, hanya 1 yang semua labelnya benar → Subset Accuracy = 1/4 = **0.25**.

---

## 5. **Zero-One Loss**

- **Definisi**: Kebalikan dari subset accuracy.
- **Formula**:

  $$
  \text{Zero-One Loss} = 1 - \text{Subset Accuracy}
  $$

- **Contoh**:

  - Subset Accuracy = 0.25 → Zero-One Loss = 0.75.

- **Catatan**: Metrik ini “jahat”, karena 1 label salah = penalti penuh.

---

# 🔑 Ringkasan Perbandingan

| Metrik              | Level Perhitungan        | Karakteristik Utama                              | Kapan Dipakai                 |
| ------------------- | ------------------------ | ------------------------------------------------ | ----------------------------- |
| **Micro-F1**        | Global (gabungan label)  | Bagus untuk dataset imbalanced                   | Performance overall           |
| **Macro-F1**        | Per-label lalu dirata    | Adil ke semua label, sensitif ke label minoritas | Fairness across labels        |
| **Hamming Loss**    | Per-label per-sample     | Mengukur proporsi label salah, fleksibel         | Multilabel umum               |
| **Subset Accuracy** | Per-sample (exact match) | Sangat ketat, 1 salah = gagal total              | Kasus all-or-nothing          |
| **Zero-One Loss**   | Per-sample               | Lawan subset accuracy (penalti penuh)            | Analisis keras (strict check) |

---
