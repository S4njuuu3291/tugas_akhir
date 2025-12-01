# 🎯 Multilabel Classification Techniques

## 1. Binary Relevance (BR)

- **Cara kerja**: Satu label = satu model terpisah (anggap semua label independen).
- **Kelebihan**:
  - Simpel, mudah diimplementasikan.
  - Bisa dilatih paralel.
- **Kekurangan**:
  - Tidak mempertimbangkan hubungan antar label.

---

## 2. One-vs-Rest (OvR)

- **Mirip BR**, istilah ini biasanya dipakai di multi**class** → tapi bisa dipakai juga di multi**label**.
- **Cara kerja**: Tiap label dilatih classifier biner.
- **Kelebihan**: Implementasi mudah, stabil.
- **Kekurangan**: Sama seperti BR → abaikan korelasi antar label.

---

## 3. Classifier Chains (CC)

- **Cara kerja**: Model dilatih berantai, prediksi label sebelumnya jadi fitur tambahan untuk label berikutnya.
- **Kelebihan**:
  - Menangkap hubungan antar label.
- **Kekurangan**:
  - Error bisa menular ke label selanjutnya (error propagation).
  - Training/prediksi lebih lambat.

---

## 4. Probabilistic Classifier Chains (PCC)

- **Cara kerja**: Versi probabilistik dari CC → coba berbagai urutan label dengan probabilitas.
- **Kelebihan**:
  - Lebih robust, akurasi bisa lebih tinggi.
- **Kekurangan**:
  - Komputasi sangat berat (eksponensial).

---

## 5. Label Powerset (LP)

- **Cara kerja**: Kombinasi label dianggap satu kelas baru → masalah multilabel diubah jadi multiclass.
- **Kelebihan**:
  - Tangkap hubungan label penuh.
- **Kekurangan**:
  - Jumlah kelas bisa meledak kalau label banyak (combinatorial explosion).

---

## 6. RAkEL (Random k-Labelsets)

- **Cara kerja**: Ambil subset kecil dari label (misalnya 3 label), buat model multiclass, lalu ensemble hasilnya.
- **Kelebihan**:
  - Lebih scalable daripada LP.
  - Masih bisa menangkap sebagian hubungan label.
- **Kekurangan**:
  - Lebih kompleks, butuh tuning subset size.

---

## 7. Neural Network Multilabel

- **Cara kerja**:
  - Satu model NN dengan output neuron sebanyak jumlah label.
  - Output pakai **sigmoid** (bukan softmax).
- **Kelebihan**:
  - Natural untuk multilabel.
  - Hidden layer bisa belajar hubungan antar label implicit.
- **Kekurangan**:
  - Butuh data besar & tuning hati-hati.

---

## 8. Ensemble Methods

- **Cara kerja**: Gabungan beberapa pendekatan (misalnya OvR + Bagging, CC dengan random order, voting antar model).
- **Kelebihan**:
  - Bisa meningkatkan akurasi.
- **Kekurangan**:
  - Lebih berat, lebih rumit.

---

# 📊 Tabel Perbandingan Singkat

| Teknik            | Menangkap Korelasi Label | Kompleksitas       | Catatan Utama             |
| ----------------- | ------------------------ | ------------------ | ------------------------- |
| **BR / OvR**      | ❌ Tidak                 | ⭐ Rendah          | Simpel, cepat             |
| **CC**            | ✅ Ya                    | ⭐⭐ Sedang        | Bisa error domino         |
| **PCC**           | ✅ Ya (probabilistik)    | ⭐⭐⭐ Tinggi      | Akurasi tinggi, lambat    |
| **LP**            | ✅ Ya (penuh)            | ⭐⭐⭐ Tinggi      | Kombinasi label = kelas   |
| **RAkEL**         | ✅ Ya (parsial)          | ⭐⭐ Sedang        | Lebih scalable dari LP    |
| **NN Multilabel** | ✅ Ya (implicit)         | ⭐⭐ Sedang        | Cocok untuk dataset besar |
| **Ensemble**      | ✅ Ya (variasi)          | ⭐⭐ Sedang–Tinggi | Gabungkan kekuatan model  |
