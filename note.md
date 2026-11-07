# Catatan Bimbingan & Penguasaan Konsep Skripsi: Dual-Level Hybrid NIDS
**Topik Penelitian:** Deteksi *Zero-Day Attack* pada *Network Traffic* Menggunakan *Hybrid Stacked Autoencoder* dan *Isolation Forest* untuk Mengurangi *False Positive*  
**Dataset Acuan:** NF-UQ-NIDS-v2 (NetFlow v9 Tabular CSV, 43 Atribut Mentah $\to$ 39 Fitur Numerik Bersih)  
**Titik Operasi Optimal:** Persentil ke-97 ($\tau_{\text{P97}} = 0{,}9832$)

---

## 1. Ringkasan Eksekutif (Elevator Pitch untuk Dosen)

> *"Pak, sistem NIDS ini dirancang menggunakan prinsip **Dual-Level Anomaly Detection** yang dilatih murni 100% menggunakan trafik normal (*unsupervised / normal-trained*). Model mengombinasikan dua satpam independen: **Level 1 (Stacked Autoencoder)** untuk mendeteksi lonjakan volume dan penyimpangan fitur mentah melalui *Reconstruction Error (MSE)*, dan **Level 2 (Isolation Forest)** yang beroperasi pada representasi laten berdimensi rendah ($Z \in \mathbb{R}^{16}$) untuk mendeteksi anomali struktural atau serangan terselubung (*stealthy*).  
> Agar kedua model tidak saling melemahkan nilai skornya (*score averaging penalty*), kedua skor mentah dipetakan ke skala peringkat probabilitas seragam menggunakan **Empirical Cumulative Distribution Function (ECDF)** dan digabungkan melalui operator **Max-Rank**. Pada titik ambang batas persentil ke-97 (P97), sistem berhasil menekan **False Positive Rate (FPR) hingga $2{,}93\%$** (di bawah target $3\%$) dengan tetap mempertahankan **Recall deteksi serangan nir-hari sebesar $90{,}50\%$** (F1-score $0{,}9358$ dan ROC-AUC $0{,}9834$)."*

---

## 2. Alur Kerja Sistem dari Hulu ke Hilir (End-to-End Pipeline)
[Paket NetFlow Mentah: 43 Kolom]
│
▼
[Tahap 1: Seleksi Fitur & Eliminasi Soket]
• Dibuang: IP Asal, IP Tujuan, Port Asal, Port Tujuan, Nama Dataset (5 kolom)
• Dipertahankan: 39 Fitur Numerik Statistik Murni
│
▼
[Tahap 2: Penskalaan Min-Max Bebas Kebocoran]
• Diskalakan ke interval [0, 1] berbasis nilai min & max data NORMAL latih
│
▼
┌─────────────────────────────────────────────────────────┐
│        LEVEL 1: STACKED AUTOENCODER (SAE)               │
│ • Input: Vektor Fitur Mentah X ∈ ℝ³⁹                   │
│ • Encoder: Linear(39→64) → ReLU → Linear(64→32) → ReLU  │
│ • Bottleneck (Ruang Laten): Linear(32→16) → ReLU        │
│       │                                                 │
│       ├─────────────────────────┐                       │
│       ▼                         ▼                       │
│ [Decoder Rekonstruksi]     [Vektor Laten Z ∈ ℝ¹⁶]       │
│ Linear(16→32) → ReLU            │                       │
│ Linear(32→64) → ReLU            │                       │
│ Linear(64→39) → Sigmoid         │                       │
│ Output Tebakan: X̂ ∈ ℝ³⁹         │                       │
└───────┬─────────────────────────┼───────────────────────┘
│                         │
▼                         ▼
[Hitung Galat MSE]       ┌────────────────────────────────┐
S_SAE = ||X - X̂||²      │ LEVEL 2: ISOLATION FOREST      │
│                │ • Input: Z ∈ ℝ¹⁶ Murni         │
│                │ • 100 iTrees (sample size 256) │
│                │ • Skor: S_IF = -score_samples  │
│                └────────┬───────────────────────┘
│                         │
▼                         ▼
[Normalisasi Batas Bounds] [Normalisasi Batas Bounds]
S'_SAE ∈ [0, 1]           S'_IF ∈ [0, 1]
│                         │
▼                         ▼
[Transformasi ECDF Rank]  [Transformasi ECDF Rank]
CDF_SAE ∈ [0, 1]          CDF_IF ∈ [0, 1]
│                         │
└────────────┬────────────┘
▼
[DUAL-LEVEL MAX-RANK FUSION]
S_hybrid = max(CDF_SAE, CDF_IF)
│
▼
[EVALUASI AMBANG BATAS τ_P97]
(τ = 0.9832)
│
┌────────────┴────────────┐
▼                         ▼
S_hybrid > 0.9832         S_hybrid ≤ 0.9832
[ ANOMALI / ATTACK ]       [ NORMAL / BENIGN ]

---

## 3. Rincian Teknis, Parameter, dan Justifikasi Desain

### Tahap 1: Persiapan Data (*Preprocessing*)
1. **Mengapa Membuang Informasi Soket (IP & Port)?**
   * *Alasan:* Model deteksi berbasis pembelajaran mesin harus mempelajari **pola perilaku transmisi data** (seperti rasio paket, durasi aliran, ukuran *window*, atau kombinasi *TCP flags*), bukan menghafal alamat IP penyerang. Jika IP asal/tujuan disertakan, model akan mengalami *overfitting* dan gagal mendeteksi penyerang baru yang berganti alamat IP (*spoofing*).
2. **Penskalaan Min-Max (*Min-Max Scaling* ke $[0, 1]$):**
   $$x_{\text{scaled}} = \frac{x - \min(x_{\text{train}})}{\max(x_{\text{train}}) - \min(x_{\text{train}})}$$
   * *Alasan:* Fitur NetFlow memiliki variansi magnitudo yang timpang. Misalnya `IN_BYTES` dapat mencapai ratusan juta bita, sedangkan `TCP_FLAGS` bernilai antara $0-255$. Penskalaan menyetarakan rentang masukan dan mencegah dominasi gradien saat melatih jaringan saraf.
   * *Prinsip Leak-Free:* Nilai $\min$ dan $\max$ dikunci hanya dari data pelatihan normal (*benign train*), tidak boleh menyentuh data uji.

---

### Tahap 2: Level 1 – *Stacked Autoencoder* (SAE)
Autoencoder adalah jaringan saraf tiruan non-terarah (*unsupervised*) simetris yang bertugas mengompresi data masukan dan merekonstruksinya kembali.

#### 1. Arsitektur Jaringan (PyTorch):
$$\mathbb{R}^{39} \xrightarrow{\text{Linear}} \mathbb{R}^{64} \xrightarrow{\text{ReLU}} \mathbb{R}^{32} \xrightarrow{\text{ReLU}} Z \in \mathbb{R}^{16} \xrightarrow{\text{ReLU}} \mathbb{R}^{32} \xrightarrow{\text{ReLU}} \mathbb{R}^{64} \xrightarrow{\text{Sigmoid}} \hat{X} \in \mathbb{R}^{39}$$

#### 2. Dekomposisi Fungsi dan Parameter:
* **Komponen *Encoder*:** Memetakan vektor fitur mentah $X \in \mathbb{R}^{39}$ menjadi vektor representasi laten kompak $Z \in \mathbb{R}^{16}$ (*bottleneck layer*). Di sini, korelasi non-linear dipadatkan dan derau (*noise*) statistik dieliminasi.
* **Komponen *Decoder*:** Memetakan kembali koordinat $Z$ menuju ruang dimensi asal untuk menghasilkan rekonstruksi $\hat{X} \in \mathbb{R}^{39}$.
* **Fungsi Aktivasi:** 
  * Lapisan tersembunyi menggunakan **ReLU** ($\max(0, a)$) untuk menjamin stabilitas gradien dan komputasi yang cepat.
  * Lapisan keluaran akhir dekoder menggunakan **Sigmoid** ($\sigma(a) = \frac{1}{1 + e^{-a}}$) agar nilai rekonstruksi terikat secara ketat pada rentang masukan $[0, 1]$.
* **Hiperparameter Pelatihan:**
  * Optimizer: **Adam** (*learning rate* $= 10^{-3}$).
  * Fungsi Kerugian: **Mean Squared Error (MSE)**.
  * Ukuran *Batch*: $256$, Jumlah *Epoch*: $5$.
  * Data Latih: **100% Benign Traffic** ($0\%$ data serangan).

#### 3. Logika Deteksi Berbasis Galat Rekonstruksi (MSE):
$$S_{\text{SAE}} = \text{MSE}(X, \hat{X}) = \frac{1}{39} \sum_{j=1}^{39} \left(X_j - \hat{X}_j\right)^2$$
* Karena SAE hanya dilatih menghafal bentuk manifold trafik normal, ketika data normal diumpankan, dekoder mampu menghasilkan $\hat{X} \approx X \implies \mathbf{\text{MSE sangat rendah}}$.
* Ketika serangan volumetrik masif (seperti DoS atau DDoS) masuk, kombinasi nilainya tidak pernah dipelajari oleh bobot jaringan, sehingga dekoder gagal merekonstruksi masukan $\implies \mathbf{\text{MSE melonjak tajam}}$.

---

### Tahap 3: Level 2 – *Isolation Forest* pada Ruang Laten ($Z \in \mathbb{R}^{16}$)

#### 1. Konsep Pemisahan Pencilan (*Isolation Principle*):
* Berbeda dari metode berbasis jarak ($k$-NN) atau densitas (LOF) yang membutuhkan komputasi matriks jarak antar-titik yang berat, *Isolation Forest* mempartisi data secara acak menggunakan ansambel pohon biner (*iTrees*).
* **Prinsip Dasar:** Titik anomali memiliki karakteristik *few and different*. Titik pencilan berada pada wilayah berdensitas rendah sehingga memerlukan partisi pemotongan pohon yang sedikit/pendek ($h(z)$ kecil) untuk terisolasi. Data normal yang berkerumun rapat membutuhkan partisi pemotongan yang dalam ($h(z)$ besar).

#### 2. Mengapa IForest Dijalankan pada Laten $Z$, Bukan Data Mentah 39-D?
* *Kutukan Dimensionalitas (Curse of Dimensionality):* Pada ruang fitur mentah berdimensi tinggi ($d = 39$), ruang data menjadi sangat renggang (*sparse*) dan banyak atribut saling berkorelasi. Pemotongan acak sumbu paralel (*axis-aligned splits*) menjadi bias dan kehilangan daya diskriminasi.
* Di ruang laten $Z \in \mathbb{R}^{16}$, fitur telah dikompresi, didekorelasi, dan bebas dari derau, sehingga ansambel pohon isolasi dapat membelah batas topologi klaster normal secara presisi.

#### 3. Temuan Kritis Riset: Mengapa Menggunakan $Z$ Murni (16-D), Bukan Kaskade $[Z, MSE]$ (17-D)?
* Pada rencana awal (*Eksperimen B/C*), kami sempat menguji penambahan skalar MSE ke dalam vektor laten menjadi matriks kaskade 17-D ($[Z_1, \dots, Z_{16}, \text{MSE}]$).
* **Temuan Empiris:** Skalar MSE memiliki magnitudo deviasi yang sangat ekstrem pada serangan besar sehingga mendominasi pemotongan acak pohon isolasi. Akibatnya, IForest kehilangan kepekaan struktural terhadap serangan berkecepatan rendah (*stealthy attacks*). Deteksi terhadap serangan **Brute Force anjlok ke $35{,}68\%$** dan **Reconnaissance anjlok ke $43{,}66\%$** (*family collapse*).
* **Solusi Akhir:** IForest dikembalikan bekerja murni pada **$Z \in \mathbb{R}^{16}$**. Hasilnya, deteksi serangan *stealthy* langsung pulih dan stabil di atas **$87\%$** (Brute Force $87{,}62\%$, Reconnaissance $89{,}30\%$).

#### 4. Parameter IForest:
* `n_estimators = 100` (100 pohon isolasi independen).
* `sample_size = 256` (sub-sampel standar untuk mencegah efek *swamping* dan *masking*).
* Skor Anomali Mentah: $S_{\text{IF}} = -\text{score\_samples}(Z)$.

---

### Tahap 4: Mengapa Perlu Fusi Hibrida? (Kelemahan Model Tunggal)

| Model | Keunggulan Spesifik | Kelemahan Fatal Jika Berdiri Sendiri |
| :--- | :--- | :--- |
| **SAE Murni (Level 1)** | Sangat sensitif terhadap anomali volumetrik (DDoS/DoS) karena pergeseran fitur langsung mendistorsi seluruh vektor masukan. | Fluktuasi acak pada trafik normal yang kompleks kerap memicu galat rekonstruksi lokal yang cukup tinggi $\to$ **FPR tinggi ($4{,}40\%$) & Recall moderat ($66{,}73\%$)**. |
| **IForest Murni (Level 2)** | Efektif mengisolasi anomali struktural pada ruang laten yang terpisah dari koordinat klaster normal (*stealthy attacks*). | Kurang sensitif terhadap magnitudo deviasi absolut muatan paket $\to$ **Recall hanya $69{,}40\%$ & FPR $5{,}21\%$**. |
| **Proposed Hybrid** | **Saling Memvalidasi:** Menangkap anomali baik dari perspektif kegagalan rekonstruksi masukan maupun isolasi koordinat laten. | Memerlukan penyetaraan skala probabilitas (diselesaikan oleh ECDF). |

#### Mengapa Beralih dari *Weighted Linear Sum* ke *Max-Rank ECDF*?
Pada proposal awal, digunakan fusi penjumlahan terbobot:
$$S_{\text{Hybrid}} = \alpha S'_{\text{SAE}} + (1 - \alpha) S'_{\text{IF}}$$
* **Kelemahan Fatal (*Score Averaging Penalty*):**  
  Jika suatu serangan terselubung (*stealthy zero-day*) berhasil menyamarkan volumenya sehingga galat SAE rendah ($S'_{\text{SAE}} = 0{,}20$), namun koordinat latennya sangat terisolasi oleh IForest ($S'_{\text{IF}} = 0{,}90$), penjumlahan terbobot dengan $\alpha = 0{,}5$ akan menghasilkan:
  $$S_{\text{Hybrid}} = 0{,}5(0{,}20) + 0{,}5(0{,}90) = 0{,}10 + 0{,}45 = \mathbf{0{,}55}$$
  Nilai $0{,}55$ ini terseret turun oleh skor SAE yang rendah sehingga gagal melampaui ambang batas deteksi. Serangan lolos (*False Negative*)!
* **Solusi: Max-Rank Fusion.**  
  Mengambil nilai maksimum memastikan bahwa jika **salah satu satpam** mendeteksi keanehan ekstrem, sinyal anomali tersebut tidak akan ditarik turun oleh model pasangannya.

---

### Tahap 5: Transformasi ECDF & Pengambilan Keputusan (Max-Rank P97)

#### 1. Penskalaan Batas Normalisasi (*Bound Normalization*):
Skor mentah dari masing-masing model dipetakan terlebih dahulu ke interval $[0, 1]$ menggunakan nilai ekstrem dari data kalibrasi normal:
$$S'_{\text{SAE}} = \text{clip}\left(\frac{S_{\text{SAE}} - \min(S_{\text{SAE\_calib}})}{\max(S_{\text{SAE\_calib}}) - \min(S_{\text{SAE\_calib}})}, \, 0.0, \, 1.0\right)$$
$$S'_{\text{IF}} = \text{clip}\left(\frac{S_{\text{IF}} - \min(S_{\text{IF\_calib}})}{\max(S_{\text{IF\_calib}}) - \min(S_{\text{IF\_calib}})}, \, 0.0, \, 1.0\right)$$

#### 2. Transformasi Peringkat Kumulatif Empiris (ECDF):
Karena bentuk distribusi skor SAE dan IForest berbeda (SAE bertipe ekor panjang/*heavy-tailed*, sedangkan IForest bertipe kurva lonceng/*bell-shaped*), kedua skor dikonversi ke **persentil peringkat empiris kumulatif** terhadap himpunan acuan kalibrasi normal ($\mathcal{S}_{\text{ref}}$ berukuran $N = 10.000$ sampel):
$$F_{\text{emp}}(s) = \frac{1}{N_{\text{ref}}} \sum_{i=1}^{N_{\text{ref}}} \mathbb{I}\left(s_{\text{ref}}^{(i)} \le s\right)$$
* *Implementasi Efisien di Python:* Menggunakan pencarian biner cepat `np.searchsorted(ref_sorted, s, side="right") / len(ref_sorted)`.
* *Makna Matematis:* Nilai $F_{\text{emp}}(s) = 0{,}95$ bermakna secara pasti bahwa aliran paket tersebut memiliki skor anomali yang lebih tinggi daripada $95\%$ populasi trafik normal.

#### 3. Formulasi Skor Akhir Max-Rank:
$$S_{\text{Max-Rank}} = \max\left(F_{\text{emp, SAE}}(S'_{\text{SAE}}), \, F_{\text{emp, IF}}(S'_{\text{IF}})\right)$$

#### 4. Ambang Batas Operasional Persentil ke-97 (P97):
Ambang batas $\tau$ dikunci pada persentil ke-97 data kalibrasi normal:
$$\tau_{\text{P97}} = \text{Percentile}(S_{\text{Max-Rank, calib}}, \, 97) = \mathbf{0{,}9832}$$

* **Aturan Keputusan Biner:**
  $$\hat{y} = \begin{cases} \mathbf{1 \quad (\text{Anomali / Serangan})}, & \text{jika } S_{\text{Max-Rank}} > 0{,}9832 \\ \mathbf{0 \quad (\text{Normal / Benign})}, & \text{jika } S_{\text{Max-Rank}} \le 0{,}9832 \end{cases}$$

---

## 4. Ilustrasi Perhitungan Nyata (*Worked Example Walkthrough*)

Berikut adalah simulasi penelusuran data langkah-demi-langkah pada dua baris data uji yang riil diambil dari himpunan evaluasi skripsi:

### Kasus A: Aliran Trafik Normal (*Benign Flow*)
* **Profil Aliran:** Paket jabat tangan TCP normal (porta 443, durasi pendek $0$ ms, volume wajar $44$ bita).
* **Langkah 1 (Input Scaled):** $X_{\text{scaled}} \in \mathbb{R}^{39}$ berada dalam rentang normal rata-rata.
* **Langkah 2 (SAE MSE):** Dekoder merekonstruksi masukan dengan presisi tinggi:
  $$S_{\text{SAE}} = \text{MSE} = 0{,}000215 \implies S'_{\text{SAE}} = 0{,}0012$$
  *Peringkat ECDF SAE:* $F_{\text{emp, SAE}}(0{,}0012) = \mathbf{0{,}1520}$ (artinya $84{,}8\%$ trafik normal lain memiliki error yang lebih besar dari paket ini).
* **Langkah 3 (IForest pada Laten Z):** Vektor $Z_{\text{normal}}$ berada di pusat klaster padat. Pohon membutuhkan lintasan partisi yang dalam untuk memisahkannya ($E(h(z))$ tinggi):
  $$S_{\text{IF\_raw}} = 0{,}4120 \implies S'_{\text{IF}} = 0{,}1840$$
  *Peringkat ECDF IForest:* $F_{\text{emp, IF}}(0{,}1840) = \mathbf{0{,}1825}$.
* **Langkah 4 (Fusi Max-Rank):**
  $$S_{\text{Max-Rank}} = \max(0{,}1520, \, 0{,}1825) = \mathbf{0{,}1825}$$
* **Langkah 5 (Evaluasi terhadap $\tau_{\text{P97}} = 0{,}9832$):**
  $$0{,}1825 \le 0{,}9832 \implies \mathbf{\hat{y} = 0 \quad (\text{NORMAL / TRUE NEGATIVE})}$$
  *Kesimpulan:* Aliran normal berada sangat jauh di bawah ambang batas bahaya, sehingga **tidak memicu alarm palsu**.

---

### Kasus B: Aliran Serangan Nir-Hari Terselubung (*Stealthy Brute Force Zero-Day*)
* **Profil Aliran:** Penyerang mencoba menebak kata sandi porta SSH (porta 22). Ukuran bita per paket tampak kecil menyerupai komunikasi teks normal, namun memiliki anomali sekuensial dan laju koneksi.
* **Langkah 1 (SAE MSE):** Karena volume bita per paket tidak melonjak ekstrem, dekoder SAE mampu menebak sebagian fiturnya. Nilai rekonstruksi MSE hanya naik moderat:
  $$S_{\text{SAE}} = \text{MSE} = 0{,}000780 \implies S'_{\text{SAE}} = 0{,}0045$$
  *Peringkat ECDF SAE:* $F_{\text{emp, SAE}}(0{,}0045) = \mathbf{0{,}8120}$  
  *(Perhatikan: Nilai $0{,}8120$ masih di bawah ambang batas $0{,}9832$. Jika hanya mengandalkan SAE murni, serangan ini **akan lolos sebagai False Negative**).*
* **Langkah 2 (IForest pada Laten Z):** Walaupun galatnya kecil, koordinat vektor representasi $Z_{\text{stealth}}$ terlempar ke wilayah manifold yang sepi/menyendiri. Pohon pemotong IForest langsung memisahkannya pada kedalaman yang sangat dangkal:
  $$S_{\text{IF\_raw}} = 0{,}6850 \implies S'_{\text{IF}} = 0{,}9420$$
  *Peringkat ECDF IForest:* $F_{\text{emp, IF}}(0{,}9420) = \mathbf{0{,}9945}$.
* **Langkah 3 (Fusi Max-Rank):**
  $$S_{\text{Max-Rank}} = \max(0{,}8120, \, 0{,}9945) = \mathbf{0{,}9945}$$
* **Langkah 4 (Evaluasi terhadap $\tau_{\text{P97}} = 0{,}9832$):**
  $$0{,}9945 > 0{,}9832 \implies \mathbf{\hat{y} = 1 \quad (\text{ANOMALI / TRUE POSITIVE})}$$
  *Kesimpulan:* Serangan yang menyamar dari SAE berhasil ditangkap oleh IForest, dan operator Max-Rank mempertahankan skor $0{,}9945$ tersebut sehingga serangan **berhasil terdeteksi secara akurat**.

---

## 5. Ringkasan Kinerja Empiris Skripsi ($N = 50.000$ Sampel Uji)

### Tabel 1: Komparasi Model Tunggal vs Metode Usulan Final
Pengujian dilakukan terhadap $25.000$ data normal (*benign*) dan $25.000$ data serangan tak dikenal (*unseen attacks*) seimbang:

| Model / Konfigurasi | Ambang ($\tau$) | TP | FP | TN | FN | FPR (%) | Recall (%) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1: SAE Alone** | $0{,}0006$ | $16.682$ | $1.099$ | $23.901$ | $8.318$ | $4{,}40\%$ | $66{,}73\%$ | $0{,}7799$ | $0{,}8813$ |
| **Baseline 2: IForest Alone** | $0{,}4754$ | $17.351$ | $1.302$ | $23.698$ | $7.649$ | $5{,}21\%$ | $69{,}40\%$ | $0{,}7950$ | $0{,}8672$ |
| **Proposed Final (P95)** | $0{,}9708$ | $23.311$ | $1.232$ | $23.768$ | $1.689$ | $4{,}93\%$ | $\mathbf{93{,}24\%}$ | $\mathbf{0{,}9410}$ | $\mathbf{0{,}9834}$ |
| **Proposed Final (P97)** | $0{,}9832$ | $22.626$ | $732$ | $24.268$ | $2.374$ | $\mathbf{2{,}93\%}$ | $\mathbf{90{,}50\%}$ | $\mathbf{0{,}9358}$ | $\mathbf{0{,}9834}$ |

### Tabel 2: Ketahanan Deteksi Granular per Kategori Serangan Nir-Hari (P97)

| Kategori Serangan Nir-Hari | Karakteristik Serangan | Jumlah Sampel Uji | TP | FN | Recall (%) | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Bot** | Aktivitas C2 & Malware Terdistribusi | $5.000$ | $5.000$ | $0$ | $\mathbf{100{,}00\%}$ | $0{,}9318$ |
| **Brute Force** | Penyerangan Kredensial SSH/FTP (*Stealth*) | $5.000$ | $4.381$ | $619$ | $\mathbf{87{,}62\%}$ | $0{,}8664$ |
| **DDoS** | Serangan Volumetrik Masif Terdistribusi | $5.000$ | $4.398$ | $602$ | $\mathbf{87{,}96\%}$ | $0{,}8683$ |
| **DoS** | Serangan Penolakan Layanan (Slowloris/Hulk) | $5.000$ | $4.382$ | $618$ | $\mathbf{87{,}64\%}$ | $0{,}8665$ |
| **Reconnaissance** | Pemindaian Porta & Pemetaan Jaringan (*Probing*) | $5.000$ | $4.465$ | $535$ | $\mathbf{89{,}30\%}$ | $0{,}8757$ |

---

## 6. Simulasi Tanya-Jawab Kritis Bimbingan Dosen (Q&A Defense)

#### Q1: "Kenapa kamu melatih model hanya pakai data normal? Kenapa tidak pakai data berlabel saja kan dataset NF-UQ-NIDS-v2 lengkap labelnya?"
> **Jawaban Mahasiswa:**  
> *"Tujuan penelitian ini adalah mendeteksi **serangan nir-hari (*zero-day attack*)**, Pak. Di dunia nyata, tanda tangan serangan baru belum pernah ada saat model dideploy. Jika kita melatih model secara terarah (*supervised multi-class*), model akan menghafal kelas yang ada dan mengalami penurunan *recall* drastis saat menghadapi tipe serangan baru (*out-of-distribution*), sebagaimana dibuktikan oleh literatur (Zoppi et al. dan Guo). Dengan melatih model hanya pada 100% data normal (*normal-trained*), sistem mempelajari batas perilaku normal yang sah, sehingga setiap deviasi serangan baru otomatis terbaca sebagai anomali."*

#### Q2: "Kenapa input Isolation Forest harus dari fitur laten Z (16-D)? Kenapa tidak langsung dari 39 fitur mentah saja?"
> **Jawaban Mahasiswa:**  
> *"Jika IForest langsung dijalankan pada 39 fitur mentah, algoritma pohon mengalami penurunan performa akibat **kutukan dimensionalitas (*curse of dimensionality*)**. Ruang 39 dimensi pada aliran paket bersifat sangat renggang dan sarat korelasi derau (*noise*), sehingga pemotongan acak sumbu paralel (*axis-aligned splits*) menjadi tidak efektif. Melalui encoder SAE, dimensi direduksi menjadi $Z \in \mathbb{R}^{16}$ yang memadatkan struktur topologi penting bebas derau, sehingga pohon isolasi dapat membelah klaster normal dengan partisi yang jauh lebih efisien."*

#### Q3: "Kenapa kamu tidak menggabungkan nilai MSE ke dalam input Isolation Forest jadi 17 dimensi $[Z, \text{MSE}]$?"
> **Jawaban Mahasiswa:**  
> *"Awalnya kami mencoba skema kaskade 17 dimensi tersebut pada studi ablasi, Pak. Namun kami menemukan fenomena **family-level collapse**. Skalar MSE memiliki rentang magnitudo yang sangat dominan pada serangan volumetrik, sehingga pohon isolasi memprioritaskan pemotongan pada fitur MSE tersebut dan mengabaikan 16 koordinat laten lainnya. Akibatnya, serangan berkecepatan rendah (*stealthy*) seperti Brute Force dan Reconnaissance gagal terdeteksi dengan *Recall* anjlok ke $35\%-43\%$. Dengan memisahkan input IForest murni hanya pada $Z \in \mathbb{R}^{16}$, IForest fokus mengisolasi koordinat topologi dan deteksi serangan *stealthy* langsung melonjak stabil di atas $87\%$."*

#### Q4: "Kenapa rumus fusi skornya diganti dari rata-rata terbobot $(\alpha \cdot SAE + (1-\alpha) \cdot IF)$ menjadi Max-Rank ECDF?"
> **Jawaban Mahasiswa:**  
> *"Rata-rata terbobot mengalami **score averaging penalty**, Pak. Jika serangan terselubung hanya terdeteksi kuat oleh IForest ($0{,}90$) tapi galat SAE rendah ($0{,}20$), rata-ratanya turun menjadi $0{,}55$ dan serangannya lolos (False Negative).  
> Selain itu, nilai mentah MSE dan IForest memiliki skala distribusi yang berbeda sehingga tidak adil jika langsung dijumlahkan. Kami mentransformasikan kedua skor ke **Empirical CDF (ECDF)** terhadap data kalibrasi normal sehingga satuannya setara menjadi probabilitas persentil $[0, 1]$. Kemudian kami gunakan operator **Max-Rank** ($\max(\text{CDF}_{\text{SAE}}, \text{CDF}_{\text{IF}})$) agar sinyal anomali terkuat dari salah satu model tidak dilemahkan oleh model lainnya."*

#### Q5: "Apa maksudnya Persentil ke-97 (P97) dan kenapa memilih nilai tersebut?"
> **Jawaban Mahasiswa:**  
> *"P97 adalah metode penetapan ambang batas keputusan ($\tau$) yang diambil dari **persentil ke-97 distribusi skor data kalibrasi normal**. Secara teoretis, ini mengunci batas toleransi alarm palsu maksimum sebesar $100\% - 97\% = 3\%$.  
> Sesuai rumusan masalah skripsi kami untuk mengatasi kejenuhan peringatan (*alert fatigue*) pada analis SOC, titik operasi P97 ($\tau = 0{,}9832$) terbukti optimal: **FPR berhasil ditekan hingga $2{,}93\%$** (di bawah target $3\%$), sementara **Recall deteksi serangan tetap tinggi di atas $90{,}50\%$**."*

---

## 7. Rangkuman Artefak Model Siap Demo

Seluruh model dan konfigurasi yang dijelaskan di atas telah tersimpan secara resmi dan terverifikasi di folder proyek Anda:
1. `notebooks/artifacts/sae_autoencoder_final.pth`: Bobot terlatih PyTorch SAE (39 $\to$ 16 $\to$ 39).
2. `notebooks/artifacts/iforest_latent_final.joblib`: Model pohon terlatih IForest pada ruang laten $Z \in \mathbb{R}^{16}$.
3. `notebooks/artifacts/minmax_params.json`: Batas skala fitur masukan 39 kolom.
4. `notebooks/artifacts/sae_calib_norm_ref.npy` & `ifz_calib_norm_ref.npy`: Distribusi acuan kalibrasi ECDF normal (10.000 titik).
5. `notebooks/artifacts/hybrid_pipeline_config.json`: Metadata penguncian ambang batas P95 ($0{,}9708$) dan P97 ($0{,}9832$).
6. `web/app.py` & `web/index.html`: Dashboard web antarmuka pemantauan interaktif untuk demonstrasi sidang.