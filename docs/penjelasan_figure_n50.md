# Penjelasan Hasil Eksperimen — Grid Final (N = 50, Heterogenitas Perangkat)

> Bahan presentasi ke dosen pembimbing. Sumber angka: `results_system/analysis_n50/` dan `notebooks/figures/results_system_n50/` (CSV ANOVA, Kruskal–Wallis, waktu simulasi), ditambah uji Mann–Whitney pada *time-to-target*. Semua nilai adalah rerata dari 5 *seed* kecuali disebutkan lain.

---

## 0. Konteks Eksperimen

### 0.1 Konfigurasi

| Komponen | Nilai |
|---|---|
| Algoritma agregasi | FedAvg (McMahan dkk., 2017) |
| Jumlah klien (N) / klien per round (m) | 50 / 5 (fraksi partisipasi 10%) |
| Communication round | 100 |
| Local epoch / learning rate / batch | 3 / 0,01 (SGD, momentum 0,9) / 32 |
| Dataset & model | MNIST (CNN sederhana), CIFAR-10 (CNN dengan BatchNorm) |
| Label skew | Partisi Dirichlet, α ∈ {0,1; 0,5; 1,0} (Hsu dkk., 2019) |
| Seed | 42, 123, 456, 789, 1024 → 3 strategi × 2 dataset × 3 α × 5 seed = **90 eksperimen** |
| Target akurasi (A2) | MNIST 85%, CIFAR-10 70% |
| Heterogenitas perangkat | Disimulasikan (lihat 0.3) |

### 0.2 Strategi yang dibandingkan

| Strategi | Mekanisme seleksi | Rujukan |
|---|---|---|
| **Random** | Sampling seragam tanpa pengembalian; baseline FedAvg | McMahan dkk. (2017) |
| **Performance-based (Oort)** | *Statistical utility* `|Bᵢ|·√(mean Loss²)` + *temporal uncertainty* + *system utility* `(T/tᵢ)^α` dengan α = 2, *pacer* yang melonggarkan T, eksplorasi ε-greedy (0,9 → 0,2) yang memprioritaskan perangkat cepat | Lai dkk. (2021) |
| **Fairness-aware (FairFedCS)** | Reputasi Beta `rᵢ = (aᵢ+1)/(aᵢ+bᵢ+2)` yang diperbarui berdasarkan tanda Shapley Value, *virtual queue* Lyapunov `Qᵢ`, seleksi top-m berdasarkan `CSIᵢ = σ·rᵢ + Qᵢ` (σ = 0,6) | Shi dkk. (2023) |

### 0.3 Model heterogenitas perangkat (asumsi simulasi)

Setiap klien diberi kecepatan komputasi dan *bandwidth* tetap yang diambil dari distribusi log-normal (median 200 sampel/s dan 5 MB/s, σ = 1). Kedua nilai ini diacak dari aliran RNG terpisah, sehingga **tidak berkorelasi dengan data/label skew**. Durasi round klien adalah `tᵢ = |Bᵢ|·E/speedᵢ + 2·ukuran_model/bandwidthᵢ`, dan durasi satu round sinkron = klien terpilih yang paling lambat. Pendekatan emulasi ini mengikuti metodologi evaluasi Oort sendiri, yang juga mengemulasikan *runtime* perangkat alih-alih mengukurnya pada perangkat fisik. **Seluruh metrik waktu dalam dokumen ini adalah waktu simulasi**, bukan waktu nyata komputasi.

### 0.4 Penyimpangan implementasi dari paper (wajib disampaikan secara terbuka)

1. **Oort — batas partisipasi (*blacklist*) tidak diaktifkan.** Paper Oort (§7.1) mengeluarkan klien dari daftar eksploitasi setelah terpilih lebih dari 10 kali. Pada implementasi ini parameter tersebut ada, tetapi bernilai nonaktif (`max_participation=None`). Ini berkontribusi langsung pada tingginya ketimpangan partisipasi Oort (B3) yang dibahas di Figure 8–9.
2. **Oort — jendela *pacer* W = 5 round** (paper: W = 20 untuk ratusan round). T dinyatakan sebagai persentil durasi klien (awal persentil ke-30, Δ = +5 poin), mengikuti implementasi resmi Oort di FedScale.
3. **FairFedCS — exact Shapley Value** (enumerasi 2⁵ = 32 subset per round), bukan aproksimasi GTG-Shapley seperti pada paper. Secara definisi ini lebih presisi; biaya komputasinya layak karena m = 5.
4. **Metrik B1/B2** dihitung dari akurasi model global pada **data latih lokal** setiap klien, dan hanya dari **round terakhir**.

### 0.5 Catatan statistik

- n = 5 per sel. Asumsi normalitas residual ANOVA **dilanggar** untuk A1, B1, dan B2 (Shapiro–Wilk p < 0,03); untuk B3, asumsi homogenitas varians dilanggar (Levene p ≤ 0,036). Karena itu, setiap kesimpulan ANOVA juga dikonfirmasi dengan uji non-parametrik **Kruskal–Wallis** per (dataset, α).
- Banyak uji dilakukan tanpa koreksi perbandingan ganda. Nilai p yang berada tepat di sekitar 0,05 sebaiknya dibaca sebagai indikasi, bukan bukti kuat.
- Dengan n = 5 per kelompok, nilai p terkecil yang mungkin pada uji Mann–Whitney dua sisi adalah 0,0079.

---

## 1. Figure 01 — Distribusi Label per Klien (`01_partition_per_alpha.png`)

**Yang ditampilkan.** Histogram jumlah sampel per kelas untuk 10 dari 50 klien MNIST (seed 42), pada tiga nilai α (baris).

**Hasil.**
- Pada **α = 0,1**, sebagian besar klien didominasi satu atau dua kelas. Contohnya, klien C2 memiliki 99% sampel kelas 4, dan C1 memiliki 81% sampel kelas 3. Ukuran data antarklien juga sangat timpang, dari 67 hingga 2.994 sampel.
- Pada **α = 1,0**, setiap klien memiliki hampir semua kelas, dengan proporsi kelas dominan sekitar 20–54%.
- Statistik partisi (seluruh seed): proporsi kelas dominan rata-rata turun dari ±65% (α = 0,1) ke ±28–29% (α = 1,0). Rata-rata jumlah kelas yang hilang per klien turun dari ±3,6–3,8 menjadi ±0. Klien terkecil pada CIFAR-10 α = 0,1 hanya memiliki **2 sampel** (MNIST: 16 sampel).

**Kaitan teori.** Ini mengonfirmasi bahwa manipulasi label skew berjalan sesuai rancangan: konsentrasi Dirichlet yang kecil menghasilkan distribusi label per klien yang terkonsentrasi pada sedikit kelas (Hsu dkk., 2019). Pengaturan ini merupakan standar *benchmark* yang juga dipakai Li dkk. (2022).

**Catatan objektif.** Partisi Dirichlet per kelas **juga menghasilkan *quantity skew*** (ukuran data antarklien sangat berbeda), terutama pada α kecil. Li dkk. (2022) memperlakukan *quantity skew* sebagai kategori Non-IID tersendiri. Jadi efek yang kita sebut "efek label skew" sebenarnya adalah efek gabungan *label skew* dan *quantity skew*. Ini perlu disebutkan di Batasan Penelitian.

---

## 2. Figure 02 — Heatmap Proporsi Label (`02_partition_heatmap.png`)

**Yang ditampilkan.** Proporsi (%) setiap kelas (kolom) pada ke-50 klien (baris), MNIST seed 42.

**Hasil.** Pada α = 0,1, heatmap tampak "jarang" dengan banyak sel gelap terisolasi, artinya setiap klien hanya memiliki sedikit kelas dengan proporsi tinggi. Pada α = 1,0, warna merata dan pucat, artinya distribusi label tiap klien mendekati distribusi global.

**Kaitan teori.** Figure ini memvisualisasikan definisi *label distribution skew*: distribusi kelas tiap klien menyimpang dari distribusi global (Zhang dkk., 2022). Kondisi pada α = 0,1 identik dengan ilustrasi rumah sakit di Latar Belakang proposal, yaitu setiap institusi didominasi jenis kasus tertentu.

---

## 3. Figure 03 — Heatmap Rerata Akurasi Global A1 (`03_heatmap_A1.png`)

**Yang ditampilkan.** Rerata akurasi global (%) pada round terakhir untuk setiap kombinasi strategi × α.

**Hasil.**

| | Random | Oort | FairFedCS |
|---|---|---|---|
| MNIST α = 0,1 | 98,27 | 98,47 | 98,83 |
| MNIST α = 0,5 | 99,19 | 99,14 | 99,16 |
| MNIST α = 1,0 | 99,18 | 99,22 | 99,24 |
| CIFAR-10 α = 0,1 | 52,32 | 55,64 | 54,72 |
| CIFAR-10 α = 0,5 | 70,57 | 70,05 | 69,75 |
| CIFAR-10 α = 1,0 | 73,14 | 72,29 | 72,61 |

Two-way ANOVA:

| Dataset | Efek strategi | Efek α | Interaksi |
|---|---|---|---|
| MNIST | F = 0,89; p = 0,419; η²p = 0,047 | F = 13,07; **p < 0,001**; η²p = 0,421 | p = 0,564 |
| CIFAR-10 | F = 0,20; p = 0,820; η²p = 0,011 | F = 189,6; **p < 0,001**; η²p = 0,913 | p = 0,453 |

Kruskal–Wallis per (dataset, α) juga tidak signifikan untuk A1 di keenam sel (p = 0,15–0,83).

**Interpretasi.** **Strategi seleksi klien tidak berpengaruh signifikan terhadap akurasi global akhir**, sedangkan label skew berpengaruh sangat besar. Pada CIFAR-10, akurasi turun ±18–21 poin persentase dari α = 1,0 ke α = 0,1, dan α menjelaskan ±91% variasi (η²p = 0,913).

**Kaitan teori.** Penurunan akurasi FedAvg di bawah label skew sejalan dengan Hsu dkk. (2019) dan Li dkk. (2022), yang menempatkan label skew sebagai bentuk Non-IID paling merugikan. Mekanismenya dijelaskan Wang dkk. (2020): pembaruan lokal pada data yang sangat berbeda menyebabkan *objective inconsistency* (*client drift*), sehingga model global menyimpang dari optimum global. Paper Oort melaporkan peningkatan akurasi akhir sebesar 1,3–9,8%; **klaim ini tidak tereplikasi** pada eksperimen ini.

**Catatan objektif.**
- **Efek plafon (*ceiling effect*) pada MNIST:** semua sel berada di 98,3–99,2%, sehingga ruang untuk membedakan strategi sangat sempit. Skala warna MNIST hanya 96,5–100, yang membuat selisih di bawah 1 poin terlihat lebih besar dari sebenarnya.
- **Temuan tambahan (sensitivitas metrik):** jika akurasi dirata-rata atas 5 round terakhir, efek strategi pada MNIST menjadi signifikan (p = 0,007), dengan Oort sedikit lebih rendah di α = 0,1 (98,25% vs 98,52–98,60%). Besarnya kurang dari 0,4 poin, jadi signifikan secara statistik tetapi tidak bermakna secara praktis.

---

## 4. Figure 04 — Distribusi A1 per Strategi × α (`04_boxplot_A1.png`)

**Yang ditampilkan.** Sebaran akurasi akhir dari 5 seed untuk setiap sel.

**Hasil.**
- Pada **CIFAR-10 α = 0,1**, sebaran FairFedCS paling lebar: 46,85–65,68% (simpangan baku 7,14), dibandingkan Random 47,19–56,43% (3,43) dan Oort 53,6–57,3% (1,63).
- Rerata FairFedCS di sel ini dinaikkan oleh satu seed (1024: 65,68%), sedangkan median-nya 54,78%.
- Seed 123 menghasilkan akurasi terendah untuk Random (47,19%) maupun FairFedCS (46,85%). Ini menunjukkan sebagian besar variasi berasal dari **partisi data** (seed menentukan partisi), bukan dari strategi.
- Pada MNIST α = 0,1, terdapat satu *outlier* Random di ±96,2%.

**Kaitan teori.** FairFedCS memilih klien secara **deterministik** (top-m berdasarkan CSI), dan reputasi diperbarui secara kumulatif. Akibatnya, lintasan seleksi bersifat *path-dependent*: perbedaan kecil pada round awal (yang ditentukan partisi) dapat menghasilkan himpunan klien "berreputasi tinggi" yang sangat berbeda. Oort memakai sampling probabilistik dari *pool* berutilitas tinggi, sehingga lebih stabil antar-seed.

**Catatan objektif.** Stabilitas antar-seed (varians) adalah sudut pandang yang tidak dirumuskan di proposal, tetapi relevan secara praktis. Pada skew berat, Oort memberikan hasil yang paling dapat diprediksi.

---

## 5. Figure 05 — A2: Round Menuju Target (`05_A2_rounds_to_target.png`)

**Yang ditampilkan.** Rerata (± simpangan baku) jumlah round hingga akurasi global pertama kali mencapai target.

**Hasil.**

| | Random | Oort | FairFedCS |
|---|---|---|---|
| MNIST α = 0,1 | 6,4 | 7,2 | 5,8 |
| MNIST α = 0,5 | 1,8 | 2,0 | 2,0 |
| MNIST α = 1,0 | 1,0 | 1,2 | 1,2 |
| CIFAR-10 α = 0,5 | 65,8 | **74,6** | 60,0 |
| CIFAR-10 α = 1,0 | 43,0 | **54,4** | 46,2 |
| CIFAR-10 α = 0,1 | target tidak tercapai (0/5) | 0/5 | 0/5 |

**Interpretasi.** Diukur dalam **jumlah round**, Oort justru paling lambat di CIFAR-10 (±9–11 round lebih banyak dari Random). Pada CIFAR-10 α = 0,1, tidak ada strategi yang mencapai 70% dalam 100 round, sehingga titik tersebut kosong pada grafik.

**Kaitan teori — ini bukan anomali.** Oort mengoptimalkan *time-to-accuracy* (waktu nyata), bukan *round-to-accuracy*. Komponen *system utility* `(T/tᵢ)^α` sengaja menurunkan prioritas klien lambat, termasuk klien lambat yang secara statistik sangat berguna. Paper Oort secara eksplisit menyebut ini sebagai *trade-off* antara efisiensi statistik dan efisiensi sistem, yang diatur oleh *pacer*. Konsekuensinya, setiap round Oort lebih singkat tetapi sedikit kurang informatif secara statistik, sehingga dibutuhkan lebih banyak round. Figure 13–14 menunjukkan bahwa secara waktu, Oort tetap jauh lebih cepat.

**Catatan objektif.**
- A2 pada MNIST α ≥ 0,5 jenuh (1–2 round), sehingga tidak dapat membedakan strategi.
- Target 70% untuk CIFAR-10 tidak tercapai pada α = 0,1. Batasan proposal yang menyebut target ini "realistis" perlu direvisi.
- A2 tidak dimasukkan ke ANOVA, karena run yang tidak mencapai target menghasilkan nilai kosong yang akan membiaskan perbandingan.

---

## 6. Figure 06 — Progres Akurasi per Round (`06_accuracy_progression.png`)

**Yang ditampilkan.** Kurva akurasi global per round, MNIST, satu seed (42).

**Hasil.** Ketiga strategi mencapai plateau ±98–99%. Pada α = 0,1, Oort tertinggal di ±20 round pertama dan lebih berosilasi. Random juga menunjukkan penurunan sesaat di sekitar round ke-10. Pada α ≥ 0,5, kurva ketiga strategi praktis berimpit.

**Kaitan teori.** Osilasi pada α = 0,1 adalah manifestasi *client drift* (Wang dkk., 2020). Dengan hanya 5 klien per round yang masing-masing didominasi sedikit kelas, model global "ditarik" ke arah kelas yang kebetulan terwakili pada round tersebut. Keterlambatan awal Oort konsisten dengan fase eksplorasinya (ε awal 0,9), yang memprioritaskan klien cepat, bukan klien paling informatif.

**Catatan objektif.** Figure ini hanya satu seed dan hanya MNIST, sehingga bersifat ilustratif. Untuk kesimpulan gunakan Figure 13, yang merata-ratakan 5 seed dan mencakup CIFAR-10.

---

## 7. Figure 07 — Fairness Akurasi Antarklien B1 & B2 (`07_B1_B2_vs_alpha.png`)

**Yang ditampilkan.** B1 (simpangan baku akurasi antarklien) dan B2 (koefisien Gini akurasi antarklien) terhadap α. Arah sumbu-x dibalik, sehingga ke kanan berarti skew makin berat.

**Hasil.**
- Pada kedua dataset, B1 dan B2 meningkat tajam pada α = 0,1. Label skew memperburuk ketimpangan akurasi antarklien.
- Pada **CIFAR-10** terjadi **persilangan (*crossover*)** antarstrategi:

| B1 CIFAR-10 | Random | Oort | FairFedCS |
|---|---|---|---|
| α = 1,0 | **0,055** (terbaik) | 0,075 | 0,072 |
| α = 0,5 | **0,074** (terbaik) | 0,081 | 0,092 |
| α = 0,1 | 0,216 (terburuk) | **0,179** (terbaik) | 0,188 |

- ANOVA CIFAR-10: efek utama strategi tidak signifikan (B1 p = 0,580; B2 p = 0,437), tetapi **interaksi strategi × α signifikan** (B1 F = 5,56; p = 0,0014; η²p = 0,382. B2 F = 3,38; p = 0,019; η²p = 0,273). Pada MNIST, interaksi tidak signifikan (p > 0,49).
- Kruskal–Wallis CIFAR-10: signifikan untuk B1 pada α = 0,1 (p = 0,048) dan α = 1,0 (p = 0,009), serta B2 pada α = 1,0 (p = 0,012).

**Interpretasi.** Tidak ada strategi yang selalu paling adil secara akurasi. Pada skew ringan hingga sedang, Random menghasilkan akurasi antarklien yang paling merata. Pada skew berat (CIFAR-10 α = 0,1), Random justru paling timpang.

**Kaitan teori (sebagai hipotesis penjelas).**
- Pada skew ringan, sampling seragam merupakan estimator tak bias dari objektif FedAvg dengan partisipasi penuh. Cho dkk. (2022) menunjukkan bahwa seleksi yang bias, misalnya ke klien dengan *loss* tinggi, mempercepat konvergensi tetapi menambahkan **bias solusi** yang tidak hilang. Bias ini dapat membuat model lebih condong ke sebagian klien, sehingga dispersi akurasi naik. Ini konsisten dengan Random terbaik di α ≥ 0,5.
- Pada skew berat, *statistical utility* Oort memprioritaskan klien dengan *loss* tinggi, yaitu klien yang kelasnya paling buruk direpresentasikan model global. Ini berfungsi seperti mekanisme korektif terhadap klien minoritas, sejalan dengan motivasi Zhang dkk. (2022) tentang penurunan akurasi pada klien berdistribusi minoritas.

**Catatan objektif.**
- **Temuan interaksi ini sensitif terhadap cara pengukuran.** Jika B1 dirata-rata atas 5 round terakhir, interaksinya tidak lagi signifikan (p = 0,125). Artinya sebagian efek berasal dari fluktuasi round terakhir. Sampaikan sebagai *indikasi*, bukan temuan yang kokoh.
- B1 diukur pada data latih lokal, bukan data uji lokal. Ini membatasi interpretasi sebagai "fairness performa" yang sesungguhnya.
- Label "Accuracy Variance" pada B1 secara teknis adalah simpangan baku, bukan varians. Sebaiknya istilahnya disesuaikan di skripsi.

---

## 8. Figure 08 — Fairness Partisipasi B3 (`08_B3_participation_fairness.png`)

**Yang ditampilkan.** Simpangan baku jumlah kali setiap klien dipilih selama 100 round. Semakin rendah, semakin merata.

**Hasil.**

| B3 | Random | Oort | FairFedCS |
|---|---|---|---|
| MNIST α = 1,0 / 0,5 / 0,1 | 3,16 / 3,17 / 3,19 | 4,43 / 4,60 / **5,84** | 4,43 / 4,42 / 4,49 |
| CIFAR-10 α = 1,0 / 0,5 / 0,1 | 3,19 / 3,17 / 3,19 | 5,80 / 6,16 / **8,35** | 4,49 / 4,82 / 4,70 |

- ANOVA: efek strategi **sangat signifikan** dengan efek besar (MNIST F = 50,9, η²p = 0,739; CIFAR-10 F = 182,4, η²p = 0,910). Interaksi juga signifikan (MNIST p = 0,013; CIFAR-10 p < 0,001), yang didorong oleh B3 Oort yang memburuk seiring skew.
- Kruskal–Wallis signifikan pada **keenam** sel (p = 0,002–0,012), sehingga hasil ini robust terhadap pelanggaran asumsi.
- Urutan yang konsisten di semua kondisi: **Random < FairFedCS < Oort**.

**Validasi internal.** Untuk seleksi seragam dengan peluang m/N = 0,1 selama R = 100 round, jumlah seleksi tiap klien berdistribusi Binomial(100; 0,1), dengan simpangan baku teoretis √(100·0,1·0,9) = **3,0**. Nilai B3 Random yang teramati (3,16–3,19) sesuai dengan teori ini, yang mengonfirmasi bahwa implementasi seleksi acak benar. B3 Random juga identik lintas α untuk seed yang sama, karena seleksi acak memang tidak bergantung pada data.

**Kaitan teori dan anomali utama.** Judul figure memuat hipotesis "fairness strategy seharusnya terendah", dan **hipotesis ini tidak terpenuhi**. Penjelasannya terletak pada definisi fairness di FairFedCS (Shi dkk., 2023):
- *Virtual queue* bertambah sebesar `cᵢ = ε·rᵢ` ketika klien tidak terpilih (ε = m/N). Klien berreputasi tinggi mengakumulasi "utang fairness" lebih cepat dan karena itu lebih sering dipilih.
- Dengan demikian, FairFedCS menjamin keadilan **sebanding dengan reputasi/kontribusi**, bukan pemerataan absolut. Klien yang kontribusi Shapley-nya sering negatif secara sengaja lebih jarang dipilih.
- Random, yang tidak memiliki preferensi apa pun, secara definisi menghasilkan sebaran partisipasi paling merata.

Oort paling timpang karena tiga mekanisme sekaligus: eksploitasi utilitas statistik, penalti klien lambat (*system utility*), dan dalam implementasi ini, tidak aktifnya batas partisipasi 10 kali (lihat 0.4). Ketimpangan Oort memburuk pada α = 0,1, karena *loss* antarklien jauh lebih bervariasi sehingga utilitas lebih terkonsentrasi.

**Catatan objektif.** Rumusan Latar Belakang proposal ("FairFedCS secara eksplisit menyeimbangkan partisipasi antarklien") perlu diperhalus menjadi "menyeimbangkan partisipasi **sebanding dengan reputasi klien**". Judul figure ini juga sebaiknya diganti agar netral, misalnya "B3: Participation Fairness per Strategi vs Alpha".

---

## 9. Figure 09 — Pola Partisipasi per Klien (`09_participation_pattern.png`)

**Yang ditampilkan.** Jumlah kali setiap dari 50 klien dipilih (MNIST α = 0,1, seed 42). Garis merah menunjukkan rerata (10 kali).

**Hasil.**
- **Random** (B3 = 2,81): semua klien terpilih 4–16 kali, tersebar merata di sekitar rerata.
- **Oort** (B3 = 5,88): satu klien (C5) dipilih **34 kali**, sementara beberapa klien hanya 1–3 kali.
- **FairFedCS** (B3 = 4,40): rentang 4–20 kali, lebih timpang dari Random tetapi tanpa dominasi ekstrem.
- Pada CIFAR-10 α = 0,1, klien paling sering dipilih mencapai 29–43 kali pada Oort, dibandingkan 16–19 (Random) dan 19–23 (FairFedCS). **Tidak ada klien yang tidak pernah dipilih** pada strategi mana pun, karena fase eksplorasi Oort dan *queue* FairFedCS memastikan setiap klien pernah dicoba.

**Kaitan teori.** Figure ini adalah bukti mikro dari Figure 08. Dominasi C5 pada Oort menggambarkan risiko yang diantisipasi paper Oort sendiri, yaitu ketergantungan pada sedikit klien berutilitas tinggi. Paper Oort mengatasinya dengan membatasi partisipasi, sedangkan pada implementasi ini batas tersebut tidak aktif.

**Catatan objektif.** Figure ini hanya satu seed, jadi gunakan sebagai ilustrasi. Pola kualitatif yang sama konsisten di seluruh seed (lihat angka CIFAR-10 di atas).

---

## 10. Figure 10 — Sebaran A1 vs B1 (`10_scatter_A1_vs_B1.png`)

**Yang ditampilkan.** Setiap titik adalah satu eksperimen (90 titik). Warna menunjukkan strategi, bentuk menunjukkan α.

**Hasil.**
- Titik-titik membentuk **kluster berdasarkan α, bukan berdasarkan strategi**: lingkaran (α = 0,1) berada di kiri atas, sedangkan kotak dan segitiga di kanan bawah. Di dalam setiap kluster, ketiga warna saling bertumpuk.
- Hubungan A1–B1 **negatif**: eksperimen dengan akurasi global lebih tinggi cenderung juga memiliki dispersi akurasi antarklien lebih rendah.
- Satu titik FairFedCS α = 0,1 pada CIFAR-10 (±65,7%, seed 1024) terpisah jauh dari kelompoknya dan mendekati kluster α = 0,5.

**Interpretasi.** Pada eksperimen ini, **tidak terlihat *trade-off* antara akurasi global dan fairness akurasi** dalam arti "meningkatkan satu mengorbankan yang lain". Keduanya bergerak bersama, dan penggerak utamanya adalah tingkat skew.

**Kaitan teori.** Proposal mengasumsikan adanya *trade-off* akurasi–fairness. Hasil ini menunjukkan bahwa di bawah label skew, ketimpangan akurasi antarklien terutama merupakan **gejala dari model global yang buruk** akibat *client drift* (Wang dkk., 2020), bukan konsekuensi dari strategi yang memprioritaskan akurasi. Ini adalah temuan yang berguna untuk Pembahasan dan relevan dengan pertanyaan terbuka tentang fairness di FL yang diidentifikasi Kairouz dkk. (2021).

---

## 11. Figure 11 — Sebaran A1 vs B3 (`11_scatter_A1_vs_B3.png`)

**Yang ditampilkan.** Akurasi global terhadap fairness partisipasi, per eksperimen.

**Hasil.** Berbeda dengan Figure 10, titik-titik **berlapis secara vertikal berdasarkan strategi**: Random di bawah (B3 ±2,8–3,8), FairFedCS di tengah (±4–5,5), dan Oort di atas (±4–10). Pada sumbu akurasi, ketiganya tetap bertumpuk.

**Interpretasi.** B3 adalah satu-satunya metrik yang benar-benar "dikendalikan" oleh strategi seleksi. Namun, konsentrasi partisipasi yang lebih tinggi (Oort) **tidak menghasilkan akurasi global yang lebih tinggi**. Dari sisi akurasi akhir, ketimpangan partisipasi Oort tidak memberi imbal balik; imbal baliknya ada pada dimensi waktu (Figure 13–14).

---

## 12. Figure 12 — Perbandingan Ranking Antar-Dataset (`12_cross_dataset_ranking.png`)

**Yang ditampilkan.** Rerata A1 per strategi untuk MNIST dan CIFAR-10, per α.

**Hasil.** Batang dalam satu dataset hampir sama tinggi. Urutan rerata berbeda antar-dataset:

| α | Urutan MNIST | Urutan CIFAR-10 |
|---|---|---|
| 0,1 | FairFedCS > Oort > Random | Oort > FairFedCS > Random |
| 0,5 | Random > FairFedCS > Oort | Random > Oort > FairFedCS |
| 1,0 | FairFedCS > Oort > Random | Random > FairFedCS > Oort |

**Interpretasi.** Satu-satunya pola yang konsisten adalah **Random terendah pada α = 0,1** di kedua dataset. Namun, karena perbedaan A1 antarstrategi tidak signifikan di semua sel (Figure 03), urutan-urutan ini **tidak dapat ditafsirkan sebagai ranking yang bermakna**.

**Catatan objektif.** Secara visual, figure ini lemah karena sumbu-y dimulai dari 0, sehingga perbedaan yang relevan (<3 poin) tidak terlihat. Untuk skripsi disarankan diganti dengan *dot plot* rerata ± interval kepercayaan per dataset.

---

## 13. Figure 13 — Time-to-Accuracy (`13_time_to_accuracy_cifar10.png`, `13_time_to_accuracy_mnist.png`)

**Yang ditampilkan.** Akurasi global terhadap **waktu simulasi kumulatif**, rata-rata 5 seed. Kurva yang naik lebih cepat ke kiri berarti lebih efisien waktu.

**Hasil.**
- Kurva Oort bergeser jelas ke kiri di semua kondisi. Oort menyelesaikan 100 round dalam ±3.200–4.100 detik, sedangkan Random dan FairFedCS membutuhkan ±6.000–9.200 detik.
- *Time-to-target* (detik simulasi):

| | Random | Oort | FairFedCS | Percepatan Oort | Mann–Whitney (Oort vs R / vs F) |
|---|---|---|---|---|---|
| MNIST α = 0,1 | 556 | **174** | 647 | 3,2–3,7× | p = 0,008 / 0,008 |
| MNIST α = 0,5 | 118 | **44** | 111 | 2,5–2,7× | p = 0,016 / 0,008 |
| MNIST α = 1,0 | 41 | **32** | 63 | 1,3–2,0× | p = 0,116 / 0,095 (tidak signifikan) |
| CIFAR-10 α = 0,5 | 3.983 | **2.507** | 3.713 | 1,5–1,6× | p = 0,016 / 0,016 |
| CIFAR-10 α = 1,0 | 2.485 | **1.699** | 2.818 | 1,5–1,7× | p = 0,016 / 0,008 |
| CIFAR-10 α = 0,1 | tidak tercapai | tidak tercapai | tidak tercapai | — | — |

- Pada CIFAR-10 α = 0,1, Oort mencapai ±55% dalam ±3.200 detik, sedangkan strategi lain baru mencapai ±52–54% setelah ±8.600 detik. Untuk **anggaran waktu yang sama**, Oort jauh lebih unggul.

**Kaitan teori.** Ini adalah **replikasi temuan utama Oort**: paper melaporkan percepatan *time-to-accuracy* 1,2×–14,1×, dan percepatan 1,5×–3,7× pada eksperimen ini berada dalam rentang tersebut. Mekanismenya adalah Persamaan 1 Oort, yaitu perkalian *statistical utility* dengan *system utility* `(T/tᵢ)^α`, ditambah eksplorasi yang memprioritaskan perangkat cepat. Percepatan paling besar terjadi pada MNIST α = 0,1, karena di sana heterogenitas ukuran data (Figure 01) memperbesar perbedaan durasi antarklien.

**Catatan objektif.**
1. Hasil ini **bergantung pada asumsi distribusi kecepatan perangkat** (log-normal, σ = 1). Dengan heterogenitas lebih kecil, keunggulan Oort akan mengecil; tanpa heterogenitas, keunggulan ini hilang. Pada eksperimen awal tanpa heterogenitas, Oort tidak lebih cepat.
2. Waktu simulasi hanya memodelkan pelatihan dan komunikasi klien. **Biaya komputasi sisi server tidak dimodelkan**: FairFedCS memerlukan 32 evaluasi model per round untuk Shapley Value, sehingga waktu nyatanya ±4–6× lebih lama dibanding strategi lain. Dalam praktik, FairFedCS akan tampak lebih lambat dari yang ditunjukkan figure ini.
3. Sumbu-x adalah rerata waktu kumulatif dari 5 seed. Kurva menunjukkan perilaku rata-rata, bukan run individual.

---

## 14. Figure 14 — Rata-rata Durasi Round (`14_mean_round_duration.png`)

**Yang ditampilkan.** Waktu simulasi rata-rata per round.

**Hasil.**

| Detik/round | Random | Oort | FairFedCS |
|---|---|---|---|
| MNIST α = 0,1 / 0,5 / 1,0 | 91,7 / 72,6 / 68,4 | **41,2 / 40,2 / 39,3** | 82,4 / 70,5 / 66,7 |
| CIFAR-10 α = 0,1 / 0,5 / 1,0 | 89,0 / 61,8 / 59,9 | **31,7 / 36,4 / 37,1** | 86,0 / 63,6 / 62,4 |

- Durasi round Oort ±45–65% lebih singkat (ANOVA total waktu: efek strategi p < 0,001; F = 45,5 MNIST, 28,6 CIFAR-10).
- Random dan FairFedCS **tidak berbeda** (Mann–Whitney p > 0,5 di semua sel).

**Interpretasi dan kaitan teori.**
- Kesamaan Random dan FairFedCS sesuai teori, karena keduanya tidak menggunakan informasi kecepatan perangkat.
- Durasi round Random dan FairFedCS **lebih panjang pada α = 0,1** (±89–92 detik vs ±60–68 detik). Penyebabnya adalah *quantity skew* dari partisi Dirichlet: pada α kecil, terdapat klien dengan data sangat banyak, sehingga peluang terpilihnya klien yang lambat menjadi lebih besar (durasi round = klien terlambat).
- Oort **kebal** terhadap efek ini karena *system utility* secara aktif menghindari klien lambat. Ini merupakan ilustrasi langsung dari masalah *straggler* yang menjadi motivasi Oort.
- Bersama Figure 05, figure ini menjelaskan "paradoks" Oort: **lebih banyak round, tetapi setiap round jauh lebih singkat**, sehingga secara keseluruhan lebih cepat.

---

## 15. Figure 15 — Pareto Frontier (`15_pareto_*.png`, 4 figure)

**Yang ditampilkan.** Rerata setiap strategi pada bidang A1 (maksimalkan) vs B1 atau B3 (minimalkan), per α. Titik bertepi hitam adalah titik Pareto-optimal; garis putus-putus adalah *frontier*.

**Hasil.**

| Dataset, α | A1 vs B1 | A1 vs B3 | A1 vs B2/Gini (`analyze_results.py`) |
|---|---|---|---|
| CIFAR-10, 0,1 | Oort mendominasi | R, F, O (ketiganya di frontier) | Oort |
| CIFAR-10, 0,5 | Random mendominasi | Random mendominasi | Random |
| CIFAR-10, 1,0 | Random mendominasi | Random mendominasi | Random |
| MNIST, 0,1 | FairFedCS mendominasi | Random, FairFedCS | FairFedCS |
| MNIST, 0,5 | Random mendominasi | Random mendominasi | Random |
| MNIST, 1,0 | R, O, F | Random, FairFedCS | FairFedCS, Random |

**Interpretasi.**
- **Random paling sering Pareto-optimal**, terutama pada α ≥ 0,5. Ini wajar, karena Random unggul pada B3 dan tidak kalah signifikan pada A1.
- **FairFedCS optimal hanya pada MNIST** (α = 0,1 dan 1,0), dan **tidak pernah mendominasi pada CIFAR-10**. Satu-satunya pengecualian adalah α = 1,0 pada versi *robust* (rata-rata 5 round terakhir).
- *Trade-off* yang sesungguhnya (ketiga strategi berada di frontier) hanya muncul pada **CIFAR-10 α = 0,1 untuk A1 vs B3**: Random paling merata partisipasinya, Oort paling akurat, dan FairFedCS di antaranya.

**Jawaban Rumusan Masalah 3.** **Tidak ditemukan threshold α yang konsisten** di mana FairFedCS menjadi strategi optimal pada kedua dataset.

**Catatan objektif.**
- Pareto dihitung dari **rerata** tanpa memperhitungkan varians. Pada MNIST, selisih akurasi antarstrategi hanya ±0,05 poin, jauh di bawah simpangan baku antar-seed (0,05–0,2). Dominasi pada MNIST karena itu **berada dalam rentang noise** dan tidak boleh ditafsirkan kuat.
- Hasil Pareto bergantung pada metrik fairness yang dipilih (B1, B2, atau B3). Skripsi perlu menetapkan dan menjustifikasi metrik mana yang menjadi acuan utama.

---

## 16. Sintesis: Jawaban Rumusan Masalah

**RM1 — Pengaruh strategi terhadap akurasi global dan fairness.**
Strategi seleksi **tidak berpengaruh signifikan terhadap akurasi global akhir** (p = 0,42 dan 0,82). Strategi **sangat berpengaruh terhadap pemerataan partisipasi** (η²p 0,74–0,91), dengan urutan Random < FairFedCS < Oort. Untuk fairness akurasi (B1/B2), pengaruhnya bergantung pada α (interaksi signifikan di CIFAR-10), tetapi temuan ini sensitif terhadap cara pengukuran. Di bawah heterogenitas perangkat, Oort **unggul signifikan dalam efisiensi waktu** (1,5–3,7×).

**RM2 — Pengaruh label skew terhadap trade-off dan signifikansinya.**
α adalah faktor paling dominan dan signifikan pada semua metrik akurasi dan fairness akurasi (η²p hingga 0,94). Skew berat menurunkan akurasi dan meningkatkan ketimpangan akurasi antarklien secara bersamaan. Akurasi dan fairness akurasi bergerak searah, bukan saling bertukar. *Trade-off* yang sebenarnya hanya muncul antara akurasi dan fairness **partisipasi**, dan hanya pada skew berat CIFAR-10.

**RM3 — Kondisi di mana fairness-aware selection optimal.**
Tidak ditemukan threshold α yang konsisten antara MNIST dan CIFAR-10. FairFedCS Pareto-optimal pada sebagian kondisi MNIST, dengan selisih yang berada dalam noise, dan tidak pada CIFAR-10. Panduan praktis yang didukung data:
- prioritas **efisiensi waktu dengan perangkat heterogen** → Oort;
- prioritas **pemerataan partisipasi atau baseline yang stabil** → Random;
- FairFedCS **tidak terbukti unggul** pada konfigurasi ini, dan biaya komputasi servernya paling tinggi.

---

## 17. Daftar Anomali dan Temuan Tak Terduga (ringkas)

| # | Temuan | Status | Penjelasan |
|---|---|---|---|
| 1 | Oort butuh round terbanyak (A2) | Bukan bug | *Trade-off* efisiensi statistik vs sistem (paper Oort, Pers. 1) |
| 2 | FairFedCS tidak paling merata (B3) | Bukan bug | Fairness FairFedCS proporsional terhadap reputasi (`cᵢ = ε·rᵢ`) |
| 3 | Random paling merata (B3 ≈ 3,0) | Sesuai teori | Binomial(100; 0,1), simpangan baku teoretis 3,0 |
| 4 | Random paling timpang di B1 CIFAR α = 0,1, tetapi paling merata di α ≥ 0,5 | Indikasi, belum kokoh | Interaksi tidak signifikan pada versi *robust* |
| 5 | Varians A1 FairFedCS sangat besar di CIFAR α = 0,1 | Temuan | Seleksi deterministik dan *path-dependent* |
| 6 | Tidak ada *trade-off* A1–B1 | Temuan | Keduanya digerakkan α (*client drift*) |
| 7 | Durasi round Random/FairFedCS naik pada α = 0,1 | Temuan | *Quantity skew* dari partisi Dirichlet |
| 8 | Batas partisipasi Oort tidak aktif | **Penyimpangan implementasi** | Menaikkan B3 Oort; wajib diungkap |

---

## 18. Saran Alur Presentasi (±15 menit)

1. **Setup dan validasi manipulasi** (Figure 01–02, ringkas): label skew terbentuk sesuai rancangan.
2. **Pesan 1 — α dominan, strategi tidak memengaruhi akurasi** (Figure 03–04, tabel ANOVA A1).
3. **Pesan 2 — Oort unggul secara waktu, bukan secara round** (Figure 05 → 14 → 13). Urutan ini membangun "paradoks" lalu menjelaskannya.
4. **Pesan 3 — fairness bersifat multidimensi** (Figure 07, 08, 09, 10, 11): FairFedCS tidak otomatis paling adil; fairness akurasi dan fairness partisipasi tidak sejalan.
5. **Jawaban RM3** (Figure 15): tidak ada threshold yang konsisten.
6. **Keterbatasan dan rencana revisi** (bagian 0.4, 0.5, dan 17).

---

## 19. Antisipasi Pertanyaan Dosen Pembimbing

**"Kenapa FairFedCS kalah dari Random dalam fairness?"**
Karena definisi fairness yang diukur berbeda. B3 mengukur pemerataan absolut, sedangkan FairFedCS menjamin keadilan relatif terhadap reputasi: *queue* tumbuh sebesar `ε·rᵢ`. Random, yang tidak memiliki preferensi, secara teoretis menghasilkan sebaran paling merata, dan nilainya sesuai prediksi Binomial (3,0).

**"Apakah implementasi Oort dan FairFedCS sudah sesuai paper?"**
Komponen inti sudah sesuai: *statistical utility*, *temporal uncertainty*, *system utility*, *pacer*, dan eksplorasi untuk Oort; reputasi Beta, Shapley Value, *virtual queue*, dan CSI untuk FairFedCS. Penyimpangannya diungkap terbuka: batas partisipasi Oort tidak aktif, *pacer* W = 5, dan exact Shapley menggantikan GTG-Shapley. Ke-47 unit test, termasuk verifikasi aksioma efisiensi Shapley, lolos.

**"Waktu simulasi itu valid?"**
Valid sebagai **emulasi** dengan asumsi eksplisit, mengikuti pendekatan evaluasi paper Oort sendiri. Distribusi kecepatan perangkat adalah asumsi, dan hal itu dicantumkan di Batasan Penelitian. Keunggulan Oort bersyarat pada adanya heterogenitas perangkat.

**"Kenapa hasilnya banyak yang tidak signifikan? Apakah sampelnya kurang?"**
Sebagian ya: n = 5 per sel memberi daya uji terbatas untuk efek kecil. Namun untuk A1, efek strateginya juga **sangat kecil secara praktis** (η²p 0,01–0,05), sedangkan efek α sangat besar. Penambahan seed kemungkinan besar tidak mengubah kesimpulan bahwa α jauh lebih dominan.

**"Kenapa jumlah klien, seed, dan round berbeda dari proposal?"**
Pada eksperimen awal (10 klien, 50% dipilih per round), cara memilih hampir tidak berpengaruh karena separuh klien selalu ikut, dan mekanisme eksplorasi Oort habis dalam 2 round. Skala 50 klien dengan 10% partisipasi lebih sesuai dengan rancangan kedua algoritma. Seed ditambah menjadi 5 untuk meningkatkan daya uji statistik, dan round ditambah menjadi 100 agar tiap klien rata-rata tetap terpilih ±10 kali.

**"Apa kontribusi penelitian ini kalau hipotesisnya tidak terbukti?"**
Ada tiga temuan empiris yang dapat dipertahankan:
1. Di bawah label skew, akurasi dan fairness akurasi didominasi tingkat skew, bukan strategi seleksi.
2. Fairness seleksi (FairFedCS) tidak identik dengan pemerataan partisipasi maupun pemerataan akurasi.
3. Keunggulan Oort bersifat **sistemik** (waktu), bukan statistik (akurasi atau round).

Ketiganya relevan dengan pertanyaan terbuka tentang interaksi fairness seleksi dan heterogenitas data (Kairouz dkk., 2021).

---

## 20. Rekomendasi Perbaikan (opsional, bila diminta pembimbing)

1. Aktifkan batas partisipasi Oort (10 kali, sesuai paper) dan jalankan ulang 30 eksperimen Oort, untuk menguji seberapa besar B3 Oort turun.
2. Ukur akurasi per-klien pada **data uji lokal** dan rata-ratakan atas beberapa round terakhir, agar B1/B2 lebih kokoh.
3. Analisis sensitivitas heterogenitas perangkat (σ log-normal 0,5 dan 1,5), untuk menguji ketahanan keunggulan waktu Oort.
4. Terapkan koreksi perbandingan ganda (Holm–Bonferroni) pada uji per sel.
