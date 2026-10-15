# BAB II
# METODE PENELITIAN

## 2.1 Jenis dan Pendekatan Penelitian

Penelitian ini merupakan penelitian eksperimental kuantitatif yang bersifat komparatif. Tujuan utamanya adalah membandingkan kinerja tiga strategi seleksi klien pada algoritma FedAvg, yaitu *random selection*, *performance-based selection*, dan *fairness-aware selection*, pada kondisi *label distribution skew* dengan tingkat keparahan yang berbeda.

Penelitian dilakukan melalui simulasi Federated Learning (FL). Pendekatan simulasi dipilih karena memungkinkan kontrol penuh terhadap variabel bebas, yaitu strategi seleksi klien, tingkat *label skew* yang dikendalikan melalui parameter konsentrasi Dirichlet α, dan dataset. Dengan kontrol tersebut, pengaruh setiap faktor dan interaksinya dapat diisolasi dan diuji secara statistik. Simulasi juga memungkinkan pemodelan heterogenitas kemampuan perangkat klien secara terkendali dan dapat direproduksi, mengikuti pendekatan evaluasi yang lazim pada penelitian seleksi klien (Lai dkk., 2021).

## 2.2 Tempat dan Waktu Penelitian

Penelitian dilaksanakan di [NAMA LOKASI PENELITIAN, mis. Laboratorium ... Departemen Teknik Informatika, Fakultas Teknik, Universitas Hasanuddin] pada rentang waktu [BULAN TAHUN] hingga [BULAN TAHUN]. Tahapan penelitian meliputi studi literatur, implementasi strategi seleksi klien, uji pendahuluan, eksekusi eksperimen, dan analisis data.

## 2.3 Lingkungan dan Instrumen Penelitian

### 2.3.1 Perangkat Keras

Eksperimen utama dijalankan pada satu mesin komputer di [NAMA LABORATORIUM] dengan spesifikasi [CPU, jumlah inti, RAM; isi sesuai komputer lab]. Pelatihan dilakukan sepenuhnya menggunakan CPU. Pengembangan, pengujian kode, dan analisis data dilakukan pada komputer pribadi dengan prosesor Intel Core i9-11900H (16 *thread*) dan RAM 16 GB. Seluruh 90 eksperimen utama dijalankan pada mesin yang sama untuk meminimalkan variasi yang berasal dari perbedaan perangkat keras.

### 2.3.2 Perangkat Lunak

Implementasi menggunakan bahasa pemrograman Python 3.12.3. Simulasi FL dibangun menggunakan kerangka kerja Flower (flwr) versi 1.32.0 (Beutel dkk., 2020) dengan mesin simulasi berbasis Ray 2.55.1. Pelatihan model menggunakan PyTorch 2.12.1 dan torchvision 0.27.1. Analisis data dan visualisasi menggunakan NumPy 2.5.0, pandas 3.0.3, SciPy 1.18.0, pingouin 0.6.1 (Vallat, 2018), matplotlib 3.11.0, dan seaborn 0.13.2. Ringkasan perangkat lunak disajikan pada Tabel 2.1.

**Tabel 2.1** Perangkat lunak yang digunakan

| Perangkat lunak | Versi | Fungsi |
|---|---|---|
| Python | 3.12.3 | Bahasa pemrograman |
| Flower (flwr) | 1.32.0 | Kerangka kerja simulasi Federated Learning |
| Ray | 2.55.1 | Mesin eksekusi klien virtual pada simulasi Flower |
| PyTorch / torchvision | 2.12.1 / 0.27.1 | Pelatihan model dan pemuatan dataset |
| NumPy, pandas | 2.5.0, 3.0.3 | Pengolahan data hasil eksperimen |
| SciPy, pingouin | 1.18.0, 0.6.1 | Uji statistik |
| matplotlib, seaborn | 3.11.0, 0.13.2 | Visualisasi |

## 2.4 Dataset dan Partisi Data Non-IID

### 2.4.1 Dataset

Penelitian ini menggunakan dua dataset *benchmark* klasifikasi citra dengan tingkat kompleksitas yang berbeda.

1. **MNIST** (LeCun dkk., 1998) terdiri atas 70.000 citra *grayscale* digit tulisan tangan berukuran 28×28 piksel dalam 10 kelas, dengan pembagian 60.000 data latih dan 10.000 data uji.
2. **CIFAR-10** (Krizhevsky, 2009) terdiri atas 60.000 citra RGB berukuran 32×32 piksel dalam 10 kelas objek, dengan pembagian 50.000 data latih dan 10.000 data uji.

Penggunaan dua dataset bertujuan menguji konsistensi pola hasil pada tingkat kompleksitas tugas yang berbeda. Pada tahap pra-pemrosesan, citra dikonversi menjadi tensor dan dinormalisasi per kanal menggunakan nilai rata-rata dan simpangan baku yang lazim digunakan untuk masing-masing dataset, yaitu (0,1307; 0,3081) untuk MNIST dan rata-rata (0,4914; 0,4822; 0,4465) dengan simpangan baku (0,2023; 0,1994; 0,2010) untuk CIFAR-10. Tidak dilakukan augmentasi data. Data latih dibagi ke klien sesuai prosedur pada Subbab 2.4.2. Data uji tidak dibagi dan digunakan secara utuh oleh server untuk mengevaluasi model global.

### 2.4.2 Simulasi Label Skew

Kondisi *label distribution skew* dibangkitkan menggunakan partisi berbasis distribusi Dirichlet, mengikuti pendekatan Hsu dkk. (2019) yang juga diadopsi sebagai standar *benchmark* Non-IID oleh Li dkk. (2022). Untuk setiap kelas *k* ∈ {1, …, 10}, vektor proporsi alokasi data kelas tersebut ke *N* klien diambil dari distribusi Dirichlet simetris:

$$
\mathbf{p}_k = (p_{k,1}, p_{k,2}, \dots, p_{k,N}) \sim \mathrm{Dir}(\alpha \cdot \mathbf{1}_N) \tag{2.1}
$$

dengan $p_{k,i}$ menyatakan proporsi data kelas *k* yang dialokasikan ke klien *i*. Jumlah sampel kelas *k* yang diterima klien *i* adalah $\lfloor p_{k,i} \cdot |D_k| \rfloor$, dengan $|D_k|$ jumlah data latih kelas *k*. Sisa sampel akibat pembulatan dibagikan secara bergiliran agar seluruh data latih terdistribusi.

Penelitian ini menggunakan *N* = 50 klien dan tiga nilai konsentrasi α ∈ {0,1; 0,5; 1,0}. Nilai α yang kecil menghasilkan distribusi label per klien yang terkonsentrasi pada sedikit kelas (skew berat), sedangkan nilai α yang besar menghasilkan distribusi yang mendekati distribusi global (skew ringan). Karakteristik partisi yang dihasilkan disajikan pada Tabel 2.2.

**Tabel 2.2** Karakteristik partisi data (rata-rata seluruh *seed*, *N* = 50)

| Dataset | α | Proporsi kelas dominan per klien | Rata-rata kelas tidak dimiliki per klien | Jumlah data klien terkecil |
|---|---|---|---|---|
| MNIST | 0,1 | 64,9% | 3,57 | 16 |
| MNIST | 0,5 | 37,1% | 0,37 | 122 |
| MNIST | 1,0 | 29,2% | 0,04 | 582 |
| CIFAR-10 | 0,1 | 65,3% | 3,80 | 2 |
| CIFAR-10 | 0,5 | 36,6% | 0,38 | 87 |
| CIFAR-10 | 1,0 | 28,2% | 0,07 | 341 |

Karena proporsi alokasi diambil secara independen untuk setiap kelas, partisi Dirichlet juga menghasilkan perbedaan jumlah data antarklien (*quantity skew*), terutama pada α kecil. Dengan demikian, pengaruh yang dianalisis dalam penelitian ini merupakan pengaruh gabungan dari *label skew* dan *quantity skew* yang melekat pada partisi Dirichlet (Li dkk., 2022). Partisi dibangkitkan satu kali untuk setiap kombinasi dataset, α, dan *seed*, lalu disimpan dan digunakan ulang oleh ketiga strategi, sehingga setiap strategi dievaluasi pada partisi data yang identik.

## 2.5 Arsitektur Model

Model yang digunakan adalah *Convolutional Neural Network* (CNN) yang disesuaikan dengan karakteristik masing-masing dataset. Arsitektur dipilih berdasarkan kesesuaian dengan dataset, bukan untuk optimasi akurasi, sehingga fokus penelitian tetap pada perilaku strategi seleksi klien. Rincian arsitektur disajikan pada Tabel 2.3.

**Tabel 2.3** Arsitektur CNN

| Lapisan | MNIST (SimpleCNN) | CIFAR-10 (CIFARCNN) |
|---|---|---|
| Masukan | 1×28×28 | 3×32×32 |
| Konvolusi 1 | 32 filter, kernel 5×5, ReLU | 32 filter, kernel 3×3, *padding* 1, *Batch Normalization*, ReLU |
| *Pooling* 1 | *Max pooling* 2×2 | *Max pooling* 2×2 |
| Konvolusi 2 | 64 filter, kernel 5×5, ReLU | 64 filter, kernel 3×3, *padding* 1, *Batch Normalization*, ReLU |
| *Pooling* 2 | *Max pooling* 2×2 | *Max pooling* 2×2 |
| *Fully connected* 1 | 1.024 → 128, ReLU | 4.096 → 256, ReLU, *Dropout* (*p* = 0,5) |
| Keluaran | 128 → 10 | 256 → 10 |
| Jumlah parameter | 184.586 | 1.070.986 |

Kedua model dilatih menggunakan fungsi kerugian *cross-entropy*. Model CIFAR-10 menggunakan *Batch Normalization* dan *Dropout* untuk menstabilkan pelatihan dan mengurangi *overfitting* pada data lokal yang terbatas dan timpang.

## 2.6 Prosedur Pelatihan Federated Learning

Algoritma agregasi yang digunakan adalah *Federated Averaging* (FedAvg) (McMahan dkk., 2017). Misalkan *N* adalah jumlah klien, *m* jumlah klien yang dipilih pada setiap ronde, dan $S_t$ himpunan klien terpilih pada ronde *t*. Setiap ronde komunikasi *t* = 1, 2, …, *T* terdiri atas tahapan berikut.

1. **Seleksi klien.** Server memilih *m* klien $S_t$ menggunakan strategi seleksi yang diuji (Subbab 2.7).
2. **Distribusi model.** Server mengirimkan parameter model global $w_t$ kepada klien terpilih.
3. **Pelatihan lokal.** Setiap klien *i* ∈ $S_t$ melatih model pada data lokalnya $D_i$ selama *E* epoch menggunakan *stochastic gradient descent* (SGD), menghasilkan parameter lokal $w_{t+1}^i$.
4. **Agregasi.** Server menggabungkan parameter lokal melalui rata-rata berbobot jumlah data:

$$
w_{t+1} = \sum_{i \in S_t} \frac{n_i}{\sum_{j \in S_t} n_j} \, w_{t+1}^{i} \tag{2.2}
$$

dengan $n_i = |D_i|$ jumlah data latih klien *i*.

5. **Evaluasi.** Model global $w_{t+1}$ dievaluasi terhadap data uji terpusat untuk memperoleh kurva akurasi per ronde. Model global juga dikirim ke seluruh klien untuk mengukur akurasi pada data lokal setiap klien (Subbab 2.10).

## 2.7 Strategi Seleksi Klien

Ketiga strategi memilih jumlah klien yang sama pada setiap ronde, yaitu *m* = 5 dari *N* = 50 klien (fraksi partisipasi 10%), dan menggunakan prosedur pelatihan serta agregasi yang identik. Perbedaan antarstrategi hanya terletak pada cara pemilihan $S_t$.

### 2.7.1 Random Selection

*Random selection* berfungsi sebagai *baseline* netral. Pada setiap ronde, *m* klien dipilih secara seragam tanpa pengembalian, sehingga setiap klien memiliki peluang terpilih yang sama:

$$
P(i \in S_t) = \frac{m}{N} \tag{2.3}
$$

Strategi ini merupakan mekanisme seleksi bawaan FedAvg (McMahan dkk., 2017) dan tidak menggunakan informasi apa pun tentang data maupun perangkat klien.

### 2.7.2 Performance-Based Selection (Oort)

*Performance-based selection* diimplementasikan mengikuti Oort (Lai dkk., 2021). Oort memprioritaskan klien yang memberikan manfaat terbesar bagi pelatihan per satuan waktu, dengan menggabungkan utilitas statistik dan utilitas sistem dalam kerangka *multi-armed bandit*.

**a. Utilitas statistik.** Manfaat statistik klien *i* diukur dari besarnya *loss* pelatihan pada data lokalnya:

$$
U(i) = |B_i| \sqrt{\frac{1}{|B_i|} \sum_{k \in B_i} \mathrm{Loss}(k)^2} \tag{2.4}
$$

dengan $|B_i|$ jumlah sampel data lokal klien *i* dan $\mathrm{Loss}(k)$ nilai *cross-entropy* sampel *k* yang dicatat selama pelatihan lokal terakhir klien tersebut. Klien dengan *loss* tinggi dianggap memiliki data yang belum dipelajari dengan baik oleh model global. Untuk mengurangi fluktuasi, nilai akar rata-rata kuadrat *loss* dirata-ratakan atas maksimal 10 laporan terakhir setiap klien.

**b. Ketidakpastian temporal.** Untuk mengakomodasi kemungkinan utilitas yang sudah usang (*stale*), ditambahkan suku eksplorasi bagi klien yang lama tidak dipilih:

$$
U'(i) = U(i) + \sqrt{\frac{0{,}1 \, \log R}{L(i)}} \tag{2.5}
$$

dengan *R* nomor ronde saat ini dan *L(i)* ronde terakhir klien *i* dipilih.

**c. Utilitas sistem.** Klien yang durasi pelatihannya melebihi durasi ronde yang diinginkan *T* diberi penalti:

$$
\mathrm{Util}(i) = U'(i) \times \left(\frac{T}{t_i}\right)^{\mathbb{1}(T < t_i) \times \alpha_s} \tag{2.6}
$$

dengan $t_i$ durasi satu ronde pelatihan klien *i* (Subbab 2.8), $\mathbb{1}(\cdot)$ fungsi indikator, dan $\alpha_s = 2$ faktor penalti *straggler* sesuai nilai bawaan Oort. Klien yang lebih cepat dari *T* tidak diberi penalti maupun bonus. Notasi $\alpha_s$ dibedakan dari parameter Dirichlet α.

**d. Pacer.** Durasi ronde yang diinginkan *T* ditetapkan sebagai persentil ke-30 dari durasi seluruh klien, mengikuti implementasi resmi Oort pada FedScale (Lai dkk., 2022). Setiap *W* ronde, jika total utilitas statistik pada *W* ronde terakhir lebih kecil daripada *W* ronde sebelumnya, *T* dilonggarkan dengan menaikkan persentilnya sebesar Δ = 5 poin. Nilai *W* = 5 digunakan; paper Oort menggunakan *W* = 20 untuk pelatihan ratusan ronde, sehingga nilai tersebut disesuaikan dengan jumlah ronde penelitian ini.

**e. Seleksi eksplorasi–eksploitasi.** Pada setiap ronde, sebanyak $(1-\varepsilon)m$ klien dipilih melalui eksploitasi dan $\varepsilon m$ klien melalui eksplorasi.
- *Eksploitasi*: utilitas klien yang pernah dipilih dibatasi pada persentil ke-95 untuk ketahanan terhadap *outlier*, lalu klien diurutkan berdasarkan utilitas. Dari kumpulan kandidat teratas, klien dipilih secara acak dengan peluang sebanding dengan utilitasnya.
- *Eksplorasi*: klien yang belum pernah dipilih diambil secara acak dengan peluang sebanding dengan kecepatannya ($1/t_i$), sehingga perangkat cepat lebih diprioritaskan.
- Nilai ε dimulai dari 0,9 dan berkurang dengan faktor 0,98 setiap ronde hingga batas bawah 0,2, sesuai konfigurasi Oort (Lai dkk., 2021).

Mekanisme pembatasan partisipasi pada paper Oort, yaitu mengeluarkan klien dari daftar eksploitasi setelah terpilih lebih dari 10 kali, tidak diaktifkan dalam penelitian ini. Hal ini dicatat sebagai batasan penelitian.

### 2.7.3 Fairness-Aware Selection (FairFedCS)

*Fairness-aware selection* diimplementasikan mengikuti FairFedCS (Shi dkk., 2023). FairFedCS menyeimbangkan kinerja model dan keadilan seleksi melalui optimasi Lyapunov, dengan mempertimbangkan reputasi klien dan riwayat partisipasinya.

**a. Penilaian kontribusi dengan Shapley Value.** Kontribusi klien *i* pada suatu ronde diukur menggunakan *Shapley Value* (Shapley, 1953) terhadap koalisi klien terpilih $S_t$:

$$
\phi_i = \frac{1}{m} \sum_{S \subseteq S_t \setminus \{i\}} \binom{m-1}{|S|}^{-1} \left[ f(w_{S \cup \{i\}}) - f(w_S) \right] \tag{2.7}
$$

dengan $w_S$ model hasil agregasi FedAvg dari subset klien *S* dan $f(\cdot)$ akurasi model pada data uji. Untuk subset kosong, $w_\emptyset$ adalah model global sebelum pelatihan ronde tersebut. Paper FairFedCS menggunakan aproksimasi GTG-Shapley (Liu dkk., 2022) untuk efisiensi pada jumlah klien yang besar. Karena *m* = 5, penelitian ini menghitung Shapley Value secara **eksak** melalui enumerasi seluruh $2^5 = 32$ subset koalisi pada setiap ronde, yang secara definisi lebih presisi daripada aproksimasinya.

**b. Reputasi.** Reputasi klien dimodelkan dengan *Beta Reputation System* (Jøsang & Ismail, 2002):

$$
r_i = \frac{a_i + 1}{a_i + b_i + 2} \tag{2.8}
$$

dengan $a_i$ dan $b_i$ jumlah kontribusi positif dan negatif klien *i*. Nilai awal $a_i = b_i = 0$ sehingga $r_i = 0{,}5$. Setelah setiap ronde, klien terpilih dengan $\phi_i \geq 0$ memperoleh $a_i \leftarrow a_i + 1$, sedangkan klien dengan $\phi_i < 0$ memperoleh $b_i \leftarrow b_i + 1$.

**c. Antrean virtual (*virtual queue*).** Ketidakadilan yang terakumulasi pada klien *i* dicatat dalam antrean virtual:

$$
Q_i(t+1) = \max\left[0,\; Q_i(t) + c_i(t) - x_i(t)\right] \tag{2.9}
$$

$$
c_i(t) = \epsilon \, r_i(t) \, \mathbb{1}[x_i(t) = 0], \qquad \epsilon = \frac{m}{N} \tag{2.10}
$$

dengan $x_i(t) \in \{0, 1\}$ status terpilihnya klien *i* pada ronde *t*. Antrean klien bertambah saat klien tidak terpilih dan berkurang saat terpilih. Laju pertambahannya sebanding dengan reputasi, sehingga klien bereputasi tinggi lebih cepat "berhak" untuk dipilih kembali.

**d. Indeks kesesuaian klien.** Pada setiap ronde, klien diurutkan berdasarkan *Client Suitability Index*:

$$
\mathrm{CSI}_i(t) = \sigma \, r_i(t) + Q_i(t) \tag{2.11}
$$

lalu *m* klien dengan CSI tertinggi dipilih. Parameter σ = 0,6 mengatur bobot relatif antara kinerja (reputasi) dan keadilan (antrean), sesuai nilai pada paper FairFedCS.

Perlu ditekankan bahwa keadilan yang dikejar FairFedCS adalah **keadilan seleksi yang sebanding dengan reputasi klien**, bukan pemerataan partisipasi mutlak. Shi dkk. (2023) secara eksplisit menyatakan bahwa seleksi yang sepenuhnya merata dapat dianggap tidak adil oleh klien yang berkontribusi besar.

## 2.8 Simulasi Heterogenitas Perangkat

Seluruh klien dijalankan pada satu mesin, sehingga tidak terdapat perbedaan kemampuan perangkat yang nyata. Untuk mengevaluasi strategi pada kondisi perangkat klien yang heterogen, sebagaimana pada sistem FL nyata (Lai dkk., 2021), setiap klien *i* diberi profil perangkat tetap, yaitu kecepatan komputasi $s_i$ (sampel per detik) dan *bandwidth* $b_i$ (MB per detik), yang diambil dari distribusi log-normal:

$$
s_i \sim \mathrm{LogNormal}(\ln 200,\; 1), \qquad b_i \sim \mathrm{LogNormal}(\ln 5,\; 1) \tag{2.12}
$$

Distribusi log-normal dengan simpangan baku logaritmik 1 menghasilkan perbedaan kemampuan hingga sekitar satu orde besaran antara perangkat tercepat dan terlambat, sesuai dengan sebaran kemampuan komputasi dan jaringan perangkat nyata yang dilaporkan Lai dkk. (2021). Profil perangkat dibangkitkan dari aliran bilangan acak yang terpisah dari partisi data, sehingga kecepatan perangkat **tidak berkorelasi** dengan distribusi label klien.

Durasi satu ronde pelatihan klien *i* dimodelkan sebagai waktu komputasi ditambah waktu pengiriman dan penerimaan model:

$$
t_i = \frac{|D_i| \cdot E}{s_i} + \frac{2 \cdot M}{b_i} \tag{2.13}
$$

dengan *M* ukuran parameter model dalam MB (0,74 MB untuk MNIST dan 4,28 MB untuk CIFAR-10). Karena FedAvg bersifat sinkron, durasi satu ronde ditentukan oleh klien terpilih yang paling lambat:

$$
\tau_t = \max_{i \in S_t} t_i \tag{2.14}
$$

Waktu simulasi kumulatif hingga ronde *t* adalah $\sum_{u=1}^{t} \tau_u$. Hanya Oort yang menggunakan informasi $t_i$ dalam proses seleksi. Random dan FairFedCS tidak menggunakan informasi tersebut, tetapi waktu simulasinya tetap dicatat dengan cara yang sama agar dapat dibandingkan. Waktu simulasi ini tidak mencakup biaya komputasi di sisi server, termasuk perhitungan Shapley Value pada FairFedCS.

## 2.9 Konfigurasi Eksperimen

Konfigurasi hiperparameter yang digunakan pada seluruh skenario disajikan pada Tabel 2.4.

**Tabel 2.4** Konfigurasi hiperparameter eksperimen

| Kategori | Parameter | Nilai | Alasan |
|---|---|---|---|
| Simulasi | Jumlah klien (*N*) | 50 | Populasi cukup besar agar mekanisme seleksi berpengaruh |
| | Klien per ronde (*m*) | 5 (10%) | Fraksi partisipasi kecil, mendekati praktik FL lintas perangkat |
| | Jumlah ronde (*T*) | 100 | Setiap klien terpilih rata-rata *T*·*m*/*N* = 10 kali |
| Pelatihan lokal | *Local epoch* (*E*) | 3 | Keseimbangan pembelajaran lokal dan *client drift* (McMahan dkk., 2017) |
| | *Optimizer* | SGD, *momentum* 0,9 | Standar pelatihan CNN |
| | *Learning rate* | 0,01 | Standar pelatihan CNN |
| | *Batch size* | 32 | Standar |
| Oort | ε awal / *decay* / batas bawah | 0,9 / 0,98 / 0,2 | Sesuai Lai dkk. (2021) |
| | Penalti *straggler* ($\alpha_s$) | 2 | Sesuai Lai dkk. (2021) |
| | Persentil *T* awal / Δ / *W* | 30 / 5 / 5 ronde | Implementasi FedScale; *W* disesuaikan jumlah ronde |
| | Persentil *clipping* utilitas | 95 | Sesuai Lai dkk. (2021) |
| FairFedCS | σ | 0,6 | Sesuai Shi dkk. (2023) |
| | ε | *m*/*N* = 0,1 | Sesuai Shi dkk. (2023) |
| | Penilaian kontribusi | *Shapley Value* eksak (32 subset) | Layak dihitung eksak karena *m* = 5 |
| Heterogenitas perangkat | Kecepatan komputasi | LogNormal(ln 200, 1) sampel/detik | Sebaran satu orde besaran (Lai dkk., 2021) |
| | *Bandwidth* | LogNormal(ln 5, 1) MB/detik | Idem |
| Target akurasi | MNIST | 85% | Ambang praktis untuk metrik waktu |
| | CIFAR-10 | 70% | Ambang praktis untuk metrik waktu |

Konfigurasi ini merupakan revisi dari rancangan awal yang menggunakan 10 klien dengan 5 klien per ronde (50%) selama 20 ronde. Uji pendahuluan dengan konfigurasi awal tersebut menunjukkan bahwa ketika separuh populasi klien selalu dilibatkan setiap ronde, cara pemilihan klien hampir tidak memengaruhi hasil, dan fase eksplorasi Oort selesai dalam dua ronde. Konfigurasi 50 klien dengan fraksi partisipasi 10% lebih sesuai dengan kondisi yang menjadi rancangan kedua algoritma. Jumlah ronde ditingkatkan menjadi 100 agar frekuensi rata-rata keterpilihan setiap klien tetap sebanding (10 kali).

## 2.10 Rancangan Eksperimen

Rancangan eksperimen menggunakan desain faktorial lengkap dengan tiga faktor, yaitu strategi seleksi klien (3 level), konsentrasi Dirichlet α (3 level), dan dataset (2 level), dengan lima *seed* sebagai replikasi. Total eksperimen adalah 3 × 3 × 2 × 5 = 90 run, sebagaimana disajikan pada Tabel 2.5.

**Tabel 2.5** Rancangan eksperimen

| Faktor | Level |
|---|---|
| Strategi seleksi klien | Random, Performance-based (Oort), Fairness-aware (FairFedCS) |
| Konsentrasi Dirichlet α | 0,1; 0,5; 1,0 |
| Dataset | MNIST, CIFAR-10 |
| *Seed* (replikasi) | 42, 123, 456, 789, 1024 |
| **Total eksperimen** | **3 × 3 × 2 × 5 = 90** |

Untuk setiap kombinasi dataset, α, dan *seed*, ketiga strategi dijalankan pada partisi data, inisialisasi model, dan profil perangkat yang identik. Dengan demikian, perbedaan hasil antarstrategi dapat diatribusikan pada strategi seleksi itu sendiri. *Seed* mengatur pembangkit bilangan acak pada Python, NumPy, dan PyTorch, sedangkan profil perangkat menggunakan *seed* turunan yang terpisah.

## 2.11 Metrik Evaluasi

Evaluasi dilakukan menggunakan metrik yang dikelompokkan ke dalam tiga dimensi, yaitu performa model, efisiensi waktu, dan keadilan (*fairness*). Dimensi keadilan dibedakan menjadi keadilan akurasi dan keadilan partisipasi, karena keduanya mengukur aspek yang berbeda. Ringkasan metrik disajikan pada Tabel 2.6.

**Tabel 2.6** Metrik evaluasi penelitian

| Kode | Metrik | Dimensi | Arah lebih baik | Peran |
|---|---|---|---|---|
| A1 | Akurasi global | Performa | ↑ | Utama |
| A_auc | Rata-rata akurasi sepanjang ronde | Performa | ↑ | Pendukung |
| A2 | Ronde menuju target | Performa | ↓ | Deskriptif |
| T | Waktu menuju target (simulasi) | Efisiensi waktu | ↓ | Utama |
| B1 | Simpangan baku akurasi antarklien | Keadilan akurasi | ↓ | Utama |
| B2 | Koefisien Gini akurasi antarklien | Keadilan akurasi | ↓ | Konfirmasi B1 |
| B3 | Simpangan baku partisipasi | Keadilan partisipasi | ↓ | Utama |
| B4 | *Jain's Fairness Index* berbobot kualitas | Keadilan seleksi | ↑ | Utama |

### 2.11.1 Metrik Performa

**A1 — Akurasi global.** Akurasi model global pada data uji terpusat setelah ronde terakhir:

$$
\mathrm{A1} = \frac{1}{|D_{uji}|} \sum_{(x, y) \in D_{uji}} \mathbb{1}\left[\hat{y}(x; w_T) = y\right] \times 100\% \tag{2.15}
$$

**A_auc — Rata-rata akurasi sepanjang ronde.** Rata-rata akurasi global dari ronde 1 hingga *T*, $\frac{1}{T}\sum_{t=1}^{T} \mathrm{Acc}(w_t)$. Metrik ini mencerminkan kecepatan konvergensi dalam satuan ronde dan tetap terdefinisi meskipun target akurasi tidak tercapai.

**A2 — Ronde menuju target.** Ronde pertama *t* ketika akurasi global mencapai target (85% untuk MNIST dan 70% untuk CIFAR-10). Apabila target tidak tercapai hingga ronde *T*, nilai A2 dinyatakan tidak terdefinisi. Karena itu A2 hanya dilaporkan secara deskriptif dan tidak diuji secara inferensial.

### 2.11.2 Metrik Efisiensi Waktu

**T — Waktu menuju target.** Waktu simulasi kumulatif (Persamaan 2.14) hingga akurasi global pertama kali mencapai target:

$$
\mathrm{T} = \sum_{u=1}^{\mathrm{A2}} \tau_u \tag{2.16}
$$

Metrik ini mengukur efisiensi waktu (*time-to-accuracy*), yaitu kriteria evaluasi utama Oort (Lai dkk., 2021). Sebagai pelengkap juga dilaporkan total waktu simulasi seluruh *T* ronde.

### 2.11.3 Metrik Keadilan Akurasi

Akurasi per klien $a_i$ diperoleh dengan mengevaluasi model global pada akhir ronde terakhir terhadap data lokal setiap klien, untuk seluruh *N* = 50 klien termasuk klien yang tidak terpilih pada ronde tersebut. Data lokal yang digunakan adalah data latih lokal klien, karena partisi tidak menyediakan data uji lokal terpisah. Hal ini dicatat sebagai batasan penelitian.

**B1 — Simpangan baku akurasi antarklien:**

$$
\mathrm{B1} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (a_i - \bar{a})^2}, \qquad \bar{a} = \frac{1}{N}\sum_{i=1}^{N} a_i \tag{2.17}
$$

**B2 — Koefisien Gini akurasi antarklien.** Dengan $a_{(1)} \leq a_{(2)} \leq \dots \leq a_{(N)}$ akurasi yang diurutkan menaik:

$$
\mathrm{B2} = \frac{2 \sum_{i=1}^{N} i \, a_{(i)}}{N \sum_{i=1}^{N} a_{(i)}} - \frac{N+1}{N} \tag{2.18}
$$

Nilai 0 menunjukkan akurasi yang sepenuhnya merata antarklien.

Untuk mengurangi pengaruh fluktuasi antarronde, B1 dan B2 juga dihitung sebagai rata-rata dari lima ronde terakhir sebagai uji ketahanan.

### 2.11.4 Metrik Keadilan Partisipasi

Misalkan $y_i$ adalah jumlah ronde klien *i* terpilih selama *T* ronde, dengan rata-rata $\bar{y} = T \cdot m / N = 10$.

**B3 — Simpangan baku partisipasi:**

$$
\mathrm{B3} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \bar{y})^2} \tag{2.19}
$$

B3 mengukur pemerataan kesempatan berpartisipasi secara mutlak. Sebagai acuan, seleksi acak seragam menghasilkan $y_i \sim \mathrm{Binomial}(T, m/N)$ dengan simpangan baku teoretis $\sqrt{100 \times 0{,}1 \times 0{,}9} = 3{,}0$.

**B4 — *Jain's Fairness Index* berbobot kualitas.** Karena FairFedCS mengejar keadilan relatif terhadap kualitas klien, bukan pemerataan mutlak, digunakan pula metrik evaluasi yang dipakai pada paper FairFedCS, yaitu *Jain's Fairness Index* (Jain dkk., 1984) berbobot kualitas data (Shi dkk., 2023, Pers. 15):

$$
\mathrm{B4} = \frac{\left(\sum_{i=1}^{N} y_i / z_i\right)^2}{N \sum_{i=1}^{N} (y_i / z_i)^2} \tag{2.20}
$$

dengan $z_i$ kualitas data klien *i*. Mengikuti protokol Shi dkk. (2023), $z_i = n_{class,i} \times (1 - p_{noisy,i})$ yang dinormalisasi *min-max* ke rentang [0,5; 1], dengan $n_{class,i}$ jumlah kelas yang dimiliki klien dan $p_{noisy,i}$ proporsi label yang rusak. Karena penelitian ini tidak menyuntikkan *label noise* ($p_{noisy,i} = 0$), kualitas klien direpresentasikan oleh jumlah kelas yang dimilikinya. B4 bernilai antara 1/*N* (paling timpang) dan 1 (paling adil, yaitu partisipasi sebanding dengan kualitas). Sebagai analisis pendukung, dihitung korelasi Spearman antara $y_i$ dan $n_{class,i}$ untuk menguji apakah suatu strategi cenderung memilih klien berkualitas lebih sering.

*Jain's Fairness Index* tanpa bobot ($z_i = 1$) tidak dilaporkan terpisah, karena dengan $\bar{y}$ konstan nilainya merupakan transformasi langsung dari B3, yaitu $1 / (1 + (\mathrm{B3}/\bar{y})^2)$.

## 2.12 Teknik Analisis Data

Analisis data dilakukan dalam empat tahap.

**Tahap 1 — Statistik deskriptif.** Untuk setiap kombinasi strategi, α, dan dataset, setiap metrik dilaporkan sebagai rata-rata ± simpangan baku dari lima *seed*, disertai visualisasi kurva akurasi per ronde, kurva akurasi terhadap waktu simulasi, dan distribusi hasil antar-*seed*.

**Tahap 2 — Uji pengaruh faktor.** Pengaruh strategi seleksi, α, dan interaksinya diuji menggunakan *two-way* ANOVA pada setiap dataset secara terpisah, dengan strategi dan α sebagai faktor antar-subjek serta metrik sebagai variabel terikat. Besar efek dilaporkan menggunakan *partial eta squared*:

$$
\eta_p^2 = \frac{SS_{efek}}{SS_{efek} + SS_{galat}} \tag{2.21}
$$

dengan interpretasi 0,01 (kecil), 0,06 (sedang), dan 0,14 (besar) (Cohen, 1988). Uji ini diterapkan pada A1, A_auc, B1, B2, B3, B4, dan total waktu simulasi. A2 dan T tidak diuji dengan ANOVA karena tidak terdefinisi pada run yang tidak mencapai target.

**Tahap 3 — Uji asumsi dan uji non-parametrik.** Normalitas residual diuji dengan Shapiro–Wilk dan homogenitas varians dengan Levene. Karena jumlah replikasi per sel kecil (*n* = 5) dan asumsi tersebut tidak selalu terpenuhi, hasil ANOVA dikonfirmasi dengan uji non-parametrik Kruskal–Wallis (Kruskal & Wallis, 1952) untuk pengaruh strategi pada setiap kombinasi dataset dan α. Perbandingan dua strategi, termasuk pada metrik T, menggunakan uji Mann–Whitney *U* (Mann & Whitney, 1947). Tingkat signifikansi yang digunakan adalah 0,05.

**Tahap 4 — Analisis *Pareto frontier*.** *Trade-off* antara akurasi global dan keadilan dianalisis menggunakan *Pareto frontier* pada setiap kombinasi dataset dan α, berdasarkan nilai rata-rata setiap strategi. Strategi *A* mendominasi strategi *B* apabila *A* memiliki A1 yang lebih besar atau sama dan nilai keadilan (B2, B1, atau B3) yang lebih kecil atau sama, dengan minimal satu perbandingan yang bersifat strik. Strategi yang tidak didominasi strategi lain berada pada *Pareto frontier*. Analisis ini digunakan untuk mengidentifikasi kondisi α ketika suatu strategi memberikan keseimbangan terbaik antara akurasi dan keadilan, serta menguji konsistensinya antara MNIST dan CIFAR-10.

Keterkaitan antara rumusan masalah, metrik, dan teknik analisis disajikan pada Tabel 2.7.

**Tabel 2.7** Keterkaitan rumusan masalah, metrik, dan teknik analisis

| Rumusan masalah | Metrik | Teknik analisis |
|---|---|---|
| RM1: Pengaruh strategi seleksi terhadap akurasi global dan keadilan | A1, A_auc, T, B1, B3, B4 | *Two-way* ANOVA (efek strategi), η²p, Kruskal–Wallis, Mann–Whitney |
| RM2: Pengaruh tingkat *label skew* terhadap *trade-off* akurasi–keadilan dan signifikansinya | A1, B1, B2, B3 terhadap α | *Two-way* ANOVA (efek α dan interaksi), η²p, korelasi Spearman A1–B1 |
| RM3: Kondisi α ketika *fairness-aware selection* optimal dan konsistensinya antar dataset | A1 terhadap B2 (utama), B1, B3 | *Pareto frontier* per dataset dan α |

## 2.13 Validasi dan Reproduktibilitas

Untuk menjamin validitas implementasi dan perbandingan, dilakukan langkah-langkah berikut.

1. **Pengujian unit.** Komponen utama diuji dengan 52 *unit test*, termasuk rumus utilitas Oort, *pacer*, pembaruan reputasi dan antrean FairFedCS, serta verifikasi aksioma efisiensi *Shapley Value* ($\sum_i \phi_i = f(w_{S_t}) - f(w_\emptyset)$) pada contoh sederhana yang dapat dihitung manual.
2. **Uji pendahuluan.** Sebelum eksperimen utama, seluruh *pipeline* dijalankan satu ronde untuk setiap strategi guna memastikan tidak terjadi kesalahan dan nilai metrik berada pada rentang yang valid.
3. **Validasi implementasi seleksi acak.** Nilai B3 *random selection* dibandingkan dengan nilai teoretis distribusi Binomial (3,0).
4. **Kontrol variabel.** Seluruh skenario menggunakan konfigurasi identik kecuali faktor yang diuji. Partisi data, inisialisasi model, dan profil perangkat identik untuk ketiga strategi pada *seed* yang sama.
5. **Reproduktibilitas.** *Seed* diatur secara eksplisit. Konfigurasi, metrik per ronde, catatan partisipasi, profil perangkat, dan *log* pelatihan setiap run disimpan sehingga setiap eksperimen dapat direproduksi.
6. **Robustness.** Konsistensi pola hasil antara MNIST dan CIFAR-10, serta antara metrik keadilan versi ronde terakhir dan rata-rata lima ronde terakhir, digunakan sebagai indikasi ketahanan temuan.

## 2.14 Batasan Metode

Batasan metodologis yang perlu diperhatikan dalam menafsirkan hasil penelitian adalah sebagai berikut.

1. Heterogenitas perangkat dan waktu pelatihan merupakan hasil simulasi berdasarkan asumsi distribusi log-normal, bukan pengukuran pada perangkat fisik. Waktu simulasi tidak mencakup biaya komputasi di sisi server.
2. Akurasi per klien untuk metrik B1 dan B2 diukur pada data latih lokal karena partisi tidak menyediakan data uji lokal terpisah.
3. Mekanisme pembatasan partisipasi Oort (maksimal 10 kali terpilih untuk eksploitasi) tidak diaktifkan.
4. Penelitian tidak menyuntikkan *label noise*, sehingga skenario klien berkualitas rendah yang menjadi fokus evaluasi FairFedCS pada paper aslinya tidak diuji. Kualitas klien pada metrik B4 direpresentasikan oleh jumlah kelas yang dimiliki.
5. Partisi Dirichlet menghasilkan *label skew* sekaligus *quantity skew*.
6. Jumlah replikasi lima *seed* per kombinasi membatasi daya uji statistik untuk efek yang kecil.
7. Eksperimen dilakukan pada satu konfigurasi skala (*N* = 50, *m* = 5).

## 2.15 Ringkasan Bab

Bab ini menguraikan rancangan penelitian eksperimental komparatif yang mencakup dataset MNIST dan CIFAR-10 dengan partisi Dirichlet pada tiga tingkat *label skew*, arsitektur CNN, prosedur pelatihan FedAvg, implementasi tiga strategi seleksi klien (Random, Oort, dan FairFedCS) beserta formulasi matematisnya, simulasi heterogenitas perangkat, konfigurasi hiperparameter, rancangan faktorial sebanyak 90 eksperimen, delapan metrik evaluasi dalam dimensi performa, efisiensi waktu, dan keadilan, serta teknik analisis statistik yang meliputi *two-way* ANOVA, uji non-parametrik, dan *Pareto frontier*. Hasil dari rancangan ini disajikan pada Bab III.

---

## Daftar Pustaka (referensi yang digunakan pada Bab II)

Beutel, D. J., Topal, T., Mathur, A., Qiu, X., Fernandez-Marques, J., Gao, Y., Sani, L., Li, K. H., Parcollet, T., de Gusmão, P. P. B., & Lane, N. D. (2020). Flower: A friendly federated learning research framework. *arXiv preprint arXiv:2007.14390*. https://arxiv.org/abs/2007.14390

Cohen, J. (1988). *Statistical power analysis for the behavioral sciences* (2nd ed.). Lawrence Erlbaum Associates.

Hsu, T. M. H., Qi, H., & Brown, M. (2019). Measuring the effects of non-identical data distribution for federated visual classification. *Workshop on Federated Learning for Data Privacy and Confidentiality, NeurIPS 2019*. arXiv:1909.06335. https://arxiv.org/abs/1909.06335

Jain, R., Chiu, D.-M., & Hawe, W. R. (1984). *A quantitative measure of fairness and discrimination for resource allocation in shared computer systems* (Technical Report DEC-TR-301). Digital Equipment Corporation. arXiv:cs/9809099.

Jøsang, A., & Ismail, R. (2002). The beta reputation system. *Proceedings of the 15th Bled Electronic Commerce Conference*, 41–55.

Krizhevsky, A. (2009). *Learning multiple layers of features from tiny images* (Technical Report). University of Toronto.

Kruskal, W. H., & Wallis, W. A. (1952). Use of ranks in one-criterion variance analysis. *Journal of the American Statistical Association, 47*(260), 583–621.

Lai, F., Zhu, X., Madhyastha, H. V., & Chowdhury, M. (2021). Oort: Efficient federated learning via guided participant selection. *Proceedings of the 15th USENIX Symposium on Operating Systems Design and Implementation (OSDI)*, 19–35. https://arxiv.org/abs/2010.06081

Lai, F., Dai, Y., Singapuram, S. S., Liu, J., Zhu, X., Madhyastha, H. V., & Chowdhury, M. (2022). FedScale: Benchmarking model and system performance of federated learning at scale. *Proceedings of the 39th International Conference on Machine Learning (ICML)*, PMLR 162, 11814–11827. https://arxiv.org/abs/2105.11367

LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE, 86*(11), 2278–2324.

Li, Q., Diao, Y., Chen, Q., & He, B. (2022). Federated learning on non-IID data silos: An experimental study. *Proceedings of the 38th IEEE International Conference on Data Engineering (ICDE)*, 965–978. https://arxiv.org/abs/2102.02079

Liu, Z., Chen, Y., Yu, H., Liu, Y., & Cui, L. (2022). GTG-Shapley: Efficient and accurate participant contribution evaluation in federated learning. *ACM Transactions on Intelligent Systems and Technology, 13*(4), 1–21. https://arxiv.org/abs/2109.02053

Mann, H. B., & Whitney, D. R. (1947). On a test of whether one of two random variables is stochastically larger than the other. *The Annals of Mathematical Statistics, 18*(1), 50–60.

McMahan, B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017). Communication-efficient learning of deep networks from decentralized data. *Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS)*, PMLR 54, 1273–1282. https://arxiv.org/abs/1602.05629

Shapley, L. S. (1953). A value for n-person games. Dalam H. W. Kuhn & A. W. Tucker (Ed.), *Contributions to the Theory of Games II* (hlm. 307–317). Princeton University Press.

Shi, Y., Liu, Z., Shi, Z., & Yu, H. (2023). Fairness-aware client selection for federated learning. *2023 IEEE International Conference on Multimedia and Expo (ICME)*, 324–329. https://arxiv.org/abs/2307.10738

Vallat, R. (2018). Pingouin: Statistics in Python. *Journal of Open Source Software, 3*(31), 1026. https://doi.org/10.21105/joss.01026

---

> **Catatan untuk penulis (hapus sebelum dikumpulkan):**
> 1. Lengkapi bagian bertanda [ ... ]: lokasi dan waktu penelitian (2.2) serta spesifikasi komputer lab (2.3.1).
> 2. Periksa kembali detail bibliografi yang ditandai di bawah ini terhadap sumber aslinya sebelum digunakan, karena ditulis dari ingatan dan belum diverifikasi langsung: nomor halaman ICME pada Shi dkk. (2023), daftar penulis lengkap Beutel dkk. (2020) dan Lai dkk. (2022), serta detail publikasi Liu dkk. (2022) dan Jøsang & Ismail (2002).
> 3. Penomoran persamaan (2.1–2.21) dan tabel (2.1–2.7) berurutan; sesuaikan bila format kampus berbeda.
> 4. Bila referensi ini dipindahkan ke Daftar Pustaka utama skripsi, samakan tahun FairFedCS (2023) dan FedConcat (2024) dengan yang tertulis di Bab I.
