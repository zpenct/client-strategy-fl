Proposal Penelitian
Analisis Perbandingan Strategi Seleksi Klien pada Algoritma FedAvg
dalam Sistem Federated Learning dengan Data Label-Skewed
OLEH:
MUHAMMAD FAYZUL HAQ
D121221003
DEPARTEMEN TEKNIK INFORMATIKA
FAKULTAS TEKNIK
UNIVERSITAS HASANUDDIN
2026

LEMBAR PENGESAHAN
Judul: Analisis Perbandingan Strategi Seleksi Klien pada Algoritma FedAvg
dalam Sistem Federated Learning dengan Data Label-Skewed
Nama: Muhammad Fayzul Haq
NIM: D121221003
Departemen: Teknik Informatika
Diajukan sebagai salah satu syarat akademik pada Program Sarjana
Universitas Hasanuddin.
Menyetujui,
Calon Pembimbing
Dr. Muhammad Abdillah Rahmat, S.T., M.T.
NIP 19951026202412300

I. Judul
Analisis Perbandingan Strategi Seleksi Klien pada Algoritma FedAvg dalam
Sistem Federated Learning dengan Data Label-Skewed
II. Latar Belakang
Federated Learning (FL) merupakan paradigma pembelajaran mesin terdistribusi
yang memungkinkan sejumlah klien melatih model bersama tanpa harus
mentransmisikan data mentah ke server pusat (McMahan dkk., 2017). Pendekatan
ini relevan untuk berbagai domain yang memiliki kendala privasi, seperti layanan
kesehatan, perbankan, dan sistem IoT, di mana data sensitif tidak dapat
dikumpulkan secara terpusat (Kairouz dkk., 2021). Kendati demikian, FL
menghadapi tantangan mendasar yang belum sepenuhnya terpecahkan, salah
satunya adalah heterogenitas data atau kondisi non-independent and identically
distributed (Non-IID).
Gambar 1. Federated averaging workflow mechanism
Di antara berbagai bentuk Non-IID, label distribution skew atau label skew
merupakan kondisi yang paling sering dijumpai dalam praktik. Label skew terjadi
ketika distribusi kelas pada setiap klien berbeda secara signifikan dari distribusi
global. Sebagai ilustrasi, dalam sistem FL untuk diagnosis medis yang melibatkan

sepuluh rumah sakit, satu rumah sakit mungkin didominasi data penyakit
kardiovaskular sementara rumah sakit lain didominasi penyakit pernapasan.
Ketika model global dilatih menggunakan algoritma agregasi standar seperti
FedAvg, model cenderung bias terhadap distribusi mayoritas, menghasilkan
penurunan akurasi hingga 26% pada klien dengan distribusi minoritas di bawah
kondisi ekstrem (Zhang dkk., 2022). Ketimpangan akurasi antarklien inilah yang
melahirkan persoalan fairness dalam FL.
Sejumlah penelitian telah mengusulkan solusi untuk mengatasi label skew dari sisi
strategi agregasi. FedConcat (Diao dkk., 2023) mengubah mekanisme agregasi
dari weighted averaging menjadi concatenation untuk menggabungkan
representasi fitur model lokal secara efektif. Pendekatan ini terbukti meningkatkan
akurasi rata-rata sebesar 4% pada CIFAR-10 dan hingga 8% pada CIFAR-100
dalam berbagai kondisi heterogenitas data (label skew) ekstrim dibandingkan
metode terfederasi standar. Meskipun demikian, pendekatan berbasis agregasi
tersebut belum mempertimbangkan dimensi keadilan dalam proses pemilihan
klien itu sendiri. Padahal, keadilan dalam FL tidak hanya ditentukan oleh
bagaimana model diagregasi, melainkan juga oleh klien mana yang mendapatkan
kesempatan untuk berkontribusi pada setiap putaran pelatihan.
Aspek krusial ini mengarah pada strategi seleksi klien. Pada setiap
communication round, server hanya dapat memilih subset dari seluruh klien yang
tersedia karena keterbatasan sumber daya dan tidak semua perangkat aktif secara
bersamaan (Fu dkk., 2023). Tiga strategi utama yang umum digunakan
merepresentasikan tiga filosofi desain yang berbeda. Random selection menjadi
asumsi dasar di FedAvg (McMahan dkk., 2017) dan memilih klien secara acak
seragam sehingga memberikan peluang yang sama kepada semua klien;
pendekatan ini merepresentasikan filosofi neutral atau baseline.
Performance-based selection, yang direpresentasikan oleh Oort (Lai dkk., 2021)
dan Power-of-Choice (Cho dkk., 2022), memprioritaskan klien berdasarkan utility
score, terbukti meningkatkan performa time-to-accuracy sebesar 1,2x–14,1x dan
akurasi akhir model sebesar 1,3%–9,8%. Fairness-aware selection, seperti yang
diusulkan oleh Shi dkk. (2023) dalam FairFedCS, secara eksplisit
menyeimbangkan partisipasi antarklien dengan mempertimbangkan kontribusi
klien, mencegah dominasi klien tertentu dalam proses pelatihan . Ketiga filosofi
ini memungkinkan analisis trade-off yang komprehensif antara performa dan
keadilan dalam satu kerangka eksperimen yang sama.
Tinjauan literatur mengungkap kesenjangan penelitian yang signifikan. Di satu
sisi, terdapat banyak publikasi mengenai label-skew dari perspektif agregasi.
Kairouz dkk. (2021) dalam survei komprehensif mereka mengidentifikasi bahwa

interaksi antara fairness dalam seleksi klien dan heterogenitas data merupakan
salah satu open problem utama dalam FL yang belum terjawab secara sistematis.
Penelitian yang secara eksplisit mengkaji bagaimana ketiga strategi seleksi klien
tersebut berinteraksi dengan label skew pada berbagai tingkat keparahan, serta
menguji konsistensi pola tersebut pada lebih dari satu dataset benchmark, masih
sangat terbatas.
Penelitian ini berupaya mengisi kesenjangan tersebut dengan menganalisis secara
empiris bagaimana random selection, performance-based selection, dan
fairness-aware selection berinteraksi dengan label skew pada berbagai tingkat
keparahan (α = 0,1; 0,5; 1,0), serta menguji konsistensi pola trade-off antara
global accuracy dan per-client fairness pada dua dataset dengan tingkat
kompleksitas berbeda, yaitu MNIST dan CIFAR-10.
III. Rumusan Masalah
1. Bagaimana pengaruh tiga strategi seleksi klien — random selection,
performance-based selection, dan fairness-aware selection — terhadap
akurasi global dan fairness antarklien pada berbagai tingkat label skew
dalam simulasi Federated Learning menggunakan framework Flower?
2. Sejauh mana tingkat keparahan label skew mempengaruhi trade-off antara
akurasi global dan per-client fairness untuk masing-masing strategi seleksi
klien, dan apakah pengaruh tersebut signifikan secara statistik?
3. Pada kondisi label skew seperti apa (nilai α berapa) fairness-aware
selection secara empiris menjadi strategi yang lebih optimal dibandingkan
random selection maupun performance-based selection, dan apakah
kondisi threshold tersebut konsisten antara MNIST dan CIFAR-10?
IV. Tujuan Penelitian
1. Mengimplementasikan dan membandingkan tiga strategi seleksi klien —
random selection, performance-based selection, dan fairness-aware
selection — dalam simulasi Federated Learning menggunakan framework
Flower pada dataset MNIST dan CIFAR-10 dengan label skew yang
dikontrol melalui parameter Dirichlet α.
2. Menganalisis trade-off antara akurasi global dan per-client fairness pada
setiap tingkat label skew untuk masing-masing strategi seleksi klien,
menggunakan ANOVA untuk menguji signifikansi pengaruh dan interaksi

antar faktor, serta analisis Pareto frontier untuk mengidentifikasi strategi
yang memberikan keseimbangan terbaik antara akurasi dan fairness.
3. Mengidentifikasi kondisi optimal penggunaan masing-masing strategi
seleksi klien berdasarkan tingkat keparahan label skew, dan menguji
konsistensi temuan tersebut antara dataset MNIST dan CIFAR-10 sebagai
indikasi robustness hasil.
V. Manfaat Penelitian
1. Manfaat Teoritis
Penelitian ini berkontribusi dalam mengisi keterbatasan studi sebelumnya
yang belum banyak menguji interaksi antara label skew dan strategi
seleksi klien dari perspektif fairness pada berbagai tingkat keparahan
skew. Temuan mengenai karakteristik trade-off accuracy-fairness yang
divalidasi pada dua dataset benchmark diharapkan dapat
menginformasikan desain algoritma FL yang lebih seimbang. Penelitian
ini juga menyediakan fondasi metodologi yang reproducible bagi
penelitian lanjutan yang mengkaji fairness dalam setting Non-IID yang
lebih kompleks.
2. Manfaat Praktis
Hasil penelitian memberikan panduan konkret bagi pengembang sistem FL
mengenai strategi seleksi klien yang sebaiknya digunakan berdasarkan
karakteristik distribusi data, khususnya tingkat label-skew. Panduan ini
didasarkan pada bukti empiris dari dua dataset sehingga memiliki validitas
yang lebih luas.
VI. Batasan Penelitian
1. Penelitian dilakukan sepenuhnya dalam lingkungan simulasi menggunakan
framework Flower pada satu mesin dengan 10 klien virtual. Tidak
melibatkan perangkat fisik terdistribusi maupun skenario deployment
nyata.
2. Dataset yang digunakan adalah MNIST dan CIFAR-10, keduanya dipartisi
menggunakan distribusi Dirichlet dengan α ∈ {0.1, 0.5, 1.0} mengikuti
pendekatan yang diperkenalkan oleh Hsu dkk. (2019) dan diadopsi sebagai
standar benchmark oleh Li dkk. (2022). Dataset di luar kedua benchmark

ini tidak digunakan.
3. Eksperimen pada MNIST dan CIFAR-10 menggunakan arsitektur model
CNN yang disesuaikan dengan karakteristik masing-masing dataset,
dengan konfigurasi FL yang identik agar hasil dapat dibandingkan secara
fair.
4. Algoritma agregasi yang digunakan adalah FedAvg (McMahan dkk.,
2017) sebagai standar. Perbandingan dengan algoritma agregasi alternatif
berada di luar cakupan penelitian ini.
5. Evaluasi mencakup lima metrik utama: A1 (global accuracy), A2
(rounds-to-target), B1 (accuracy variance), B2 (Gini coefficient), dan B3
(participation fairness). Hasil analisis dilengkapi dengan dua teknik
analisis: (1) Pareto frontier untuk visualisasi trade-off accuracy-fairness,
dan (2) two-way ANOVA untuk validasi statistik signifikansi perbedaan
antar strategi.
6. Penelitian tidak berfokus pada optimasi arsitektur model maupun
peningkatan akurasi secara spesifik. Model CNN yang digunakan dipilih
berdasarkan kesesuaian dengan karakteristik dataset, bukan performa
optimal

VII. Alur Sistem

| Kategori  | Parameter  |     | Nilai  |     | Alasan  |
| --------- | ---------- | --- | ------ | --- | ------- |
Eksperimen  Strategi Selection  3 (Random,  Fokus utama penelitian
Performance, Fairness)
  Dirichlet Alpha  0.1, 0.5, 1.0  Standar FL research; kontrol
Non-IID level
|     | Dataset  | MNIST, CIFAR-10  |     | Uji konsistensi across  |     |
| --- | -------- | ---------------- | --- | ----------------------- | --- |
complexity

|           | Seeds            | 42, 123, 456  | Reproducibility & mean±std  |
| --------- | ---------------- | ------------- | --------------------------- |
| Simulasi  | Total Klien      | 10            | Tractable simulasi          |
|           | Klien per Round  | 5 (50%)       | Cukup besar untuk           |
convergence
|     | Rounds  | 20  | Dari preliminary test, sudah  |
| --- | ------- | --- | ----------------------------- |
converge/plateau
| Training  | Local Epochs  | 3   | McMahan et al. (2017);  |
| --------- | ------------- | --- | ----------------------- |
trade-off antara learning &
client drift
|         | Learning Rate     | 0.01       | Standar untuk CNN             |
| ------- | ----------------- | ---------- | ----------------------------- |
|         | Batch Size        | 32         | Standar PyTorch               |
| Target  | Accuracy (MNIST)  | 85%        | Practically useful threshold  |
|         | Accuracy          |       70%  | Realistis untuk 20 rounds     |
(CIFAR-10)

| Metrik  | Performance  |     A1, A2  | Accuracy & speed            |
| ------- | ------------ | ----------- | --------------------------- |
|         | Fairness     | B1, B2, B3  | Per-client & participation  |
fairness
| Analisis  |     | Pareto Frontier,  | Trade-off & statistical  |
| --------- | --- | ----------------- | ------------------------ |
|           |     | ANOVA             | significance             |
Tabel 1. Hyperparameter Konfigurasi Eksperimen

VIII. Alur Penelitian

VIII. Penelitian Terkait
| Judul  | Penulis  | Penerbit  | Metodologi  |     |     | Hasil Utama   |     |
| ------ | -------- | --------- | ----------- | --- | --- | ------------- | --- |
(Tahun)
Communicati McMah Proceedings  Mengusulkan  Pada  dataset  MNIST  IID
on-Efficient  an, dkk.  of  the  20th  algoritma  (model CNN), FedAvg dengan
Learning  of  (2017)  International  FederatedAveraging  komputasi  lokal  yang  lebih
Deep  Conference  (FedAvg)  yang  tinggi  (E=20,  B=10) mampu
Networks  on  Artificial  mengombinasikan  mencapai target akurasi 99%
from  Intelligence  local  Stochastic  hanya  dalam  18  rounds,
Decentralized  and Statistics  Gradient  Descent  dibandingkan dengan baseline
Data  (AISTATS)  (SGD)  pada  klien  FedSGD  yang  membutuhkan
|     |     |     | dengan                | agregasi    | 626                          | rounds          | (efisiensi   |
| --- | --- | --- | --------------------- | ----------- | ---------------------------- | --------------- | ------------ |
|     |     |     | rata-rata             | tertimbang  | meningkat                    | 34,8×           | lebih        |
|     |     |     | di server. Algoritma  |             | cepat)10.                    | Pada            | model  2NN   |
|     |     |     | ini  mengevaluasi     |             | (MNIST                       | IID),           | konfigurasi  |
|     |     |     | pengaruh              | fraksi      | yang                         | sama  mencapai  | target       |
|     |     |     | klien  (𝐶),           | jumlah      | akurasi 97% dalam 32 rounds  |                 |              |
|     |     |     | local  epochs         | (𝐸),        | dibanding 1.468 rounds pada  |                 |              |
|     |     |     | dan                   | ukuran      | baseline                     | (efisiensi      | meningkat    |
45,9×).
|     |     |     | minibatch  | lokal  (𝐵   |     |     |     |
| --- | --- | --- | ---------- | ----------- | --- | --- | --- |
|     |     |     | )1.        | Eksperimen  |     |     |     |
disimulasikan secara
sinkron pada dataset
|     |     |     | MNIST,  | CIFAR-10,  |     |     |     |
| --- | --- | --- | ------- | ---------- | --- | --- | --- |
dan  Shakespeare,
|     |     |     | dengan     | metrik      |     |     |     |
| --- | --- | --- | ---------- | ----------- | --- | --- | --- |
|     |     |     | efisiensi  | utama       |     |     |     |
|     |     |     | berupa     | jumlah      |     |     |     |
|     |     |     | putaran    | komunikasi  |     |     |     |
(communication
|     |     |     | rounds)     | yang       |     |     |     |
| --- | --- | --- | ----------- | ---------- | --- | --- | --- |
|     |     |     | dibutuhkan  | untuk      |     |     |     |
|     |     |     | mencapai    | target     |     |     |     |
|     |     |     | akurasi     | pengujian  |     |     |     |
|     |     |     | spesifik.   | Metode     |     |     |     |
baseline

|     | pembanding  | yang      |     |     |     |     |
| --- | ----------- | --------- | --- | --- | --- | --- |
|     | digunakan   | adalah    |     |     |     |     |
|     | FedSGD      |           | (   |     |     |     |
|     | 𝐸 = 1,𝐵     | = ∞) dan  |     |     |     |     |
|     | sequential  | SGD       |     |     |     |     |
terpusat
Advances and  Kairouz,  Foundations  Survei komprehensif  Mengidentifikasi  secara
Open  dkk.  and  Trends  terhadap  eksplisit  bahwa  interaksi
Problems  in  (2021)  in  Machine  perkembangan,  antara  fairness dalam seleksi
Federated  Learning,  tantangan, dan open  klien  dan  heterogenitas  data
Learning  Vol.  14  No.  problems dalam FL,  Non-IID merupakan salah satu
| 1–2  | mencakup               | aspek     | open problem utama FL yang      |                       |             |           |
| ---- | ---------------------- | --------- | ------------------------------- | --------------------- | ----------- | --------- |
|      | optimasi,              | privasi,  | belum                           | terjawab. Menegaskan  |             |           |
|      | fairness,              | dan       | bahwa                           | sebagian              |             | besar     |
|      | heterogenitas          | data.     | penelitian                      | fairness              |             | dalam FL  |
|      | Melibatkan             | lebih     | belum diuji pada kondisi label  |                       |             |           |
|      | dari 50 peneliti dari  |           | skew                            | yang                  | bervariasi  | secara    |
|      | berbagai institusi.    |           | sistematis.                     |                       |             |           |
Oort:  Lai,  Proceedings  Mengusulkan  Oort  meningkatkan  performa
Efficient  dkk.  of  the  15th  performance-based  time-to-accuracy  sebesar
Federated  (2021)  USENIX  selection  bernama  1,2x–14,1x dan akurasi akhir
Learning  via  Symposium  Oort  yang  memilih  model  sebesar  1,3%–9,8%,
Guided  on Operating  klien  berdasarkan  sembari  menerapkan  kriteria
Participant  Systems  utility  score  pengujian  model  yang
Selection  Design  and  gabungan  antara  ditentukan  pengembang
Implementati statistical  utility  secara  efisien  pada  skala
on (OSDI)  (kualitas  data)  dan  jutaan  klien.  Namun  Oort
|     | system         | utility    | cenderung memilih klien yang  |           |              |           |
| --- | -------------- | ---------- | ----------------------------- | --------- | ------------ | --------- |
|     | (kecepatan     |            | sama                          | berulang  | kali         | sehingga  |
|     | perangkat).    | Diuji      | mengakibatkan                 |           |              | fairness  |
|     | pada  dataset  | CV,        | partisipasi                   | yang      | rendah pada  |           |
|     | NLP,  dan      | speech     | klien dengan performa sistem  |           |              |           |
|     | dengan         | 100–1.300  | lambat.                       |           |              |           |
klien heterogen.

Towards  Cho,  Proceedings  Menganalisis  Power-of-Choice  mencapai
Understandin dkk.  of  the  25th  dampak  biased  konvergensi  3×  lebih  cepat
g  Biased  (2022)  International  client  selection  dari  random  selection  pada
Client  Conference  (Power-of-Choice)  kondisi  Non-IID  dalam  hal
Selection  in  on  Artificial  yang  jumlah  rounds.  Namun  bias
Federated  Intelligence  memprioritaskan  seleksi terhadap klien dengan
Learning  and Statistics  klien  dengan  loss  loss  tinggi  memperbesar
(AISTATS)  tinggi.  Diuji  pada  accuracy  variance  antarklien
|     |     |     | CIFAR-10  | dan      | hingga                         | 2×  dibandingkan  |     |
| --- | --- | --- | --------- | -------- | ------------------------------ | ----------------- | --- |
|     |     |     | FEMNIST   | dengan   | random selection pada kondisi  |                   |     |
|     |     |     | kondisi   | Non-IID  | label skew.                    |                   |     |
menggunakan
|     |     |     | simulasi  | PyTorch.  |     |     |     |
| --- | --- | --- | --------- | --------- | --- | --- | --- |
Membandingkan
|     |     |     | dengan     | random   |     |     |     |
| --- | --- | --- | ---------- | -------- | --- | --- | --- |
|     |     |     | selection  | sebagai  |     |     |     |
baseline.
| Exploiting  | Diao,  | IEEE   | Mengusulkan  |     | FedConcat  |     | terbukti  |
| ----------- | ------ | ------ | ------------ | --- | ---------- | --- | --------- |
Label  Skews  dkk.  FedConcat  yang  meningkatkan  akurasi  model
| in  Federated  | (2023)  |     | mengubah  |     | global  | sebesar  | 4%  pada  |
| -------------- | ------- | --- | --------- | --- | ------- | -------- | --------- |
Learning with  mekanisme agregasi  CIFAR-109  dan  hingga  8%
| Model        |     |     | dari                | weighted      | pada                             | CIFAR-10010     |              |
| ------------ | --- | --- | ------------------- | ------------- | -------------------------------- | --------------- | ------------ |
| Concatenatio |     |     | averaging           | menjadi       | dibandingkan dengan FedAvg       |                 |              |
| n            |     |     | concatenation       | pada          | pada                             | kondisi  label  | skew         |
|              |     |     | representasi        | fitur         | ekstrem.                         | Metode          | ini  juga    |
|              |     |     | model               | lokal  untuk  | menghasilkan                     |                 | konvergensi  |
|              |     |     | melindungi          |               | yang lebih stabil dengan biaya   |                 |              |
|              |     |     | pengetahuan         | kelas         | komunikasi yang lebih efisien    |                 |              |
|              |     |     | minoritas.          | Evaluasi      | dibandingkan                     |                 | baseline     |
|              |     |     | dilakukan           |               | lainnya.                         | Studi  ini      | berfokus     |
|              |     |     | menggunakan         |               | pada  agregasi                   | model           | global       |
|              |     |     | dataset             | CIFAR-10,     | dan tidak mempertimbangkan       |                 |              |
|              |     |     | FMNIST,             | SVHN,         | pengaruh strategi seleksi klien  |                 |              |
|              |     |     | CIFAR-100,          | dan           | terhadap                         | performa        | maupun       |
|              |     |     | Tiny-ImageNet pada  |               | keadilan sistem                  |                 |              |
|              |     |     | kondisi             | Non-IID       |                                  |                 |              |
(label skew) berbasis
|     |     |     | partisi  | kelas  serta  |     |     |     |
| --- | --- | --- | -------- | ------------- | --- | --- | --- |
distribusi Dirichlet (

β ∈ {0.1,0.5}).
|     | Metode  | ini  |     |     |
| --- | ------- | ---- | --- | --- |
dibandingkan
|     | dengan             | baseline  |     |     |
| --- | ------------------ | --------- | --- | --- |
|     | FedAvg,  FedProx,  |           |     |     |
MOON, FedRS, dan
FedLC
Measuring the  Hsu,  Workshop on  Memperkenalkan  Pada α = 0.1, akurasi FedAvg
Effects  of  dkk.  Federated  penggunaan  turun signifikan dibandingkan
Non-Identical  (2019)  Learning and  distribusi  Dirichlet  kondisi IID. Semakin kecil α,
Data  Analytics,  dengan parameter α  semakin  tinggi  divergensi
Distribution  NeurIPS  untuk  mengontrol  bobot  model  dan  semakin
for  Federated  2019  tingkat  label  skew  banyak  communication
| Visual  | secara  continuous  | rounds  | dibutuhkan.  | Studi ini  |
| ------- | ------------------- | ------- | ------------ | ---------- |
Classification  dalam  simulasi  FL.  menetapkan Dirichlet sebagai
|     | Mengevaluasi  | standar partisi Non-IID yang    |           |          |
| --- | ------------- | ------------------------------- | --------- | -------- |
|     | FedAvg        | pada  diadopsi                  | luas  di  | seluruh  |
|     | CIFAR-10      | dan  penelitian FL berikutnya.  |           |          |
|     | CIFAR-100     | dengan                          |           |          |
α ∈ {0.1, 1.0, 10.0,
|     | 100.0}  dan  | 100      |     |     |
| --- | ------------ | -------- | --- | --- |
|     | klien.       | Seluruh  |     |     |
eksperimen
menggunakan
random selection.

Federated  Li,  dkk.  Proceedings  Membangun  Label  distribution  skew
Learning  on  (2022)  of  the  38th  NIID-Bench dengan  menghasilkan  akurasi
Non-IID Data  IEEE  enam strategi partisi  konvergensi terendah di antara
Silos:  An  International  Non-IID  termasuk  semua  jenis  Non-IID  yang
Experimental  Conference  label  distribution  diuji. FedAvg pada label skew
Study  on  Data  skew  via  Dirichlet.  ekstrem  menghasilkan
|     | Engineering  | Mengevaluasi  |            | accuracy                         | variance  | antarklien    |     |
| --- | ------------ | ------------- | ---------- | -------------------------------- | --------- | ------------- | --- |
|     | (ICDE)       | FedAvg,       | FedProx,   | paling                           | tinggi    | dibandingkan  |     |
|     |              | SCAFFOLD,     | dan        | jenis                            | Non-IID   | lainnya.      |     |
|     |              | FedNova       | pada       | Pengaruh strategi seleksi klien  |           |               |     |
|     |              | MNIST,        | CIFAR-10,  | tidak dikaji dalam benchmark     |           |               |     |
|     |              | dan           | CIFAR-100  | ini.                             |           |               |     |
menggunakan
|     |     | simulasi  | PyTorch.    |     |     |     |     |
| --- | --- | --------- | ----------- | --- | --- | --- | --- |
|     |     | Seluruh   | eksperimen  |     |     |     |     |
menggunakan
random selection.
Tackling  the  Wang,  Advances  in  Menganalisis  secara  Pada  kondisi  Non-IID,  bias
Objective  dkk.  Neural  teoritis  objective  model  global  meningkat
Inconsistency  (2020)  Information  inconsistency antara  secara  signifikan
Problem  in  Processing  optimasi  lokal  dan  dibandingkan  kondisi  IID
Heterogeneou Systems  global  dalam  FL  pada  CIFAR-10.  FedNova
s  Federated  (NeurIPS)  akibat data Non-IID.  mengurangi  objective
| Optimization  |     | Mengusulkan          |            | inconsistency  |               |                 | dan       |
| ------------- | --- | -------------------- | ---------- | -------------- | ------------- | --------------- | --------- |
|               |     | FedNova. Diuji pada  |            | menghasilkan   |               | akurasi         | lebih     |
|               |     | CIFAR-10             | dengan     | tinggi         | dibandingkan  |                 | FedAvg    |
|               |     | partisi              | Non-IID    | pada           | kondisi       | heterogen.      |           |
|               |     | berbasis             | Dirichlet  | Memberikan     |               | fondasi         | teoritis  |
|               |     | dan  dataset         | sintetis   | mengenai       |               | dampak          |           |
|               |     | menggunakan          |            | heterogenitas  |               | data  terhadap  |           |
|               |     | simulasi.            |            | konvergensi    |               | yang  menjadi   |           |
|               |     |                      |            | motivasi       |               | pentingnya      |           |
fairness-aware selection.

Fairness-Awa Shi, dkk  IEEE  FairFedCS  Pada  kondisi  label-skewed,
re  Client  (2022)  mengombinasikan  FairFedCS  menghasilkan
Selection  for  Beta  Reputation  keadilan seleksi (JFI) rata-rata
| Federated  | System                  |           | (BRS),  19,6% lebih tinggi dan akurasi  |               |                 |            |
| ---------- | ----------------------- | --------- | --------------------------------------- | ------------- | --------------- | ---------- |
| Learning   | GTG-Shapley,            |           | dan  uji                                | 0,73%         | lebih           | tinggi     |
|            | optimasi                | Lyapunov  | dibandingkan metode baseline            |               |                 |            |
|            | dengan                  |           | antrean  terbaik.                       |               | Pada  skenario  | bias       |
|            | virtual                 |           | untuk  kelas                            |               | ekstrem         | dengan     |
|            | menyeleksi              |           | klien  kebisingan data 20%, metode      |               |                 |            |
|            | secara dinamis tanpa    |           | ini                                     | meningkatkan  |                 | akurasi    |
|            | melakukan               |           | hingga                                  |               | 2,71%           | pada       |
|            | pemblokiran             |           | Fashion-MNIST dengan tetap              |               |                 |            |
|            | permanen.Strategi       |           | menjaga                                 |               | frekuensi       | pemilihan  |
|            | seleksi ini dievaluasi  |           | klien minoritas secara optima           |               |                 |            |
menggunakan
dataset
Fashion-MNIST dan
|     | CIFAR-10      |          | dalam  |     |     |     |
| --- | ------------- | -------- | ------ | --- | --- | --- |
|     | skenario      | Non-IID  |        |     |     |     |
|     | (label-skew)  |          | yang   |     |     |     |
|     | menyertakan   |          | klien  |     |     |     |
|     | minoritas     | pembawa  |        |     |     |     |
hard samples

Daftar Pustaka
Cho, Y. J., Wang, J., & Joshi, G. (2022). Towards understanding biased client selection in
federated learning. Proceedings of the 25th International Conference on Artificial
Intelligence and Statistics (AISTATS).
https://proceedings.mlr.press/v151/jee-cho22a/jee-cho22a.pdf
Diao, Y., Li, Q., & He, B. (2024). Exploiting label skews in federated learning with model
concatenation. Proceedings of the 38th AAAI Conference on Artificial Intelligence
(AAAI). https://arxiv.org/pdf/2312.06290v2
Fu, L., Zhang, H., Geng, G., Zhang, Z., & Li, X. (2023). Client selection in federated
learning: Principles, challenges, and opportunities. IEEE Internet of Things Journal,
10(24), 21811–21819. https://arxiv.org/pdf/2211.01549
Hsu, T. M. H., Qi, H., & Brown, M. (2019). Measuring the effects of non-identical data
distribution for federated visual classification. Workshop on Federated Learning and
Analytics, NeurIPS 2019. arXiv:1909.06335. https://arxiv.org/pdf/1909.06335
Kairouz, P., McMahan, H. B., Avent, B., Bellet, A., Bennis, M., Bhagoji, A. N., Bonawitz,
K., Charles, Z., Cormode, G., Cummings, R., D'Oliveira, R. G. L., Eichner, H., El
Rouayheb, S., Evans, D., Gardner, J., Garrett, Z., Gascón, A., Ghazi, B., Gibbons, P. B.,
... Zhao, S. (2021). Advances and open problems in federated learning. Foundations and
Trends® in Machine Learning, 14(1–2), 1–210. https://doi.org/10.1561/2200000083
Lai, F., Zhu, X., Madhyastha, H. V., & Chowdhury, M. (2021). Oort: Efficient federated
learning via guided participant selection. Proceedings of the 15th USENIX Symposium
on Operating Systems Design and Implementation (OSDI), 19–35.
https://arxiv.org/pdf/2010.06081
Li, Q., Diao, Y., Chen, Q., & He, B. (2022). Federated learning on non-IID data silos: An
experimental study. Proceedings of the 38th IEEE International Conference on Data
Engineering (ICDE), 965–978. https://arxiv.org/pdf/2102.02079
McMahan, B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017).
Communication-efficient learning of deep networks from decentralized data.
Proceedings of the 20th International Conference on Artificial Intelligence and
Statistics (AISTATS), 1273–1282. https://arxiv.org/pdf/1602.05629

Shi, Y., Liu, Z., Shi, Z., & Yu, H. (2023). Fairness-aware client selection for federated
learning. In 2023 IEEE International Conference on Multimedia and Expo (ICME).
arXiv:2307.10738. https://arxiv.org/pdf/2307.10738
Wang, J., Liu, Q., Liang, H., Joshi, G., & Poor, H. V. (2020). Tackling the objective
inconsistency problem in heterogeneous federated optimization. Advances in Neural
Information Processing Systems (NeurIPS), 33, 7611–7623.
https://arxiv.org/pdf/2007.07481
Zhang, J., Li, Z., Li, B., Xu, J., Wu, S., Ding, S., & Wu, C. (2022). Federated learning with
label distribution skew via logits calibration. In Proceedings of the International
Conference on Machine Learning (Vol. 162, pp. 26311–26329). PMLR. Retrieved from
https://arxiv.org/abs/2209.00189
