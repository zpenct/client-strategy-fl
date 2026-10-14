# Presentasi: Performa Model & Efisiensi Waktu

**Fokus presentasi ini:** akurasi model dan kecepatan mencapai target akurasi.
**Di luar cakupan presentasi ini:** metrik fairness (B1, B2, B3, JFI) dan analisis statistik lanjutan, yang akan dibahas di pertemuan berikutnya.

Semua gambar ada di `notebooks/figures/results_system_n50/`. Angka diambil dari 90 eksperimen di `results_system/`.

---

## Slide 1 — Setup Eksperimen

**Narasi:**
> "Saya membandingkan tiga strategi pemilihan klien pada FedAvg. Pertama **Random**, sebagai baseline. Kedua **Oort**, yang mewakili strategi berbasis performa (Lai dkk., 2021). Ketiga **FairFedCS**, yang mewakili strategi berbasis fairness (Shi dkk., 2023).
>
> Ada 50 klien, dan setiap round server memilih 5 klien, jadi 10%. Training berjalan 100 round. Saya memakai dua dataset, MNIST dan CIFAR-10, dengan tiga tingkat label skew, α = 0,1, 0,5, dan 1,0. Setiap kombinasi diulang dengan 5 seed, jadi totalnya 90 eksperimen.
>
> Saya juga menyimulasikan **perbedaan kecepatan perangkat antar klien**. Ada klien yang cepat dan ada yang lambat, seperti di dunia nyata. Ini penting karena Oort dirancang khusus untuk kondisi seperti itu."

**Tabel untuk slide:**

| Komponen | Nilai |
|---|---|
| Strategi | Random, Oort (performance-based), FairFedCS (fairness-aware) |
| Klien | 50 total, 5 dipilih per round |
| Round | 100 |
| Dataset | MNIST, CIFAR-10 |
| Label skew (α Dirichlet) | 0,1 (berat) · 0,5 (sedang) · 1,0 (ringan) |
| Ulangan | 5 seed → 90 eksperimen |
| Target akurasi | MNIST 85%, CIFAR-10 70% |

**Catatan untuk dosen (kalau ditanya kenapa berbeda dari proposal):** proposal awalnya memakai 10 klien dengan 5 dipilih per round (50%). Dengan setting itu, setiap round sudah mengikutkan separuh klien, sehingga cara memilih klien hampir tidak berpengaruh. Setting 50 klien dengan 10% partisipasi lebih sesuai dengan rancangan kedua algoritma.

---

## Slide 2 — Bagaimana Label Skew Terbentuk

**Gambar:** `02_partition_heatmap.png`

**Cara membaca gambar:** setiap baris adalah satu klien, dan setiap kolom adalah satu kelas (angka 0–9). Warna gelap berarti klien itu punya banyak data dari kelas tersebut.

**Narasi:**
> "Gambar ini menunjukkan bagaimana data dibagi ke 50 klien. Pada α = 0,1 di kiri, warnanya terpusat, artinya setiap klien hanya punya satu atau dua kelas dominan. Misalnya ada klien yang 99% datanya angka 4. Pada α = 1,0 di kanan, warnanya merata, artinya setiap klien punya hampir semua kelas.
>
> Jadi semakin kecil α, semakin berat label skew-nya. Ini yang ingin saya uji: bagaimana ketiga strategi bekerja ketika data antar klien sangat berbeda."

---

## Slide 3 — Akurasi Akhir Model (A1)

**Gambar:** `03_heatmap_A1.png`

**Cara membaca gambar:** setiap kotak berisi rata-rata akurasi akhir dari 5 seed. Merah berarti akurasi rendah, hijau berarti tinggi.

**Data:**

| | Random | Oort | FairFedCS |
|---|---|---|---|
| MNIST α = 0,1 | 98,27% | 98,47% | 98,83% |
| MNIST α = 0,5 | 99,19% | 99,14% | 99,16% |
| MNIST α = 1,0 | 99,18% | 99,22% | 99,24% |
| CIFAR-10 α = 0,1 | 52,32% | 55,64% | 54,72% |
| CIFAR-10 α = 0,5 | 70,57% | 70,05% | 69,75% |
| CIFAR-10 α = 1,0 | 73,14% | 72,29% | 72,61% |

**Narasi:**
> "Ini akurasi akhir setelah 100 round. Ada dua hal yang bisa dilihat.
>
> Pertama, **perbedaan antar strategi sangat kecil**. Dalam satu baris, selisihnya umumnya di bawah 1 poin, paling besar sekitar 3 poin di CIFAR-10 α = 0,1. Jadi strategi pemilihan klien hampir tidak memengaruhi akurasi akhir.
>
> Kedua, **label skew sangat berpengaruh**. Di CIFAR-10, akurasi turun dari sekitar 73% pada α = 1,0 menjadi sekitar 52–56% pada α = 0,1, turun hampir 20 poin. Ini sesuai literatur: label skew membuat model lokal tiap klien berbeda arah, sehingga model global sulit konvergen (Wang dkk., 2020; Li dkk., 2022).
>
> Di MNIST semua strategi sudah mencapai 98–99%, karena MNIST relatif mudah. Jadi perbedaan strategi lebih terlihat di CIFAR-10."

**Kalimat kunci:**
> **Faktor utama yang menentukan akurasi adalah tingkat label skew, bukan strategi pemilihan klien.**

---

## Slide 4 — Konsistensi Hasil Antar Seed

**Gambar:** `04_boxplot_A1.png`

**Cara membaca gambar:** setiap kotak menunjukkan sebaran akurasi dari 5 seed. Kotak yang tinggi atau panjang berarti hasilnya tidak konsisten antar ulangan.

**Data (CIFAR-10 α = 0,1):**

| Strategi | Rentang akurasi 5 seed | Simpangan baku |
|---|---|---|
| Random | 47,2% – 56,4% | 3,4 |
| **Oort** | **53,6% – 57,3%** | **1,6** (paling stabil) |
| FairFedCS | 46,9% – 65,7% | 7,1 (paling tidak stabil) |

**Narasi:**
> "Selain rata-rata, saya juga melihat konsistensinya. Pada kondisi paling sulit, CIFAR-10 α = 0,1, **Oort paling stabil**, dengan hasil 5 seed berkisar 54–57%. **FairFedCS paling tidak stabil**, dari 47% sampai 66%.
>
> Kemungkinan penyebabnya, FairFedCS memilih klien secara deterministik berdasarkan skor reputasi yang terakumulasi dari round ke round. Perbedaan kecil di awal training bisa membawa ke kelompok klien yang sangat berbeda. Oort memilih klien secara acak berbobot, sehingga hasilnya lebih stabil."

---

## Slide 5 — Kecepatan dalam Jumlah Round (A2)

**Gambar:** `05_A2_rounds_to_target.png`

**Cara membaca gambar:** sumbu-y adalah jumlah round sampai akurasi menyentuh target. **Semakin rendah semakin cepat.**

**Data:**

| Round sampai target | Random | Oort | FairFedCS |
|---|---|---|---|
| CIFAR-10 α = 0,5 | 65,8 | **74,6** (paling lambat) | 60,0 |
| CIFAR-10 α = 1,0 | 43,0 | **54,4** (paling lambat) | 46,2 |
| CIFAR-10 α = 0,1 | tidak tercapai | tidak tercapai | tidak tercapai |
| MNIST α = 0,1 | 6,4 | 7,2 | 5,8 |

**Narasi:**
> "Kalau kecepatan diukur dalam **jumlah round**, Oort justru paling lambat. Di CIFAR-10, Oort butuh sekitar 9–11 round lebih banyak daripada Random.
>
> Ini sekilas bertentangan dengan klaim paper Oort yang menyebut Oort lebih cepat. Tapi ternyata ada penjelasannya, yang akan saya tunjukkan di slide berikutnya."

**Catatan:** pada CIFAR-10 α = 0,1, tidak ada strategi yang mencapai 70% dalam 100 round, jadi titiknya tidak muncul di grafik. Target 70% ternyata terlalu tinggi untuk skew seberat itu.

---

## Slide 6 — Durasi Setiap Round (kunci penjelasan)

**Gambar:** `14_mean_round_duration.png`

**Cara membaca gambar:** tinggi batang adalah rata-rata durasi satu round dalam detik (waktu simulasi). **Semakin rendah semakin cepat.**

**Data (detik per round):**

| | Random | Oort | FairFedCS |
|---|---|---|---|
| MNIST α = 0,1 / 0,5 / 1,0 | 92 / 73 / 68 | **41 / 40 / 39** | 82 / 70 / 67 |
| CIFAR-10 α = 0,1 / 0,5 / 1,0 | 89 / 62 / 60 | **32 / 36 / 37** | 86 / 64 / 62 |

**Narasi:**
> "Di sinilah kuncinya. Dalam FL, setiap round harus menunggu **klien paling lambat** selesai training. Satu klien lambat saja membuat seluruh round ikut lambat.
>
> Oort punya komponen yang disebut *system utility*: Oort memberi penalti pada klien yang lambat, sehingga jarang memilihnya. Hasilnya, **setiap round Oort hanya sekitar 32–41 detik, kurang dari separuh** Random dan FairFedCS yang 60–92 detik.
>
> Random dan FairFedCS hampir sama, karena keduanya tidak memperhatikan kecepatan perangkat sama sekali."

**Kalimat kunci:**
> **Oort butuh lebih banyak round, tetapi setiap round-nya jauh lebih singkat.**

---

## Slide 7 — Kecepatan Sebenarnya: Akurasi terhadap Waktu

**Gambar:** `13_time_to_accuracy_cifar10.png` (utama) dan `13_time_to_accuracy_mnist.png`

**Cara membaca gambar:** sumbu-x adalah **waktu** (bukan round), sumbu-y adalah akurasi, dan garis merah putus-putus adalah target. **Kurva yang naik lebih cepat ke kiri berarti lebih cepat mencapai akurasi tinggi.**

**Data — waktu sampai target (detik simulasi):**

| | Random | Oort | FairFedCS | Oort lebih cepat |
|---|---|---|---|---|
| MNIST α = 0,1 | 556 | **174** | 647 | **3,2–3,7×** |
| MNIST α = 0,5 | 118 | **44** | 111 | 2,5–2,7× |
| MNIST α = 1,0 | 41 | **32** | 63 | 1,3–2,0× |
| CIFAR-10 α = 0,5 | 3.983 | **2.507** | 3.713 | 1,5–1,6× |
| CIFAR-10 α = 1,0 | 2.485 | **1.699** | 2.818 | 1,5–1,7× |

**Narasi:**
> "Kalau kecepatan diukur dalam **waktu**, hasilnya berbalik. Kurva oranye (Oort) selalu berada di paling kiri.
>
> Oort mencapai target **1,5 sampai 3,7 kali lebih cepat** daripada strategi lain. Misalnya di MNIST α = 0,1, Oort butuh 174 detik, sedangkan Random 556 detik dan FairFedCS 647 detik. Perbedaan ini signifikan secara statistik di hampir semua kondisi; satu-satunya yang tidak signifikan adalah MNIST α = 1,0.
>
> Ini **sesuai dengan klaim utama paper Oort**, yang melaporkan percepatan *time-to-accuracy* 1,2 sampai 14,1 kali. Hasil saya berada dalam rentang itu.
>
> Pada CIFAR-10 α = 0,1, panel kiri, tidak ada yang mencapai 70%. Tapi bisa dilihat bahwa Oort mencapai sekitar 55% dalam sekitar 3.200 detik, sedangkan strategi lain butuh sekitar 8.600 detik untuk mencapai akurasi yang setara. Jadi dengan waktu yang sama, Oort tetap jauh lebih unggul."

**Kalimat kunci:**
> **Keunggulan Oort ada pada efisiensi waktu, bukan pada akurasi akhir atau jumlah round. Ini sesuai dengan tujuan rancangan Oort.**

---

## Slide 8 — Kesimpulan Sementara (Performa & Waktu)

**Narasi:**
> "Untuk performa dan efisiensi waktu, ada tiga kesimpulan:
>
> 1. **Akurasi akhir hampir sama** untuk ketiga strategi. Yang paling menentukan akurasi adalah tingkat label skew. Semakin berat skew, semakin rendah akurasinya, dan efeknya jauh lebih besar di CIFAR-10 daripada MNIST.
> 2. **Oort paling efisien secara waktu**, 1,5–3,7 kali lebih cepat mencapai target, karena menghindari klien yang lambat. Konsekuensinya, Oort butuh sedikit lebih banyak round.
> 3. **Oort juga paling stabil** pada kondisi skew berat, sedangkan FairFedCS paling bervariasi antar ulangan.
>
> Jadi kalau kondisi perangkat klien beragam dan waktu training penting, Oort adalah pilihan terbaik dari sisi efisiensi. Bagaimana dampaknya terhadap fairness akan saya bahas di pertemuan berikutnya."

**Ringkasan untuk slide:**

| Aspek | Temuan |
|---|---|
| Akurasi akhir | Hampir sama; ditentukan tingkat label skew |
| Jumlah round ke target | Oort paling banyak |
| Durasi per round | Oort paling singkat (< ½ strategi lain) |
| **Waktu ke target** | **Oort tercepat (1,5–3,7×)** |
| Stabilitas (skew berat) | Oort paling stabil, FairFedCS paling bervariasi |

---

## Catatan Penting yang Sebaiknya Disampaikan Jujur

Sampaikan di akhir presentasi atau saat ditanya:

1. **Waktu yang ditampilkan adalah waktu simulasi.** Semua klien berjalan di satu komputer. Perbedaan kecepatan perangkat disimulasikan dengan memberi tiap klien kecepatan komputasi dan bandwidth yang berbeda, diambil dari distribusi log-normal. Pendekatan ini mirip dengan cara paper Oort sendiri mengevaluasi, yang juga mengemulasikan kecepatan perangkat. Tapi hasil ini **bergantung pada asumsi tersebut**.
2. **Biaya komputasi server tidak termasuk dalam waktu simulasi.** FairFedCS menghitung Shapley Value dengan 32 kali evaluasi model setiap round. Di komputer lab, satu eksperimen FairFedCS rata-rata butuh **±105–126 menit**, dibandingkan Random ±10–25 menit dan Oort ±15–19 menit. Jadi dalam praktik, FairFedCS paling berat.
3. **Target 70% CIFAR-10 tidak tercapai pada α = 0,1** oleh strategi mana pun. Ini perlu dicatat di batasan penelitian.

---

## Antisipasi Pertanyaan Dosen

**"Kenapa Oort butuh lebih banyak round padahal katanya lebih cepat?"**
> Karena Oort mengoptimalkan **waktu**, bukan jumlah round. Oort kadang melewatkan klien yang datanya bagus tapi perangkatnya lambat. Akibatnya setiap round sedikit kurang informatif, tetapi selesai jauh lebih cepat. Paper Oort sendiri menyebut ini sebagai trade-off antara efisiensi statistik dan efisiensi sistem.

**"Apakah perbedaan kecepatan perangkatnya realistis?"**
> Kecepatan dan bandwidth tiap klien diambil dari distribusi log-normal, sehingga ada perbedaan hingga sekitar satu orde besaran antara perangkat cepat dan lambat. Ini meniru temuan paper Oort bahwa perangkat nyata berbeda kecepatan sekitar satu orde besaran. Kecepatan ini diacak terpisah dari data, jadi tidak berkaitan dengan label skew.

**"Kalau tanpa perbedaan kecepatan perangkat, apa yang terjadi?"**
> Pada eksperimen awal saya yang tanpa simulasi kecepatan perangkat, Oort tidak lebih cepat. Jadi keunggulan Oort memang bersumber dari kemampuannya menghindari perangkat lambat.

**"Kenapa akurasinya tidak berbeda antar strategi?"**
> Tingkat label skew jauh lebih dominan. Ketika data antar klien sangat berbeda, model global sulit konvergen apa pun strategi pemilihannya. Selain itu, dengan 5 klien per round selama 100 round, setiap strategi tetap melihat sebagian besar klien.

**"Kenapa MNIST hampir tidak ada bedanya?"**
> MNIST relatif mudah, sehingga semua strategi sudah mencapai 98–99% (efek plafon). Itulah alasan CIFAR-10 dipakai sebagai dataset yang lebih menantang.

---

## Urutan Gambar untuk Slide

| Slide | Gambar | Pesan |
|---|---|---|
| 1 | — (tabel setup) | Konfigurasi eksperimen |
| 2 | `02_partition_heatmap.png` | Label skew terbentuk sesuai rancangan |
| 3 | `03_heatmap_A1.png` | Akurasi hampir sama; ditentukan α |
| 4 | `04_boxplot_A1.png` | Oort paling stabil |
| 5 | `05_A2_rounds_to_target.png` | Oort butuh round terbanyak (paradoks) |
| 6 | `14_mean_round_duration.png` | ...karena round Oort jauh lebih singkat |
| 7 | `13_time_to_accuracy_cifar10.png` (+ mnist) | Oort tercepat secara waktu |
| 8 | — (tabel ringkasan) | Kesimpulan |

**Tips:** urutan slide 5 → 6 → 7 sengaja dibuat seperti cerita: muncul "paradoks" (Oort lambat dalam round), lalu dijelaskan (round Oort singkat), lalu ditunjukkan hasil akhirnya (Oort tercepat secara waktu). Urutan ini membuat pesannya lebih mudah diingat.
