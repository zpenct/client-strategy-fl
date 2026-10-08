# Warning
- implementasi pada codebase ini dibagi atas 3 yaitu data loader(sekaligus pertisi), running experiment/strategi, dan evaluasi hasil experiment
- Pada kode ini aku mau kamu memperthaanakan proses partisi data/data loadernya, karena aku merasa itu sudah cukup, dan juga datasetnya sudah terbagi (kamu bisa lihat di folder /data)
- yg perlu kamu ubah adalah pada tahap simulasinya (kode strategi seleksi cleunt dan running experimentnya) soalnya pada kode sekarang semua strategynya itu belum berdasarkan pada paper2 yg seharusnya yaitu "FAIRNESS-AWARE CLIENT SELECTION FOR FEDERATED LEARNING" dan "Oort: Efficient Federated Learning via Guided Participant Selection". Selain itu pada tahap evaluasinya juga masih belum sesuai dgn yg ada di proposal jadi tolong bantu sesuaikan juga kode evaluasinya
- pastikan hasil dari expeiment/kode ini bisa menjawab semua pertanayn pada proposal kita 
- setiap ada pembaruan penting tolong berikan updatenya di file ini


# Tutorial: Up-scaling eksperimen (N=50 klien) — partisi → run → analisis

> Ditulis 2026-10-05. Kode untuk skala N≠10 sudah siap (lihat Refactor Log 2026-10-05 di bawah). Data & hasil 10 klien **tidak tersentuh**: semua artefak N=50 memakai akhiran `_n50`.

**Konfigurasi yang disarankan:** 50 klien, 5 per round (10%), 100 round, 3 local epoch, seed 42/123/456/789/1024, α 0.1/0.5/1.0. Dengan 100 round, tiap klien rata-rata dipilih 10×, sama dengan grid 10 klien × 20 round. Exact Shapley tetap dipakai (m=5 → 32 evaluasi/round).

Semua perintah dijalankan dari root project, dengan venv aktif (`source venv/bin/activate`).

### Step 0 — Persiapan (sekali saja)
1. Tarik kode terbaru ke komputer lab (`git pull`). Pastikan file-file ini ada: `src/strategies/client_ids.py`, `src/system/device_model.py`, dan argumen `num_clients` di `src/data/partitioner.py::_get_partition_dir`.
2. **Cek ruang disk.** Satu partisi = ±590 MB (CIFAR-10) / ±190 MB (MNIST), untuk N berapa pun (total data sama, hanya dibagi lebih banyak file). 5 seed × 3 α × 2 dataset ≈ **12 GB**. Hanya CIFAR-10 ≈ **9 GB**. Cek dengan `df -h .`. Laptop ini tinggal ±3 GB kosong, jadi **jalankan di komputer lab**.
3. Jalankan test: `python -m pytest tests -q` (harus 47 passed).

### Step 1 — Generate partisi 50 klien
```bash
python experiments/prepare_data.py --num_clients 50 \
    --datasets cifar10 mnist --seeds 42 123 456 789 1024
```
- Hasil: `data/partitions/<dataset>/alpha01_seed42_n50/` dst (`client_0.pt` … `client_49.pt` + `partition_info.json`). Folder lama tanpa `_n50` tetap utuh.
- ±20 detik per partisi (30 partisi ≈ 10 menit).
- Catatan α=0.1: beberapa klien hanya dapat sangat sedikit data (uji coba CIFAR-10 α=0.1 seed 42: klien terkecil 2 sampel). Ini konsekuensi wajar Dirichlet dengan N besar; tulis di Batasan Penelitian. Tidak ada klien dengan 0 sampel di 5 seed yang disarankan (sudah disimulasikan).
- Cek ulang tanpa generate: tambahkan `--verify_only`.

### Step 2 — Validasi sebelum batch
```bash
python experiments/validate.py --datasets cifar10 mnist \
    --num_clients 50 --clients_per_round 5 --seeds 42 123 456 789 1024
```
Level 4 menjalankan 1 round × 3 strategi pada N=50 (hasil di `results/_validation/`, boleh dihapus). Semua harus PASS. Untuk cek cepat tanpa pipeline, tambahkan `--skip_pipeline`.

### Step 3 — Jalankan eksperimen
**3a. Tanpa heterogenitas perangkat** (pembanding langsung dengan grid 10 klien):
```bash
python experiments/run_batch.py --num_clients 50 --clients_per_round 5 --rounds 100 \
    --seeds 42 123 456 789 1024 --skip_existing
```
→ hasil di `results/<strategy>_<dataset>_a<α>_s<seed>_n50/`

**3b. Dengan heterogenitas perangkat:**
```bash
python experiments/run_batch.py --num_clients 50 --clients_per_round 5 --rounds 100 \
    --seeds 42 123 456 789 1024 --system_hetero --skip_existing
```
→ hasil di `results_system/..._n50/`

Tips:
- Total 90 eksperimen per mode (3 strategi × 2 dataset × 3 α × 5 seed). Bisa dicicil per strategi/dataset dengan `--strategies fairness --datasets cifar10`, dst. Kalau terputus, jalankan ulang perintah yang sama; `--skip_existing` melewati yang sudah selesai.
- Cek daftar dulu tanpa menjalankan: tambahkan `--dry_run`.
- Estimasi waktu nyata kasar (CPU, dari grid lama ×5 karena round 5×): Random ±25–40 mnt, Oort ±35–50 mnt, FairFedCS ±2,5–3 jam per eksperimen. FairFedCS paling dominan, jadi jalankan semalaman. Ukur dari 1–2 eksperimen pertama lalu sesuaikan.
- Pacer Oort tetap W=5 round (`OORT_PACER_WINDOW` di `run_single.py`). Dengan 100 round, nilai W=20 sesuai paper juga masuk akal; kalau mau diganti, putuskan **sebelum** batch dijalankan dan catat di metodologi.

### Step 4 — Analisis lintas eksperimen
```bash
python experiments/analyze_results.py --results_dir results        --num_clients 50
python experiments/analyze_results.py --results_dir results_system --num_clients 50
```
Output: `results/analysis_n50/` dan `results_system/analysis_n50/` (summary, Pareto, ANOVA, threshold; versi heterogen juga berisi metrik waktu simulasi). `--num_clients` **wajib**, karena folder `results/` berisi campuran 10 dan 50 klien; tanpa flag ini script akan berhenti dengan pesan error, bukan diam-diam mencampur.

### Step 5 — Notebook
Di cell setup (`# ── Style konsisten`), ubah 2 baris lalu **Restart kernel → Run All**:
```python
RESULTS_SUBDIR = 'results'          # atau 'results_system' (heterogen)
NUM_CLIENTS    = 50
```
- Seed, jumlah round, dan klien/round dibaca otomatis dari `final_metrics.json`. Cek output cell load: harus `Loaded : 90 eksperimen ... (N=50)`, `Config : 5 klien/round | 100 round | seeds=[42, 123, 456, 789, 1024]`, `Missing : 0`.
- Figure & CSV tersimpan di `notebooks/figures/results_n50/` atau `results_system_n50/`.
- Partisi N=50 harus ada di mesin yang menjalankan notebook untuk section 1 (visualisasi partisi). Kalau tidak ada, section itu dilewati otomatis; section lain tetap jalan.

### Ringkasan lokasi artefak
| | 10 klien (lama) | 50 klien (baru) |
|---|---|---|
| Partisi | `data/partitions/<ds>/alpha01_seed42/` | `data/partitions/<ds>/alpha01_seed42_n50/` |
| Hasil homogen | `results/random_cifar10_a0.1_s42/` | `results/random_cifar10_a0.1_s42_n50/` |
| Hasil heterogen | `results_system/random_cifar10_a0.1_s42/` | `results_system/random_cifar10_a0.1_s42_n50/` |
| Analisis | `results*/analysis/` | `results*/analysis_n50/` |


# Refactor Log:

## 2026-10-08 — Fix error cell load notebook (`'DataFrame' object has no attribute 'dataset'`)

Penyebab: notebook yang dijalankan di komputer lab menyusun daftar eksperimen tanpa akhiran `_n50` (loader versi lama, belum memakai `NUM_CLIENTS`), sehingga 0 hasil termuat. `df` kosong lalu crash di `df.dataset`. Folder hasil juga bernama `results_system-n-20`, bukan `results_system`.

Perbaikan loader (`# ── Load semua hasil eksperimen`):
- Jumlah klien tiap run dideteksi berurutan dari `final_metrics.json` → `config.json` → akhiran nama folder `_n<N>` → default 10. Run tetap terbaca walau salah satu file tidak mencatat `num_clients`.
- Kalau 0 hasil termuat, cell mencetak **diagnosis**: folder ada/tidak, jumlah subfolder & `final_metrics.json`, `num_clients` yang terdeteksi, dan daftar folder `results*` di root. Setelah itu baru berhenti dengan pesan jelas, bukan `AttributeError`.

Verifikasi: notebook dijalankan pada **90 hasil asli N=50 heterogen** (`results_system/`, disalin dari lab) — 0 error, `Loaded 90 (N=50)`, `100 round`, `seeds=[42,123,456,789,1024]`, `Missing 0`, 24 figure/CSV di `notebooks/figures/results_system_n50/`. Juga diuji pada folder bernama `results_system-n-20`, run tanpa `num_clients`, dan N salah (memunculkan diagnosis).

Catatan: di laptop, folder `results/` saat ini **kosong** (54 hasil 10-klien tidak ada di disk, tapi masih aman di git commit `68e6d60`; pulihkan dengan `git checkout -- results/`).

## 2026-10-08 — Deep check pipeline analisis untuk hasil N=50 (sebelum batch 90 eksperimen selesai)

**Kesimpulan: siap**, setelah perbaikan di bawah. Diverifikasi dengan hasil sintetis N=50 (format file persis sama dengan output `run_single.py`; nilai acak, hanya untuk menguji kode).

Masalah yang ditemukan & diperbaiki:
1. `notebooks/analysis.ipynb` **tidak bisa** membaca hasil N=50 (folder `_n50`, seed 3 hardcode, partisi diasumsikan 10 klien). Diperbaiki:
   - Saklar `NUM_CLIENTS` (+ `SEEDS_EXPECTED` opsional). Hasil dipindai langsung dari folder & disaring berdasarkan `num_clients` di `final_metrics.json`. Seed, round, dan klien/round terdeteksi otomatis. Ada peringatan kalau konfigurasi tercampur.
   - Section 1 (partisi): membaca folder partisi sesuai N. 1A menampilkan 10 klien pertama, heatmap 1B menampilkan semua klien, 1C ditambah `min_samples`.
   - Label "3 seeds" / "20 rounds" / tabel konfigurasi / seed=42 tetap → dinamis.
   - 4C (participation per klien): klien yang tidak pernah terpilih kini tampil sebagai 0. Sebelumnya hilang dari grafik, padahal kasus ini mungkin terjadi di N=50.
   - ANOVA (section 9) ditulis ulang: A2 dikeluarkan dari ANOVA (run yang gagal mencapai target menjadi NaN, sehingga desain tidak seimbang dan bias). Ditambah effect size η²p, uji asumsi (Shapiro residual + Levene), uji non-parametrik Kruskal-Wallis per (dataset, α), dan ekspor CSV. Cell debug `df.info()` dan cell kosong dihapus.
   - Plot Pareto (10) sekarang disimpan ke file (`15_pareto_*.png`); sebelumnya hanya ditampilkan.
2. `experiments/analyze_results.py`: kolom metrik dikonversi ke numerik saat load. Kalau semua run di satu sel tidak mencapai target, `A2` bertipe object dan bisa crash.

Verifikasi (semua 0 error): notebook × 3 skenario (N=10 hasil asli; N=50 homogen dan N=50 heterogen pada folder campuran 10+50 klien), `analyze_results.py` × 4 skenario (termasuk menolak folder campuran tanpa `--num_clients`). Figure N=50 dicek visual (participation 50 batang, heatmap 50 baris, kurva 100 round).

Belum terverifikasi: smoke test run sungguhan N=50 di laptop terputus 2× (disk laptop 98% penuh, Ray memberi peringatan). Run 1 round N=50 sempat berjalan normal (akurasi round 1 = 38%). Batch di komputer lab yang sedang berjalan adalah verifikasi sebenarnya.

## 2026-10-05 — Dukungan skala N≠10 (persiapan up-scaling)

- `src/data/partitioner.py`: `_get_partition_dir`, `check_partition_exists`, `create_dirichlet_partition`, `load_partition_info` menerima `num_clients`. Untuk N=10 nama folder **tetap sama** (`alpha01_seed42`). Untuk N lain diberi akhiran `_n<N>`. **Logika partisi Dirichlet tidak diubah sama sekali.** Diverifikasi: algoritma yang sama mereproduksi persis jumlah sampel partisi 10-klien yang tersimpan.
- `src/data/loader.py`, `src/client/fl_client.py`: meneruskan `num_clients` agar klien memuat folder partisi yang benar.
- `experiments/run_single.py`: helper `make_experiment_id()`, dengan akhiran `_n<N>` untuk N≠10. `final_metrics.json` kini mencatat `num_clients`, `clients_per_round`, `num_rounds`.
- `experiments/run_batch.py`: memakai `make_experiment_id`, jadi `--skip_existing` mengenali hasil per skala.
- `experiments/validate.py`: argumen `--num_clients`, `--clients_per_round`, `--seeds`.
- `experiments/analyze_results.py`: argumen `--num_clients`, menolak mencampur skala berbeda, output default `analysis_n<N>/`.
- Verifikasi: 47/47 test pass, `validate.py --skip_pipeline` (N=10) pass, `analyze_results.py` pada grid 10 klien tetap jalan. Uji coba partisi N=50 (CIFAR-10 α=0.1 seed 42) berhasil, dan partisi 10 klien di folder lama tidak berubah. Smoke test run 50 klien **terputus di tengah** karena sesi berakhir dan disk laptop hampir penuh (Ray memperingatkan >95% terpakai), jadi **belum terverifikasi**. Jalankan Step 2 (`validate.py` Level 4) di komputer lab sebelum batch.
- Partisi uji `data/partitions/cifar10/alpha01_seed42_n50/` (587 MB) masih ada di laptop; aman dihapus atau dipakai ulang.

## 2026-10-05 — Notebook disesuaikan untuk hasil heterogenitas perangkat (`results_system/`)

User sudah menjalankan 54 eksperimen `--system_hetero` di komputer lab, plus `analyze_results.py`.

Perubahan `notebooks/analysis.ipynb`:
- Cell setup: saklar **`RESULTS_SUBDIR`** (`'results'` / `'results_system'`). Figure & CSV ditulis ke `notebooks/figures/<RESULTS_SUBDIR>/`, jadi kedua set hasil tidak saling menimpa. (Figure lama di `notebooks/figures/*.png` dibiarkan; versi barunya ada di `figures/results/`.)
- Cell load: membaca `sim_total_time_seconds`, `A2_time_to_target_seconds`, `sim_cumulative_time`, `system_hetero`.
- Section **11 baru** (otomatis dilewati untuk `results/`): 11A tabel time-to-target & total waktu simulasi (+ `summary_sim_time.csv`), 11B kurva time-to-accuracy (akurasi vs waktu simulasi kumulatif), 11C rata-rata durasi round + ANOVA total waktu simulasi.

Verifikasi: notebook dieksekusi headless dua kali, **0 error** di kedua mode: (1) `results/` asli, (2) fixture `results_system/` tiruan, yaitu metrik waktu yang dihitung ulang dari participation log lama + device model. Fixture hanya untuk uji kode; angkanya bukan hasil eksperimen. Notebook yang disimpan adalah hasil eksekusi mode `results/`.

## 2026-10-03 — Opsi C: heterogenitas perangkat tersimulasi + Oort system utility (kode siap, batch BELUM dijalankan)

Setelah bimbingan: opsi A (sudah selesai 2026-09-29) dan C dikerjakan, opsi B (N=50) ditunda.

**File baru/diubah:**
- `src/system/device_model.py` (baru): tiap klien diberi `compute_speed` (sampel/detik, log-normal median 200, σ=1) dan `bandwidth` (MB/s, log-normal median 5, σ=1). Diacak dengan seed+10000, sehingga independen dari partisi data dan tidak berkorelasi dengan label skew. Durasi klien `t_i = |B_i|·E/speed_i + 2·model_MB/bandwidth_i`. Durasi round = klien terpilih paling lambat (FedAvg sinkron). Pendekatan ini mengikuti metodologi Oort sendiri, yang juga mengemulasi runtime perangkat (§7.1, trace AI Benchmark/MobiPerf); spread ~±1 orde besaran sesuai Fig. 2 paper.
- `src/strategies/performance_strategy.py`: bila diberi `device_model`, Oort memakai Eq. 1 penuh: `Util(i) = (U(i)+staleness)·(T/t_i)^α` jika `t_i>T`, dengan α=2. **Pacer** (Alg. 1 baris 7–8): jika utilitas statistik W round terakhir < W round sebelumnya, T dinaikkan. T dinyatakan sebagai persentil durasi klien (awal 30, Δ=+5 poin), mengikuti implementasi resmi Oort di FedScale. W=5 (paper W=20 untuk ratusan round; pacer baru bisa aktif setelah 2W round). **Exploration** memakai SampleBySpeed (peluang ∝ 1/t_i). Tanpa `device_model`, perilakunya identik dengan versi lama.
- `experiments/run_single.py` & `run_batch.py`: flag `--system_hetero`. Hasilnya ke folder **`results_system/`** (terpisah, tidak menimpa `results/`). Output tambahan: `device_profiles.json`, `sim_round_durations`, `sim_total_time_seconds`, **`A2_time_to_target_seconds`** (waktu simulasi sampai target akurasi). Random & FairFedCS tetap tidak memakai info kecepatan (sesuai algoritmanya), tapi waktu simulasi mereka tetap dicatat agar bisa dibandingkan.
- `experiments/analyze_results.py`: membaca `sim_total_time_seconds` & `A2_time_to_target_seconds` (summary) + ANOVA untuk `sim_total_time_seconds`. Pakai `--results_dir results_system`.
- `tests/test_system_heterogeneity.py` (baru): 16 test (device model deterministik & independen dari data, rumus durasi, penalti straggler, pacer, SampleBySpeed, metrik waktu simulasi). Total **47/47 test pass**.

**Bug yang ditemukan & diperbaiki saat implementasi — memengaruhi Oort di grid 54 eksperimen lama:**
Di Flower 1.32, `ClientProxy.cid` adalah *node id acak 64-bit*, bukan nomor partisi data. Ketiga strategi memetakan cid → index dengan mengurutkan cid, sehingga "klien 3" di strategi **bukan** klien yang memegang `client_3.pt` (urutannya acak per run). Fix: `src/strategies/client_ids.py`, memetakan via `proxy.partition_id`.
- Dampak ke hasil lama: **Oort** memakai `client_num_samples[index]` (|B_i| di U(i)), jadi pada grid lama |B_i| tertukar antar klien. Utility Oort memakai jumlah sampel klien yang salah. Komponen loss-nya tetap benar karena loss & seleksi memakai index yang sama.
- **Random**: tidak berdampak pada seleksi; hanya label log/participation_count yang permutasi (B3 tidak berubah karena std invarian terhadap permutasi).
- **FairFedCS**: tidak berdampak. Reputasi/antrean/Shapley hanya memakai index yang konsisten dalam 1 run dan tidak memakai metadata partisi; hanya label klien di log yang tidak cocok dengan nomor file partisi.
- Rekomendasi: jalankan ulang 18 eksperimen Oort pada grid lama (`run_batch.py --strategies performance`, setelah folder lama dipindah) agar U(i) benar. Belum dikerjakan — butuh keputusan user.

**Smoke test** (MNIST α=0.1, 2 round, 1 epoch, `--system_hetero`): random & performance jalan tanpa error. Durasi round & `device_profiles.json` tersimpan, dan `analyze_results.py --results_dir` bisa membaca hasilnya. Contoh: waktu simulasi 2 round Oort 83s vs Random 109s.

**Cara menjalankan (belum dijalankan):**
```
python experiments/run_batch.py --system_hetero --skip_existing
python experiments/analyze_results.py --results_dir results_system
```
Estimasi waktu nyata ~ sama dengan grid lama (simulasi waktu tidak menambah komputasi): ±16 jam untuk 54 eksperimen, didominasi FairFedCS.

## 2026-09-28 — Rangkuman hasil 54 eksperimen & jawaban Rumusan Masalah

Sumber: `results/analysis/{summary_table.csv, anova_results.json, pareto_data.csv, fairness_threshold_summary.csv}` + `accuracy_history` tiap run. Setup: N=10 klien, m=5/round, 20 round, 3 local epoch, 3 seed per sel (n=3). Semua angka = rata-rata 3 seed.

### Tabel ringkas (mean)

| Dataset | α | Strategi | A1 Akurasi (%) | A2 Round→target | B1 Std akurasi | B2 Gini | B3 Std partisipasi |
|---|---|---|---|---|---|---|---|
| CIFAR-10 | 0.1 | Random | 54.13 ± 5.86 | tidak tercapai | 0.230 | 0.240 | **2.31** |
| | | Oort | 55.08 ± 5.35 | tidak tercapai | 0.238 | 0.243 | 3.42 |
| | | FairFedCS | **58.10** ± 4.17 | tidak tercapai | **0.223** | **0.237** | 3.17 |
| CIFAR-10 | 0.5 | Random | 69.25 ± 4.97 | 14.0 (3/3) | **0.094** | **0.070** | **1.91** |
| | | Oort | 69.21 ± 1.02 | 15.5 (2/3) | 0.105 | 0.072 | 3.55 |
| | | FairFedCS | **70.63** ± 1.13 | 14.0 (3/3) | 0.125 | 0.085 | 2.77 |
| CIFAR-10 | 1.0 | Random | **72.60** | 12.3 | **0.045** | **0.031** | 1.91 |
| | | Oort | 72.26 | **12.0** | 0.057 | 0.039 | 2.79 |
| | | FairFedCS | 71.76 | 13.3 | 0.054 | 0.036 | **1.30** |
| MNIST | 0.1 | Random | **98.29** | 4.0 | **0.014** | **0.008** | **1.91** |
| | | Oort | 96.95 | 3.3 | 0.034 | 0.019 | 2.82 |
| | | FairFedCS | 97.96 | **3.0** | 0.019 | 0.010 | 2.29 |
| MNIST | 0.5 | semua | 99.12–99.21 | 1.0 (semua) | ≈0.002 | ≈0.001 | R 1.91 / O 2.43 / F 2.48 |
| MNIST | 1.0 | semua | 99.20–99.23 | 1.0 (semua) | ≈0.0015 | ≈0.0008 | O **1.77** / R 1.88 / F 2.44 |

ANOVA dua arah (strategy × α, per dataset, n=3/sel):

| Dataset | Metrik | p strategi | p α | p interaksi |
|---|---|---|---|---|
| CIFAR-10 | A1 | 0.618 | <0.001 | 0.818 |
| CIFAR-10 | B1 | 0.694 | <0.001 | 0.816 |
| CIFAR-10 | B2 | 0.971 | <0.001 | 0.998 |
| CIFAR-10 | B3 | **<0.001** | 0.001 | 0.054 |
| MNIST | A1 | 0.130 | <0.001 | 0.103 |
| MNIST | B1 | **0.010** | <0.001 | **0.002** |
| MNIST | B2 | **0.017** | <0.001 | **0.005** |
| MNIST | B3 | 0.135 | 0.466 | 0.399 |

Pareto (akurasi vs Gini, per dataset×α): CIFAR-10 α=0.1 → FairFedCS; α=0.5 → FairFedCS & Random; α=1.0 → Random. MNIST α=0.1 → Random; α=0.5 → FairFedCS & Oort; α=1.0 → Oort & Random.

### RM1 — Pengaruh strategi terhadap akurasi global & fairness

Pada seluruh kondisi, **tingkat label skew (α) jauh lebih menentukan daripada pilihan strategi**. Perbedaan akurasi antar strategi pada satu α umumnya 1–4 poin persen dan berada di dalam rentang simpangan baku antar-seed; ANOVA tidak menemukan efek strategi yang signifikan terhadap akurasi global (CIFAR-10 p=0.62, MNIST p=0.13).

Secara deskriptif, FairFedCS unggul akurasi pada skew berat–sedang di CIFAR-10 (α=0.1: 58.1% vs 55.1% Oort vs 54.1% Random; α=0.5: 70.6% dan satu-satunya selain Random yang mencapai target 70% di ketiga seed), dengan varians antar-seed yang juga lebih kecil. Keunggulan ini hilang pada α=1.0, di mana Random sedikit lebih baik. Di MNIST, semua strategi mencapai ≥97% dan target 85% tercapai sejak round 1 untuk α≥0.5 (efek plafon), sehingga dataset ini kurang mampu membedakan strategi kecuali pada α=0.1.

Untuk fairness per-klien (B1/B2), efek strategi signifikan hanya di MNIST, dan efek itu terutama digerakkan oleh **Oort yang paling tidak adil pada α=0.1** (std akurasi 0.034 vs 0.019 FairFedCS vs 0.014 Random) — terlihat juga dari interaksi strategi×α yang signifikan (p≈0.002–0.005): perbedaan antar strategi hanya muncul saat skew berat. Untuk fairness partisipasi (B3), strategi berpengaruh signifikan di CIFAR-10 (p<0.001): Oort konsisten menghasilkan partisipasi paling timpang, Random paling merata pada α rendah, FairFedCS paling merata pada α=1.0.

### RM2 — Label skew terhadap trade-off akurasi–fairness & signifikansinya

α berpengaruh signifikan (p<0.001) terhadap A1, B1, dan B2 di kedua dataset. Makin berat skew, akurasi turun dan ketimpangan antar-klien naik, dan efeknya jauh lebih besar pada CIFAR-10 (akurasi turun ~14–18 poin dan Gini naik ~7× dari α=1.0 ke α=0.1) daripada MNIST (turun ~1–2 poin). Trade-off akurasi–fairness baru benar-benar muncul pada skew berat, yang terlihat dari interaksi strategi×α yang signifikan untuk B1/B2 di MNIST. Pada skew ringan (α=1.0), ketiga strategi praktis setara. Di CIFAR-10 interaksi untuk A1/B1/B2 tidak signifikan: dengan n=3 seed dan varians antar-seed yang tinggi pada α=0.1 (std ±4–6 poin), perbedaan deskriptif yang ada belum dapat dikonfirmasi secara statistik.

### RM3 — Threshold α di mana FairFedCS optimal & konsistensinya

Berdasarkan Pareto frontier (akurasi vs Gini), FairFedCS optimal di CIFAR-10 pada **α=0.1 (satu-satunya titik optimal) dan α=0.5**, tetapi tidak pada α=1.0. Jadi FairFedCS menguntungkan pada skew berat–sedang dan kehilangan keunggulannya saat data mendekati IID. Pola ini **tidak konsisten di MNIST**: pada α=0.1 justru Random yang optimal, dan FairFedCS hanya ikut optimal pada α=0.5, dengan selisih yang sangat kecil (orde 0.01–0.1 poin, dalam noise). Ketidakkonsistenan ini wajar, karena MNIST terlalu mudah untuk model CNN ini sehingga hampir semua strategi mendekati plafon akurasi. Catatan: Pareto dihitung dari rata-rata, dan perbedaan A1/B2 antar strategi di CIFAR-10 tidak signifikan secara ANOVA, jadi klaim threshold sebaiknya ditulis sebagai kecenderungan empiris, bukan kesimpulan statistik.

### Mengapa beberapa hasil terlihat "berlawanan" dengan sifat strategi (untuk pembahasan)

1. **Oort bukan yang tercepat.** Keunggulan kecepatan Oort di paper aslinya berasal terutama dari *system utility* (memilih klien cepat untuk mengurangi durasi round/wall-clock), dan komponen ini sengaja di-drop di sini. Yang tersisa hanya *statistical utility* `|B_i|·sqrt(mean loss²)`, yang memilih klien dengan **loss tinggi**, bukan klien "berperforma bagus". Di bawah label skew berat, klien loss-tinggi adalah klien dengan distribusi label paling menyimpang, sehingga model global terdorong ke arah klien tersebut. Itu sebabnya Oort paling tidak adil di MNIST α=0.1 (std akurasi 0.034) dan akurasinya terendah di sana (96.9%). A2 diukur dalam round, bukan waktu, jadi keuntungan Oort memang tidak bisa terlihat di metrik ini.
2. **Mekanisme bandit Oort hampir tidak bekerja pada N=10, m=5.** ε awal 0.9 → round 1–2 hampir semuanya eksplorasi, dan setelah ~2 round semua 10 klien sudah "explored" sehingga eksplorasi habis. Selanjutnya seleksi adalah eksploitasi proporsional utility pada pool kecil. Oort dirancang untuk ribuan klien, sehingga pada skala ini perilakunya mendekati "random terbobot loss". Hal ini menjelaskan hasil yang mirip Random tetapi dengan partisipasi lebih timpang (B3 tertinggi di hampir semua sel).
3. **FairFedCS tidak selalu punya B3 terendah.** FairFedCS tidak mengejar partisipasi *sama rata*. Ia mengejar keadilan *proporsional terhadap reputasi*: antrean virtual `Q_i` tumbuh sebesar `ε·r_i`, jadi klien bereputasi rendah (Shapley negatif) memang sengaja lebih jarang dipilih. Random dengan 20 round × 5/10 memberi partisipasi yang secara statistik sudah cukup merata. B3 Random bernilai sama (1.9121) di 4 dari 6 sel. Ini kemungkinan karena urutan sampling acak Flower terutama ditentukan oleh seed, bukan oleh α/dataset, dan ini wajar untuk baseline random (belum diverifikasi di kode, 2 sel lain berbeda).
4. **Perbedaan kecil & banyak yang tidak signifikan.** Dengan hanya 10 klien (m/N=50%), setiap strategi tetap melihat sebagian besar klien tiap 2 round, sehingga ruang pembeda antar strategi sempit. Ditambah n=3 seed, daya uji statistik menjadi rendah. Ini perlu dicantumkan di bagian keterbatasan.
5. **Biaya komputasi.** Rata-rata waktu per run: Random 4.9/8.0 menit, Oort 7.2/10.3 menit, FairFedCS 29.5/33.1 menit (MNIST/CIFAR). FairFedCS ~4–6× lebih lambat karena 32 evaluasi subset Shapley per round.

### Anomali tambahan (dicek dari participation_log.json & log Shapley)

6. **Random justru paling adil di akurasi per-klien (B1/B2) di banyak kondisi.** Random punya B1/B2 terendah di CIFAR α=0.5, CIFAR α=1.0, dan MNIST α=0.1. FairFedCS di CIFAR α=0.5 malah punya B1 terburuk (0.125), padahal akurasinya tertinggi. Penyebabnya: "fairness" pada FairFedCS adalah keadilan *seleksi* yang proporsional terhadap reputasi, bukan pemerataan *akurasi* per klien. Model global yang lebih bias ke klien berkontribusi tinggi bisa naik akurasinya, tapi klien minoritas makin tertinggal.
7. **FairFedCS tetap membuat partisipasi timpang.** Contoh CIFAR α=0.1 s42: klien 3 dipilih 17/20 round, sedangkan klien 6 hanya 6/20. Pola 17× ini muncul di ketiga seed. Klien dengan Shapley selalu positif (klien 3 & 7: φ≈+2 sampai +11 tiap round) reputasinya terus naik, lalu antrean Q klien itu juga tumbuh lebih cepat (`c_i = ε·r_i`). Akibatnya klien itu makin sering dipilih dan terjadi efek "rich get richer".
8. **Seleksi awal FairFedCS deterministik & identik di semua seed.** Round 1–3 selalu {0-4}, {5-9}, {0-4}. Saat semua reputasi masih 0.5 dan Q=0, CSI semua klien sama, sehingga tie-break memakai indeks klien. Perilaku ini sesuai algoritma (seleksi top-m yang deterministik, bukan sampling), tapi artinya seed hanya memengaruhi FairFedCS lewat partisi data dan inisialisasi model, bukan lewat seleksi.
9. **Shapley Value kehilangan daya beda pada data mudah/IID.** Di MNIST α=1.0, setelah round 1 nilai φ hanya berkisar ±0.01–0.2 poin akurasi, pada level noise. Karena reputasi di-update berdasarkan *tanda* φ, reputasi praktis diisi noise. Di kondisi ini FairFedCS berperilaku mirip round-robin + noise, sehingga B3-nya tidak lebih baik dari Random. Sebaliknya, di CIFAR α=0.1 Shapley sangat informatif (rentang −6 s/d +11). Ini konsisten dengan temuan bahwa FairFedCS hanya unggul pada skew berat.
10. **Nilai φ round 1 sangat besar (MNIST: +15 s/d +19).** Baseline f(∅) adalah model inisialisasi acak (~10%), jadi semua klien mendapat φ positif besar dan +1 reputasi "gratis". Ini wajar, tapi round 1 tidak membedakan kualitas klien.
11. **B3 Random identik (1.9121) di beberapa sel** — sudah dicek: yang sama adalah *multiset* jumlah partisipasi, sedangkan ID klien yang terpilih berbeda. Sampling Flower memakai urutan RNG dari seed yang sama terhadap daftar CID, sehingga pola hitungannya sama walaupun datasetnya berbeda. Ini bukan bug, dan wajar karena seleksi random memang tidak bergantung pada data.
12. **Oort sering "melupakan" klien tertentu.** Contoh: di MNIST α=0.1 s42 klien 8 hanya dipilih 3/20, di CIFAR α=0.5 s42 klien 8 dipilih 4/20. Klien yang loss-nya kecil (datanya sudah "dipelajari") utility-nya rendah dan jarang terpilih lagi. Ini sumber B3 tertinggi pada Oort.
13. **Metrik A2 jenuh/kosong di dua ekstrem.** Di MNIST α≥0.5 semua strategi mencapai target 85% di round 1, jadi A2 = 1 untuk semua dan tidak membedakan apa pun. Di CIFAR α=0.1 tidak ada strategi yang mencapai 70% (A2 = NaN semua). Metrik "kecepatan konvergensi" hanya informatif di CIFAR α=0.5/1.0 dan MNIST α=0.1. Sebagai pelengkap bisa dipakai rata-rata akurasi sepanjang round (AUC); dengan metrik ini FairFedCS tertinggi di CIFAR α=0.1 (46.7 vs 44.4/44.1) dan MNIST α=0.1.
14. **Variansi antar-seed sangat besar di CIFAR α=0.1** (std A1 ±4–6 poin, lebih besar dari selisih antar strategi). Partisi Dirichlet berbeda per seed, dan pada α=0.1 pembagian label antar klien sangat bergantung pada hasil sampling. Inilah penyebab utama efek strategi tidak signifikan di ANOVA.

### Akar penyebab anomali & opsi perbaikan (2026-09-29, belum dikerjakan — menunggu keputusan user)

Akar penyebab (terverifikasi dari kode & log):
- **Skala eksperimen (paling dominan).** N=10 dan m=5 (50%): Random saja sudah memilih tiap klien ±10×/20 round, sehingga hampir tidak ada ruang bagi strategi pintar untuk berbeda. Oort (ribuan klien) dan FairFedCS (ratusan klien, m/N kecil) dirancang untuk populasi besar. Eksplorasi Oort sudah habis di round ~2.
- **Beda definisi fairness.** FairFedCS mengejar keadilan *seleksi* yang proporsional terhadap reputasi (Q tumbuh `ε·r_i`), bukan pemerataan *akurasi*. Ini menimbulkan efek rich-get-richer: klien 3 dipilih 17/20. Oort statistical utility memilih klien loss-tinggi, yang di bawah skew adalah klien paling menyimpang.
- **Cara ukur.** B1/B2 diambil dari 1 snapshot round terakhir. Akurasi per-klien dievaluasi pada data *train* lokal klien (`fl_client.evaluate` memakai `self.trainloader`). A2 jenuh (MNIST round 1) atau NaN (CIFAR α=0.1). n=3 seed dengan variansi besar.
- **Shapley.** Dihitung pada test set global. Nilainya noise pada data mudah, sementara reputasi hanya memakai tanda φ.

Opsi perbaikan yang sah (bukan mengakali hasil):
1. Tanpa re-run: B1/B2 rata-rata k round terakhir, metrik AUC akurasi, reframing pembahasan. Hasil "Random kompetitif pada N kecil" adalah temuan yang valid.
2. Re-run dengan desain lebih tepat: N lebih besar (mis. 50, m=5 / 10%), seed ditambah, evaluasi per-klien pada split test lokal. Mengubah Batasan proposal (N=10) → perlu persetujuan dosen.
3. Oort + heterogenitas sistem tersimulasi + metrik wall-clock (asumsi latency harus dicatat eksplisit).
Tidak sah: tuning σ/parameter atau memilih seed sampai hasil "sesuai harapan".

### Opsi A dieksekusi (2026-09-29)

`experiments/analyze_results.py` ditambah metrik robust dari `metrics_per_round.json` (tanpa re-run): `A_auc_accuracy` (rata-rata akurasi 20 round, proxy kecepatan konvergensi), `A1_lastK_accuracy`, `B1_lastK_accuracy_std`, `B2_lastK_gini` (rata-rata 5 round terakhir). Metrik ini masuk ke `summary_table.csv` dan `anova_results.json`. Output baru: `pareto_data_robust.csv` dan `fairness_threshold_summary_robust.csv`.

Hasil utama:
- AUC akurasi: FairFedCS tertinggi di CIFAR α=0.1 (46.7 vs 44.4 Oort vs 44.1 Random) dan MNIST α=0.1 (92.7 vs 92.6 vs 91.7). Oort tertinggi di CIFAR α=0.5 (61.8), Random tertinggi di α=1.0.
- Setelah dirata-rata 5 round, keunggulan "Random paling adil" mengecil: selisih B1/B2 antar strategi jadi sangat tipis. Efek strategi yang tadinya signifikan di MNIST (B1 p=0.010, B2 p=0.017) **hilang** (p≈0.85–0.89). Artinya signifikansi itu berasal dari fluktuasi round terakhir, bukan efek yang stabil.
- ANOVA semua metrik robust: efek α signifikan (p<0.001), efek strategi & interaksi tidak signifikan (p>0.5).
- Pareto robust: FairFedCS optimal di CIFAR α=0.1 (bersama Random), MNIST α=0.1 & 0.5. Random optimal di hampir semua sel. Threshold "FairFedCS optimal pada skew berat (α=0.1)" kini **konsisten di kedua dataset**, tapi tidak eksklusif.
- Kesimpulan jujur: dengan N=10/m=5, pengaruh strategi seleksi terhadap akurasi & fairness per-klien tidak signifikan. Label skew yang dominan. FairFedCS cenderung unggul (deskriptif) pada α=0.1.

### Catatan untuk opsi B (belum dikerjakan)
`configs/experiment_config.yaml` **tidak dibaca kode**. Parameter diambil dari argumen CLI `run_batch.py` (`--num_clients`, `--clients_per_round`, `--rounds`, `--seeds`). Namun sebelum re-run dengan N≠10 wajib ada 2 perubahan kecil:
1. Folder partisi `data/partitions/<ds>/alpha{a}_seed{s}/` tidak memuat N. Menjalankan N=50 akan membuat partisi baru di folder yang sama dan **menimpa `partition_info.json` partisi 10-klien**.
2. `experiment_id` hasil (`results/<strategy>_<ds>_a<a>_s<s>`) juga tidak memuat N, sehingga `--skip_existing` akan melewati semua eksperimen dan hasil lama bisa tertimpa.

### Saran penulisan/lanjutan (opsional, belum dikerjakan)
- Tulis keunggulan FairFedCS di CIFAR-10 α≤0.5 sebagai temuan deskriptif, dan dukung dengan uji post-hoc/effect size alih-alih klaim signifikansi.
- Jika waktu memungkinkan: tambah seed (mis. 5), dan/atau tambah metrik "AUC kurva akurasi" (rata-rata akurasi 20 round) sebagai proxy kecepatan konvergensi, karena A2 di MNIST jenuh di round 1.

## 2026-09-28 — Verifikasi & perbaikan pipeline analisis end-to-end (setelah full batch 54 eksperimen selesai)

User sudah menjalankan full batch 54 eksperimen baru (hasil di `results/`). Diminta verifikasi kode analisis (`experiments/analyze_results.py`, `notebooks/analysis.ipynb`, dan file terkait) bebas error sebelum dipakai untuk tahap analisis skripsi.

**2 bug nyata ditemukan & diperbaiki, keduanya baru ketahuan setelah dijalankan terhadap data 54-eksperimen asli:**

1. `experiments/analyze_results.py` — `build_pareto_frontier()` grouping salah: `groupby("dataset")` mencampur SEMUA nilai alpha jadi satu perbandingan dominance, padahal Pareto frontier harus dibandingkan PER (dataset, alpha) — akibatnya banyak titik yang seharusnya optimal (mis. `random` di MNIST α=0.1 yang terbaik di akurasi DAN Gini sekaligus) malah ditandai `pareto_optimal=False`. Fix: groupby `["dataset","alpha"]`.
2. `src/metrics/evaluator.py` — `run_two_way_anova()` hardcode nama kolom pingouin `"p-unc"` (strip), padahal versi pingouin terinstal (0.6.1) pakai `"p_unc"` (underscore) — selalu `KeyError`. Fix: resolve nama kolom p-value secara dinamis (menerima kedua ejaan).
3. `notebooks/analysis.ipynb` — bug yang SAMA (`'p-unc'`) juga ada ter-duplikasi langsung di cell notebook (bukan cuma di evaluator.py), plus 2 cell "Pareto Frontier" detail (multi-metrik) memakai nama kolom panjang (`A1_global_accuracy`, dst) yang TIDAK ADA di DataFrame notebook (kolomnya pendek: `A1`,`A2`,`B1`,`B2`,`B3`) — akan `KeyError` kalau dijalankan. Root cause: `notebooks/add_improvements.py` adalah script one-time yang dulu inject cell-cell ini ke notebook dengan asumsi nama kolom yang salah. **Jangan jalankan ulang `add_improvements.py`** — akan menyuntik ulang bug yang sama di atas notebook yang sudah diperbaiki.
4. Notebook juga dirapikan: hapus 1 cell mati (`savefig2`, typo `bbox_inches='thight'`, tidak pernah dipanggil), hapus 1 fungsi duplikat mati (`run_two_way_anova_fixed`, tidak pernah dipanggil karena cell pemanggilnya di-comment), dan penomoran section dirapikan (8→Early Insights, 9→ANOVA Detail, 10→Pareto Detail).

**Verifikasi:** `analysis.ipynb` dieksekusi headless end-to-end (`python -m nbconvert --execute`) terhadap 54 hasil eksperimen asli — **0 error di 38 cell**. Angka F/p ANOVA dari notebook dicek silang dengan `analyze_results.py` — identik. Notebook (dengan output baru yang benar) sudah menggantikan versi lama yang tersimpan (sebelumnya berisi path mesin lama + error `'p-unc'` + angka dari strategi lama/salah).

**Belum ditindak (butuh keputusan user), tidak diubah sendiri:**
- `notebooks/analysis_fixed.ipynb` — nyaris duplikat `analysis.ipynb` versi lebih lama, bug yang sama, tidak direferensikan README. Perlu diputuskan: dihapus / disamakan / dibiarkan.
- `notebooks/analysis_baseline.ipynb` + `.py` — analisis cepat 3-eksperimen (α=0.1, seed=42), dari commit Juli, kemungkinan sudah superseded oleh `analysis.ipynb` yang mencakup 54 eksperimen penuh.
- `notebooks/add_improvements.py` — script codegen one-time, sudah selesai tugasnya (jangan dijalankan ulang, lihat poin 3 di atas).

## 2026-09-22 — Strategi seleksi klien dirombak sesuai paper Oort & FairFedCS

**Keputusan desain (dikonfirmasi user):**
- Oort: komponen system-utility/pacer/straggler-penalty di-DROP sepenuhnya (tidak relevan untuk simulasi 1 mesin tanpa heterogenitas device nyata, dan proposal tidak mengukur efisiensi sistem). Implementasi fokus pada statistical utility (loss-based) + full exploration-exploitation bandit.
- FairFedCS: kontribusi klien dinilai via **exact Shapley Value** (enumerasi 2^m subset koalisi per round), bukan aproksimasi GTG-Shapley dari paper asli — feasible karena m=5 klien/round, dan lebih presisi.

**File yang diubah:**
- `src/client/fl_client.py` — tambah metrik `loss_rms` (sqrt(mean(per-sample loss²))) di `fit()`, dibutuhkan Oort untuk U(i).
- `src/strategies/performance_strategy.py` — rewrite total jadi Oort: statistical utility, explored-set tracking, temporal uncertainty (staleness bonus), robustness clipping (percentile 95), cutoff-pool proportional sampling (exploitation), uniform random sampling klien belum-pernah-dipilih (exploration), epsilon decay dgn floor 0.2. Menambahkan override `aggregate_fit()` untuk menangkap `loss_rms` per klien dengan identitas benar (lihat catatan bug di bawah).
- `src/strategies/fairness_strategy.py` — rewrite total jadi FairFedCS: Beta Reputation System (`r_i=(a_i+1)/(a_i+b_i+2)`), virtual fairness queue (Lyapunov), Client Suitability Index (`CSI_i=sigma*r_i+Q_i`, sigma=0.6), override `aggregate_fit()` untuk hitung exact Shapley Value tiap round & update reputasi berdasarkan tanda kontribusi.
- `experiments/run_single.py` — `build_strategy()` diperbarui (hapus latency generation, tambah `make_shapley_eval_fn()` — scratch model terpisah untuk evaluasi subset Shapley agar tidak menimpa global model utama). **Bug fix**: `make_callbacks()` sebelumnya salah mengira elemen pertama tuple `evaluate_metrics_aggregation_fn`/`fit_metrics_aggregation_fn` sebagai client_id — padahal API Flower mengembalikan `(num_examples, metrics)`, BUKAN `(client_id, metrics)`. Ini menyebabkan `per_client_losses` di `metrics_per_round.json` salah label (dan hook Oort lama `update_client_loss` menerima cid palsu). Fix: per-client bookkeeping yang butuh identitas asli (Oort loss_rms, FairFedCS Shapley) sekarang dilakukan di dalam `aggregate_fit()` strategi masing-masing (punya akses `ClientProxy.cid` asli), bukan lewat callback generik Flower.
- `experiments/analyze_results.py` — BARU. Agregasi seluruh `results/*/final_metrics.json`, hasilkan `summary_table.csv`, `pareto_data.csv` (C1), `anova_results.json` (C2), `fairness_threshold_summary.csv` (jawaban Rumusan Masalah 3).
- `tests/test_performance_strategy.py`, `tests/test_fairness_strategy.py` (baru), `tests/smoke_test_performance.py` — direvisi/dibuat mengikuti API baru. Shapley Value diverifikasi manual di toy example 3-klien (efficiency axiom: sum(phi_i) == f(full)-f(empty)) — semua 31 test pass.
- `configs/experiment_config.yaml`, `README.md`, `requirements.txt` (+`ray`) — disesuaikan dengan implementasi baru.
- 54 hasil eksperimen lama (strategi lama, tidak valid lagi) dipindah ke `results_archive_old/`.

**Verifikasi:**
- `experiments/validate.py` (full, termasuk Level 4 pipeline smoke test end-to-end 1 round × 3 strategi di MNIST): **SEMUA PASS** (Level 1: 22/22, Level 2: 19/19, Level 3: 9/9, Level 4: 3/3).
- `pytest tests/` — 31/31 test pass, termasuk verifikasi manual Shapley Value pada toy example 3-klien (efficiency axiom `sum(phi_i) == f(full)-f(empty)` terbukti exact).
- `tests/smoke_test_performance.py` standalone — pass.

**Catatan runtime penting (perlu diperhatikan sebelum full batch 54 eksperimen):**
Dari pipeline smoke test (1 round, 1 local epoch, MNIST alpha=0.5): random=67s, performance=93s, **fairness=228s**. Selisih ~135-160s untuk fairness berasal dari 32 evaluasi subset Shapley per round (≈4-5s/eval). Untuk eksperimen penuh (20 round), overhead Shapley diestimasi **+45-55 menit per eksperimen FairFedCS** di atas baseline. Dengan 18 eksperimen FairFedCS (9 MNIST + 9 CIFAR-10) dari total 54, full batch 54 eksperimen kemungkinan makan waktu **jauh lebih lama dari estimasi lama di README (~15-20 jam)** — kemungkinan 20+ jam tergantung hardware. Belum dijalankan full batch — itu keputusan terpisah yang diserahkan ke user.


