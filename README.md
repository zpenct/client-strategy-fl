
# Analisis Perbandingan Strategi Seleksi Klien pada Federated Learning
**Judul:** Analisis Perbandingan Strategi Seleksi Klien pada Federated Learning dengan Data Label-Skewed: Studi Simulasi menggunakan Framework Flower


## Daftar Isi

- [Gambaran Umum](#gambaran-umum)
- [Desain Eksperimen](#desain-eksperimen)
- [Setup (Ubuntu / Linux / macOS)](#setup-ubuntu--linux--macos)
- [Setup (Windows)](#setup-windows)
- [Menjalankan Eksperimen](#menjalankan-eksperimen)
- [Format Output](#format-output)
- [Analisis Hasil (Notebook)](#analisis-hasil-notebook)
- [Catatan Teknis](#catatan-teknis)

## Gambaran Umum

Repository ini berisi sistem simulasi Federated Learning (FL) untuk membandingkan tiga strategi seleksi klien pada kondisi data Non-IID (label-skewed) menggunakan framework [Flower (flwr)](https://flower.ai/).

**Tiga strategi yang dibandingkan:**

| Strategi | Deskripsi | Paper acuan |
|---|---|---|
| `random` | Seleksi klien sepenuhnya acak per round | FedAvg baseline (McMahan dkk., 2017) |
| `performance` | Statistical utility berbasis loss lokal (`U(i)=|Bi|*sqrt(mean(Loss(k)^2))`) + bandit exploration-exploitation (staleness bonus, robustness clipping, cutoff-pool proportional sampling). Pada grid utama (`results/`) komponen system-utility/pacer tidak aktif (tanpa heterogenitas device). Dengan `--system_hetero` (`results_system/`), system utility `(T/t_i)^2`, pacer, dan exploration berbasis kecepatan diaktifkan di atas heterogenitas perangkat tersimulasi (`src/system/device_model.py`). | Oort (Lai dkk., 2021) |
| `fairness` | Beta Reputation System (dari exact Shapley Value kontribusi klien tiap round) + virtual fairness queue (Lyapunov optimization), dikombinasikan jadi Client Suitability Index `CSI_i = sigma*r_i + Q_i` untuk pilih top-m klien | FairFedCS (Shi dkk., 2023) |

**Non-IID disimulasikan** menggunakan distribusi Dirichlet dengan parameter α:

| α | Tingkat Skew | Keterangan |
|---|---|---|
| 0.1 | Sangat ekstrem | Tiap klien hampir hanya punya 1-2 kelas |
| 0.5 | Sedang | Label terdistribusi tidak merata |
| 1.0 | Ringan | Mendekati IID |


## Desain Eksperimen

**Grid: 3 × 3 × 2 × 3 = 54 eksperimen**

| Dimensi | Nilai |
|---|---|
| Strategi | random, performance, fairness |
| Alpha (Dirichlet) | 0.1, 0.5, 1.0 |
| Dataset | MNIST, CIFAR-10 |
| Seed | 42, 123, 456 |
| **Total** | **54 eksperimen** |

**Konfigurasi per eksperimen:**

| Parameter | Nilai |
|---|---|
| Total klien | 10 |
| Klien per round | 5 |
| Communication rounds | 20 |
| Local epochs | 3 |
| Learning rate | 0.01 |
| Target accuracy (MNIST) | 85% |
| Target accuracy (CIFAR-10) | 70% |

**Metrik evaluasi (7 metrik, 3 grup):**

| Kode | Nama | Grup | Interpretasi |
|---|---|---|---|
| A1 | Global Accuracy | A — Performance | Akurasi model global di test set (%) |
| A2 | Rounds to Target | A — Performance | Round pertama mencapai target accuracy |
| B1 | Accuracy Variance (σ) | B — Fairness | Std dev akurasi antar klien (lebih rendah = lebih fair) |
| B2 | Gini Coefficient | B — Fairness | Ketidakmerataan akurasi (0=fair, 1=unfair) |
| B3 | Participation Fairness | B — Fairness | Std dev jumlah seleksi per klien (lebih rendah = lebih fair) |
| C1 | Pareto Data | C — Trade-off | Data untuk analisis Pareto frontier |
| C2 | Two-Way ANOVA | C — Trade-off | Signifikansi statistik perbedaan strategi × alpha |

> **CIFAR-10:** Jika server resmi (`cs.toronto.edu`) tidak bisa diakses, sistem otomatis fallback ke Hugging Face Hub. Tidak perlu intervensi manual.

---

## Setup (Ubuntu / Linux / macOS)

```bash

# 2. Buat virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set PYTHONPATH (wajib — ulangi setiap buka terminal baru)
export PYTHONPATH="$PWD"

# 5. Generate partisi data (jalankan SEKALI, ±5-10 menit)
python experiments/prepare_data.py --datasets mnist cifar10

# 6. Validasi setup
python experiments/validate.py
```

**Tip:** Supaya tidak perlu set PYTHONPATH ulang setiap saat, tambahkan ke `~/.bashrc`:
```bash
echo 'export PYTHONPATH="$HOME/path/to/<repo>"' >> ~/.bashrc
source ~/.bashrc
```

---

## Setup (Windows)

### Opsi A — Git Bash (direkomendasikan)

Git Bash disertakan bersama [Git for Windows](https://git-scm.com/download/win). Syntax-nya sama dengan Linux.

```bash

# 2. Buat virtual environment
python -m venv venv
source venv/Scripts/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set PYTHONPATH
export PYTHONPATH="$PWD"

# 5. Generate partisi data
python experiments/prepare_data.py --datasets mnist cifar10

# 6. Validasi setup
python experiments/validate.py
```

### Opsi B — PowerShell

```powershell

# 2. Buat virtual environment
python -m venv venv
venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set PYTHONPATH
$env:PYTHONPATH = $PWD

# 5. Generate partisi data
python experiments\prepare_data.py --datasets mnist cifar10

# 6. Validasi setup
python experiments\validate.py
```

### VS Code (Windows)

Buat file `.env` di root project:
```
PYTHONPATH=.
```
VS Code akan otomatis load ini saat membuka terminal terintegrasi.

---

## Menjalankan Eksperimen

> Pastikan virtual environment aktif dan `PYTHONPATH` sudah di-set sebelum menjalankan command apapun.

### Validasi sebelum batch (sangat direkomendasikan)

```bash
# Validasi penuh termasuk pipeline smoke test (~15 menit)
python experiments/validate.py

# Validasi cepat tanpa pipeline (~30 detik)
python experiments/validate.py --skip_pipeline
```

### Satu eksperimen spesifik

```bash
python experiments/run_single.py \
  --strategy random \
  --dataset mnist \
  --alpha 0.1 \
  --seed 42 \
  --rounds 20
```

### Batch per dataset

```bash
# Hanya MNIST (27 eksperimen)
python experiments/run_batch.py --datasets mnist --skip_existing

# Hanya CIFAR-10 (27 eksperimen)
python experiments/run_batch.py --datasets cifar10 --skip_existing
```

### Eksperimen heterogenitas perangkat (opsi C)

```bash
python experiments/run_batch.py --system_hetero --skip_existing        # semua 54, ke results_system/
python experiments/run_batch.py --system_hetero --datasets cifar10 --skip_existing
python experiments/analyze_results.py --results_dir results_system
```

Tiap klien diberi kecepatan komputasi & bandwidth tetap (log-normal, diacak per seed, independen dari data). Durasi klien `t_i = |B_i|*E/speed_i + 2*model_MB/bandwidth_i`, durasi round = klien terpilih paling lambat. Hasil tambahan per eksperimen: `device_profiles.json`, `sim_total_time_seconds`, `A2_time_to_target_seconds`.

> **Catatan runtime — strategi `fairness` (FairFedCS) jauh lebih lambat.** Strategi ini menghitung exact Shapley Value tiap round (32 evaluasi subset koalisi untuk m=5 klien), menambah ±45-55 menit per eksperimen di atas baseline `random`/`performance`. Dari smoke test 1-round: random≈67s, performance≈93s, fairness≈228s (1 round saja). Untuk 20 round penuh, eksperimen `fairness` bisa memakan >1 jam masing-masing — pertimbangkan menjalankan batch semalaman atau per-strategi terpisah (`--strategies fairness`) agar progres tetap terpantau.

### Batch semua 54 eksperimen

```bash
python experiments/run_batch.py --skip_existing
```

### Resume jika terputus

```bash
# Jalankan command yang sama — eksperimen yang sudah ada di-skip otomatis
python experiments/run_batch.py --skip_existing

# Atau mulai dari nomor tertentu
python experiments/run_batch.py --skip_existing --start_from 15
```

### Dry run

```bash
python experiments/run_batch.py --dry_run
```

---

## Format Output

```
results/random_mnist_a0.1_s42/
├── config.json              # Parameter eksperimen
├── metrics_per_round.json   # Metrik tiap round
├── final_metrics.json       # 7 metrik akhir (A1–B3)
└── participation_log.json   # Riwayat pemilihan klien
```

**Contoh `final_metrics.json`:**

```json
{
  "A1_global_accuracy": 98.45,
  "A2_rounds_to_target": 4,
  "B1_accuracy_variance": 0.012597,
  "B2_gini_coefficient": 0.007324,
  "B3_participation_fairness": 2.898275,
  "target_reached": true,
  "accuracy_history": [74.1, 81.25, 79.84, 95.23],
  "per_client_final": [0.981, 0.976, 0.983],
  "strategy": "random",
  "dataset": "mnist",
  "alpha": 0.1,
  "seed": 42,
  "total_time_seconds": 1054.3
}
```

---

## Analisis Lintas-Eksperimen (C1 Pareto + C2 ANOVA)

Setelah batch selesai (atau sebagian selesai), jalankan:

```bash
python experiments/analyze_results.py
```

Script ini membaca semua `results/*/final_metrics.json`, lalu menghasilkan di `results/analysis/`:

| File | Isi |
|---|---|
| `summary_table.csv` | Mean±std tiap metrik per (dataset, strategi, alpha), diagregasi lintas 3 seed |
| `pareto_data.csv` | Titik Pareto-optimal (C1) per dataset berdasarkan trade-off akurasi vs Gini |
| `anova_results.json` | Two-way ANOVA (C2): signifikansi strategi × alpha × interaksi, per dataset per metrik |
| `fairness_threshold_summary.csv` | Ringkasan strategi mana yang Pareto-optimal di tiap α — jawaban langsung untuk Rumusan Masalah 3 |

## Analisis Hasil (Notebook)

```bash
pip install jupyter 
jupyter notebook notebooks/analysis.ipynb
```
