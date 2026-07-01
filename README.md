# FL Client Selection Experiment System


**Judul:** Analisis Perbandingan Strategi Seleksi Klien pada Federated Learning dengan Data Label-Skewed: Studi Simulasi menggunakan Framework Flower

---

## Struktur Eksperimen

| Dimensi | Nilai |
|---|---|
| Strategi | random, performance, fairness |
| Alpha (Dirichlet) | 0.1, 0.5, 1.0 |
| Dataset | MNIST, CIFAR-10 |
| Seed | 42, 123, 456 |
| **Total** | **54 eksperimen** |
| Rounds per eksperimen | 20 |
| Klien total | 10 (5 dipilih per round) |
| Local epochs | 3 |
| Learning rate | 0.01 |

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Pre-generate semua partisi data (jalankan sekali)

```bash
PYTHONPATH="$PWD" python experiments/prepare_data.py
```

Output: 180 file `.pt` di `data/partitions/` + 18 `partition_info.json`.

### 3. Smoke test (verifikasi pipeline, 1 round)

```bash
PYTHONPATH="$PWD" python experiments/run_single.py \
  --strategy random --dataset mnist --alpha 0.1 --seed 42 \
  --rounds 1 --trace
```

---

## Menjalankan Eksperimen

### Satu eksperimen

```bash
PYTHONPATH="$PWD" python experiments/run_single.py \
  --strategy random \
  --dataset mnist \
  --alpha 0.1 \
  --seed 42 \
  --rounds 20
```

### Semua 54 eksperimen (incremental — aman jika laptop mati)

```bash
PYTHONPATH="$PWD" python experiments/run_batch.py --skip_existing
```

### Dry run (lihat daftar tanpa eksekusi)

```bash
PYTHONPATH="$PWD" python experiments/run_batch.py --dry_run
```

### Resume dari eksperimen ke-10

```bash
PYTHONPATH="$PWD" python experiments/run_batch.py --start_from 10 --skip_existing
```

---

## Strategi Seleksi Klien

### Random (`random`)
Seleksi seragam acak per round. Baseline. Menggunakan default Flower FedAvg.

### Performance-Based (`performance`)
Pilih K klien dengan latency simulasi terendah (exponential distribution, seed-fixed). Analogous to Oort / Power-of-Choice.

### Fairness-Aware (`fairness`)
Probabilistic selection dengan bobot `1/(count+1)` — klien yang jarang dipilih mendapat probabilitas lebih tinggi. Based on Huang et al. (2021).

---

## Metrik Evaluasi

| Kode | Nama | Grup |
|---|---|---|
| A1 | Global Accuracy | A — Global Performance |
| A2 | Rounds to Target | A — Global Performance |
| B1 | Accuracy Variance (σ) | B — Per-Client Fairness |
| B2 | Gini Coefficient | B — Per-Client Fairness |
| B3 | Participation Fairness | B — Per-Client Fairness |
| C1 | Pareto Data | C — Trade-off |
| C2 | Two-Way ANOVA | C — Trade-off |

Target accuracy: MNIST ≥ 85%, CIFAR-10 ≥ 70%.

---

## Struktur Output

```
results/
└── random_mnist_a0.1_s42/
    ├── config.json              # Konfigurasi eksperimen
    ├── metrics_per_round.json   # Metrik tiap round
    ├── final_metrics.json       # 7 metrik akhir
    └── participation_log.json   # Riwayat pemilihan klien
```

### Format `final_metrics.json`

```json
{
  "A1_global_accuracy": 94.23,
  "A2_rounds_to_target": 8,
  "B1_accuracy_variance": 0.0312,
  "B2_gini_coefficient": 0.1823,
  "B3_participation_fairness": 0.0410,
  "target_reached": true,
  "strategy": "random",
  "dataset": "mnist",
  "alpha": 0.1,
  "seed": 42
}
```

---

## Struktur Folder

```
fl_experiment/
├── configs/
│   └── experiment_config.yaml
├── data/
│   ├── raw/                    # Auto-download MNIST & CIFAR-10
│   └── partitions/             # Hasil partisi Dirichlet
│       ├── mnist/
│       │   └── alpha01_seed42/
│       │       ├── client_0.pt ... client_9.pt
│       │       └── partition_info.json
│       └── cifar10/
├── experiments/
│   ├── prepare_data.py         # Pre-generate partisi
│   ├── run_single.py           # 1 eksperimen
│   └── run_batch.py            # 54 eksperimen
├── src/
│   ├── client/fl_client.py
│   ├── data/
│   │   ├── loader.py
│   │   └── partitioner.py
│   ├── metrics/evaluator.py
│   ├── models/
│   │   ├── cifar_cnn.py
│   │   └── mnist_cnn.py
│   ├── strategies/
│   │   ├── fairness_strategy.py
│   │   ├── performance_strategy.py
│   │   └── random_strategy.py
│   └── utils/
│       ├── logger.py
│       └── tracer.py
├── logs/
├── notebooks/
│   └── analysis.ipynb          # (buat manual untuk analisis akhir)
├── results/
├── requirements.txt
└── README.md
```

---

## Reproducibility

- Partisi data dibangkitkan **sekali** dengan seed fixed dan disimpan ke disk.
- Model diinisialisasi dengan seed yang sama sebelum tiap run (`set_all_seeds()`).
- Latency untuk performance strategy dibangkitkan dengan seed yang sama di semua eksperimen.
- `config.json` disimpan di setiap folder hasil.

---

---

## Kompatibilitas Versi Flower

Sistem ini **sudah diuji dan terbukti jalan end-to-end** pada Flower versi lama (1.13.0) maupun terbaru (1.32.0), karena `requirements.txt` tidak mengunci versi spesifik (`flwr[simulation]>=1.13.0`).

Catatan teknis penting: pada Flower versi terbaru, `ClientProxy.cid` yang dilihat oleh strategi server (`client_manager.all()`) berupa node-ID internal (mirip UUID), bukan lagi `"0","1","2"...` sederhana. Karena `PerformanceBasedStrategy` dan `FairnessAwareStrategy` butuh ID numerik stabil untuk mapping latency dan partisi data, kedua strategi ini sudah dimodifikasi untuk membangun **mapping index stabil** (`"0".."9"`) berdasarkan urutan sortir cid setiap round, alih-alih bergantung langsung pada nilai cid mentah dari Flower. Ini membuat sistem tetap benar terlepas dari skema cid internal Flower di versi manapun.

Selain itu, `make_client_fn()` di `src/client/fl_client.py` memanggil `.to_client()` pada `NumPyClient` agar kompatibel dengan flwr ≥1.13 yang mewajibkan instance `Client`, bukan `NumPyClient` langsung.

Jika `pip install -r requirements.txt` masih gagal karena masalah versi `grpcio`/`cryptography` di Python kamu, install Flower tanpa pin ketat:

```bash
pip install "flwr[simulation]"
```

`[simulation]` wajib disertakan — tanpa ini, `fl.simulation.start_simulation()` akan gagal dengan error `Unable to import module 'ray'`.

---



- **Laptop sering mati?** Gunakan `--skip_existing` — eksperimen yang sudah selesai tidak akan diulang.
- **Debug mode?** Tambahkan `--trace` untuk melihat shape tensor dan distribusi data.
- **Cek hasil cepat?** Buka `results/{experiment_id}/final_metrics.json`.
- **PYTHONPATH** harus di-set ke root project agar absolute imports bekerja.
