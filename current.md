# Warning
- implementasi pada codebase ini dibagi atas 3 yaitu data loader(sekaligus pertisi), running experiment/strategi, dan evaluasi hasil experiment
- Pada kode ini aku mau kamu memperthaanakan proses partisi data/data loadernya, karena aku merasa itu sudah cukup, dan juga datasetnya sudah terbagi (kamu bisa lihat di folder /data)
- yg perlu kamu ubah adalah pada tahap simulasinya (kode strategi seleksi cleunt dan running experimentnya) soalnya pada kode sekarang semua strategynya itu belum berdasarkan pada paper2 yg seharusnya yaitu "FAIRNESS-AWARE CLIENT SELECTION FOR FEDERATED LEARNING" dan "Oort: Efficient Federated Learning via Guided Participant Selection". Selain itu pada tahap evaluasinya juga masih belum sesuai dgn yg ada di proposal jadi tolong bantu sesuaikan juga kode evaluasinya
- pastikan hasil dari expeiment/kode ini bisa menjawab semua pertanayn pada proposal kita 
- setiap ada pembaruan penting tolong berikan updatenya di file ini


# Refactor Log:

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


