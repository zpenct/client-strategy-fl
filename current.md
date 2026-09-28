# Warning
- implementasi pada codebase ini dibagi atas 3 yaitu data loader(sekaligus pertisi), running experiment/strategi, dan evaluasi hasil experiment
- Pada kode ini aku mau kamu memperthaanakan proses partisi data/data loadernya, karena aku merasa itu sudah cukup, dan juga datasetnya sudah terbagi (kamu bisa lihat di folder /data)
- yg perlu kamu ubah adalah pada tahap simulasinya (kode strategi seleksi cleunt dan running experimentnya) soalnya pada kode sekarang semua strategynya itu belum berdasarkan pada paper2 yg seharusnya yaitu "FAIRNESS-AWARE CLIENT SELECTION FOR FEDERATED LEARNING" dan "Oort: Efficient Federated Learning via Guided Participant Selection". Selain itu pada tahap evaluasinya juga masih belum sesuai dgn yg ada di proposal jadi tolong bantu sesuaikan juga kode evaluasinya
- pastikan hasil dari expeiment/kode ini bisa menjawab semua pertanayn pada proposal kita 
- setiap ada pembaruan penting tolong berikan updatenya di file ini


# Refactor Log:

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


