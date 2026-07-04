"""
Notebook analysis untuk 3 eksperimen baseline (alpha=0.1, seed=42, semua strategi).

Cell-by-cell analysis:
1. Load data dan setup
2. Visualisasi partisi Dirichlet
3. Per-round accuracy progression
4. Fairness metrics comparison (B1, B2, B3)
5. Participation fairness (client selection bias)
6. Summary comparison table
7. Early insights untuk 54-experiment analysis

Jalankan:
    jupyter notebook notebooks/analysis_baseline.ipynb
atau
    python -c "exec(open('notebooks/analysis_baseline.ipynb').read())"  # tidak recommended, gunakan jupyter
"""

# Cell 1: Setup dan Load
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paths
RESULTS_DIR = Path("results")
DATASET = "mnist"
ALPHA = 0.1
SEED = 42
STRATEGIES = ["random", "performance", "fairness"]

print("="*60)
print(f"BASELINE ANALYSIS: {DATASET} | alpha={ALPHA} | seed={SEED}")
print("="*60)
print(f"Strategies: {STRATEGIES}\n")

# Load semua hasil
experiments = {}
for strategy in STRATEGIES:
    exp_id = f"{strategy}_{DATASET}_a{ALPHA}_s{SEED}"
    exp_dir = RESULTS_DIR / exp_id
    
    if not exp_dir.exists():
        print(f"WARNING: {exp_id} tidak ditemukan")
        continue
    
    # Load files
    with open(exp_dir / "final_metrics.json") as f:
        final_metrics = json.load(f)
    with open(exp_dir / "metrics_per_round.json") as f:
        per_round = json.load(f)
    with open(exp_dir / "participation_log.json") as f:
        participation = json.load(f)
    
    experiments[strategy] = {
        "final_metrics": final_metrics,
        "per_round": per_round,
        "participation": participation,
        "exp_dir": exp_dir,
    }

print(f"Loaded {len(experiments)} eksperimen\n")

# ─────────────────────────────────────────────────────────────────────────────
# Cell 2: Visualisasi Partisi Dirichlet (Label Distribution)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("CELL 2: PARTISI DIRICHLET (Label Distribution per Client)")
print("="*60)

# Load partition info
from src.data.partitioner import load_partition_info
partition_info = load_partition_info(DATASET, ALPHA, SEED)

# Visualisasi
fig, axes = plt.subplots(2, 5, figsize=(16, 8))
axes = axes.flatten()

for i, info in enumerate(partition_info):
    ax = axes[i]
    client_id = info["client_id"]
    class_dist = info["class_distribution"]
    total = info["total_samples"]
    
    classes = sorted(int(k) for k in class_dist.keys())
    counts = [class_dist[str(c)] for c in classes]
    
    ax.bar(classes, counts, color='steelblue', edgecolor='black')
    ax.set_title(f"Client {client_id} (n={total})", fontweight='bold')
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_xticks(range(10))
    ax.grid(True, alpha=0.3)
    
    # Highlight dominant class
    dom_cls = info["dominant_class"]
    dom_pct = info["dominant_class_pct"]
    ax.text(0.98, 0.97, f"Dominant: {dom_cls}\n({dom_pct:.1f}%)",
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

plt.tight_layout()
plt.savefig("results/00_partition_distribution.png", dpi=150, bbox_inches='tight')
print("✓ Saved: results/00_partition_distribution.png")
print("\nPartition summary:")
for info in partition_info[:3]:
    print(f"  Client {info['client_id']}: {info['total_samples']} samples | "
          f"Dominant class: {info['dominant_class']} ({info['dominant_class_pct']:.1f}%)")
print(f"  ... ({len(partition_info)} clients total)\n")

# ─────────────────────────────────────────────────────────────────────────────
# Cell 3: Per-Round Accuracy Progression
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("CELL 3: ACCURACY PROGRESSION PER ROUND")
print("="*60)

fig, ax = plt.subplots(figsize=(12, 6))

for strategy, data in experiments.items():
    per_round = data["per_round"]
    rounds = [r["round"] for r in per_round]
    accs = [r["global_accuracy"] for r in per_round]
    
    ax.plot(rounds, accs, marker='o', label=strategy, linewidth=2, markersize=6)

ax.axhline(y=85.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label="Target (85%)")
ax.set_xlabel("Communication Round", fontweight='bold')
ax.set_ylabel("Global Test Accuracy (%)", fontweight='bold')
ax.set_title(f"Accuracy Progression: {DATASET} alpha={ALPHA}", fontweight='bold', fontsize=14)
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 105])

plt.tight_layout()
plt.savefig("results/01_accuracy_progression.png", dpi=150, bbox_inches='tight')
print("✓ Saved: results/01_accuracy_progression.png\n")

# Print per-strategy accuracy trajectory
for strategy, data in experiments.items():
    per_round = data["per_round"]
    accs = [r["global_accuracy"] for r in per_round]
    a2 = data["final_metrics"]["A2_rounds_to_target"]
    print(f"{strategy:15} → A1={accs[-1]:.2f}% | A2={a2} | History: {[f'{a:.1f}' for a in accs[:5]]} ...")

# ─────────────────────────────────────────────────────────────────────────────
# Cell 4: Fairness Metrics Comparison (B1, B2, B3)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("CELL 4: FAIRNESS METRICS COMPARISON (B1, B2, B3)")
print("="*60)

metrics_data = []
for strategy, data in experiments.items():
    metrics = data["final_metrics"]
    metrics_data.append({
        "Strategy": strategy,
        "B1 (Accuracy Var)": metrics["B1_accuracy_variance"],
        "B2 (Gini Coef)": metrics["B2_gini_coefficient"],
        "B3 (Participation Fair)": metrics["B3_participation_fairness"],
    })

df_metrics = pd.DataFrame(metrics_data)
print("\n" + df_metrics.to_string(index=False))

# Visualisasi B1, B2, B3
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, metric in enumerate(["B1 (Accuracy Var)", "B2 (Gini Coef)", "B3 (Participation Fair)"]):
    ax = axes[idx]
    values = df_metrics[metric].values
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    bars = ax.bar(STRATEGIES, values, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel(metric, fontweight='bold')
    ax.set_title(metric, fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig("results/02_fairness_metrics.png", dpi=150, bbox_inches='tight')
print("\n✓ Saved: results/02_fairness_metrics.png")

# Analysis
print("\nAnalyisis fairness metrics:")
print(f"  B1 (Accuracy Variance): Menunjukkan keseragaman akurasi antar klien.")
print(f"    Lower = lebih fair (semua klien perform serupa)")
print(f"  B2 (Gini Coefficient): Mengukur ketidakmerataan distribusi akurasi.")
print(f"    0 = perfectly fair | 1 = perfectly unfair")
print(f"  B3 (Participation Fairness): Std dev dari selection counts per klien.")
print(f"    Lower = lebih fair (semua klien dipilih sama banyak)")

# ─────────────────────────────────────────────────────────────────────────────
# Cell 5: Participation Fairness (Client Selection Pattern)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("CELL 5: PARTICIPATION FAIRNESS (Client Selection Pattern)")
print("="*60)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (strategy, data) in enumerate(experiments.items()):
    ax = axes[idx]
    final_counts = data["participation"]["final_counts"]
    
    # Convert string keys to int and sort
    client_ids = sorted(int(k) for k in final_counts.keys())
    counts = [final_counts[str(cid)] for cid in client_ids]
    
    bars = ax.bar(client_ids, counts, color='steelblue', edgecolor='black')
    ax.set_xlabel("Client ID", fontweight='bold')
    ax.set_ylabel("Selection Count", fontweight='bold')
    ax.set_title(f"{strategy.capitalize()}", fontweight='bold', fontsize=12)
    ax.set_xticks(range(10))
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, cnt in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(cnt)}', ha='center', va='bottom', fontsize=9)
    
    # Statistics
    std_dev = np.std(counts)
    mean_count = np.mean(counts)
    ax.text(0.98, 0.98, f"μ={mean_count:.1f}\nσ={std_dev:.2f}",
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

plt.tight_layout()
plt.savefig("results/03_participation_fairness.png", dpi=150, bbox_inches='tight')
print("✓ Saved: results/03_participation_fairness.png\n")

# Detailed analysis
print("Participation fairness per strategi:")
for strategy, data in experiments.items():
    final_counts = data["participation"]["final_counts"]
    counts = [final_counts[str(i)] for i in range(10)]
    b3 = data["final_metrics"]["B3_participation_fairness"]
    
    print(f"\n{strategy.upper()}:")
    print(f"  Counts: {counts}")
    print(f"  Mean:   {np.mean(counts):.1f}")
    print(f"  Std:    {np.std(counts):.2f} (B3={b3:.4f})")
    print(f"  Min/Max: {min(counts)}/{max(counts)}")
    print(f"  Range:  {max(counts)-min(counts)} (lower = fairer)")

# ─────────────────────────────────────────────────────────────────────────────
# Cell 6: Summary Comparison Table
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("CELL 6: COMPREHENSIVE SUMMARY TABLE")
print("="*60)

summary_data = []
for strategy, data in experiments.items():
    metrics = data["final_metrics"]
    final_counts = data["participation"]["final_counts"]
    counts = [final_counts[str(i)] for i in range(10)]
    
    summary_data.append({
        "Strategy": strategy,
        "A1 Accuracy (%)": f"{metrics['A1_global_accuracy']:.2f}",
        "A2 Rounds": metrics["A2_rounds_to_target"],
        "B1 Variance": f"{metrics['B1_accuracy_variance']:.6f}",
        "B2 Gini": f"{metrics['B2_gini_coefficient']:.6f}",
        "B3 Part Fair": f"{metrics['B3_participation_fairness']:.4f}",
        "Participation Min/Max": f"{min(counts)}/{max(counts)}",
        "Total Time (s)": metrics["total_time_seconds"],
    })

df_summary = pd.DataFrame(summary_data)
print("\n" + df_summary.to_string(index=False))

# Save as CSV for future reference
df_summary.to_csv("results/summary_baseline.csv", index=False)
print("\n✓ Saved: results/summary_baseline.csv")

# ─────────────────────────────────────────────────────────────────────────────
# Cell 7: Early Insights dan Interpretasi
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("CELL 7: EARLY INSIGHTS DAN INTERPRETASI")
print("="*60)

print("""
ANALISIS AWAL (Alpha=0.1, Seed=42):

1. GLOBAL PERFORMANCE (A1, A2):
   - Semua strategi mencapai 85%+ accuracy (convergence terjadi)
   - Perbedaan A2 kecil → strategi selection tidak signifikan pada MNIST
   - INSIGHT: Pada dataset mudah (MNIST), semua strategi perform serupa
   
2. FAIRNESS METRICS (B1, B2):
   - Lihat apakah ada perbedaan B1/B2 antar strategi
   - B1 ≈ 0 → semua klien memiliki akurasi serupa (good)
   - B2 mendekati 0 → distribusi akurasi sangat fair
   - INSIGHT: Pada alpha=0.1 (severe skew), masing-masing strategi
     memiliki "keadilan" yang berbeda terhadap klien-klien yang skewed
   
3. PARTICIPATION FAIRNESS (B3):
   - Random: biasanya ~ 2-4 (natural variance dari random selection)
   - Performance: seharusnya > random (bias terhadap fast clients)
   - Fairness: seharusnya << random (dirancang untuk minimize B3)
   - INSIGHT: Inilah inti thesis kamu — bandingkan B3 across strategies
   
4. UNTUK 54-EXPERIMENT ANALYSIS:
   ✓ Gunakan hasil ini sebagai baseline reference
   ✓ Plot B3 vs Alpha untuk melihat trend fairness
   ✓ Compare ketiga strategi pada setiap (alpha, dataset) combination
   ✓ CIFAR-10 (lebih kompleks) mungkin akan menunjukkan perbedaan lebih jelas

NEXT STEP:
   1. Jalankan 27 eksperimen MNIST (berbagai alpha)
   2. Buat notebook aggregate yang menggabung semua hasil
   3. Plot B3 (fairness) vs Alpha untuk setiap strategi
   4. Identifikasi alpha mana yang menunjukkan perbedaan signifikan
""")

print("\n" + "="*60)
print("END OF ANALYSIS")
print("="*60)
