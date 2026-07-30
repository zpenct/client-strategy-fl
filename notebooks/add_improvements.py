import nbformat
import sys

def main():
    notebook_path = 'notebooks/analysis.ipynb'
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # 1. BAGIAN ANOVA
    anova_md = nbformat.v4.new_markdown_cell("## 1. BAGIAN ANOVA (Two-Way ANOVA: Strategy × Alpha)")
    anova_code = nbformat.v4.new_code_cell('''\
import pingouin as pg
from scipy.stats import shapiro
import warnings
import pandas as pd
from IPython.display import display, HTML

metrics = [
    ('A1_global_accuracy', 'A1 (Accuracy)'),
    ('A2_rounds_to_target', 'A2 (Rounds)'),
    ('B1_accuracy_variance', 'B1 (Variance)'),
    ('B2_gini_coefficient', 'B2 (Gini)'),
    ('B3_participation_fairness', 'B3 (Fairness)')
]
datasets = ['mnist', 'cifar10']

for dataset in datasets:
    print(f"\\n{'='*60}\\nHASIL ANOVA — {dataset.upper()} Dataset\\n{'='*60}")
    
    # Kumpulkan hasil untuk ditampilkan dalam tabel
    table_data = []
    
    for metric_col, metric_name in metrics:
        sub_df = df[df['dataset'] == dataset].copy()
        
        # Drop NaN (terutama untuk A2 jika target tidak tercapai)
        sub_df = sub_df.dropna(subset=[metric_col])
        
        if len(sub_df) < 3:
            table_data.append({
                "Metrik (A1-B3)": metric_name,
                "Main Effect": "Not enough data",
                "F-stat": "-",
                "p-value": "-"
            })
            continue
            
        # Uji Normalitas dengan Shapiro-Wilk
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                stat, p_shap = shapiro(sub_df[metric_col])
                is_normal = p_shap > 0.05
            except:
                is_normal = False
                
        # Two-Way ANOVA
        try:
            aov = pg.anova(data=sub_df, dv=metric_col, between=['strategy', 'alpha'])
            
            for idx, row in aov.iterrows():
                if row['Source'] == 'strategy':
                    effect = 'Strategi'
                elif row['Source'] == 'alpha':
                    effect = 'Alpha'
                elif row['Source'] == 'strategy * alpha':
                    effect = 'Strategi×α'
                else:
                    continue
                    
                table_data.append({
                    "Metrik (A1-B3)": metric_name if effect == 'Strategi' else "",
                    "Main Effect": effect,
                    "F-stat": f"{row['F']:.3f}",
                    "p-value": f"{row['p-unc']:.4f}",
                    "Normalitas": "Normal" if is_normal else "Tidak Normal" if effect == 'Strategi' else ""
                })
        except Exception as e:
            table_data.append({
                "Metrik (A1-B3)": metric_name,
                "Main Effect": f"Error: {str(e)}",
                "F-stat": "-",
                "p-value": "-"
            })
            
    res_df = pd.DataFrame(table_data)
    display(HTML(res_df.to_html(index=False)))
    
    # Interpretasi Sederhana
    print("\\nInterpretasi Sederhana:")
    print("1. Jika p-value < 0.05, maka faktor tersebut memiliki pengaruh yang signifikan secara statistik.")
    print("2. Jika interaksi Strategi×α signifikan, berarti efektivitas strategi seleksi bergantung pada tingkat keparahan label skew.")
    print("3. Uji Normalitas (Shapiro-Wilk) disertakan sebagai referensi asumsi ANOVA.")
''')

    # 2. BAGIAN PARETO FRONTIER
    pareto_md = nbformat.v4.new_markdown_cell("## 2. BAGIAN PARETO FRONTIER")
    pareto_code = nbformat.v4.new_code_cell('''\
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def is_pareto_efficient(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True
    return is_efficient

datasets = ['mnist', 'cifar10']
tradeoffs = [
    ('A1_global_accuracy', 'B1_accuracy_variance', 'A1 (Accuracy)', 'B1 (Variance)'),
    ('A1_global_accuracy', 'B3_participation_fairness', 'A1 (Accuracy)', 'B3 (Fairness)')
]

colors = {'random': 'blue', 'performance': 'orange', 'fairness': 'green'}
alphas = [0.1, 0.5, 1.0]

for dataset in datasets:
    for x_col, y_col, x_label, y_label in tradeoffs:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Trade-off {x_label} vs {y_label} - {dataset.upper()}', fontsize=16)
        
        for idx, alpha in enumerate(alphas):
            ax = axes[idx]
            sub = df[(df['dataset'] == dataset) & (df['alpha'] == alpha)].copy()
            if sub.empty:
                continue
                
            agg = sub.groupby('strategy')[[x_col, y_col]].mean().reset_index()
            
            # Scatter Plot
            for _, row in agg.iterrows():
                ax.scatter(row[x_col], row[y_col], color=colors.get(row['strategy'], 'black'), 
                           label=row['strategy'], s=150, zorder=5)
                
            # Hitung Pareto
            # x_col: Akurasi (Maximize -> negatifkan untuk minimisasi)
            # y_col: Variance/Fairness (Minimize)
            costs = np.column_stack((-agg[x_col], agg[y_col]))
            pareto_mask = is_pareto_efficient(costs)
            pareto_pts = agg[pareto_mask].sort_values(by=x_col)
            
            # Gambar Garis Pareto
            if len(pareto_pts) > 1:
                ax.plot(pareto_pts[x_col], pareto_pts[y_col], 
                        color='gray', linestyle='dashed', linewidth=2, 
                        label='Pareto Frontier', zorder=1)
                
            # Highlight Pareto
            for _, row in pareto_pts.iterrows():
                ax.scatter(row[x_col], row[y_col], color='none', edgecolor='black', 
                           linewidth=2, s=200, zorder=6)
                
            ax.set_title(f'Alpha = {alpha}')
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            if idx == 0:
                ax.legend()
                
        plt.tight_layout()
        plt.show()
''')

    # 3. BAGIAN RINGKASAN
    summary_md = nbformat.v4.new_markdown_cell("## 3. BAGIAN RINGKASAN NUMERIK")
    summary_code = nbformat.v4.new_code_cell('''\
def get_pareto_rank(costs):
    """
    Calculate Non-dominated sorting rank
    """
    ranks = np.zeros(costs.shape[0], dtype=int)
    current_rank = 1
    remaining = np.ones(costs.shape[0], dtype=bool)
    
    while np.any(remaining):
        current_costs = costs[remaining]
        eff = is_pareto_efficient(current_costs)
        # indices di remaining array
        eff_indices = np.where(remaining)[0][eff]
        ranks[eff_indices] = current_rank
        remaining[eff_indices] = False
        current_rank += 1
    return ranks

for dataset in datasets:
    print(f"\\n{'='*80}\\nTABEL RINGKASAN — {dataset.upper()} DATASET\\n{'='*80}")
    
    summary_data = []
    
    for alpha in alphas:
        sub = df[(df['dataset'] == dataset) & (df['alpha'] == alpha)].copy()
        if sub.empty:
            continue
            
        agg = sub.groupby('strategy').agg({
            'A1_global_accuracy': 'mean',
            'A2_rounds_to_target': 'mean',
            'B1_accuracy_variance': 'mean',
            'B2_gini_coefficient': 'mean',
            'B3_participation_fairness': 'mean'
        }).reset_index()
        
        # Pareto berdasar multi-objective (Max A1, Min B1, Min B2, Min B3)
        costs = np.column_stack((
            -agg['A1_global_accuracy'].values,
            agg['B1_accuracy_variance'].values,
            agg['B2_gini_coefficient'].values,
            agg['B3_participation_fairness'].values
        ))
        
        ranks = get_pareto_rank(costs)
        is_pareto = ranks == 1
        
        for idx, row in agg.iterrows():
            summary_data.append({
                "Alpha": f"α={alpha}",
                "Strategy": row['strategy'].title(),
                "A1 (Acc%)": f"{row['A1_global_accuracy']:.2f}",
                "A2 (Rounds)": f"{row['A2_rounds_to_target']:.1f}" if pd.notna(row['A2_rounds_to_target']) else "NaN",
                "B1 (Var)": f"{row['B1_accuracy_variance']:.4f}",
                "B2 (Gini)": f"{row['B2_gini_coefficient']:.4f}",
                "B3 (Fair)": f"{row['B3_participation_fairness']:.3f}",
                "Pareto?": "Yes" if is_pareto[idx] else "No",
                "Rank": ranks[idx]
            })
            
    summary_df = pd.DataFrame(summary_data)
    display(HTML(summary_df.to_html(index=False)))
    
    print("""
Catatan: 
- A1 = Global Accuracy (%), nilai lebih tinggi lebih baik
- A2 = Rounds to Target Accuracy, nilai lebih rendah lebih baik (NaN jika target tak tercapai)
- B1 = Accuracy Variance (std dev across clients), nilai lebih rendah = fairness lebih baik
- B2 = Gini Coefficient of Accuracy, nilai lebih rendah = fairness lebih baik
- B3 = Participation Fairness (std dev participation count), nilai lebih rendah = fairness lebih baik
- Pareto? = Pareto-optimal dalam trade-off accuracy vs fairness (Rank 1)
- Rank = Peringkat strategi berdasarkan konsep Pareto dominance
    """)
''')

    # Append to notebook
    nb.cells.extend([anova_md, anova_code, pareto_md, pareto_code, summary_md, summary_code])
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
        
    print(f"Successfully added 3 improvement sections to {notebook_path}")

if __name__ == '__main__':
    main()
