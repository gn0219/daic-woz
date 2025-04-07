import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots
import matplotlib as mpl
import pandas as pd
plt.style.use(['science', 'no-latex', 'grid'])
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['axes.facecolor'] = 'white'

def plot_result(result: pd.DataFrame, col: str ='Model', metric: str = 'F1(macro)', sort: bool = False, title: str = None, save_path: str = None) -> None:
    colors = ['#FD8080', '#FEBC3B', '#46EAB3', '#26A0FC', '#94DAFB', '#6D848E'] # green: '#46EAB3', skyblue: '#94DAFB', mint: '#68D4CD'

    # Melt the result dataframe for easier plotting
    result_melted = result.melt(id_vars=col, var_name="Metric", value_name="Score")

    performance = result_melted[result_melted["Metric"] == metric]

    plt.figure(figsize=(6, 3))
    
    if 'Baseline (Majority Voting)' in result[col].values:
        baseline_value = performance[performance[col] == "Baseline (Majority Voting)"]["Score"].values[0]
        performance = performance.iloc[1:]
        print('Baseline found...')
        if sort:
            performance = performance.sort_values(by="Score", ascending=False)
        plt.axvline(x=baseline_value, color="#d7191c", linestyle='-.', label=f'Baseline ({baseline_value:.2f})')
        legend = plt.legend(frameon=True)
        legend.get_frame().set_edgecolor('gray')      
        legend.get_frame().set_linewidth(0.5)          
        legend.get_frame().set_facecolor('#f4f4f4')    
        legend.get_frame().set_alpha(0.5)

    sns.barplot(data=performance, y=col, x="Score", orient="h", palette=colors[:len(performance)])

    # Add edge color
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(0.5)
    if title is None:
        title = f"{col} Performance - {metric}"
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(metric, fontsize=12)
    plt.ylabel(col, fontsize=12)
    plt.xlim(0, 1)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)

    plt.show()

def plot_cm(cm, class_names=['ND', 'D'], 
            title="Confusion Matrix", cmap='YlGnBu', save_path=None):
    plt.figure(figsize=(4, 4)) 
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                     xticklabels=class_names, yticklabels=class_names,
                     linewidths=0.7, linecolor='black', square=True, cbar=False,
                     annot_kws={"size": 16, "weight": "bold"}) 

    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel("Predicted Label", fontsize=15, weight='bold', labelpad=10)
    ax.set_ylabel("True Label", fontsize=15, weight='bold', labelpad=10)
    ax.tick_params(axis='both', labelsize=13)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)

    plt.show()