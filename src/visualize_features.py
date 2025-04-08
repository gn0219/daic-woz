import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

plt.style.use("ggplot")

def bar_chart(df, column, title=None, option='basic', hue=None, save_path = None):
    """
    Plots a bar chart based on the specified option.

    Parameters:
    - df (DataFrame): The pandas DataFrame containing the data.
    - column (str): The column to plot.
    - title (str): The title of the chart.
    - option (str): The type of bar chart ('basic', 'grouped', 'stacked').
    - hue (str): The column to use for grouping (only for 'grouped' or 'stacked' options).
    """
    if title is None:
        title = f"Distribtuion of {column}"

    # If hue is given, ensure consistent categorical ordering (alphabetical)
    if hue is not None:
        hue_order = sorted(df[hue].dropna().unique())  # e.g., ['F', 'M']
        df[hue] = pd.Categorical(df[hue], categories=hue_order, ordered=True)

    if option == 'basic':
        plt.figure(figsize=(5, 4))
        sns.countplot(data=df, x=column, color='slateblue')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel(column, fontsize=12)
        plt.ylabel("Count", fontsize=12)

    elif option == 'stacked':
        plt.figure(figsize=(5, 4))
        if hue is None:
            raise ValueError("For 'stacked' option, 'hue' must be specified.")
        grouped_data = df.groupby([column, hue]).size().unstack(fill_value=0)
        # Reorder columns for consistency
        grouped_data = grouped_data[hue_order]
        grouped_data.plot(kind='bar', stacked=True, figsize=(5, 4), color=['salmon', 'skyblue'])
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel(column, fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.legend(title=hue, fontsize=10)

    elif option == 'grouped':
        plt.figure(figsize=(8, 4))
        if hue is None:
            raise ValueError("For 'grouped' option, 'hue' must be specified.")
        sns.countplot(data=df, x=column, hue=hue, hue_order=hue_order, palette={'F': 'salmon', 'M': 'skyblue'})
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel(column, fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.legend(title=hue, fontsize=10)

    else:
        raise ValueError("Invalid option. Choose from 'basic', 'grouped', or 'stacked'.")
    
    plt.grid(axis='y', linestyle='--', color='gray', alpha=0.2)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('gray')
        spine.set_linewidth(0.5)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(fontsize=10, rotation=0)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(f'viz/eda/{title.replace(":", "-")}.png', dpi=300, bbox_inches='tight', transparent=False)
    plt.show()

def print_stats(df, group_vars, value_col):
    print("=== Summary Statistics ===")
    
    if group_vars is None:
        mean = df[value_col].mean()
        std = df[value_col].std()
        print(f"Overall: Mean = {mean:.2f}, SD = {std:.2f}")
    else:
        grouped = df.groupby(group_vars)[value_col]
        for group, vals in grouped:
            mean = vals.mean()
            std = vals.std()
            group_str = " - ".join([f"{g}" for g in group]) if isinstance(group, tuple) else str(group)
            print(f"{group_str}: Mean = {mean:.2f}, SD = {std:.2f}")
    
    print("==========================")

def print_mannwhitney(df, group_col, value_col):
    """
    Prints Mann-Whitney U test results for two groups in a DataFrame.
    """
    from scipy.stats import mannwhitneyu

    if df[group_col].nunique() != 2:
        raise ValueError("Mann-Whitney U test requires exactly two groups.")

    group1 = df[df[group_col] == df[group_col].unique()[0]][value_col]
    group2 = df[df[group_col] == df[group_col].unique()[1]][value_col]

    stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
    
    print(f"Mann-Whitney U test: U-statistic = {stat:.2f}, p-value = {p_value:.4f}")

def plot_histogram(df, column, title, bins=30, color='slateblue', kde=True, hue=None, palette='husl', print_summary=True, save_path = None):
    """
    Plots a histogram for a specified column, optionally grouped by hue.
    Also prints summary statistics if print_summary is True.
    """
    plt.figure(figsize=(5, 4))

    if hue and hue in df.columns:
        sns.histplot(data=df, x=column, hue=hue, bins=bins, kde=kde,
                     palette=palette, element="step", stat="density", common_norm=False)
        if print_summary:
            print_stats(df, group_vars=hue, value_col=column)
    else:
        sns.histplot(df[column], kde=kde, bins=bins, color=color)
        if print_summary:
            print_stats(df, group_vars=None, value_col=column)

    plt.grid(axis='y', linestyle='--', color='gray', alpha=0.2)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('gray')
        spine.set_linewidth(0.5)

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(column, fontsize=12)
    plt.ylabel('Density' if kde else 'Frequency', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_path:
        plt.savefig(f'viz/eda/{title.replace(":", "-")}.png', dpi=300, bbox_inches='tight', transparent=False)

    plt.show()



def plot_numeric(data, columns, title, hue=None, category_col=None, palette="husl", option='box', print_summary=True, save_path = None):
    """
    Plots a boxplot or violin plot for numeric data with optional grouping.
    Also prints mean and standard deviation per group.
    """
    sns_func = sns.boxplot if option == 'box' else sns.violinplot

    # Case 1: category + hue (e.g., x-axis group and color group)
    if category_col and hue:
        if len(columns) != 1:
            raise ValueError("Only one column should be specified when using both 'category_col' and 'hue'.")
        col = columns[0]
        plt.figure(figsize=(10, 4))

        sns_func(data=data, x=category_col, y=col, hue=hue, palette=palette)

        if print_summary:
            print_stats(data[[category_col, hue, col]].dropna(), [category_col, hue], col)

        plt.xlabel(category_col)
        plt.ylabel(col)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(title=hue, fontsize=10)
        plt.xticks(rotation=0)
        plt.grid(True)

    # Case 2: category only (e.g., multiple group labels on x-axis)
    elif category_col:
        if len(columns) != 1:
            raise ValueError("Only one column should be specified when using 'category_col'.")
        col = columns[0]
        melted_data = data[[category_col, col]].copy()
        melted_data = melted_data.rename(columns={col: "Value", category_col: "Group"})

        if print_summary:
            print_stats(melted_data.dropna(), ["Group"], "Value")
            print_mannwhitney(melted_data, "Group", "Value")

        plt.figure(figsize=(6, 4))
        sns_func(data=melted_data, x="Group", y="Value", palette=palette)

        plt.xlabel(category_col)
        plt.ylabel(col)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xticks(rotation=0)
        plt.grid(True)

    # Case 3: hue only (e.g., Feature vs Value, colored by hue)
    elif hue:
        melted_data = data.melt(id_vars=hue, value_vars=columns, var_name="Feature", value_name="Value")
        
        if print_summary:
            print_stats(melted_data.dropna(), ["Feature", hue], "Value")

        plt.figure(figsize=(10, 4))
        sns_func(data=melted_data, x="Feature", y="Value", hue=hue, palette=palette)

        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(title=hue, fontsize=10)

    # Case 4: plain plot, no groupings
    else:
        
        if print_summary:
            print_stats(data.melt(value_vars=columns, var_name="Feature", value_name="Value").dropna(),
                        ["Feature"], "Value")

        plt.figure(figsize=(6, 4))
        sns_func(data=data[columns], palette=palette)
        plt.title(title, fontsize=14, fontweight='bold')

    plt.grid(axis='y', linestyle='--', color='gray', alpha=0.2)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('gray')
        spine.set_linewidth(0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(f'viz/eda/{title.replace(":", "-")}.png', dpi=300, bbox_inches='tight', transparent=False)

    plt.show()
