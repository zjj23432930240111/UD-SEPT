# Copyright 2024, LLM-PySC2 Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License at
# You may obtain a License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
UAAF Results Visualization Tool.

Generates SCI paper-quality figures from UAAF experiment data.

Figures produced:
1. win_rate_comparison.png - Grouped bar chart: 3 strategies × tasks × levels
2. entropy_distribution.png - Box plot: entropy by strategy, colored by result
3. win_entropy_correlation.png - Scatter + logistic regression: entropy vs win
4. fallback_analysis.png - Bar chart: fallback ratio per task × strategy
5. strategy_comparison_heatmap.png - Heatmap: win rate matrix

Usage:
  python -m llm_pysc2.bin.visualize_uaaf_results --results_dir uaaf_results --output_dir uaaf_figures
"""

import argparse
import json
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})

# Chinese stock convention colors (red=gain, green=loss) adapted for paper
STRATEGY_COLORS = {
    'raw': '#4C72B0',      # Blue - baseline
    'sc': '#DD8452',       # Orange - Self-Consistency
    'uaaf': '#C44E52',     # Red - our method (highlighted)
}


class UAAFVisualizer:
    """Visualize UAAF experiment results for SCI paper."""

    def __init__(self, results_dir: str, output_dir: str):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df = self._load_data()

    def _load_data(self):
        """Load all game-level summary JSON files into a pandas DataFrame."""
        summaries = []
        for f in self.results_dir.glob("*_summary.json"):
            try:
                with open(f, 'r', encoding='utf-8') as fp:
                    data = json.load(fp)
                    summaries.append(data)
            except Exception as e:
                print(f"Warning: Failed to load {f}: {e}")
        df = pd.DataFrame(summaries)
        if df.empty:
            print("Warning: No data found! Check results_dir path.")
        else:
            # Convert result to binary win indicator
            df['win'] = (df['result'] == 'win').astype(int)
            # Create task-level label
            df['task_label'] = df.apply(
                lambda r: f"Task {int(r['task'])}-L{int(r['level'])}", axis=1
            )
            print(f"Loaded {len(df)} game summaries.")
            print(f"  Tasks: {sorted(df['task'].unique())}")
            print(f"  Levels: {sorted(df['level'].unique())}")
            print(f"  Strategies: {sorted(df['strategy'].unique())}")
            print(f"  Results distribution: {df['result'].value_counts().to_dict()}")
        return df

    def plot_win_rate_comparison(self):
        """Figure 1: Grouped bar chart showing win rate for each task×level×strategy."""
        if self.df.empty:
            return

        # Compute win rate per configuration
        grouped = self.df.groupby(['task', 'level', 'strategy'])['win'].mean().reset_index()
        grouped['win_rate'] = grouped['win'] * 100
        grouped['task_label'] = grouped.apply(
            lambda r: f"T{int(r['task'])}-L{int(r['level'])}", axis=1
        )

        fig, ax = plt.subplots(figsize=(12, 6))

        # Sort task_labels for consistent ordering
        task_labels = sorted(grouped['task_label'].unique())
        x_positions = np.arange(len(task_labels))
        bar_width = 0.25

        for i, strategy in enumerate(['raw', 'sc', 'uaaf']):
            strategy_data = grouped[grouped['strategy'] == strategy]
            rates = []
            for tl in task_labels:
                row = strategy_data[strategy_data['task_label'] == tl]
                rates.append(row['win_rate'].values[0] if len(row) > 0 else 0)

            bars = ax.bar(x_positions + i * bar_width, rates, bar_width,
                          label=strategy.upper() if strategy == 'uaaf' else
                          ('Raw (1-call)' if strategy == 'raw' else 'Self-Consistency (3-call)'),
                          color=STRATEGY_COLORS[strategy],
                          edgecolor='white', linewidth=0.5)

            # Add value labels on bars
            for bar, rate in zip(bars, rates):
                if rate > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                            f'{rate:.1f}', ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Task Configuration')
        ax.set_ylabel('Win Rate (%)')
        ax.set_title('Win Rate Comparison: Raw vs Self-Consistency vs UAAF')
        ax.set_xticks(x_positions + bar_width)
        ax.set_xticklabels(task_labels)
        ax.legend(loc='upper right')
        ax.set_ylim(0, max(grouped['win_rate'].max() + 15, 100))
        ax.grid(axis='y', alpha=0.3)

        fig.savefig(self.output_dir / 'win_rate_comparison.png', dpi=300)
        plt.close(fig)
        print(f"  Saved: win_rate_comparison.png")

    def plot_entropy_distribution(self):
        """Figure 2: Box plot of entropy distribution by strategy, colored by win/lose."""
        if self.df.empty:
            return

        # Filter to SC and UAAF strategies (they have entropy data)
        subset = self.df[self.df['strategy'].isin(['sc', 'uaaf'])].copy()
        if subset.empty:
            print("  No SC/UAAF data for entropy plot.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: Entropy distribution by strategy
        sns.boxplot(data=subset, x='strategy', y='avg_entropy',
                    palette=STRATEGY_COLORS, ax=axes[0])
        axes[0].set_xlabel('Strategy')
        axes[0].set_ylabel('Average Entropy per Game')
        axes[0].set_title('Entropy Distribution by Strategy')
        axes[0].set_xticklabels(['Self-Consistency', '+UAAF'])

        # Right: Entropy by strategy, colored by result
        subset['result_label'] = subset['result'].map({'win': 'Win', 'lose': 'Lose', 'tie': 'Tie'})
        sns.boxplot(data=subset, x='strategy', y='avg_entropy',
                    hue='result_label', ax=axes[1],
                    palette={'Win': '#55A868', 'Lose': '#C44E52', 'Tie': '#8172B2'})
        axes[1].set_xlabel('Strategy')
        axes[1].set_ylabel('Average Entropy per Game')
        axes[1].set_title('Entropy Distribution: Win vs Lose')
        axes[1].set_xticklabels(['Self-Consistency', '+UAAF'])
        axes[1].legend(title='Result')

        plt.tight_layout()
        fig.savefig(self.output_dir / 'entropy_distribution.png', dpi=300)
        plt.close(fig)
        print(f"  Saved: entropy_distribution.png")

    def plot_win_entropy_correlation(self):
        """Figure 3: Scatter plot with logistic regression: avg_entropy vs win probability."""
        if self.df.empty:
            return

        subset = self.df[self.df['strategy'] == 'uaaf'].copy()
        if subset.empty:
            print("  No UAAF data for correlation plot.")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        # Scatter plot: each game as a point
        jitter_y = subset['win'] + np.random.normal(0, 0.03, len(subset))
        ax.scatter(subset['avg_entropy'], jitter_y,
                   alpha=0.4, s=20, color=STRATEGY_COLORS['uaaf'], label='Individual games')

        # Binned win rate curve
        if len(subset) > 20:
            bins = np.linspace(subset['avg_entropy'].min(), subset['avg_entropy'].max(), 10)
            subset['entropy_bin'] = pd.cut(subset['avg_entropy'], bins)
            binned = subset.groupby('entropy_bin', observed=True)['win'].mean().reset_index()
            binned['entropy_mid'] = binned['entropy_bin'].apply(
                lambda x: x.mid if pd.notna(x) else None
            ).dropna()
            if len(binned) > 2:
                ax.plot(binned['entropy_mid'], binned['win'] * 100,
                        'o-', color='#55A868', linewidth=2, markersize=8,
                        label='Binned win rate')

        ax.set_xlabel('Average Shannon Entropy')
        ax.set_ylabel('Win Rate (%)')
        ax.set_title('Win Rate vs. Decision Entropy (UAAF Strategy)')
        ax.legend()
        ax.grid(alpha=0.3)

        fig.savefig(self.output_dir / 'win_entropy_correlation.png', dpi=300)
        plt.close(fig)
        print(f"  Saved: win_entropy_correlation.png")

    def plot_fallback_analysis(self):
        """Figure 4: Bar chart showing fallback ratio per task × strategy."""
        if self.df.empty:
            return

        subset = self.df[self.df['strategy'] == 'uaaf'].copy()
        if subset.empty:
            print("  No UAAF data for fallback analysis.")
            return

        grouped = subset.groupby(['task', 'level'])['fallback_ratio'].mean().reset_index()
        grouped['task_label'] = grouped.apply(
            lambda r: f"T{int(r['task'])}-L{int(r['level'])}", axis=1
        )
        grouped['fallback_pct'] = grouped['fallback_ratio'] * 100

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(range(len(grouped)), grouped['fallback_pct'],
                      color=STRATEGY_COLORS['uaaf'], edgecolor='white')

        for bar, pct in zip(bars, grouped['fallback_pct']):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)

        ax.set_xlabel('Task Configuration')
        ax.set_ylabel('Fallback Action Ratio (%)')
        ax.set_title('UAAF Fallback Action Trigger Rate by Task')
        ax.set_xticks(range(len(grouped)))
        ax.set_xticklabels(grouped['task_label'])
        ax.set_ylim(0, max(grouped['fallback_pct'].max() + 10, 30))
        ax.grid(axis='y', alpha=0.3)

        fig.savefig(self.output_dir / 'fallback_analysis.png', dpi=300)
        plt.close(fig)
        print(f"  Saved: fallback_analysis.png")

    def plot_strategy_comparison_heatmap(self):
        """Figure 5: Heatmap of win rates (task × strategy)."""
        if self.df.empty:
            return

        # Pivot table: task_label × strategy → win rate
        pivot = self.df.groupby(['task_label', 'strategy'])['win'].mean().reset_index()
        pivot['win_rate'] = pivot['win'] * 100
        pivot_table = pivot.pivot(index='task_label', columns='strategy', values='win_rate')

        # Ensure column order
        for col in ['raw', 'sc', 'uaaf']:
            if col not in pivot_table.columns:
                pivot_table[col] = 0
        pivot_table = pivot_table[['raw', 'sc', 'uaaf']]

        # Rename columns for display
        pivot_table.columns = ['Raw (1-call)', 'Self-Consistency (3-call)', '+UAAF (3-4 call)']

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='YlOrRd',
                    linewidths=0.5, ax=ax, vmin=0,
                    cbar_kws={'label': 'Win Rate (%)'})
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Task Configuration')
        ax.set_title('Win Rate Heatmap: Strategy × Task')

        fig.savefig(self.output_dir / 'strategy_comparison_heatmap.png', dpi=300)
        plt.close(fig)
        print(f"  Saved: strategy_comparison_heatmap.png")

    def generate_ablation_table(self):
        """Generate the PromptEvo ablation table with UAAF as additional row.

        This produces a formatted table image suitable for insertion into the paper
        alongside existing Tables 4-4, 4-5, 4-6.
        """
        if self.df.empty:
            return

        # Compute win rate per task × level × strategy
        grouped = self.df.groupby(['task', 'level', 'strategy'])['win'].mean().reset_index()
        grouped['win_rate'] = grouped['win'] * 100

        # Create table data matching existing ablation format
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.axis('off')

        # Build table rows
        table_data = []
        row_labels = []

        # Group by task × level
        configs = sorted(grouped.apply(
            lambda r: (int(r['task']), int(r['level'])), axis=1
        ).unique())

        for (task, level) in configs:
            row_data = []
            task_data = grouped[(grouped['task'] == task) & (grouped['level'] == level)]
            for strategy in ['raw', 'sc', 'uaaf']:
                s_data = task_data[task_data['strategy'] == strategy]
                if len(s_data) > 0:
                    row_data.append(f"{s_data['win_rate'].values[0]:.1f}%")
                else:
                    row_data.append("-")
            table_data.append(row_data)
            row_labels.append(f"Task {task}-Level {level}")

        # Add average row
        avg_row = []
        for strategy in ['raw', 'sc', 'uaaf']:
            s_data = grouped[grouped['strategy'] == strategy]
            avg_row.append(f"{s_data['win_rate'].mean():.1f}%")
        table_data.append(avg_row)
        row_labels.append("Average")

        col_labels = ['Raw (1-call)', 'Self-Consistency\n(3-call)', '+UAAF\n(3-4 call)']

        table = ax.table(
            cellText=table_data,
            rowLabels=row_labels,
            colLabels=col_labels,
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 2.0)

        # Highlight UAAF column
        for i in range(len(table_data)):
            cell = table[i + 1, 2]  # UAAF column (index 2)
            cell.set_facecolor('#FFF2E6')
            cell.set_text_props(weight='bold')

        # Highlight average row
        avg_row_idx = len(table_data)
        for j in range(3):
            cell = table[avg_row_idx, j]
            cell.set_facecolor('#E6F3FF')

        ax.set_title('UAAF Ablation Results (DeepSeek-v4-pro)',
                     fontsize=13, pad=20, weight='bold')

        plt.tight_layout()
        fig.savefig(self.output_dir / 'promptevo_ablation_table.png', dpi=300)
        plt.close(fig)
        print(f"  Saved: promptevo_ablation_table.png")

    def run_all(self):
        """Generate all visualizations."""
        print(f"\nGenerating UAAF visualizations...")
        print(f"  Results dir: {self.results_dir}")
        print(f"  Output dir: {self.output_dir}")

        self.plot_win_rate_comparison()
        self.plot_entropy_distribution()
        self.plot_win_entropy_correlation()
        self.plot_fallback_analysis()
        self.plot_strategy_comparison_heatmap()
        self.generate_ablation_table()

        print(f"\nAll visualizations saved to {self.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize UAAF experiment results")
    parser.add_argument('--results_dir', default='uaaf_results',
                        help='Directory containing UAAF result JSON files')
    parser.add_argument('--output_dir', default='uaaf_figures',
                        help='Directory to save visualization figures')
    args = parser.parse_args()

    viz = UAAFVisualizer(args.results_dir, args.output_dir)
    viz.run_all()
