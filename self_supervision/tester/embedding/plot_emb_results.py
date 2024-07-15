import os
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plottable import ColumnDefinition, Table
from plottable.cmap import normed_cmap
from plottable.plots import bar
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl
from self_supervision.paths import RESULTS_FOLDER

mpl.rcParams.update(mpl.rcParamsDefault)

if __name__ == '__main__':

    _METRIC_TYPE = "Metric Type"
    _AGGREGATE_SCORE = "Aggregate score"


    df = pd.read_csv(os.path.join(RESULTS_FOLDER, "embedding", "all.csv"), index_col=0)
    df_pca = df.copy()

    df.index = df.index.str.replace(r'(_|)run[\d]*', '', regex=True)
    # df = df[~df.index.str.contains(r'(_|)v[\d]+$', regex=True)]
    df = df.apply(pd.to_numeric, errors='coerce')



    def plot_results_table(df, order=None, show: bool = True, save_dir: Optional[str] = None, suffix='') -> Table:
        num_embeds = len(df)-1
        cmap_fn = lambda col_data: normed_cmap(col_data, cmap=matplotlib.cm.PRGn, num_stds=2.5)
        # df = self.get_results(min_max_scale=min_max_scale)
        # Do not want to plot what kind of metric it is
        plot_df = df.drop(_METRIC_TYPE, axis=0)
        # Sort by total score
        if order:
            plot_df = plot_df.loc[order].astype(np.float64)
        else:
            plot_df = plot_df.sort_values(by="Total", ascending=False).astype(np.float64)
        plot_df["Method"] = plot_df.index
        # Split columns by metric type, using df as it doesn't have the new method col
        score_cols = df.columns[df.loc[_METRIC_TYPE] == _AGGREGATE_SCORE]
        other_cols = df.columns[df.loc[_METRIC_TYPE] != _AGGREGATE_SCORE]
        column_definitions = [
            ColumnDefinition("Method", width=1.5, textprops={"ha": "left", "weight": "bold"}),
        ]
        # Circles for the metric values
        column_definitions += [
            ColumnDefinition(
                col,
                title=col.replace(" ", "\n", 1),
                width=1,
                textprops={
                    "ha": "center",
                    "bbox": {"boxstyle": "circle", "pad": 0.25},
                },
                cmap=cmap_fn(plot_df[col]),
                group=df.loc[_METRIC_TYPE, col],
                formatter="{:.2f}",
            )
            for i, col in enumerate(other_cols)
        ]
        # Bars for the aggregate scores
        column_definitions += [
            ColumnDefinition(
                col,
                width=1,
                title=col.replace(" ", "\n", 1),
                plot_fn=bar,
                plot_kw={
                    "cmap": matplotlib.cm.YlGnBu,
                    "plot_bg_bar": False,
                    "annotate": True,
                    "height": 0.9,
                    "formatter": "{:.2f}",
                },
                group=df.loc[_METRIC_TYPE, col],
                border="left" if i == 0 else None,
            )
            for i, col in enumerate(score_cols)
        ]
        # Allow to manipulate text post-hoc (in illustrator)
        with matplotlib.rc_context({"svg.fonttype": "none"}):
            fig, ax = plt.subplots(figsize=(len(df.columns) * 1.25, 3 + 0.3 * num_embeds))
            tab = Table(
                plot_df,
                cell_kw={
                    "linewidth": 0,
                    "edgecolor": "k",
                },
                column_definitions=column_definitions,
                ax=ax,
                row_dividers=True,
                footer_divider=True,
                textprops={"fontsize": 10, "ha": "center"},
                row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
                col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
                column_border_kw={"linewidth": 1, "linestyle": "-"},
                index_col="Method",
            ).autoset_fontcolors(colnames=plot_df.columns)
        if show:
            plt.show()
        if save_dir is not None:
            if suffix:
                save_path = os.path.join(save_dir, f"scib_results_{suffix}.svg")
            else:
                save_path = os.path.join(save_dir, f"scib_results.svg")
            fig.savefig(save_path, facecolor=ax.get_facecolor(), dpi=300)
        return tab

    order = ['R-No-SSL', 'R-Masking', 'R-Masking(GP)', 'R-Masking(GP-TF)', 'R-Contrastive(BYOL)', 'R-Contrastive(BT)']
    comparisons = {'reconstruction_CN_HVG_No_SSL_CN_MLP': 'R-No-SSL', 
                'reconstruction_CN_HVG_SSL_CN_CN_HVG_MLP_50p': 'R-Masking',
                'reconstruction_CN_HVG_SSL_CN_CN_HVG_MLP_gene_program_C8_50p': 'R-Masking(GP)',
                'reconstruction_CN_HVG_SSL_CN_CN_HVG_MLP_gp_to_tf': 'R-Masking(GP-TF)',
                'reconstruction_CN_HVG_SSL_CN_CN_HVG_MLP_BYOL_Gaussian_0.001': 'R-Contrastive(BYOL)',
                'reconstruction_CN_HVG_SSL_CN_contrastive_CN_HVG_MLP_bt_Gaussian_0_001': 'R-Contrastive(BT)'
                }

    order = ['C-No-SSL', 'C-Masking', 'C-Masking(GP)', 'C-Masking(GP-TF)', 'C-Contrastive(BYOL)', 'C-Contrastive(BT)']
    comparisons = {
                'classification_No_SSL': 'C-No-SSL', 
                'classification_SSL_CN_MLP_50p': 'C-Masking',
                'classification_SSL_CN_MLP_gene_program_C8_25p': 'C-Masking(GP)',
                'classification_SSL_CN_MLP_gp_to_tf': 'C-Masking(GP-TF)',
                'classification_SSL_MLP_BYOL_Gaussian_0_001': 'C-Contrastive(BYOL)',
                'classification_SSL_contrastive_MLP_bt_Gaussian_0_01': 'C-Contrastive(BT)'
                }

    order = ['C-No-SSL', 'C-Masking', 'R-No-SSL', 'R-Masking', 'R-Masking(GP-TF)']
    comparisons = {'classification_No_SSL_CN_MLPHVG': 'C-No-SSL', 
                'reconstruction_CN_HVG_No_SSL_CN_MLP': 'R-No-SSL', 
                'reconstruction_CN_HVG_SSL_CN_CN_HVG_MLP_50p': 'R-Masking', 
                'classification_SSL_CN_HVG_CN_HVG_MLP_50p': 'C-Masking',
                'reconstruction_CN_HVG_SSL_CN_CN_HVG_MLP_gp_to_tf': 'R-Masking(GP-TF)'
                }



    df_temp = df.loc[list(comparisons.keys())].copy()
    df_temp.index = [comparisons[key] for key in df_temp.index]


    ######################## Scib table ########################
    # df_temp = df_temp.reset_index().groupby('index').mean()
    df_temp = df_temp.reset_index().sort_values(by='Total', ascending=False).drop_duplicates(subset='index', keep='first').set_index('index')

    df_temp = df_temp.applymap(str)
    plot_results_table(pd.concat([df_temp, df_pca.loc['Metric Type'].to_frame().T], axis=0), order=order,
                    save_dir=os.path.join(RESULTS_FOLDER, "embedding"), suffix='mean')


    ######################## Box plots ########################
    sns.set_theme()

    df_1 = df_temp.reset_index()[['index','Batch correction']].copy().rename(columns={'Batch correction': 'score'})
    df_2 = df_temp.reset_index()[['index','Bio conservation']].copy().rename(columns={'Bio conservation': 'score'})
    df_1['score_type']='Batch correction'
    df_2['score_type']='Bio conservation'
    df_total = pd.concat([df_1, df_2], axis=0, ignore_index=True).reset_index()

    plt.figure(figsize=(15,8))

    ############ Total ############
    plt.clf()
    s = sns.boxplot(data=df_temp.reset_index(), x="index", y="Total", order = order)
    s.yaxis.grid(True)
    plt.title('SCIB metrics ("C-": cell-type classification task, "R-": gene expression reconstruction task)')
    plt.xlabel("Pretraining method")
    plt.ylabel("Scib score total")
    s.get_figure().savefig(os.path.join(RESULTS_FOLDER, "embedding", "box_plot_overall.png"))

    ############ hue ############
    plt.clf()
    s = sns.boxplot(data=df_total,x="index",y="score", hue="score_type", order = order)
    s.yaxis.grid(True)
    plt.title('SCIB metrics ("C-": cell-type classification task, "R-": gene expression reconstruction task)')
    plt.xlabel("Pretraining method")
    plt.ylabel("Scib scores")
    s.get_figure().savefig(os.path.join(RESULTS_FOLDER, "embedding", "box_plot_separated.png"))
