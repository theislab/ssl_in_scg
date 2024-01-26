import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)


split = 'test'
RESULT_PATH: str = '/lustre/groups/ml01/workspace/mojtaba.bahrami/ssl_results'
df = pd.read_csv(os.path.join(RESULT_PATH, 'embedding', f'{split}_scib_scores.csv', index_col=0))

df = df.apply(pd.to_numeric, errors='coerce')
df.index = df.index.str.replace(r'(-|)[\d]+', '', regex=True)


order = ['PCA', 'Supervised', 'SSL', 'scVI', 'SSL-Shallow']
comparisons = {'PCA': 'PCA',
               'Supervised': 'Supervised',
               'SSL': 'SSL',
               'scVI': 'scVI',
               'SSL-Shallow': 'SSL-Shallow',
               }


df_temp = df.loc[list(comparisons.keys())].copy()
df_temp.index = [comparisons[key] for key in df_temp.index]

######################## font and color setup ########################
font = {'family': 'sans-serif', 'size': 5}  # Adjust size as needed
tick_font = {'fontsize': 5, 'fontname': 'sans-serif'}  # Adjust font size for tick labels

# Set the colorblind friendly palette
sns.set_palette("colorblind")
sns.set_theme(style="whitegrid")
# Get the list of colors in the palette
palette_colors = sns.color_palette("colorblind")

# Access the colors
color_supervised = palette_colors[0]  # First color
color_ssl = palette_colors[1]  # Second color
color_zeroshot = palette_colors[2]  # Third color
color_baseline = palette_colors[3]  # Forth color, ([3] looks similar to [0])
color_else1 = palette_colors[5]
color_else2 = palette_colors[6]
color_else3 = palette_colors[7]
my_pal = {'PCA': color_baseline,
          'Supervised': color_supervised,
          'SSL': color_ssl,
          'scVI': color_baseline,
          'SSL-Shallow': color_ssl,
          }
######################## Box plots setup ########################

df_1 = df_temp.reset_index()[['index', 'Batch correction']].copy().rename(columns={'Batch correction': 'score'})
df_2 = df_temp.reset_index()[['index', 'Bio conservation']].copy().rename(columns={'Bio conservation': 'score'})
df_1['score_type'] = 'Batch correction'
df_2['score_type'] = 'Bio conservation'
df_total = pd.concat([df_1, df_2], axis=0, ignore_index=True).reset_index()

plt.figure(figsize=(2.3, 2.3))

############ Total ############
plt.clf()
ax = sns.boxplot(data=df_temp.reset_index(), x="index", y="Total", order=order, palette=my_pal, linewidth=0.5)
ax.yaxis.grid(True)
ax.set_title('scIB metrics', fontdict=font)
ax.set_xlabel("Method", fontdict=font)
ax.set_ylabel("scIB Score Total", fontdict=font)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, **tick_font, ha='right')
ax.set_yticklabels(ax.get_yticklabels(),  **tick_font)
# plt.tight_layout()
plt.savefig(os.path.join(RESULT_PATH, 'embedding/box_plot_overall.svg'), bbox_inches='tight')
plt.savefig(os.path.join(RESULT_PATH, 'embedding/box_plot_overall.png'), bbox_inches='tight')

############ Batch correction ############
plt.clf()
ax = sns.boxplot(data=df_temp.reset_index(), x="index", y="Batch correction", order=order, palette=my_pal, linewidth=0.5)
ax.yaxis.grid(True)
ax.set_title('scIB batch correction scores',  fontdict=font)
ax.set_xlabel("Method", fontdict=font)
ax.set_ylabel("scIB Score", fontdict=font)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, **tick_font, ha='right')
ax.set_yticklabels(ax.get_yticklabels(),  **tick_font)
# plt.tight_layout()
plt.savefig(os.path.join(RESULT_PATH, 'embedding/box_plot_batch_correction.svg'), bbox_inches='tight')
plt.savefig(os.path.join(RESULT_PATH, 'embedding/box_plot_batch_correction.png'), bbox_inches='tight')

############ Bio Conservation ############
plt.clf()
ax = sns.boxplot(data=df_temp.reset_index(), x="index", y="Bio conservation", order=order, palette=my_pal, linewidth=0.5)
ax.yaxis.grid(True)
ax.set_title('scIB bio conservation scores', fontdict=font)
ax.set_xlabel("Method", fontdict=font)
ax.set_ylabel("scIB Score", fontdict=font)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, **tick_font, ha='right')
ax.set_yticklabels(ax.get_yticklabels(), **tick_font)
# plt.tight_layout()
plt.savefig(os.path.join(RESULT_PATH, 'embedding/box_plot_bio_conservation.svg'), bbox_inches='tight')
plt.savefig(os.path.join(RESULT_PATH, 'embedding/box_plot_bio_conservation.png'), bbox_inches='tight')

############ hue ############
# plt.clf()
# ax = sns.boxplot(data=df_total, x="index", y="score", hue="score_type", order=order)
# ax.yaxis.grid(True)
# plt.title('scIB metrics')
# plt.xlabel("Method")
# plt.ylabel("scIB Batch Correction Score")
# ax.set_xticklabels(ax.get_xticklabels(), **tick_font)
# ax.set_yticklabels(ax.get_yticklabels(), **tick_font)
# plt.tight_layout()
# plt.savefig(os.path.join(RESULT_PATH, 'embedding/box_plot_separated.svg'), bbox_inches='tight')
