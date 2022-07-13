# Box-plot with scatter
xticks = CELL_CLASSES.copy()
vals = []
feature_name = 'mouse_counts'
for xtick in xticks:
    vals.append([nuclei_transcriptome_raw_df.loc[_, feature_name] 
                 for _ in nuclei_transcriptome_raw_df.index.values
                 if nuclei_transcriptome_raw_df.loc[_, 'cell_type'] == xtick]) 

i = 1
# plt.violinplot(vals, showmeans=False, showextrema=False)
plt.boxplot(vals, showfliers=False)
for val in vals:
    plt.scatter([i] * len(val) + np.random.randn(len(val))/10, val, alpha=0.001)
    plt.xticks(rotation=45, ha="right")
    i += 1
# plt.errorbar(np.arange(1, 1 + len(xticks)), [np.average(_) for _ in vals], [np.std(_) for _ in vals], 
#              linestyle='None', marker='o', fmt='-o', c='k', elinewidth=2, capsize=10)
plt.ylabel(feature_name, weight='bold', fontsize=20)
plt.xticks(np.arange(1,  1 + len(xticks)), xticks, weight='bold', fontsize=20)
plt.ylim(0, np.quantile(vals[0], 0.7))
plt.title('mouse counts', weight='bold', fontsize=20)

from scipy.stats import ttest_ind
for i in range(len(vals) - 1):
    print("{} vs. {}:".format(xticks[i], xticks[i + 1]), ttest_ind(vals[i], vals[i + 1], nan_policy='omit'))
