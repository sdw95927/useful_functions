from scipy.spatial import distance
from scipy.cluster import hierarchy
import seaborn as sns
from scipy.stats import spearmanr

# Heatmap
correlation_between_image_and_genomic = np.zeros([image_feature.shape[1], image_feature.shape[1]])
for feature_id_i in range(image_feature.shape[1]):
    for feature_id_j in range(feature_id_i, image_feature.shape[1]):
        corr = spearmanr(image_feature[:, feature_id_i], 
                         gene_feature[:, feature_id_j]).correlation
        correlation_between_image_and_genomic[feature_id_i, feature_id_j] = corr
        correlation_between_image_and_genomic[feature_id_j, feature_id_i] = corr
        
row_linkage = hierarchy.linkage(
    distance.pdist(correlation_between_image_and_genomic), method='average')
col_linkage = hierarchy.linkage(
    distance.pdist(correlation_between_image_and_genomic.T), method='average')

clustergrid = sns.clustermap(correlation_between_image_and_genomic, 
               cmap='coolwarm', 
               row_cluster=True, col_cluster=True,
               row_linkage=row_linkage, col_linkage=col_linkage,
               yticklabels=temp_labels, xticklabels=temp_labels, dendrogram_ratio=0.05,
               vmin=-0.8, vmax=0.8, square=True, figsize=(6, 6), 
               row_colors=row_colors, col_colors=col_colors)
clustergrid.ax_row_dendrogram.set_visible(True)
clustergrid.ax_col_dendrogram.set_visible(False)
clustergrid.cax.set_visible(False)
clustergrid.ax_heatmap.set_xlabel('')
clustergrid.ax_heatmap.set_ylabel('')
tick_labels = clustergrid.ax_heatmap.get_xmajorticklabels()
ticks = clustergrid.ax_heatmap.get_xticks()
_ticks = []
_tick_labels = []
for i, _ in enumerate(tick_labels):
    if _.get_text() != "":
        _ticks.append(ticks[i])
        _tick_labels.append(_)
clustergrid.ax_heatmap.set_xticks(_ticks)
# clustergrid.ax_heatmap.set_xticklabels(labels=_tick_labels, rotation=45, ha='right')
clustergrid.ax_heatmap.set_xticklabels(labels=_tick_labels)
tick_labels = clustergrid.ax_heatmap.get_ymajorticklabels()
ticks = clustergrid.ax_heatmap.get_yticks()
_ticks = []
_tick_labels = []
for i, _ in enumerate(tick_labels):
    if _.get_text() != "":
        _ticks.append(ticks[i])
        _tick_labels.append(_)
clustergrid.ax_heatmap.set_yticks(_ticks)
clustergrid.ax_heatmap.set_yticklabels(labels=_tick_labels, rotation=0)
# clustergrid.ax_heatmap.set_title("Corr. between Image and Genomic Features")
plt.show()
