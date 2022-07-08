import phenograph
import seaborn as sns
import umap

k = 50  # choose k nearest neighbors
communities, graph, Q = phenograph.cluster(features_np, k=k) # run PhenoGraph

umap_2d = umap.UMAP(n_components=2, init='random', random_state=3)
umap_3d = umap.UMAP(n_components=3, init='random', random_state=3)

proj_2d = umap_2d.fit_transform(features_np)
proj_3d = umap_3d.fit_transform(features_np)

plt.figure(figsize=(8, 8))
sns.scatterplot(proj_2d[:, 0], proj_2d[:, 1], hue=communities, palette='pastel', legend='full', 
               hue_order=np.arange(len(np.unique(communities))))
plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
plt.axis('tight')
plt.show()
