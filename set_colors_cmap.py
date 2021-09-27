 ################## 1) Categorical cmap ###############

plt.figure(figsize=(10, 6))

mycmap = plt.get_cmap("Set2")

mycmap.N = N_CLASSES

mycmap.colors = [mycmap.colors[_] for _ in range(N_CLASSES)]

plt.imshow(pred_class, cmap=mycmap, vmin=0, vmax=N_CLASSES-1)

# plt.axis('off')

plt.gca().set_xticks([])

plt.gca().set_yticks([])

cbar = plt.colorbar()

cbar.ax.get_yaxis().set_ticks([])

ys = [(N_CLASSES-1)/N_CLASSES * (i + 0.5) for i in range(N_CLASSES)]

for i, category in enumerate(list(CLASSES_NAMES)):

    cbar.ax.text(10, ys[i], category, ha='left', va='center')

plt.title("Category map")

plt.show()



################ 2) Continuous cmap ################

#****** For sns ******

import matplotlib as mpl

mycmap = sns.color_palette("rocket", as_cmap=True)

mynorm = mpl.colors.Normalize(vmin=np.quantile(distance_matrix_all, 0.03)-0.1, vmax=np.quantile(distance_matrix_all, 0.91))

dist_colors = []

for _ in range(len(slide_ids)):

    dist_colors.append(mycmap(mynorm(distance_matrix_all[_*2, _*2+1])))



#****** For plt ******

    # Set color

    vmax = np.quantile(_image_feature, 0.9)

    vmin = np.quantile(_image_feature, 0.05)

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    cmap = cm.jet

    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    _mycolor = m.to_rgba(_image_feature)