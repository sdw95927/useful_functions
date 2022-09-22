### Plot a single colorbar
vmax = 2.5
vmin = -0.5
colorbar = list(np.linspace(vmax, vmin, 150))*10
colorbar = np.array(colorbar).reshape(-1, 10, order='F')
plt.imshow(colorbar, cmap='PiYG_r')
plt.xticks([], [])
plt.yticks(np.linspace(colorbar.shape[0], 0, 10), np.round(np.linspace(vmin, vmax, 10), 1))
plt.ylim(150, 0)
plt.show()

### Plot colorbar aside of a main figure
from matplotlib.colorbar import ColorbarBase

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
plt.imshow(image)
ax2 = fig.add_axes([0.95, 0.1, 0.03, 0.8])
bounds = np.linspace(vmin, vmax, 20)
cb = ColorbarBase(ax2, cmap=cmap, norm=norm,
    spacing='proportional', ticks=bounds, boundaries=bounds, format='%.2f')

ax.set_title("Slide {}, Region {}".format(slide_id, patch_id))
ax2.set_ylabel('SCC Score', size=12)


### Change xticks of colorbar
cbar = plt.colorbar()
cbar.ax.get_yaxis().set_ticks([])
ys = [(N_CLASSES-1)/N_CLASSES * (i + 0.5) for i in range(N_CLASSES)]
for i, category in enumerate(list(CLASSES_NAMES)):
    cbar.ax.text(10, ys[i], category, ha='left', va='center')

