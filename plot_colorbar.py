# Plot colorbar
vmax = 2.5
vmin = -0.5
colorbar = list(np.linspace(vmax, vmin, 150))*10
colorbar = np.array(colorbar).reshape(-1, 10, order='F')
plt.imshow(colorbar, cmap='PiYG_r')
plt.xticks([], [])
plt.yticks(np.linspace(colorbar.shape[0], 0, 10), np.round(np.linspace(vmin, vmax, 10), 1))
plt.ylim(150, 0)
plt.show()


# Change xticks of colorbar
cbar = plt.colorbar()
cbar.ax.get_yaxis().set_ticks([])
ys = [(N_CLASSES-1)/N_CLASSES * (i + 0.5) for i in range(N_CLASSES)]
for i, category in enumerate(list(CLASSES_NAMES)):
    cbar.ax.text(10, ys[i], category, ha='left', va='center')
