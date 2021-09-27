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
