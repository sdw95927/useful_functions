fig = plt.figure(figsize=(5, 5))
ax = plt.axes([0,0,1,1], frameon=False)
plt.imshow(image)
plt.axis('off')
plt.autoscale(tight=True)

# If we haven't already shown or saved the plot, then we need to
# draw the figure first...
fig.canvas.draw()

# Now we can save it to a numpy array.
data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
plt.close()
plt.imshow(data)
plt.show()
