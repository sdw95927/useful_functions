sns.heatmap(confusion_matrix,
            annot=True,
            fmt='g',
            xticklabels=['Not Spam','Spam'],
            yticklabels=['Not Spam','Spam'])

# display matrix
plt.ylabel('Actual',fontsize=12)
plt.xlabel('Prediction',fontsize=12)
plt.show()
