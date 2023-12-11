from matplotlib import font_manager

fontP = font_manager.FontProperties()
fontP.set_family('SimHei')
fontP.set_size(10)

plt.figure(figsize=(10, 10))
patches, texts, autotexts = plt.pie(num_cases, labels=labels, autopct='%1.1f%%', labeldistance=1.1, colors=matplotlib.cm.tab20.colors)
plt.setp(autotexts, fontproperties=fontP)
plt.setp(texts, fontproperties=fontP)
plt.legend(prop=fontP)
plt.savefig("8_从事行业饼图.png".format(start_time, end_time))
plt.show()
