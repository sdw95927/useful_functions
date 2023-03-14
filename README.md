# Useful functions
Useful functions in daily life

## Python
* [BIC calculation for general clustering problems](https://github.com/sdw95927/useful_functions/blob/main/BIC_calculation_for_general_clustering.py)
* [Set colors with cmap](https://github.com/sdw95927/useful_functions/blob/main/set_colors_cmap.py)
* [Plot colorbar](https://github.com/sdw95927/useful_functions/blob/main/plot_colorbar.py)
* [Seaborn clustermap](https://github.com/sdw95927/useful_functions/blob/main/sns_clustermap.py)
* [Phenograph and UMAP (with legend out of box)](https://github.com/sdw95927/useful_functions/blob/main/phenograph.py)
* [Scatter plot, boxplot, violioplot, and errorbar together](https://github.com/sdw95927/useful_functions/blob/main/scatterplot_with_boxplot.py)

### Config conda env to make it available in jupyter
```
conda install ipykernel
python -m ipykernel install --user --name=mask_rcnn_torch
```
https://medium.com/@nrk25693/how-to-add-your-conda-environment-to-your-jupyter-notebook-in-just-4-steps-abeab8b8d084

### Accelerate num checking
![image](https://user-images.githubusercontent.com/16247996/224452772-f365b577-e563-49a6-bf7f-57b6d8945996.png)

## Bash

### Check and release GPU memory

In terminal:

lsof /dev/nvidia-uvm
for i in {23723..23807}; do kill -9 $i; done;

If not sure which is the pid you want:

ps (pid)  # process status command

## Mathematical symbols
https://dept.math.lsa.umich.edu/~kesmith/295handout1-2010.pdf

## R

### When your R package is not available

Visit https://cran.r-project.org/src/contrib/Archive/.

Find the package you want to install with Ctrl + F

Click the package name

Determine which version you want to install

Open RStudio

```
install.packages("https://cran.r-project.org/src/contrib/Archive/[NAME OF PACKAGE]/[VERSION NUMBER].tar.gz", repos = NULL, type="source")
```

https://stackoverflow.com/questions/25721884/how-should-i-deal-with-package-xxx-is-not-available-for-r-version-x-y-z-wa

## Word tricks

1) Insert seperator for word format: Ctrl + Alt + Enter.

2) Show hidden markers (e.g., page break, section break): Ctrl + Shift + 8
