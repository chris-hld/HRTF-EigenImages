"""Eigen-Images from plotted HRTF SHs."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import src

from sklearn.decomposition import PCA

# close previous plots
plt.close("all")
# Plot
try:
    plt.style.use('CFH')  # Custom matplotlib stylefile
except:
    pass

# import HRTF images
dataset = 'TUB_2015'
# 'horizontal' or 'median':
mode = 'horizontal'

samplesL, imgDim1, imgDim2 = src.IO.load_HRTF_img(dataset, mode, 'left')

if mode == 'median':
    xlabeling = 'Elevation / Degree'
if mode == 'horizontal':
    xlabeling = 'Azimuth / Degree'

grid = src.IO.load_grid(dataset)
gridNow = grid.loc[grid['azimuth'] == 0]

# Remove mean
samplesL -= samplesL.mean()

# define features and labels
X_cluster = pd.DataFrame(samplesL)

# PCA

pca = PCA(n_components=40)
pca.fit(X_cluster)

X_cluster = pca.transform(X_cluster)

pca_explained_ratio = pca.explained_variance_ratio_
pca_explained = pca.explained_variance_
pca_components = pca.components_

# PCA Components
plt.figure()
plt.plot(np.arange(1, 41), pca_explained)
plt.xticks([1, 5, 10, 15, 20, 25, 30, 35, 40],
           ('1', '5', '10', '15', '20', '25', '30', '35', '40'))

# plt.title('Explained Variance by Dimension')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.tight_layout()

# reduce dimension
X_cluster10 = X_cluster[:, :10]
X_cluster20 = X_cluster[:, :20]
X_cluster40 = X_cluster[:, :40]  # Full

# show "basis pictures"
fig = plt.figure(figsize=(9.5, 3.8))
pc_sum = 0
for k in range(3):
    pc_img = pca_components[k, :] * pca_explained[k]
    pc_img = pc_img.reshape(imgDim1, imgDim2)
    pc_img = np.abs(pc_img)
    pc_sum += pc_img

    plt.subplot(1, 3, k+1)
    pc_plot = plt.imshow(pc_img)
    plt.colorbar(pc_plot, fraction=0.0635, pad=0.04)
    plt.yticks(np.linspace(1, imgDim1, 5), ('20', '15', '10', '5', '1'))
    if mode == 'horizontal':
        plt.xticks(np.linspace(1, imgDim2, 5),
                   ('0', '90', '180', '270', '360'))
    if mode == 'median':
        plt.xticks(np.linspace(1, imgDim2, 3),
                   ('-65', '90', '245'))

fig.text(0.5, 0.01, xlabeling, ha='center')
fig.text(0, 0.5, 'Frequency / kHz', va='center', rotation='vertical')
plt.tight_layout()

# sum of first 10 components
plt.figure()
# plt.title('Sum of First 10 Absolute Basis Pictures')
img_plot = plt.imshow(pc_sum)
plt.colorbar(img_plot, fraction=0.08, pad=0.03)
plt.xlabel(xlabeling)
plt.ylabel('Frequency / kHz')
plt.yticks(np.linspace(1, imgDim1, 5), ('20', '15', '10', '5', '1'))
if mode == 'horizontal':
    plt.xticks(np.linspace(1, imgDim2, 5), ('0', '90', '180', '270', '360'))
if mode == 'median':
    plt.xticks(np.linspace(1, imgDim2, 3), ('-65', '90', '245'))
plt.tight_layout()

# Denoise: X_transformed x PC_components,reduced = X_denoised

subj = 18

cmap = plt.get_cmap('RdBu_r')
fig = plt.figure(figsize=(9.5, 3.8))
# plt.suptitle('Denoised')
for i, (dim, C) in enumerate(zip([40, 20, 10],
                                 [X_cluster40, X_cluster20, X_cluster10])):
    PC_components = pca_components[:dim, :]
    X_denoised = np.matmul(C, PC_components)
    img = X_denoised[subj-1, :].reshape(imgDim1, imgDim2)
    plt.subplot(1, 3, i+1)
    img_plot = plt.imshow(img, cmap=cmap, vmin=-0.2, vmax=0.2)
    plt.yticks(np.linspace(1, imgDim1, 5), ('20', '15', '10', '5', '1'))
    if mode == 'horizontal':
        plt.xticks(np.linspace(1, imgDim2, 5),
                   ('0', '90', '180', '270', '360'))
    if mode == 'median':
        plt.xticks(np.linspace(1, imgDim2, 3),
                   ('-65', '90', '245'))
    plt.colorbar(img_plot, fraction=0.0635, pad=0.04, ticks=[-0.2, -0.1, 0, 0.1, 0.2])
    plt.title(r'$\tilde{n} = $' + str(dim))

fig.text(0.5, 0.01, xlabeling, ha='center')
fig.text(0, 0.5, 'Frequency / kHz', va='center', rotation='vertical')
plt.tight_layout()

# last: show
plt.show()
