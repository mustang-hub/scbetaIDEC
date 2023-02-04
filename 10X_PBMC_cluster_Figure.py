import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

y = np.loadtxt('/Users/mustang/PycharmProjects/PyTorch_project/11_scVIDEC/10X_PBMC_ select_2100/10x_PBMC_y_pred.txt', delimiter=',')
z_data = np.loadtxt('/Users/mustang/PycharmProjects/PyTorch_project/11_scVIDEC/10X_PBMC_ select_2100/10x_PBMC_2100_latent_sample.txt', delimiter=',')
X_tsne = TSNE(n_components=2).fit_transform(z_data.data)
fig = plt.figure(figsize=(6,8),dpi=300)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1],c = y, s=5, cmap=plt.cm.get_cmap('gist_rainbow',8))
plt.colorbar(ticks = range(8))
plt.clim(-0.5,7.5)
plt.tick_params(labelsize=12)
plt.show()








