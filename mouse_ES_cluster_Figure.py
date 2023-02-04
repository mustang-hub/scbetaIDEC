import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
y = np.loadtxt('/Users/mustang/PycharmProjects/PyTorch_project/vital/2100/mouse_ES_y_pred.txt', delimiter=',')
z_data = np.loadtxt('/Users/mustang/PycharmProjects/PyTorch_project/vital/2100/mouse_ES_cell_2100_latent_sample.txt', delimiter=',')
X_tsne = TSNE(n_components=2).fit_transform(z_data.data)
fig = plt.figure(figsize=(8,8),dpi=300)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1],c = y, s=5, cmap=plt.cm.get_cmap('gist_rainbow',4))
plt.colorbar(ticks = range(4))
plt.clim(-0.5,3.5)
plt.tick_params(labelsize=12)
plt.show()








