import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
y = np.loadtxt('/Users/mustang/PycharmProjects/PyTorch_project/vital/2100/Worm_neuron_y_pred.txt', delimiter=',')
z_data = np.loadtxt('/Users/mustang/PycharmProjects/PyTorch_project/vital/2100/Worm_neuron_2100_latent_sample.txt', delimiter=',')
X_tsne = TSNE(n_components=2).fit_transform(z_data.data)
fig = plt.figure(figsize=(8,6),dpi=300)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1],c = y, s=5, cmap=plt.cm.get_cmap('gist_rainbow',10))
plt.colorbar(ticks = range(10))
plt.clim(-0.5,9.5)
plt.tick_params(labelsize=12)
plt.show()








