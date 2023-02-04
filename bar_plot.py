import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
# plt.style.use('seaborn')
x = ['10XPBMC','Mouse_ES','Mouse_bladder','Worm_neuron']
NMI_full = [0.7942,0.9559,0.7453,0.6529]
NMI_ = [0.8190,0.9580,0.7532,0.6568]

ARI_full = [0.7867,0.9744,0.5055,0.5208]
ARI_ = [0.8288,0.9779,0.5521,0.4723]

CA_full = [0.8136,0.9891,0.6162,0.6799]
CA_ = [0.8362,0.9897,0.6943,0.6787]

plt.figure(figsize=(6,6),frameon=True,dpi=300)
ax = plt.subplot()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(True)
plt.scatter(x,NMI_full,s=30,c='green',marker='^',edgecolors='black',label='full_datasets')
plt.scatter(x,NMI_,s=30,c='red',marker='s',edgecolors='black',label='2100_datasets')
# plt.scatter(x,ARI_full,s=30,c='green',marker='^',edgecolors='black',label='full_datasets')
# plt.scatter(x,ARI_,s=30,c='red',marker='s',edgecolors='black',label='2100_datasets')
# plt.scatter(x,CA_full,s=30,c='green',marker='^',edgecolors='black',label='full_datasets')
# plt.scatter(x,CA_,s=30,c='red',marker='s',edgecolors='black',label='2100_datasets')
# plt.xticks(rotation=30)
plt.tick_params(labelsize=10)
plt.legend()
plt.show()
