import matplotlib.pyplot as plt
x = ['0.1', '0.2', '0.3', '0.4','0.5','0.6','0.7','0.8','0.9','1.0']
NMI_1 = [0.7912, 0.7942,0.7916,0.7920,0.7966,0.8189,0.7917,0.7993,0.7924,0.7923]
ARI_1 = [0.7878,0.7923,0.7897,0.7897,0.7980,0.8287,0.7898,0.8028,0.7916,0.7915]
CA_1 = [0.8162,0.8200,0.8176,0.8186,0.8210,0.8361,0.8167,0.8248,0.8190,0.8181]

NMI_2 = [0.6652,0.6962,0.6976,0.6977,0.6989,0.7154,0.7103,0.7001,0.7024,0.6950]
ARI_2 = [0.4632,0.5539,0.5333,0.5541,0.5815,0.6067,0.6081,0.5451,0.5469,0.5458]
CA_2 = [0.5790,0.6629,0.6705,0.6667,0.7300,0.7371,0.7357,0.6767,0.6795,0.6595]

NMI_3 = [0.9739,0.9739,0.9739,0.9719,0.9739,0.9739,0.9739,0.9739,0.9719,0.9739]
ARI_3 = [0.9866,0.9866,0.9866,0.9851,0.9866,0.9866,0.9866,0.9866,0.9851,0.9866]
CA_3 = [0.9943,0.9943,0.9943,0.9938,0.9943,0.9943,0.9943,0.9943,0.9938,0.9943]

NMI_4 = [0.6370,0.6447,0.6469,0.6460,0.6472,0.6568,0.6461,0.6177,0.6473,0.6449]
ARI_4 = [0.4690,0.4761,0.4773,0.4782,0.4792,0.4822,0.4770,0.4463,0.4788,0.4759]
CA_4 = [0.6338,0.6705,0.6714,0.6724,0.6743,0.6786,0.6710,0.6548,0.6719,0.6695]
# 设置画布大小
plt.figure(figsize=(5,5))
# 标题
plt.title("Worm_neuron_cells_2100",fontsize=20)

# 数据
plt.plot(x, NMI_4, label='NMI', linewidth=3, color='r', marker='o',
         markerfacecolor='blue', markersize=10)
plt.plot(x,ARI_4,label='ARI',linewidth=3,color='g',marker='*',
         markerfacecolor='black',markersize = 10)
plt.plot(x,CA_4,label='ACC',linewidth=3,color='b',marker='d',
         markerfacecolor='yellow',markersize = 10)
# 横坐标描述
plt.xlabel('Gamma',fontsize=20)
# 纵坐标描述
plt.ylabel('Metrics',fontsize=20)

# 设置数字标签
for a, b in zip(x, NMI_4):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=20)
for a, b in zip(x,ARI_4):
    plt.text(a, b,b, ha='center', va = 'bottom', fontsize = 20)
for a, b in zip(x,CA_4):
    plt.text(a, b,b, ha='center', va = 'bottom', fontsize = 20)
plt.grid(True,linestyle = '--')
plt.legend(loc="center right")
plt.tick_params(labelsize=20)
plt.show()
