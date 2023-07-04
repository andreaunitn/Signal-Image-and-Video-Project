import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

my_market1501_rank_1 = [18.3, 20.8, 20.6, 22.3, 22.5, 26.0, 25.6, 27.0]
luo_market1501_rank_1 = [24.4, 26.3, 21.5, 23.2, 23.1, 26.7, 27.5, 41.4]
my_martket1501_map = [16.4, 19.0, 18.8, 20.4, 20.6, 23.7, 23.5, 24.9]
luo_market1501_map = [12.9, 14.1, 10.2, 11.3, 11.8, 15.2, 15.0, 25.7]

my_dukemtmc_rank_1 = [46.6, 48.1, 49.1, 49.8, 50.3, 59.0, 60.0, 60.1]
luo_dukemtmc_rank_1 = [34.2, 39.7, 32.5, 36.5, 37.1, 47.7, 47.4, 54.3]
my_dukemtmc_map = [21.0, 21.4, 22.5, 22.5, 23.1, 28.7, 29.8, 29.5]
luo_dukemtmc_map = [14.5, 17.4, 13.5, 14.9, 15.4, 21.6, 21.4, 25.5]

bar_width = 0.35
index = np.arange(len(my_market1501_rank_1))

plt.bar(index, my_market1501_rank_1, width = bar_width, label = "Ours")
plt.bar(index + bar_width, luo_market1501_rank_1, width = bar_width, label = "Luo et al.")

# plt.bar(index, my_martket1501_map, width = bar_width, label = "Ours", color = "tan")
# plt.bar(index + bar_width, luo_market1501_map, width = bar_width, label = "Luo et al.", color = "salmon")

# plt.bar(index, my_dukemtmc_rank_1, width = bar_width, label = "Ours", color = "yellowgreen")
# plt.bar(index + bar_width, luo_dukemtmc_rank_1, width = bar_width, label = "Luo et al.", color = "purple")

# plt.bar(index, my_dukemtmc_map, width = bar_width, label = "Ours", color = "gold")
# plt.bar(index + bar_width, luo_dukemtmc_map, width = bar_width, label = "Luo et al.", color = "sienna")

plt.xlabel('Trick')
plt.ylabel('Rank-1 accuracy')
#plt.ylabel('mAP')
plt.title(r'Market1501 $\rightarrow$ DukeMTMC-reID')
#plt.title(r'DukeMTMC-reID $\rightarrow$ Market1501')

labels = ["Baseline", 1, 2, 3, 4, 5, 6, "6 - 2"]
plt.xticks(index + bar_width / 2, labels) 
plt.legend()
plt.tight_layout()
plt.savefig('cross_rank1_market1501.pdf')
#plt.savefig('cross_map_market1501.pdf')
#plt.savefig('cross_rank1_dukemtmc.pdf')
#plt.savefig('cross_map_dukemtmc.pdf')

