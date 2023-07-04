import matplotlib.pyplot as plt
import numpy as np

my_market1501 = [(84.0, 94.8)]
paper_market1501 = [(85.9, 94.5)]
others_market1501 = [(80.5, 93.5), (76.5, 91.0), (74.0, 89.1)]

my_dukemtmc = [(72.0, 71.6)]
paper_dukemtmc = [(76.4, 86.4)]
others_dukemtmc = [(62.6, 78.7), (56.5, 70.8), (56.5, 74.7)]

my_market1501_x, my_market1501_y = zip(*my_market1501)
paper_market1501_x, paper_market1501_y = zip(*paper_market1501)
others_market1501_x, others_market1501_y = zip(*others_market1501)

my_dukemtmc_x, my_dukemtmc_y = zip(*my_dukemtmc)
paper_dukemtmc_x, paper_dukemtmc_y = zip(*paper_dukemtmc)
others_dukemtmc_x, others_dukemtmc_y = zip(*others_dukemtmc)

plt.subplot(2,1,1)
plt.plot(others_market1501_x, others_market1501_y, "bo", label="CVPR/ECCV 2018", color="green")
plt.plot(paper_market1501_x, paper_market1501_y, "s", markersize=7, label="Reference paper")
plt.plot(my_market1501_x, my_market1501_y, marker='X', markersize=9, label="Ours")
plt.title("Market1501")
plt.ylabel('Rank-1 accuracy')
plt.xlabel('mAP')
plt.legend(handlelength=0, prop={'size': 10}, borderpad=0.8, loc='upper left')

plt.subplot(2,1,2)
plt.plot(others_dukemtmc_x, others_dukemtmc_y, "bo", label="CVPR/ECCV 2018", color="green")
plt.plot(paper_dukemtmc_x, paper_dukemtmc_y, "s", markersize=7, label="Reference paper")
plt.plot(my_dukemtmc_x, my_dukemtmc_y, marker='X', markersize=9, label="Ours")
plt.title("DukeMTMC-reID")
plt.ylabel('Rank-1 accuracy')
plt.xlabel('mAP')
plt.legend(handlelength=0, prop={'size': 10}, borderpad=0.8, loc='upper left')

plt.tight_layout()
plt.savefig('acc_plot.pdf') 