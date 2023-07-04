import matplotlib.pyplot as plt

market1501 = [(90.4, 74.8), (92.4, 78.0), (92.6, 79.2), (93.7, 81.0), (93.9, 81.5), (94.4, 82.9), (94.8, 84.0)]
dukemtmc = [(63.5, 63.6), (66.3, 66.6), (70.2, 68.2), (70.6, 69.2), (71.1, 69.9), (71.5, 72.0), (71.6, 72.0)]

market1501_x, market1501_y = zip(*market1501)
dukemtmc_x, dukemtmc_y = zip(*dukemtmc)

#plt.plot(market1501_x, label="r = 1", color = 'green', marker = 'o')
#plt.plot(market1501_y, label="mAP", color = 'red', marker = 'o')
plt.plot(dukemtmc_x, label="r = 1", marker = 'o')
plt.plot(dukemtmc_y, label="mAP", marker = 'o')

plt.title('DukeMTMC-reID')
#plt.title('Market1501')

plt.xlabel('Trick')
labels = ["Baseline", 1, 2, 3, 4, 5, 6]
plt.xticks([0,1,2,3,4,5,6], labels) 

plt.legend(loc = 'lower right')
plt.savefig('dukemtmc.pdf')
#plt.savefig('market1501.pdf')