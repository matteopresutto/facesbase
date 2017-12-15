from scipy import misc
import numpy
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle

def computeAndSaveGraphs(pklFilename,measure = 'hit@10'):
	fp = open(pklFilename,'rb')
	results = pickle.load(fp)
	top10 = [[i[0],i[1],i[2][measure]] for i in results]
	x = sorted(list(set([i[0] for i in top10])))
	xmap = {i:index for index,i in enumerate(x)}
	y = sorted(list(set([i[1] for i in top10])))
	ymap = {i:index for index,i in enumerate(y)}
	z = numpy.zeros((len(y),len(x)))
	for i in top10:
		z[ymap[i[1]],xmap[i[0]]]=i[-1]
	xv, yv = np.meshgrid(x, y)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	dem3d=ax.plot_surface(xv,yv,z,cmap='cool', linewidth=0, alpha=0.9)
	ax.set_title(measure)
	ax.set_zlabel(measure)
	ax.set_ylabel('Photos per identity')
	ax.set_xlabel('Number of identities')
	if(measure!='MAP' and measure!='FOP95' and measure!='FOP99'):
		ax.set_zlim3d(0,1)
	plt.show()
	plt.savefig(pklFilename.split('.')[0]+'_'+measure+'.png', format='png')

computeAndSaveGraphs('results500.pkl',measure = 'hit@10')
computeAndSaveGraphs('results500.pkl',measure = 'hit@3')
computeAndSaveGraphs('results500.pkl',measure = 'hit@1')
computeAndSaveGraphs('results500.pkl',measure = 'MAP')
computeAndSaveGraphs('results500.pkl',measure = 'FOP95')
computeAndSaveGraphs('results500.pkl',measure = 'FOP99')
