import os
import sys
# Extending PYTHONPATH with caffe-sphereface/python
sys.path.append(os.path.join(os.getcwd(),"sphereface/tools/caffe-sphereface/python"))
os.environ['GLOG_minloglevel'] = '2' 
import caffe
from PIL import Image
import numpy
import annoy
from annoy import AnnoyIndex
import tqdm
from evaluation import QueryEvaluator
caffe.set_mode_gpu()
caffe.set_device(0)

featuresLength = 512
t = AnnoyIndex(featuresLength)
net = caffe.Net("pretrainedModels/sphereface_deploy.prototxt", "pretrainedModels/sphereface_model.caffemodel",0)

class spherefaceAnnoyDatabase():
	def __init__(self):
		self.network = caffe.Net("pretrainedModels/sphereface_deploy.prototxt", "pretrainedModels/sphereface_model.caffemodel",0)
		self.index = AnnoyIndex(512, metric='angular') # 512 is the number of neurons in the last layer of the net
		self.indexToName = {}
		self.nameToIndex = {}

	def getEmbedding(self, imgPath):
		img = Image.open(imgPath)
        	sampleImage = numpy.array(img.resize((net.blobs['data'].data.shape[3],net.blobs['data'].data.shape[2])))
        	sampleImage = numpy.reshape(sampleImage,(1,)+sampleImage.shape).transpose(0,3,1,2).astype(numpy.float32)
        	net.blobs['data'].data[...]=sampleImage
        	net.forward()
		return net.blobs['fc5'].data[0]
	
	def addFaceWithName(self, imgPath, name):
		embedding = self.getEmbedding(imgPath)
		length = self.index.get_n_items()
        	self.index.add_item(length, embedding)
		self.indexToName[length] = name
		self.nameToIndex[name] = length
	
	def addFaceWithoutName(self, imgPath):
		embedding = self.getEmbedding(imgPath)
                length = self.index.get_n_items()
                self.index.add_item(length, embedding)
                self.indexToName[length] = imgPath
                self.nameToIndex[imgPath] = length

	def freeze(self, nTrees = 20):
		self.index.build(nTrees)

	def lookupByFace(self, imgPath, numberOfNeighbours):
		embedding = self.getEmbedding(imgPath)
		results = self.index.get_nns_by_vector(embedding, numberOfNeighbours, search_k=-1, include_distances=True)
		for i in xrange(len(results[0])):
                        results[0][i] = self.indexToName[results[0][i]]
		return results
	
	def lookupByName(self, name, numberOfNeighbours):
		results = self.index.get_nns_by_item(self.nameToIndex[name], numberOfNeighbours, search_k=-1, include_distances=True)
		for i in xrange(len(results[0])):
			results[0][i] = self.indexToName[results[0][i]]
		return results



if __name__ == "__main__":
	limit = None
	db = spherefaceAnnoyDatabase()
	print 'Populating the database...'
	counter = 0
	for root, dirs, files in tqdm.tqdm(os.walk("dataset/todb", topdown=False)):
		for name in files:
			filename = os.path.join(root,name)
			db.addFaceWithName(filename,filename)
		counter+=1
		if(counter==limit):
			break
	
	db.freeze()
	qe = QueryEvaluator()
	counter = 0
	print 'Estimating statistics...'
	for root, dirs, files in tqdm.tqdm(os.walk("dataset/tokeep", topdown=False)):
        	for name in files:
			counter+=1
			filename = os.path.join(root,name)
                	result = db.lookupByFace(filename, 50)
			identity = filename.split('/')[2]
			orderedIdentitiesRetrieved = [i.split('/')[2] for i in result[0]]
			qe.update(identity,orderedIdentitiesRetrieved)
		if(counter==limit):
			break
	print qe.getEvaluation()
