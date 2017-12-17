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
import random
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from evaluation import *
import pickle

caffe.set_mode_gpu()
caffe.set_device(0)
random.seed(0)

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
		return net.blobs['fc5'].data[0].copy()
	
	def addFaceWithName(self, imgPath, name):
		embedding = self.getEmbedding(imgPath)
		length = self.index.get_n_items()
        	self.index.add_item(length, embedding)
		self.indexToName[length] = name
		self.nameToIndex[name] = length
	
	def addEmbeddingWithName(self, embedding, name):
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
	
	def lookupByEmbedding(self, embedding, numberOfNeighbours):
		if(numberOfNeighbours==-1):
			numberOfNeighbours = self.index.get_n_items()
		results = self.index.get_nns_by_vector(embedding, numberOfNeighbours, search_k=-1, include_distances=True)
		for i in xrange(len(results[0])):
                        results[0][i] = self.indexToName[results[0][i]]
		return results
	
	def lookupByName(self, name, numberOfNeighbours):
		if(numberOfNeighbours==-1):
			numberOfNeighbours = self.index.get_n_items()
		results = self.index.get_nns_by_item(self.nameToIndex[name], numberOfNeighbours, search_k=-1, include_distances=True)
		for i in xrange(len(results[0])):
			results[0][i] = self.indexToName[results[0][i]]
		return results


if __name__ == "__main__":
	
	def getEmbeddings(path):
		db = spherefaceAnnoyDatabase()
		result={}
		for root, dirs, files in tqdm.tqdm(os.walk(path, topdown=False)):
			for name in files:
				filename = os.path.join(root,name)
				result[filename]=db.getEmbedding(filename)
		return result
	
	def testDB(toDB, toKeep):
		evaluator = QueryEvaluator()
		db = spherefaceAnnoyDatabase()
		for filename in toDB:
			embedding = toDB[filename]
			db.addEmbeddingWithName(embedding,filename)
		db.freeze()
		for filename in toKeep:
			embedding = toKeep[filename]
			result = db.lookupByEmbedding(embedding, -1)
			identity = filename.split('/')[2]
			orderedIdentitiesRetrieved = [i.split('/')[2] for i in result[0]]
			evaluator.update(identity, orderedIdentitiesRetrieved)
		return evaluator.getEvaluation()
	
	def trimData(toDB, toKeep, numberOfIdentities, numberOfPhotosPerIdentity):
		identities = [f.split('/')[2] for f in toKeepEmbeddings]
		sampledIdentities = set(random.sample(identities, numberOfIdentities))
		newToDB = {}
		toDBKeys = toDB.keys()
		random.shuffle(toDBKeys)
		identityFrequenciesToDB = {}
		for identity in sampledIdentities:
			if(not identity in identityFrequenciesToDB):
				identityFrequenciesToDB[identity] = 0
		for filename in toDBKeys:
			currentIdentity = filename.split('/')[2]
			if(currentIdentity in sampledIdentities):
				if(identityFrequenciesToDB[currentIdentity]<numberOfPhotosPerIdentity):
					newToDB[filename]=toDB[filename]
					identityFrequenciesToDB[currentIdentity]+=1
		newToKeep = {}
		for filename in toKeep:
			currentIdentity = filename.split('/')[2]
			if(currentIdentity in sampledIdentities):
				newToKeep[filename]=toKeep[filename]
		return newToDB, newToKeep
		
	toDbEmbeddings = getEmbeddings('dataset/todb')
	toKeepEmbeddings = getEmbeddings('dataset/tokeep')
	results = []
	for numberOfIdentities in tqdm.tqdm(xrange(50,5001,50)):
		for numberOfPhotosPerIdentity in xrange(1,11):
			toDB,toKeep = trimData(toDbEmbeddings, toKeepEmbeddings, numberOfIdentities, numberOfPhotosPerIdentity)
			currentResults = testDB(toDB, toKeep)
			results.append([numberOfIdentities,numberOfPhotosPerIdentity,currentResults])
	fp = open('results500.pkl','wb')
	pickle.dump(results,fp)
	fp.flush()
	fp.close()
	
	
