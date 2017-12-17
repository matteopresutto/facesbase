import tarfile
import tqdm
import os
import random

random.seed(0)

identitiesFD = tarfile.open("identities_tight_cropped.tar.gz")

numberOfIdentities = 5000
numberOfPhotosPerIdentity = 11

print "Retrieving files in archive..."
filelist = {}
for tarinfo in tqdm.tqdm(identitiesFD):
	tmp = tarinfo.name.split("/")
	if(len(tmp)>=2):
		fatherDir = os.path.join(tmp[0], tmp[1])
		if(not fatherDir in filelist):
			filelist[fatherDir] = set()
		if(tarinfo.name[-4:]=='.jpg'):
			filelist[fatherDir].add(tarinfo.name)

print "\nRemoving directories with less than "+str(numberOfPhotosPerIdentity)+" examples..."

trimmedFilelist = {}
for key in tqdm.tqdm(filelist):
	if(len(filelist[key])>=numberOfPhotosPerIdentity):
		trimmedFilelist[key] = filelist[key]

print "\nSelecting "+str(numberOfIdentities)+" identities at random and "+str(numberOfPhotosPerIdentity)+" images each at random"
sampledIdentities = random.sample(trimmedFilelist.keys(), numberOfIdentities)
finalSamples = []
for id in sampledIdentities:
	finalSamples.append(random.sample(trimmedFilelist[id],numberOfPhotosPerIdentity))

if not os.path.exists("dataset"):
	os.makedirs("dataset")

if not os.path.exists("dataset/todb"):
	os.makedirs("dataset/todb")

if not os.path.exists("dataset/tokeep"):
        os.makedirs("dataset/tokeep")

todb = set()
tokeep = set()
for identity in finalSamples:
	tokeep.add(identity[0])
	for i in xrange(1,len(identity)):
		todb.add(identity[i])

identitiesFD.close()


identitiesFD = tarfile.open("identities_tight_cropped.tar.gz")

print "Extracting sampled files from archive (must full scan the tar.gz)..."
filelist = {}
for tarinfo in tqdm.tqdm(identitiesFD):
	whereToPlace = None
	if(tarinfo.name in todb):
		whereToPlace = "todb"
	elif(tarinfo.name in tokeep):
		whereToPlace = "tokeep"
	if(not (whereToPlace is None)):
		fatherdir = tarinfo.name.split("/")
		if not os.path.exists("dataset/"+whereToPlace+"/"+fatherdir[1]):
			os.makedirs("dataset/"+whereToPlace+"/"+fatherdir[1])
		tarinfo.name = os.path.basename(tarinfo.name)
		identitiesFD.extract(tarinfo, "dataset/"+whereToPlace+"/"+fatherdir[1])

print "\nSampled files are in the directory \"dataset\"."
