class QueryEvaluator():
	def __init__(self):
		self.numberOfElements=0
		self.summedHitAt10=0
		self.summedHitAt3=0
		self.summedHitAt1=0
		self.summedAveragePrecision=0

	def hitAt10(self, x, queryResults):
		return x in queryResults[:10]
	
	def hitAt3(self, x, queryResults):
        	return x in queryResults[:3]
	
	def hitAt1(self, x, queryResults):
		return x==queryResults[0]
	
	def averagePrecision(self, x, queryResults):
		counter = 0
		hit = 0
		ap = 0
		for r in queryResults:
			counter += 1
			if(x==r):
				hit += 1
			ap += hit/float(counter)
		ap /= float(len(queryResults))
		return ap

	def update(self, query, result):
		self.numberOfElements+=1
		self.summedHitAt10+=self.hitAt10(query,result)
		self.summedHitAt3+=self.hitAt3(query,result)
		self.summedHitAt1+=self.hitAt1(query,result)
		self.summedAveragePrecision+=self.averagePrecision(query,result)
	
	def getEvaluation(self):
		result = {}
		result['hit@10']=self.summedHitAt10/float(self.numberOfElements)
		result['hit@3']=self.summedHitAt3/float(self.numberOfElements)
		result['hit@1']=self.summedHitAt1/float(self.numberOfElements)
		result['MAP']=self.summedAveragePrecision/float(self.numberOfElements)
		return result
