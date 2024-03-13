import json
from konlpy.tag import Komoran

pos_table = ['NNG','NNP','NNB','NP','NR','VV','VA','VCP','VCN','MM','MAG','MAJ','IC','XR']

class KnuSL():

	def data_list(wordname):	
		with open('./data/SentiWord_info.json', encoding='utf-8-sig', mode='r') as f:
			data = json.load(f)
		result = [0,0] # 긍정/부정 정도	
		for i in range(0, len(data)):

			if  wordname == data[i]['word_root'] :
				# print("극성: ",data[i]['polarity'])
				if int(data[i]['polarity']) > 0:
					
					result[0] += int(data[i]['polarity'])
				elif int(data[i]['polarity']) < 0:
					result[1] += int(data[i]['polarity'])
				break
			
		
		return result[0],result[1]

def check_sentiment_absence(sentence):
	
	ksl = KnuSL
	kmo = Komoran()

	
	#print("-2:매우 부정, -1:부정, 0:중립 or Unkwon, 1:긍정, 2:매우 긍정")
	wordname = sentence
	wordname = wordname.strip(" ")		
	positive,negative = 0,0
	
	for word in wordname.split():
	
		for w_p in kmo.pos(word,flatten=False,join= True)[0]:
			w,p = w_p.split("/")
			if p in pos_table:
				# print(w,p)
				ret =  ksl.data_list(w)

				positive += ret[0]
				negative += ret[1]
			
	return (positive,negative) if positive + abs(negative) > 1 else (0,0)



