def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from nltk.tokenize import sent_tokenize, word_tokenize 
import warnings 
from sklearn.externals import joblib


warnings.filterwarnings(action = 'ignore') 

import gensim 
from gensim.models import Word2Vec 

sample = open("C:\\Users\\DELL\\ncorpusop.txt", "r") 
s = sample.read() 

# Replaces escape character with space 
f = s.replace("\n", " ") 

opdata = [] 

# iterate through each sentence in the file 
for i in sent_tokenize(f): 
	temp = [] 
	
	# tokenize the sentence into words 
	for j in word_tokenize(i): 
		temp.append(j.lower()) 

	opdata.append(temp) 

opmodell= Word2Vec.load("C:\\Users\\DELL\\opcbow.model")

phrase=input("Enter : ")

#code to clean the phrase.
import re

#phase=re.sub(r"'", " ",phase)
phrase = re.sub(r'\d','',phrase)
phrase= re.sub(r'.com', ' ', phrase)

phrase = re.sub(r'^b\s+', ' ', phrase)
phrase=re.sub(r'co \S+', ' ', phrase)
phrase = re.sub(r'http\S+', ' ', phrase)
phrase = re.sub(r'\W', ' ',phrase)
phrase = re.sub(r'http\S+', ' ', phrase)
phrase=re.sub(r'www\S+', ' ', phrase)
phrase = re.sub(r'\s+[a-zA-Z]\s+', ' ', phrase)
phrase= re.sub(r'\s+', ' ', phrase, flags=re.I)

#phrase

phrase=phrase.lower()
phrase=phrase.split(" ")
filename2 = 'C:\\Users\\DELL\\Log_model_optimized.sav'

loaded_model2 = joblib.load(filename2)
c=[]
d=[]
for i in phrase:
    if loaded_model2.predict(([i]))!='O':
        #d.append(loaded_model2.predict(vect.transform([i])))
        c.append(i)
        d.append(loaded_model2.predict([i]))
#d    
#for i in d:
#    print (i)
#for i in c:
#    print (i)
    
c=[]
d=[]
for i in phrase:
    if loaded_model2.predict(([i]))!='O':
        #d.append(loaded_model2.predict(vect.transform([i])))
        c.append(i)
        d.append(loaded_model2.predict([i]))
#d    
#for i in d:
#    print (i)
#for i in c:
#    print (i)

#phrase=input("Enter a sentence : ")
#phrase=phrase.lower()
phrase=c
l1=[]
l2=[]
l3=[]
entity=c      #phrase.split(' ')
for d in entity:  
    #for i in data:
    for i in opdata:
          for j in i:
                l1.append(d)
                l2.append(j)
                #l3.append(modell.similarity(d, j))
                l3.append(opmodell.similarity(d, j))
from json.encoder import JSONEncoder

import numpy as numpy
if len((l3))==0:
    #print ("error")
    final_entity = { "cat": ["O"]}
    # directly called encode method of JSON
    print (JSONEncoder().encode(final_entity))
else:
    posq2=numpy.argmax(l3)
    topred=l2[posq2]
    #print(loaded_model.predict(vect.transform([topred])))
    arr=loaded_model2.predict(([topred]))

    # import JSONEncoder class from json
    from json.encoder import JSONEncoder
    final_entity = { "cat": [arr[0]]}
    # directly called encode method of JSON
    print (JSONEncoder().encode(final_entity))