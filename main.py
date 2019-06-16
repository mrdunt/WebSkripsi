
from flask import Flask, request, render_template,jsonify,Response
import os
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import pandas as pd
pd.set_option('display.max_colwidth', 3000)
pd.set_option('display.max_rows', None)

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import json 
 
import re
 
from sklearn import svm
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Normalizer
import numpy as np
from ast import literal_eval
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

app = Flask(__name__)

#Set Random seed
np.random.seed(400)
#import data csv
trainData = pd.read_csv('D:\Tugas\Skripsi\Data\dataset(final).csv',encoding ="ISO-8859-1",sep=';')
test_pd = pd.DataFrame(trainData) #makes this into a panda data frame

@app.route("/")
@app.route("/index")
def index():
	trainNew = test_pd
	
	trainNew = trainNew.drop("case_folding", axis=1)
	trainNew = trainNew.drop("StopWords Removal", axis=1)
	trainNew = trainNew.drop("Stemming", axis=1)
	trainNew = trainNew.drop("final", axis=1)
	trainNew = trainNew.drop("Tokenization", axis=1)
	label = test_pd['Label']
	pos = label[label=='pos']
	neg = label[label=='neg']
	jml_pos = pos.count()
	jml_neg = neg.count()
	jml_label = label.count()
	persen_pos = str(round((jml_pos/jml_label)*100,2))
	persen_neg = str(round((jml_neg/jml_label)*100,2))
	labels = ["Positif","Negative"]
	data = [jml_pos,jml_neg]
	colors = [ "#00a65a", "#dd4b39"]
	img = os.path.join('static', 'img')
	imgNeg = os.path.join(img, 'neg.png')
	imgPos = os.path.join(img, 'pos.png')
	return render_template('index.html',persen_pos=persen_pos,persen_neg=persen_neg, imgNeg=imgNeg,imgPos=imgPos, tables=[trainNew.to_html(classes='table table-hover table-bordered',header='true',justify='justify',table_id='tabel')], set=zip(data, labels, colors))


#CASE FOLDING & REMOVE REGEX
def preprocess(text):
    clean_data = []
    for x in (text[:]): #this is Df_pd for Df_np (text[:])
        new_text = re.sub('<.*?>', '', x)   # remove HTML tags
        new_text = re.sub(r'[^\w\s]', '', new_text) # remove punc.
        new_text = re.sub(r'\d+','',new_text)# remove numbers
        new_text = new_text.lower() # lower case, .upper() for upper          
        if new_text != '':
            clean_data.append(new_text)
    return clean_data
content = test_pd['Content']
caseFolding = preprocess(content)
test_pd['case_folding']=caseFolding
case_folding = test_pd['case_folding']

#TOKENIZATION
def identify_tokens(row):
    review = row['case_folding']
    tokens = word_tokenize(review)
    # taken only words (not punctuation)
    token_words = [w for w in tokens if w.isalpha()]
    return token_words
test_pd['Tokenization'] = test_pd.apply(identify_tokens,axis=1)
tokens=test_pd['Tokenization']

#REMOVE STOPWORDS
stops = set(stopwords.words("indonesian"))
def remove_stops(row):
    my_list = row['Tokenization']
    meaningful_words = [w for w in my_list if not w in stops]
    return (meaningful_words)
test_pd['StopWords Removal'] = test_pd.apply(remove_stops, axis=1)
stopword=test_pd['StopWords Removal']
 
#STEMMING
def stem_list(row):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    my_list = row['StopWords Removal']
    stemmed_list = [stemmer.stem(word) for word in my_list]
    return (stemmed_list)
test_pd['Stemming'] = test_pd.apply(stem_list, axis=1)
stem=test_pd['Stemming']
test_pd['final']=stem.astype(str) #array list ke string
text_final = test_pd['final'] #final column

#Split Data Test
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(text_final,test_pd['Label'],test_size=0.3)

#Vectorize Dataset
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(text_final) 
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

#Print Tfidf
# get the first vector out (for the first document)
first_vector_tfidfvectorizer=Test_X_Tfidf[1]
 
# place tf-idf values in a pandas data frame
df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=Tfidf_vect.get_feature_names(), columns=["tfidf"])
df.sort_values(by=["tfidf"],ascending=False)
 

 
# print(df)
# print("Ew")
# print(Test_X_Tfidf)

#Using SVM Model
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)

#predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
accuracy = round(accuracy_score(predictions_SVM,Test_Y)*100)
print("SVM Accuracy Score -> ",accuracy)
print(predictions_SVM)

@app.route("/casefolding")
def casedata():
	trainData = pd.read_csv('D:\Tugas\Skripsi\Data\dataset(final).csv',encoding="ISO-8859-1",sep=';')
	test_pd = pd.DataFrame(trainData) #makes this into a panda data frame
	content = test_pd['Content']  
	caseFolding = preprocess(content)
	test_pd['Case Folding']=caseFolding
	case_folding = test_pd['Case Folding']
	test_pd = test_pd.drop("Label", axis=1)
	return render_template('casefolding.html',  tables=[test_pd.to_html(classes='table table-hover table-bordered',header='true',justify='justify',table_id='tabel')])


@app.route("/token")
def tokendata():
	trainNew = test_pd
	trainNew = trainNew.drop("Label", axis=1)
	trainNew = trainNew.drop("case_folding", axis=1)
	trainNew = trainNew.drop("StopWords Removal", axis=1)
	trainNew = trainNew.drop("Stemming", axis=1)
	trainNew = trainNew.drop("final", axis=1)

	return render_template('token.html',  tables=[trainNew.to_html(classes='table table-hover table-bordered',header='true',justify='justify',table_id='tabel')])

@app.route("/stopwords")
def stopwordsdata():
	trainNew = test_pd
	trainNew = trainNew.drop("Label", axis=1)
	trainNew = trainNew.drop("case_folding", axis=1)
	trainNew = trainNew.drop("Tokenization", axis=1)
	trainNew = trainNew.drop("Stemming", axis=1)
	trainNew = trainNew.drop("final", axis=1)

	return render_template('stopwords.html',  tables=[trainNew.to_html(classes='table table-hover table-bordered',header='true',justify='justify',table_id='tabel')])

@app.route("/stemming")
def stemmingdata():
	trainNew = test_pd
	trainNew = trainNew.drop("Label", axis=1)
	trainNew = trainNew.drop("case_folding", axis=1)
	trainNew = trainNew.drop("Tokenization", axis=1)
	trainNew = trainNew.drop("StopWords Removal", axis=1)
	trainNew = trainNew.drop("final", axis=1)

	return render_template('stemming.html',  tables=[trainNew.to_html(classes='table table-hover table-bordered',header='true',justify='justify',table_id='tabel')])

@app.route("/sentiment")
def sentimentData():
	dataNew = pd.DataFrame(Test_X)
	dataPredict = pd.DataFrame(predictions_SVM)
	#dataNew = pd.concat([Test_X,dataPredict],ignore_index=True)
	dataNew.columns = ['Data Test']
	dataPredict.columns =['Sentiment Prediction']
	label = dataPredict['Sentiment Prediction']
	pos = label[label=='pos']
	neg = label[label=='neg']
	jml_pos = pos.count()
	jml_neg = neg.count()
	jml_label = label.count()
	persen_pos = str(round((jml_pos/jml_label)*100,2))
	persen_neg = str(round((jml_neg/jml_label)*100,2))
	labels = ["Positif","Negative"]
	data = [jml_pos,jml_neg]
	colors = [ "#00a65a", "#dd4b39"]
	tfidf = df
	return render_template('sentiment.html', persen_pos=persen_pos,persen_neg=persen_neg, accuracy=accuracy,tables=[dataNew.to_html(classes='table table-hover table-bordered',header='true',justify='justify',table_id='tabel')],tablePredict=[dataPredict.to_html(classes='table table-hover table-bordered',header='true',justify='justify',table_id='tabel2',index=False)],set=zip(data, labels, colors),tableTfidf=[tfidf.to_html(classes='table table-hover table-bordered',header='true',justify='justify',table_id='tabel3')])
 
if __name__ == '__main__':
	app.run(debug=True)