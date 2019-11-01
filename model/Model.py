'''
Training Naive Bayes Classification Model:
- convert HS code to labeled datasets as the trainning data.
- implement the Naive Bayes algorithm to train the model.
'''
'''
1. Import the Nessary Packages and Files
'''
import pandas as pd
import numpy as np
import nltk
import re
import pickle
import gensim
from nltk.corpus import stopwords
stop_list = stopwords.words('english')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
# input/output file names
train_file = '../data/HS Code and Detailed Description.xlsx'
stopwords_file = '../data/stopwords.txt'

output_model = './Cargo_classifier.pickle'
output_dict = './dictionary.dic'
# import HS Code data to dataframe
df = pd.read_excel(train_file , sheet_name='HS_2002_20190523_084803')

'''
2. Prepare the Stopwords and Function
'''
# add more stop words to remove the noise
stop_words = open(stopwords_file, 'r')
for line in stop_words.readlines():
    stop_list.extend(line.split(','))
#processing for stopwords, alphabetic words, stemming
def preprocess_corpus(all_docs):
    # combine the categories of the corpus
    all_docs1 = [[' '.join(doc)] for doc in all_docs]
    # tokenization
    all_docs2 = [nltk.word_tokenize(v[0]) for v in all_docs1]
    # text pre-processing, including lower, alpha, short string and stop word removal, stemming, etc.
    all_docs3 = [[w.lower() for w in doc] for doc in all_docs2]
    all_docs4 = [[w for w in doc if re.search('^[a-z]+$',w)] for doc in all_docs3]
    all_docs4 = [[w for w in doc if len(w) >2] for doc in all_docs4]
    all_docs5 = [[w for w in doc if w not in stop_list] for doc in all_docs4]
    all_docs6 = [[stemmer.stem(w) for w in doc] for doc in all_docs5]
    all_docs7 = [[w for w in doc if w not in stop_list] for doc in all_docs6]
    return all_docs7

'''
3. Split HS Code Data by Sections(Level 1) as the Corpus and Preprocess the Corpus
'''
# get the column indexes of each level1 and level2 (Used to separate data by levels)
level1 = df[df['Level']==1].index.tolist()
level2 = df[df['Level']==2].index.tolist()
level1.append(7193)
level2.extend(level1)
level2.sort()
# get the labels (Sections/level1)
labels = []
for i in level1[:-1]:
    # for each level1, get the description/label
    name = df.loc[i]["Description"]
    labels.append(name)
# create corpus 
# we make a single list of docs and create a label for each item in nested list
Train_docs1 = []
Test_docs1 = []
num_docs = [0]
for i in range(len(level1)-1):
    # for each level1, create a empty list 
    label_Train = []
    for j in range(len(level2)):
        # for each level1, get all the level2 index which under this level1
        if level1[i] < level2[j] < level1[i+1]:
            # for each level2, create a empty list
            train_data = []
            # add the level2 description data to the list
            train_data.extend(df[level2[j]:level2[j+1]]["Description"][1:])
            # add the whole level2 data to level1
            label_Train.append(train_data)
    # record the range of each level
    num_docs.append(len(label_Train) + num_docs[-1])
    # combine all the level1 data together, as our corpus
    Train_docs1 += label_Train

# preprocess the corpus
Train_docs = preprocess_corpus(Train_docs1)

'''
4. Prepare Labeled Training Dictionary
'''
#Create dictionary
dictionary = gensim.corpora.Dictionary(Train_docs)
# save the dictionary
dictionary.save(output_dict)

# Convert all documents to TF Vectors
Train_tf_vectors = [dictionary.doc2bow(doc) for doc in Train_docs]
# Convert TF Vectors to dictionary format(Label the trained data).
Train_data_as_dict = [{id:1 for (id, tf_value) in vec} for vec in Train_tf_vectors]
# Create a empty list 
Train_labeled_data = []
for i in range(len(labels)):  
    # for each level2 data, given the level1-labels
    Train_labeled_data.extend([(d, labels[i]) for d in Train_data_as_dict[num_docs[i]:num_docs[i+1]]])
'''
5. Build the Classification Model
'''
#Generate the trained classifier
classifier = nltk.NaiveBayesClassifier.train(Train_labeled_data)
# save the Model
f = open(output_model, 'wb')
pickle.dump(classifier, f)
f.close()