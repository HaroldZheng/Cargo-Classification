'''
Classification with trained model:
- classify the new datasets by the trained model.
'''
'''
1. Import the Nessary Packages, Load the Trained Model and Dictionary
'''
import pandas as pd
import numpy as np
import pickle
import gensim
import nltk
import re
from nltk.corpus import stopwords
stop_list = stopwords.words('english')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
# input/output file names  
test_file = './data/MIP Cargo.xlsx'
model_file = './model/Cargo_classifier.pickle'
dict_file = './model/dictionary.dic'
stopwords_file = './data/stopwords.txt'

output_file = './output/MIP_labels.csv'
# load the trained model we have saved
f = open(model_file, 'rb')
classifier = pickle.load(f) 
f.close()
# load the dictionary we have saved
dictionary = gensim.corpora.Dictionary.load(dict_file)

'''
2. Import Files and Preprocess the Test Data
'''
# add more stop words to remove the noise
stop_words = open(stopwords_file, 'r')
for line in stop_words.readlines():
    stop_list.extend(line.split(','))

labels =['SECTION I - LIVE ANIMALS; ANIMAL PRODUCTS',
         'SECTION II - VEGETABLE PRODUCTS',
         'SECTION III - ANIMAL OR VEGETABLE FATS AND OILS AND THEIR CLEAVAGE PRODUCTS; PREPARED EDIBLE FATS; ANIMAL OR VEGETABLE WAXES',
         'SECTION IV - PREPARED FOODSTUFFS; BEVERAGES, SPIRITS AND VINEGAR; TOBACCO AND MANUFACTURED TOBACCO SUBSTITUTES',
         'SECTION V - MINERAL PRODUCTS',
         'SECTION VI - PRODUCTS OF THE CHEMICAL OR ALLIED INDUSTRIES',
         'SECTION VII - PLASTICS AND ARTICLES THEREOF; RUBBER AND ARTICLES THEREOF',
         'SECTION VIII - RAW HIDES AND SKINS, LEATHER, FURSKINS AND ARTICLES THEREOF; SADDLERY AND HARNESS; TRAVEL GOODS, HANDBAGS AND SIMILAR CONTAINERS; ARTICLES OF ANIMAL GUT (OTHER THAN SILKWORM GUT)',
         'SECTION IX - WOOD AND ARTICLES OF WOOD; WOOD CHARCOAL; CORK AND ARTICLES OF CORK; MANUFACTURES OF STRAW, OF ESPARTO OR OF OTHER PLAITING MATERIALS; BASKETWARE AND WICKERWORK',
         'SECTION X - PULP OF WOOD OR OF OTHER FIBROUS CELLULOSIC MATERIAL; RECOVERED (WASTE AND SCRAP) PAPER OR PAPERBOARD; PAPER AND PAPERBOARD AND ARTICLES THEREOF',
         'SECTION XI - TEXTILES AND TEXTILE ARTICLES',
         'SECTION XII - FOOTWEAR, HEADGEAR, UMBRELLAS, SUN UMBRELLAS, WALKING-STICKS, SEAT-STICKS, WHIPS, RIDING-CROPS AND PARTS THEREOF; PREPARED FEATHERS AND ARTICLES MADE THEREWITH; ARTIFICIAL FLOWERS; ARTICLES OF HUMAN HAIR',
         'SECTION XIII - ARTICLES OF STONE, PLASTER, CEMENT, ASBESTOS, MICA OR SIMILAR MATERIALS; CERAMIC PRODUCTS; GLASS AND GLASSWARE',
         'SECTION XIV - NATURAL OR CULTURED PEARLS, PRECIOUS OR SEMI-PRECIOUS STONES, PRECIOUS METALS, METALS CLAD WITH PRECIOUS METAL, AND ARTICLES THEREOF; IMITATION JEWELLERY; COIN',
         'SECTION XV - BASE METALS AND ARTICLES OF BASE METAL',
         'SECTION XVI - MACHINERY AND MECHANICAL APPLIANCES; ELECTRICAL EQUIPMENT; PARTS THEREOF; SOUND RECORDERS AND REPRODUCERS, TELEVISION IMAGE AND SOUND RECORDERS AND REPRODUCERS, AND PARTS AND ACCESSORIES OF SUCH ARTICLES',
         'SECTION XVII - VEHICLES, AIRCRAFT, VESSELS AND ASSOCIATED TRANSPORT EQUIPMENT',
         'SECTION XVIII - OPTICAL, PHOTOGRAPHIC, CINEMATOGRAPHIC, MEASURING, CHECKING, PRECISION, MEDICAL OR SURGICAL INSTRUMENTS AND APPARATUS; CLOCKS AND WATCHES; MUSICAL INSTRUMENTS; PARTS AND ACCESSORIES THEREOF',
         'SECTION XIX - ARMS AND AMMUNITION; PARTS AND ACCESSORIES THEREOF',
         'SECTION XX - MISCELLANEOUS MANUFACTURED ARTICLES',
         "SECTION XXI - WORKS OF ART, COLLECTORS' PIECES AND ANTIQUES"]
def openfile(fname):
    if fname[-4:] == 'xlsx':
        df = pd.read_excel(fname, sheet_name= 0, errors='ignore')
    elif fname[-4:] == '.csv':
        df = pd.read_csv(fname,encoding='latin1')
    return df
#processing for stopwords, alphabetic words, stemming
def preprocess(docs):
    # remove the string which is not alphabet
    docs = [''.join([i for i in str(s) if not re.search(r'[^a-zA-Z\s\,]', i)]) for s in docs]
    # tokenization
    docs1 = [nltk.word_tokenize(str(v)) for v in docs]
    # text pre-processing, including lower, alpha, short string and stop word removal, stemming, etc.
    docs2 = [[w.lower() for w in doc] for doc in docs1]
    docs2 = [[w for w in doc if len(w) >2] for doc in docs2]
    docs3 = [[w for w in doc if re.search('^[a-z]+$',w)] for doc in docs2]
    docs4 = [[w for w in doc if w not in stop_list] for doc in docs3]
    docs5 = [[stemmer.stem(w) for w in doc] for doc in docs4]
    docs6 = [[w for w in doc if w not in stop_list] for doc in docs5]
    return docs6
# import data to dataframe
df = openfile(test_file)
# convert the dataframe to array/list
test_docs = np.array(df[df.columns[0]]).tolist()
# preprocess the test_data
test_docs6 = preprocess(test_docs)
# Associate hs code numbers with the related tags 
num_1 = [0, 5, 14, 15, 24, 27, 38, 40, 43, 46, 49, 63, 67, 70, 71, 83, 85, 89, 92, 93, 96, 99]
hs_dict = {}
for i in range(len(num_1)-1):
    for j in range(num_1[i]+1,num_1[i+1]+1):
        hs_dict[j] = labels[i]
'''
3. use the trained naive Bayes classifier to predict the labels of the data
'''       
# add the 'label' column to dataframe
df["label"] = df.index
# Convert documents into dict representation. This is document-label representation
test_tf_vectors = [dictionary.doc2bow(doc) for doc in test_docs6]
# Store in test_data_as_dict
test_data_as_dict = [{id:1 for (id, tf_value) in vec} for vec in test_tf_vectors]
#For each file, classify and print the label.
for i in range(len(test_docs6)):
    # find the string contains ('h','s', 4 consecutive numbers)
    hs_code = re.search(r'[h]+\.*[s]+.*[0-9]{4}',str(test_docs[i]).lower())
    # if find such string, label it by hs code number.
    if hs_code:
        # get the top 2 number of hs code.
        hs_code_2 = re.split(r'([0-9]{4})',re.split(r'([h]+\.*[s]+)',str(test_docs[i]).lower())[2])
        # in case some string has more than one 'hs', we should get the one followed by numbers.
        if len(hs_code_2) == 1:
            # get the top 2 number of hs code.
            hs_code_3 = re.split(r'([0-9]{4})',re.split(r'([h]+\.*[s]+)',str(test_docs[i]).lower())[-1])
            # according to the top 2 number, add the label
            df["label"][i] = hs_dict[int(hs_code_3[1][:2])]
        else:
            # according to the top 2 number, add the label
            df["label"][i] = hs_dict[int(hs_code_2[1][:2])]
    else:
        # If there are no elements in the dictionary, means it cannot be classified.
        if test_data_as_dict[i]=={}:
            df["label"][i] = "none"
        else:
            df["label"][i] = classifier.classify(test_data_as_dict[i]) 
        
'''
4. Save the results
'''
# save the dataframe to csv.file
df.to_csv(output_file)