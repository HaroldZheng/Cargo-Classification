import sys
import os
import pandas as pd
import numpy as np
import gensim
import pickle
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QPushButton, QHBoxLayout, QVBoxLayout, QTableWidget, QAbstractItemView, QComboBox, QWidget, QFileDialog, QProgressDialog, QMessageBox, QTableWidgetItem, QDesktopWidget, QDialog, QHeaderView
stop_list = stopwords.words('english')
# add more stop words to remove the noise
stop_words = ['white', 'black', 'blue', 'red', 'green', 'yellow', 'brown', 'hide', 'skin','whole', 'cargo',
              'linear', 'densiti', 'free', 'children', 'part', 'ab', 'chines', 'addit', 'partial', 'order',
              'one', 'two', 'three', 'four', 'five', 'six', 'eight', 'nine', 'ten', 'group', 'packag', 'human',
              'whether', 'refin', 'high', 'puriti', 'natur', 'sweet', 'pcb', 'pct', 'gross', 'set', 'anim',
              'dri', 'spread', 'fresh', 'chill', 'frozen', 'raw', 'pure', 'low', 'mix', 'origin', 'day',
              'bos', 'konteyner', 'konteyn', 'contain', 'container', 'box', 'block', 'exclud', 'hot', 'cod',
              'fcl', 'total', 'empty', 'chapter', 'other', 'of', 'by', 'weight', 'medium', 'size', 'big',
              'weigh', 'code', 'new', 'net', 'pack', 'bag', 'pack', 'buyer', 'mark', 'number', 'type', 
              'round', 'case', 'long', 'short', 'bulk', 'metric', 'per', 'flexibl', 'basic', 'line','air',
              'open', 'made', 'safeti', 'outer', 'upper', 'product','i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 
              'viii', 'ix', 'x', 'xi', 'xii', 'xiii', 'xiv', 'xv', 'xvi', 'xvii', 'xviii', 'xix', 'xx', 'xxx']
stop_list.extend(stop_words) 
# labels 
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
# Associate hs code numbers with the related tags 
num_1 = [0, 5, 14, 15, 24, 27, 38, 40, 43, 46, 49, 63, 67, 70, 71, 83, 85, 89, 92, 93, 96, 99]
hs_dict = {}
for i in range(len(num_1)-1):
    for j in range(num_1[i]+1,num_1[i+1]+1):
        hs_dict[j] = labels[i]

class Example(QWidget):
    # initial header label
    choice_list = ['Raw_Cargo_Description']

    def __init__(self):
        '''
        initial settings
        '''
        super().__init__()
        self.initUI()
        self.center()

    # center the window
    def center(self):
        '''
        center the window
        '''
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
    
    # layout, add buttons 
    def initUI(self):
        '''
        ui initial settings
        '''
        # set window
        # setGeometry(left,top,width,height) 
        self.setGeometry(500, 500, 800, 600)
        # window title
        self.setWindowTitle('Cargo Classification')
        # horizontal box layout
        self.hbox = QHBoxLayout(self)
        # Vertical box layout
        self.vbox = QVBoxLayout(self)

        # an item-based table view 
        self.tableWidget = QTableWidget(0,0)
        # setGeometry(left,top,width,height) 
        self.tableWidget.setGeometry(20, 20, 300, 270)
        # Allow editing of table widget item only on double click event
        self.tableWidget.setEditTriggers(QAbstractItemView.DoubleClicked)

        # add table view to hbox layout
        self.hbox.addWidget(self.tableWidget)
        # add hbox to main window
        self.setLayout(self.hbox) 
        
        # button created with label
        self.bt1 = QPushButton('Input File', self)   
        self.bt2 = QPushButton('NLP Model', self)
        self.bt2_1 = QPushButton('Work_Progress_Model', self)
        self.bt3 = QPushButton('Output File', self)

        # connect the function to the button 
        self.bt1.clicked.connect(self.showfile)
        self.bt2.clicked.connect(self.runmodel)
        self.bt2_1.clicked.connect(self.detail)
        self.bt3.clicked.connect(self.savefile)       
        
        # add button to vbox layout
        self.vbox.addWidget(self.bt1)
        self.vbox.addWidget(self.bt2)
        self.vbox.addWidget(self.bt2_1)
        self.vbox.addWidget(self.bt3)        
        
        # add vbox view to hbox layout
        self.hbox.addLayout(self.vbox)    
        
        self.show()

    # allow users to select files and convert the file to dataframe
    def openfile(self):
        '''
        open file
        '''
        # provides a dialog that allow users to select files or directories. 
        fname = QFileDialog.getOpenFileName(self, 'open file', './',"Excel Files (*.csv *.xlsx)")
        if fname[0][-4:] == 'xlsx':
            # convert xlsx file to dataframe
            df = pd.read_excel(fname[0], sheet_name= 0, errors='ignore')
        elif fname[0][-4:] == '.csv':
            # convert csv file to dataframe
            df = pd.read_csv(fname[0],encoding='latin1')
        else:
            df = pd.DataFrame()
            # Message box reminds the user to open the file
            QMessageBox.information(self, 'Wrong message', 'Please open the excel file')
            
        return df
    
    # show the data in table view
    def showfile(self):  
        # set new dataframe to prepocess
        global df_new
        # using openfile function convert the file to dataframe
        df = self.openfile()
        # if data is not empty
        if len(df):
            # only catch the first column data
            df_new = pd.DataFrame(df, columns=[df.columns[0]])
            # count the rows
            nRows = len(df_new.index)
            # count the columns (default 1)
            nColumns = len(df_new.columns)
            # clear the previous data from table view before insert new data
            self.tableWidget.clear()
            # Set the number of columns 
            self.tableWidget.setColumnCount(nColumns)
            # Set the number of rows
            self.tableWidget.setRowCount(nRows)
            # Set the horizontal header labels using choice_list
            self.tableWidget.setHorizontalHeaderLabels(self.choice_list)
            # set columnwidth 
            self.tableWidget.setColumnWidth(0, 200)

            # progressDialog provides feedback on the progress of a slow operation
            progress = QProgressDialog(self)
            progress.setWindowTitle("Open file")  
            progress.setLabelText("Please wait...")
            progress.setCancelButtonText("Cancel")
            progress.setMinimumDuration(5)
           
            progress.setWindowModality(Qt.WindowModal)   
            # set the progress range
            progress.setRange(0, nRows)
            
            for i in range(nRows):
                for j in range(nColumns):
                    # Sets the item for the given row and column to item.
                    x = df_new.loc[i][j]
                    self.tableWidget.setItem(i, j, QTableWidgetItem(str(x)))
                progress.setValue(i)
                # if canceled, Message box remind the user "Failed to import data"
                if progress.wasCanceled():
                    QMessageBox.warning(self, "Warning", "Failed to import data")
                    break
            else:
                # Message box remind the user "Data import completed sucessfully"
                progress.setValue(nRows)
                QMessageBox.information(self, "Import Data", "Data import completed sucessfully")
    
    # processing for stopwords, alphabetic words, stemming
    def preprocess(self, docs):
        # remove the string which is not alphabet
        docs1 = [''.join([i for i in str(s) if not re.search(r'[^a-zA-Z\s\,]', i)]) for s in docs]
        # tokenization
        docs1 = [word_tokenize(str(v)) for v in docs1]
        # text pre-processing, including lower, alpha, short string and stop word removal, stemming, etc.
        docs2 = [[w.lower() for w in doc] for doc in docs1]
        docs2 = [[w for w in doc if len(w) >2] for doc in docs2]
        docs3 = [[w for w in doc if re.search('^[a-z]+$',w)] for doc in docs2]
        docs4 = [[w for w in doc if w not in stop_list] for doc in docs3]
        docs5 = [[stemmer.stem(w) for w in doc] for doc in docs4]
        docs6 = [[w for w in doc if w not in stop_list] for doc in docs5]
        return docs6

    # use the trained naive Bayes classifier to predict the labels of the data 
    def runmodel(self):
        '''
        run model
        '''
        # progressDialog provides feedback on the progress of a slow operation
        progress = QProgressDialog(self)
        progress.setWindowTitle("Run the Model")  
        progress.setLabelText("Please wait...")
        progress.setCancelButtonText("Cancel")
        progress.setMinimumDuration(5)          
        progress.setWindowModality(Qt.WindowModal)  
        progress.setRange(0, len(df_new))
        # list of raw data
        global test_docs
        test_docs = np.array(df_new[df_new.columns[0]]).tolist()
        
        # input/output file names
        '''
        #filename = os.path.join(os.path.dirname(sys.executable), 'Cargo_classifier.pickle')
        #filename_2 = os.path.join(os.path.dirname(sys.executable), 'dictionary.dic')
        #f = open(filename, 'rb')
        #dictionary = gensim.corpora.Dictionary.load(filename_2)
        '''
        f = open('../model/Cargo_classifier.pickle', 'rb')
        classifier = pickle.load(f)
        f.close()    
        dictionary = gensim.corpora.Dictionary.load('../model/dictionary.dic')

        # preprocess the test_data
        docs_preprocess = self.preprocess(test_docs) 
        # docs_preprocess_2 is for Detail_of_Cargo_Description
        global docs_preprocess_2
        docs_preprocess_2 = [[i for i in doc if i in dictionary.token2id] for doc in docs_preprocess ]

        # Convert documents into dict representation. This is document-label representation
        tf_vectors = [dictionary.doc2bow(doc) for doc in docs_preprocess]
        # Store in test_data_as_dict
        data_as_dict = [{id:1 for (id, tf_value) in vec} for vec in tf_vectors]
        # Set the number of columns
        self.tableWidget.setColumnCount(2)
        # add header laber to choice_list
        self.choice_list.append('Processed_Cargo_Description_Level1')
        # Set the horizontal header labels using choice_list
        self.tableWidget.setHorizontalHeaderLabels(self.choice_list)
        # Set column width
        self.tableWidget.setColumnWidth(1, 200)
        # add new column to dataframe
        df_new['Processed_Cargo_Description_Level1'] = 'N/A'
        
        #For each file, classify and print the label.
        for i in range(len(df_new)):
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
                    self.tableWidget.setItem(i, 1, QTableWidgetItem(hs_dict[int(hs_code_3[1][:2])]))
                    df_new['Processed_Cargo_Description_Level1'][i] = hs_dict[int(hs_code_3[1][:2])]
                else:
                    # according to the top 2 number, add the label
                    self.tableWidget.setItem(i, 1, QTableWidgetItem(hs_dict[int(hs_code_2[1][:2])]))
                    df_new['Processed_Cargo_Description_Level1'][i] = hs_dict[int(hs_code_2[1][:2])]
            else:
                if data_as_dict[i]=={}:
                    # If there are no elements in the dictionary, means it cannot be classified.
                    self.tableWidget.setItem(i, 1, QTableWidgetItem('none'))
                    df_new['Processed_Cargo_Description_Level1'][i] = "none"
                else:
                    self.tableWidget.setItem(i, 1, QTableWidgetItem(classifier.classify(data_as_dict[i])))
                    df_new['Processed_Cargo_Description_Level1'][i] = classifier.classify(data_as_dict[i])      
            progress.setValue(i)
            # if canceled, message box remind "Failed to run the model"
            if progress.wasCanceled():
                QMessageBox.warning(self, "Warning", "Failed to run the model")
                break
        else:
            # Message box remind the user "Processing sucessfully"
            progress.setValue(len(df_new))
            QMessageBox.information(self, "Run the Model", "Processing sucessfully")
    
    # show the details (after preprocessing, find the words still in the dictionary)
    def detail(self):
        '''
        show the details
        '''
        # progressDialog provides feedback on the progress of a slow operation
        progress = QProgressDialog(self)
        progress.setWindowTitle("show the details")  
        progress.setLabelText("Please wait...")
        progress.setCancelButtonText("Cancel")
        progress.setMinimumDuration(5)          
        progress.setWindowModality(Qt.WindowModal)  
        progress.setRange(0, len(df_new))
        # Set the number of columns
        self.tableWidget.setColumnCount(3)
        # add header laber to choice_list
        self.choice_list.append('Detail_of_Cargo_Description')
        # Set the horizontal header labels using choice_list
        self.tableWidget.setHorizontalHeaderLabels(self.choice_list)
        # Set column width
        self.tableWidget.setColumnWidth(2, 200)  
        # add new column to dataframe
        df_new['Detail_of_Cargo_Description'] = 'N/A' 
        #For each file, show the details.  
        for i in range(len(df_new)):
            # find the string contains ('h','s', 4 consecutive numbers)
            hs_code = re.search(r'[h]+\.*[s]+.*[0-9]{4}',str(test_docs[i]).lower())
            # if find such string, get hs code number.
            if hs_code:
                # get the top 2 number of hs code.
                hs_code_2 = re.split(r'([0-9]{4})',re.split(r'([h]+\.*[s]+)',str(test_docs[i]).lower())[2])
                # in case some string has more than one 'hs', we should get the one followed by numbers.
                if len(hs_code_2) == 1:
                    # get the top 2 number of hs code.
                    hs_code_3 = re.split(r'([0-9]{4})',re.split(r'([h]+\.*[s]+)',str(test_docs[i]).lower())[-1])
                    self.tableWidget.setItem(i, 2, QTableWidgetItem(str(hs_code_3[1][:2])))
                    df_new['Detail_of_Cargo_Description'][i] = str(hs_code_3[1][:2])
                else:
                    # get the top 2 number of hs code.
                    self.tableWidget.setItem(i, 2, QTableWidgetItem(str(hs_code_2[1][:2])))
                    df_new['Detail_of_Cargo_Description'][i] = str(hs_code_2[1][:2])
            else:
                self.tableWidget.setItem(i, 2, QTableWidgetItem(str(docs_preprocess_2[i])))
                df_new['Detail_of_Cargo_Description'][i] = str(docs_preprocess_2[i])
            progress.setValue(i)
            # if canceled, message box remind "Failed to show the details"
            if progress.wasCanceled():
                QMessageBox.warning(self, "Warning", "Failed to show the details")
                break
        else:
            # Message box remind the user "Processing sucessfully"
            progress.setValue(len(df_new))
            QMessageBox.information(self, "Run the Model", "Processing sucessfully")
    
    # save the dataframe to csv file
    def savefile(self):
        '''
        save file
        '''
        df_output = df_new
        # Uses the file name selected by the user to save the file
        savePath = QFileDialog.getSaveFileName(self, "Export Data","./", "CSV files (*.csv)")
           
        if savePath[0]:
            df_output.to_csv((savePath[0]), index = 0)
            QMessageBox.information(self, "Export Data", "Data export completed sucessfully")
        else:
            QMessageBox.warning(self, "Warning", "Failed to save file")
        
        

        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())