# Cargo Classification Project
## Short text classification (nltk NaiveBayes Classifier)
### Files in the folder
- `data/` 
  - `HS Code and Detailed Description.xlsx`
- `model/`
  - `Cargo_classifier.pickle`
  - `Dictionary.dic`
  - `Model.py`
- `ouput/`

- `gui/`
  - `Cargo_Analytics.py`
- `Main.py`

### Required packages
The code has been tested running under Python 3.7.4, with the following packages installed (along with their dependencies):
- pandas == 0.25.1
- numpy == 1.16.5
- gensim == 3.7.3
- smart_open == 1.8.0
- pyqt5 == 5.6
- pyinstaller == 3.5
### Running the code
#### Build Model
```
$ cd model
$ python Model.py (get the model file and dictionary file)
```
 
#### Classify data
```
$ python Main.py (get the results in output folder)
```
##### Note: when classifying new data
1.	Put the new data into data folder.
2.	Change the test_file and output_file names (Main.py). 

 
#### Create GUI (Graphical User Interface) application
```
$ cd gui
$ python Cargo_Analytics.py (using the model and dictionary we built)
```
 
#### Convert GUI.py to exe.file
Note: before converting, change the filename path:
``` 
$ cd gui
$ pyinstaller -F Cargo_Analytics.py
```

