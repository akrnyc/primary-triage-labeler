import pandas as pd
from preprocessing import doc_preprocess
from sklearn.preprocessing import LabelBinarizer
from transformers import pipeline

data = pd.read_csv('minintdb.csv',
                   index_col = 'INC_KEY',
                   usecols = ['INC_KEY',
                              'YOADMIT',
                              'DCODEDESCR',
                              'label'])

data.label.value_counts()
data.YOADMIT.value_counts()

# process target, y
lb = LabelBinarizer()
lb.fit(data['label'].values)
lb.classes_
target = lb.transform(data['label'])

#process features, X
data['corpus'] = data['DCODEDESCR'].apply(doc_preprocess, trim='lemma')

extractor = pipeline(model = "bart-base", 
                     task = "feature-extraction",
                     framework = "tf",
                     ) #https://huggingface.co/facebook/bart-base

data['features'] = data['corpus'].apply(extractor)
