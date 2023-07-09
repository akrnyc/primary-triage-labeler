import pandas as pd

data = pd.read_csv('minintdb.csv',
                   usecols = ['INC_KEY',
                              'YOADMIT',
                              'DCODEDESCR',
                              'label'])

data.label.value_counts()

data.YOADMIT.value_counts()
