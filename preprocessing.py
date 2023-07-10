import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

#Remove emojis and unicode chars
#fxn from https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python

def deEmojify(text):
    regex = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regex.sub(r'',text)


def deSymbolify(text):
    regex = re.compile('[^a-zA-Z]')
    return regex.sub(r'', text)

def dePunctify(text):
    regex = re.compile('[^\w\s]')
    return regex.sub(r' ', text)


def doc_preprocess(document, trim=None):
    #function check
    if trim not in ['lemma', 'stem', None]:
        return print("Invalid token trimmer arg. Select stem, lemma, or None.")

    #convert to lowercase, remove punctuation
    document = dePunctify(document.lower())
    
    #tokenize
    tokens = word_tokenize(document, language='english', preserve_line=False)

    #remove stop words
    stopword = stopwords.words('english')
    tokens = [x for x in tokens if x not in stopword]

    #remove symbols, non-ascii, digit-inclusive strings, and too long and too short tokens
    tokens = [deSymbolify(deEmojify(word)) for word in tokens]
    tokens = [word for word in tokens if word.isascii()==True]
    tokens = [word for word in tokens if not any(ch.isdigit() for ch in word)]
    tokens = [word for word in tokens if (len(word) >= 4 and len(word) < 12)]
    
    if trim == 'stem':
        #stem tokens
        stemmer = SnowballStemmer("english")
        return [stemmer.stem(y) for y in tokens]
    
    elif trim == 'lemma':
        #lemmatize tokens
        wnl = WordNetLemmatizer()
        return [wnl.lemmatize(y) for y in tokens]

    elif trim == None:
        #return tokens
        return tokens

    else:
        print("Invalid token trimmer arg. Select stem, lemma, or None.")
