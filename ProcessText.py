#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os,re,nltk
import numpy as np

nltk.data.path.append(os.path.join(os.getcwd(), "nltk_data"))

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download("stopwords")
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('cmudict')

from nltk.stem.wordnet import WordNetLemmatizer, SnowballStemmer
from nltk.corpus import cmudict

from collections import Counter, defaultdict

from bs4 import BeautifulSoup

def pattern_replace(pat,strrep,str):
    # replace all strings with the pattern "pat", in str, with strrep
    s = re.search(pat, str)
    if s != None:
        inipos = s.span()[0]
        endpos = s.span()[1]
        str_ret = str.replace(str[inipos:endpos],strrep)
    else:
        str_ret = str
    return str_ret

def print_html_header_paragraph(filepath):
    # Open the HTML file and create a BeautifulSoup Object
    with open(filepath) as f:
        page_content = BeautifulSoup(f, 'lxml')    
    # Print only the text from all the <h2> and <p> tags inside the <div> tags that have a class="section" attribute
    for section in page_content.find_all('div', class_='section'):
        header = section.h2.get_text()
        print(header)
        paragraph = section.p.get_text()
        print(paragraph)

def lemmatize_text(all_texts):
    # use Porter stemmer
    # preprocessing: normalize, remove punctuations and stop words
    # input: list of sentences
    # output: list of processed sentences
    stopwds = stopwords.words('english')
    stmr = PorterStemmer()
    all_texts_2 = []
    for text in all_texts:
        txt = text.lower() # convert to lower case
        txt = word_tokenize(txt) # tokenize
        txt2 = [tx for tx in txt if re.match(r'[^a-zA-Z0-9]', tx) == None ] # remove punctuations 
        txt2 = [stmr.stem(tx) for tx in txt2 if tx not in stopwds ] # remove stop words and stemming
        all_texts_2.append(txt2)
    return all_texts_2

def stem_text(all_texts, pos='n'):
    # use general lemmetization
    # preprocessing: normalize, remove punctuations and stop words
    # input: 
    ## all_texts: list of sentences to be processed
    ## pos: class of the word to be lemmatized, default 'n'
    # output: list of processed sentences
    stopwds = stopwords.words('english')
    lmtz = WordNetLemmatizer()
    all_texts_2 = []
    for text in all_texts:
        txt = text.lower() # convert to lower case
        txt = word_tokenize(txt) # tokenize
        txt2 = [tx for tx in txt if re.match(r'[^a-zA-Z0-9]', tx) == None ] # remove punctuations 
        txt2 = [lmtz.lemmatize(tx, pos) for tx in txt2 if tx not in stopwds ] # remove stop words and stemming
        all_texts_2.append(txt2)
    return all_texts_2

def sentence_count_naive(text):
    return len([w for w in text if w == '.'])

def sentence_count_regex(text):
    return len(re.compile(r'[^\.]+\.').finditer(text))

def sentence_count_token(text):
    return len(sent_tokenize(text))

def word_count_naive(sent):
    return len([w for w in sent.replace('.',' ') if w == ' '])

def word_count_regex(sent,nonumber=0):
    sent2 = sent.replace('.',' ')
    sent_compile = re.compile(r'[.+(\s\d+)?\s]+') if nonumber == 1 else re.compile(r'[.+\s]+')
    return len(sent_compile.finditer(sent2))

def word_count_token(sent):
    return len(word_tokenize(sent))

def syllable_count(word):
    d = cmudict.dict()
    try:
        return np.min([len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]])
    except KeyError:
        #if word not found in cmudict
        return _syllables(word)

def _syllables(word):
#referred from stackoverflow.com/questions/14541303/count-the-number-of-syllables-in-a-word
    count = 0
    vowels = 'aeiouy'
    word = word.lower()
    if word[0] in vowels:
        count +=1
    for index in range(1,len(word)):
        if word[index] in vowels and word[index-1] not in vowels:
            count +=1
    if word.endswith('e'):
        count -= 1
    if word.endswith('le'):
        count+=1
    if count == 0:
        count +=1
    return count

def hard_word_count(sent):
    # TO DO
    return len([w for w in word_tokenize(sent) if syllable_count(WordNetLemmatizer().lemmatize(w, pos='v')) >= 3])

# Below are two readability test indices: Flesch–Kincaid Grade and Gunning-Fog Grade
def flesch_index(text):
    # TO DO
    number_of_sentences, sents = sentence_count(text)
    number_of_words = np.sum([len(word_tokenize(sent)) for sent in sents])
    number_of_syllabi = np.sum([np.sum([syllable_count(w) for w in word_tokenize(sent)])\
                                for sent in sents])
    print('There are ' + str(number_of_sentences) + ' sentences in the input text.')
    print('There are ' + str(number_of_words) + ' words in the input text.')
    print('There are ' + str(number_of_syllabi) + ' syllabi in the input text.')
    return 0.39*(number_of_words/number_of_sentences) + \
        11.8*(number_of_syllabi/number_of_words) - 15.59
        
def fog_index(text):
    # TO DO
    number_of_sentences, sents = sentence_count(text)
    number_of_words = np.sum([len(word_tokenize(sent)) for sent in sents])
    number_of_hard_words = np.sum([len([w for w in word_tokenize(sent) if \
            syllable_count(WordNetLemmatizer().lemmatize(w, pos='v')) >= 3]) for sent in sents])
    print('There are ' + str(number_of_sentences) + ' sentences in the input text.')
    print('There are ' + str(number_of_words) + ' words in the input text.')
    print('There are ' + str(number_of_hard_words) + ' hard words in the input text.')
    return 0.4*(number_of_words/number_of_sentences) + \
        40*(number_of_hard_words/number_of_words)

# clean text to prepare for the conversion to bag-of-words
# lowercase, lemmatized w.r.t. nouns and verbs
# no digits, no stopwords
# stemmized and contains at least three letters
def clean_text(txt):
    lemm_txt = [ wnl.lemmatize(wnl.lemmatize(w.lower(),'n'),'v') \
                for w in word_tokenizer.tokenize(txt) if \
                w.isalpha() and w not in stop_words ]
    return [ sno.stem(w) for w in lemm_txt if w not in stop_words and len(w) > 2 ]

# convert a list of words into a "bag_of_words" dictionary, including 
def bag_of_words(words):
    bags = dict()
    for w in words:
        #bags = bags.update({w: bags.get(w)+1}) if w in bags.keys() else bags.setdefault({w: 1})
        bags[w] = bags.get(w) + 1 if w in bags.keys() else 1
    return bags


# In[ ]:


# request and process/parse html text from the webpages
# Example 1: Amazon wikipage: https://en.wikipedia.org/wiki/Amazon_(company)
from bs4 import BeautifulSoup
import requests

import pandas as pd
import numpy as np
import re

def pattern_replace(pat,strrep,str):
    # replace all strings with the pattern "pat", in str, with strrep
    s = re.search(pat, str)
    if s != None:
        inipos = s.span()[0]
        endpos = s.span()[1]
        str_ret = str.replace(str[inipos:endpos],strrep)
    else:
        str_ret = str
    return str_ret

# Create a Response object
r = requests.get('https://en.wikipedia.org/wiki/Amazon_(company)')

# Get HTML data
html_data = r.text

# Create a BeautifulSoup Object
page_content = BeautifulSoup(html_data,'html.parser')

# Find financial table
wikitable = page_content.find('table', {'class': 'wikitable float-left'})

# Find all column titles
wikicolumns = wikitable.tbody.findAll('tr')[0].findAll('th')

# Loop through column titles and store into Python array
df_columns = []
for tag in wikicolumns:
    txt = tag.get_text(strip=True, separator=" ")
    df_columns.append(pattern_replace('\[.+\]','',txt))
print(df_columns)

# Loop through the data rows and store into Python array
df_data = []
for row in wikitable.tbody.findAll('tr')[1:]:
    row_data = []
    for td in row.findAll('td'):
        txt = td.get_text(strip=True, separator=" ")
        row_data.append(pattern_replace('\[.+\]','',txt))
    df_data.append(np.array(row_data))

# Print financial data in DataFrame format and set `Year` as index
dataframe = pd.DataFrame(data=df_data, columns=df_columns)
dataframe.set_index('Year', inplace=True)
dataframe


# In[ ]:




