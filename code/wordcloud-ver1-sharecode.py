#!/usr/bin/env python
# coding: utf-8
# FAPA OCT 2022
# For SLPA Survey 2022

# Import libraries and download datasets from NLTK library. You only need to 
# download these datasets once. 

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np 
import re
import string
from nltk.stem import WordNetLemmatizer
import nltk
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('omw-1.4')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
stopwords=stopwords.words('english')
lemmatizer = WordNetLemmatizer()

#####
# Our corpus is defined as responses from latinx postdocs and non-latinx postdocs. 
# We do the same pre-processing for both datasets, here we specify which dataset
# we are processing, and load the corresponding 'tsv' file. 
####

corpus='Not_ltx_q18'
q18_text=open(corpus+'.tsv','r')
test_q18=q18_text.readlines()


######
# pre-processing of the dataset. We changed all text to lowercase, remove all characters that were not letters.
# We also removed words that we determined were not informative, but were very common, such as 'monthly', 'week', 'per'
# We noticed that some answers refered to the same concept, so we used only one term to represent them. For example:
# 'partner' was used to represent 'spouse', 'wife', 'husband'. 
# We also removed all stopwords, as defined in the NLTK dataset. And we used a lemmatizer. 
######

train_q18=[]

for i in range(0, len(test_q18)): 
    review = test_q18[i].lower()
    review = re.sub('n/a', ' ', review)
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = re.sub('monthly', ' ', review)
    review = re.sub('month', ' ', review)
    review = re.sub('year', ' ', review)
    review = re.sub('weekly', ' ', review)
    review = re.sub('week', ' ', review)
    review = re.sub('etc', ' ', review)
    review = re.sub(' per ', ' ', review)
    review = re.sub('none', ' ', review)
    review = re.sub(' mo ', ' ', review)
    review = re.sub('xx', '', review)
    review = re.sub('stanford', '', review)
    review = re.sub(' ca ', '', review)
    review = re.sub('applicable', '', review)
    review = re.sub('child care', 'childcare', review)
    review = re.sub('daycare', 'childcare', review)
    review = re.sub('nanny', 'childcare', review)
    review = re.sub('spouse', 'partner', review)
    review = re.sub('aproximatly', '', review)
    review = re.sub('approximately', '', review)
    review = re.sub('wife', 'partner', review)
    review = re.sub('husband', 'partner', review)
    review = re.sub('co pay', 'copay', review)
    review = re.sub('copays', 'copay', review)
    review = re.sub('health care', 'healthcare', review)
    review = re.sub('tuiton', 'tuition', review)
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords)]
    review = ' '.join(review)
    train_q18.append(review)


######
# Create and save the wordcloud. 
# We chose a max of 100 words per wordcloud. 
# In the case of latinx postdocs, after pre-processing there were 62 unique words left, 
# while for non-latinx postdocs, there were 223 unique words.  
#####


text=' '.join(train_q18)

wordcloud = WordCloud(font_path = '/Library/Fonts/Arial Unicode.ttf', background_color="white", 
                      width=3000, height=2000, max_words=100).generate(text)

plt.figure(figsize=[30,20])
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig(corpus+'.png', facecolor='k', bbox_inches='tight')

######
#tf idf
#Here we just cound how many samples we had, and how many unique words were present.
######

tf_idf = TfidfVectorizer()
X_train_q18 = tf_idf.fit_transform(train_q18)
print("n_samples: %d, n_features: %d" % X_train_q18.shape)



# ## References 
#
# I used the following sites as reference on how to make the wordclouds and pre-processing of the data
# 
# https://www.analyticsvidhya.com/blog/2021/09/creating-a-movie-reviews-classifier-using-tf-idf-in-python/ \
# https://towardsdatascience.com/how-to-make-word-clouds-in-python-that-dont-suck-86518cdcb61f
#
#####


