#!/usr/bin/env python
# coding: utf-8

# In[163]:



from nltk.corpus import names

print(names.words()[:10])

print(len(names.words()))



from nltk.tokenize import word_tokenize
sent = '''I am Hossein Moharrer Derakhsahandeh .
          I'm learning Python Machine Learning By Dr. Rahmani,
         in the Fall of 1400. don't forget to send $20 your exam brfore 12:00 AM O'clock I AM Fresh in America.'''

print(word_tokenize(sent))


# In[170]:



sent2 = 'I am from I.R.A.N and have been to U.K. and U.S.A. , $20 in America , worked at P.T.K or Pars.Telephone.Kar P.T.Po p.t.kk .0.Pp. .Ka. Rr'
print(word_tokenize(sent2))


# In[171]:



import spacy

nlp = spacy.load('en_core_web_sm')
tokens2 = nlp(sent2)

print([token.text for token in tokens2])


# In[129]:



from nltk.tokenize import sent_tokenize
print(sent_tokenize(sent))


# In[130]:


import nltk
tokens = word_tokenize(sent)
print(nltk.pos_tag(tokens))


# In[131]:


nltk.help.upenn_tagset('PRP')
nltk.help.upenn_tagset('VBP')
nltk.help.upenn_tagset('JJ')
nltk.help.upenn_tagset('NNP')
nltk.help.upenn_tagset('CD')
nltk.help.upenn_tagset('DT')


# In[132]:


print([(token.text, token.pos_) for token in tokens2])


# In[173]:



tokens3 = nlp('The book written by Hayden Liu in 2020 was sold at $30 in america ,20 in IRAN')
print([(token_ent.text, token_ent.label_) for token_ent in tokens3.ents])


# In[150]:


from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()
porter_stemmer.stem('machines')
porter_stemmer.stem('learning')


# In[97]:



from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lemmatizer.lemmatize('machines')

