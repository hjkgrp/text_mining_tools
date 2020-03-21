#!/usr/local/bin/python
from text_mining_tools.query import Query
from text_mining_tools.article import Article
import pickle
import glob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

"""
DO NOT try to automate ACS article downloading. If you do,
MIT libraries will get in trouble and our privileges will
be removed. It violates the ACS license agreement. I have
placed precautions on this to specifically not download
ACS papers in an automated way. If you have a few papers
that you would like to download, you can download them by
hand using the file structure outlined in article.py.
Doing so will allow you to instantiate article classes for
future mining.

This file contains the necessary functions for mining 
full texts based on keywords, and doing additional analysis
on top of the actual texts. Do NOT put any analysis tools
in the article class as analysis is person dependent!
"""

def execute_query(basepath, keywords, elsevier_key=None,journal_limit=False, \
                  number_of_results=10000, query_name=False, automate_download=False):
    # query_name is the name of the query class that will be 
    #                written to basepath/AnalyzedResults/.
    #                If nothing is provided, will be query0,
    #                query1, query2, etc. Stored as json.
    # number_of_results is PER JOURNAL.
    # This function instantiates a query, then stores the query as a JSON.
    my_query = Query(basepath=basepath, keywords=keywords, 
                     elsevier_key=elsevier_key, journal_limit=journal_limit,
                     number_of_results=number_of_results,automate_download=automate_download)
    pickle_name = name_pickle(query_name=query_name)
    filename = open(str(basepath)+'/AnalyzedResults/'+str(pickle_name), 'wb')
    pickle.dumps(my_query, filename)

def VADER_analysis(sentences, keywords):
    ##### This helper function takes sentences that are already broken apart.
    ##### You should be able to already work with sentences assigned to an 
    ##### article class, but if you can write your own tokenizer if needed.
    # Input sentences should be a list of sentences. Readily provided by the
    #       Article class.
    # Input keywords should be a list of keywords.
    kw_found = False
    kw_in_list = []
    polarity_list = []
    for sent_num, sent in enumerate(sent_text):
        if any([val in sent for val in keywords]):
            if any([('not '+val) in sent for val in keywords]):
                kw_found = False
            else:
                kw_in_list.append(sent_num)
                kw_found = True
    if kw_found: # Do not do analysis if kw not found.
        sid = SentimentIntensityAnalyzer()
        for sentence in sent_text:
            ss = sid.polarity_scores(sentence)
            polarity_list.append(ss['compound'])
    # Returns two lists, one that contains indices of keyword containing
    # the keywords supplied, and the second containing the polarities.
    return kw_in_list, polarity_list

#### Function to tag words with parts of speech,
#### Can be helpful if you are looking to find
#### words that are actions, etc.
def preprocess(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent

def name_pickle(query_name = False):
    if query_name:
        if 'pickle' not in query_name:
            query_name += '.pickle'
        pickle_name = query_name
    else:
        existing_analyzed_files = glob.glob(str(basepath)+'/AnalyzedResults/query_*pickle')
        if len(existing_analyzed_files) == 0:
            pickle_name = 'query_1.pickle'
        else:
            maxval = 0
            for analyzed in existing_analyzed_files:
                current_val = int(analyzed.split('_')[1].split('.')[0])
                if current_val > maxval:
                    maxval = current_val
            pickle_name = 'query_'+str(maxval+1)+'.pickle'
    return pickle_name



