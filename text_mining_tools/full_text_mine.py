#!/usr/local/bin/python
# functions written by Aditya Nandy for Kulik Group
from text_mining_tools.query import Query
from text_mining_tools.article import Article
import pickle
import glob, os
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
    #                query1, query2, etc. Stored as pickle.
    # number_of_results is PER JOURNAL.
    # This function instantiates a query, then stores the query as a pickle.
    my_query = Query(basepath=basepath, keywords=keywords, 
                     elsevier_key=elsevier_key, journal_limit=journal_limit,
                     number_of_results=number_of_results,automate_download=automate_download)
    pickle_name = name_pickle(basepath, name=query_name)
    filename = open(str(basepath)+'/AnalyzedResults/'+str(pickle_name), 'wb')
    pickle.dump(my_query, filename)
    return my_query

def pickle_object(basepath, python_class, name=False, query=True):
    # name is the name of the query or article class that will be 
    #                written to basepath/AnalyzedResults/.
    #                If nothing is provided, will be query0,
    #                query1, query2, etc. Stored as picklee.
    pickle_name = name_pickle(basepath, name=query_name)
    filename = open(str(basepath)+'/AnalyzedResults/'+str(pickle_name), 'wb')
    pickle.dump(python_class, filename)
    return query

def VADER_analysis(sentences, keywords):
    ##### This helper function takes sentences that are already broken apart.
    ##### You should be able to already work with sentences assigned to an 
    ##### article class, but if you can write your own tokenizer if needed.
    # Input sentences should be a list of sentences. Readily provided by the
    #       Article class.
    # Input keywords should be a list of keywords.
    if not isinstance(keywords,list):
        keywords = [keywords]
    kw_found = False
    kw_in_list = []
    polarity_list = []
    for sent_num, sent in enumerate(sentences):
        if any([val in sent for val in keywords]):
            if any([('not '+val) in sent for val in keywords]):
                kw_found = False
            else:
                kw_in_list.append(sent_num)
                kw_found = True
    if kw_found: # Do not do analysis if kw not found.
        sid = SentimentIntensityAnalyzer()
        for sentence in sentences:
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

def name_pickle(basepath, name = False, query = True):
    # if query == False, assumes article.
    if name != False:
        existing_analyzed_files = glob.glob(str(basepath)+'/AnalyzedResults/'+str(name)+'.pickle')
        if len(existing_analyzed_files) != 0:
            maxval = 0
            for analyzed in existing_analyzed_files:
                analyzed_basename = os.path.basename(analyzed)
                temp = analyzed_basename.split('.pickle')[0]
                if '_' in temp:
                    current_val = temp.split('_')[1]
                    if current_val > maxval:
                        maxval = current_val
            name = str(name)+'_'+str(maxval+1)+'.pickle'
        if 'pickle' not in name:
            name += '.pickle'
        pickle_name = name
    else:
        if query:
            existing_analyzed_files = glob.glob(str(basepath)+'/AnalyzedResults/query_*pickle')
        else:
            existing_analyzed_files = glob.glob(str(basepath)+'/AnalyzedResults/article_*pickle')
        if len(existing_analyzed_files) == 0:
            if query:
                pickle_name = 'query_1.pickle'
            else:
                pickle_name = 'article_1.pickle'
        else:
            maxval = 0
            for analyzed in existing_analyzed_files:
                analyzed_basename = os.path.basename(analyzed)
                current_val = int(analyzed_basename.split('_')[1].split('.')[0])
                if current_val > maxval:
                    maxval = current_val
            if query:
                pickle_name = 'query_'+str(maxval+1)+'.pickle'
            else:
                pickle_name = 'article_'+str(maxval+1)+'.pickle'
    return pickle_name



