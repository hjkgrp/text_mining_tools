'''
This script was used to mine the text thermal stability cases
from the extant literature, from HTML files. After identifying
the cases from these manuscripts, we manually extracted the 
TGA traces.
'''


from chemdataextractor.nlp.tokenize import ChemWordTokenizer
from chemdataextractor.nlp.pos import ChemCrfPosTagger
from chemdataextractor import Document
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import wordnet as wn
import chemdataextractor
import os
import re
import json



#### First, we define where our corpus is. In this specific case, I have defined a corpus location.
#### We also start the dictionary that we will fill up by manuscript
#### Lastly, we define keywords for TGA identification. We also define units.
basedir = '/Users/adityanandy/Documents/MIT/Kulik/TextMining/Papers/CoRECorpus/'
thermal_keyword_dict = {}
phrases_to_search_for = ['TGA','TG', 'thermogravimetric analysis', 'thermo-gravimetric analysis', 'thermal analysis',
                         'thermal gravimetric','thermal-gravimetric', 'thermo gravimetric', 
                         'thermalgravimetric analysis', 'weight loss', 'temperature range', 'mass loss','decomposition']
units = ['°C','° C','degC','°F','° F','degF', '°K','° K','degK', 'K']

for journal in os.listdir(basedir):
    if not os.path.isdir(basedir+journal):
        continue
    for article in os.listdir(basedir+'/'+journal):
        if 'html' not in article:
            continue
        header_counter = 0
        sent_counter = 0
        length_counter = 0
        found_intro = False
        header_list = []
        f = open(basedir+journal+'/'+article, 'rb')
        # Use chem data extractor to parse the file
        doc = Document.from_file(f)
        flag = False
        title = False
        search_results = []
        heading = 'nothing'
        # Look through the sections as identified by ChemDataExtractor
        for elem in doc.elements:
            if isinstance(elem,chemdataextractor.doc.text.Title):
                title = elem.text
            if isinstance(elem,chemdataextractor.doc.text.Heading):
                # Skip the sections that are the references or the links
                if ('additional links' in elem.text.lower()) or ('references' in elem.text.lower()):
                    flag = True
                else: 
                    flag = False
                if flag:
                    # Skip unnecessary sections
                    continue
                # Next identify introductions
                if 'intro' in elem.text.lower():
                    found_intro = True
                    # count sentences then move on, do not allow them to be mined.
                    print('==== INTRO ====')
                    heading = 'intro'
                    for i, sent in enumerate(elem.sentences):
                        sent_counter += 1
                        length_counter += len(sent.text)
                else:
                    heading = 'nothing'
                header_list.append(elem.text)
                header_counter += 1
            elif isinstance(elem,chemdataextractor.doc.text.Paragraph) and (heading=='intro'):
                for i, sentence in enumerate(elem.sentences):
                    sent_counter += 1
                    length_counter += len(sentence.text)
            # If not an introduction or an irrelevant section, perform analysis
            elif isinstance(elem,chemdataextractor.doc.text.Paragraph) and (heading == 'nothing'):
                for i, sent in enumerate(elem.sentences):
                    for j, kw in enumerate(phrases_to_search_for):
                        if re.search(kw, sent.text, re.IGNORECASE):
                            temp_flag = True
                            count_down_limit = 8
                            count_down_flag = 0
                            to_analyze = []
                            sentence_idx_list = []
                            # Check the nearest 8 sentences and store them in case they are useful later.
                            while count_down_flag <= count_down_limit:
                                try:
                                    new_sentence = elem.sentences[i+count_down_flag]
                                except:
                                    count_down_flag += 1
                                    continue
                                for unit in units:
                                    if unit in new_sentence.text:
                                        sentence_idx_list.append(sent_counter+i+count_down_flag)
                                        to_analyze.append(new_sentence.text)
                                count_down_flag += 1
                            if len(to_analyze) == 0:
                                # if there is nothing to analyze, continue
                                continue
                            search_results.append({'filename': journal+'/'+article,
                                                   'title': title,
                                                   'sentence': sent.text, 
                                                   'sentence_counter': sent_counter,
                                                   'keyword': kw,
                                                   'additional_sentences': to_analyze,
                                                   'additional_sentence_idxs':sentence_idx_list})
                    sent_counter += 1
                    length_counter += len(sent.text)
                    ### we now have everything we could need from this manuscript
        # Next, store the data in a dictionary. Each article has one dictionary tied to it.
        temp_dict = {'num_sections':header_counter, 'sections':header_list,'num_sentences':sent_counter,'num_char':length_counter, 'search_results':search_results,'intro':found_intro}
        thermal_keyword_dict[journal+'/'+article] = temp_dict

with open('thermal_decomposition_keyword_searches.json','w') as g:
    json.dump(thermal_keyword_dict,g)
