'''
This script was used to mine the text solvent removal stability labels
from the extant literature, from HTML files.
'''


from chemdataextractor.nlp.tokenize import ChemWordTokenizer
from chemdataextractor.nlp.pos import ChemCrfPosTagger
from chemdataextractor import Document
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import wordnet as wn
import chemdataextractor
from textblob import TextBlob
import os
import re
from itertools import compress
import dill as pickle
import json
import jsonpickle
from chemdataextractor.nlp.tokenize import ChemWordTokenizer
import stanza
nlp = stanza.Pipeline('en')

#### First, we define where our corpus is. In this specific case, I have defined a corpus location.
#### We also start the dictionary that we will fill up by manuscript
#### Lastly, we define keywords for collapse, stability, and solvent.

basedir = '/Users/adityanandy/Documents/MIT/Kulik/TextMining/Papers/CoRECorpus/'
solvent_keyword_dict = {}
single_word_collapse = ['collaps', 'deform','amorph','blockage','degrad','transform','unstable','instability','destroy', 'one step weight', 
                        'one-step weight','one‐step weight', 'single-step weight','single‐step weight','single step weight','one weight']
single_word_stable = ['stability',' stable','integrity','preserv','transparency','crystallinity','coordinatively unsaturat','porosity',
                      'microporosity','retain','maintain','two step weight','two-step weight','two‐step weight','second weight','third weight','two weight']
single_word_solvent = ['solvent','solvate', 'guest', 'desolv','remov','capillary','activat','evacuat','dehydrat','eliminat', 'lose', 'losing',
                       'water','DMF', 'DMA','DEF','H2O','EtOH','MeOH','ethanol','methylamine','diamine','formamide','pyrrolidone']


def organic_linker_phrase(temp_sent):
    '''
    This identifies when there is an organic linker decomposition phrase.
    '''
    if (('decomposition of the organic' in temp_sent) or ('decomposition of the linker' in temp_sent) 
        or ('decomposition of the linker' in temp_sent) or ('linker combustion' in temp_sent)):
        linker_decomposed = True
    else:
        linker_decomposed = False
    return linker_decomposed

def water_phrase(temp_sent):
    '''
    This identifies when there is an water related phrase --> for water stability identification to eliminate false positives.
    '''
    if (('soak' in temp_sent) or ('suspend' in temp_sent) or ('boil' in temp_sent)):
        water = True
    elif (('water stability' in temp_sent) or ('water-stability' in temp_sent) or ('water stable' in temp_sent) or 
        ('water-stable' in temp_sent) or ('exposure to water' in temp_sent) or ('stable to water' in temp_sent) or 
        ('stable in water' in temp_sent) or ('water unstable' in temp_sent) or ('water instability' in temp_sent) or 
        ('stability to water' in temp_sent) or ('water treatment' in temp_sent) or ('hydrothermal stability' in temp_sent)):
        water = True
    else:
        water = False
    return water

def air_phrase(temp_sent):
    '''
    This identifies when there is an air related phrase --> for air stability identification to eliminate false positives.
    '''
    if (('air stability' in temp_sent) or ('air-stability' in temp_sent) or ('air stable' in temp_sent) or 
        ('air-stable' in temp_sent) or ('exposure to air' in temp_sent) or ('stable to air' in temp_sent) or 
        ('stability to air' in temp_sent)):
        air = True
    else:
        air = False
    return air

def check_fake_loss(temp_sent):
    '''
    This identifies when a "loss" keyword is not loss of solvent, but rather loss of crystallinity.
    '''
    if (('loss of crystallinity' in temp_sent) or ('crystallinity loss' in temp_sent) or ('crystallinity is lost' in temp_sent) or 
        ('loss of porosity' in temp_sent) or ('loss of microporosity' in temp_sent) or ('porosity loss' in temp_sent) or ('porosity is lost' in temp_sent)):
        fake = True
    else:
        fake = False
    return fake

def check_fake_kw(temp_sent):
    '''
    This identifies when a keyword is identified for the wrong reason. For instance, "activation" can refer to bond activation.
    Additionally, decomposition can occur on the guest or the solvent, and we want to root out those false positives.
    '''
    bad = False
    if ('activ' in temp_sent) and ('carbon' in temp_sent):
        bad = True
    if ('building block' in temp_sent) or ('d-block' in temp_sent) or ('d block' in temp_sent):
        bad = True
    if ('unsaturated bond' in temp_sent):
        bad = True
    if ('activ' in temp_sent) and (('bond activation' in temp_sent) or ('unsaturated bonds' in temp_sent) or 
       ('activated bonds' in temp_sent) or ('C-H activation' in temp_sent) or ('C-C activation' in temp_sent) or 
       ('C=C activation' in temp_sent) or ('activated C–H' in temp_sent) or ('small molecule activation' in temp_sent) or
       ('oxygen activation' in temp_sent)):
        bad = True 
    if ('deactivation' in temp_sent):
        bad = True
    if ('inactivation' in temp_sent):
        bad = True
    if (('decomposition product' in temp_sent) or ('solvent decomposition' in temp_sent) or ('guest decomposition' in temp_sent)
        or ('solvent decompose' in temp_sent) or ('guest decompose' in temp_sent)):
        bad = True
    if ('stable' in temp_sent) and ('homogeneous catalyst' in temp_sent):
        bad = True
    return bad

def check_dependencies_given_solvent(sent, collapse, stable):
    # This function should take in a sentence and evaluate whether it has both
    # solvent and stability keywords, using dependencies to determine relationships between them.
    doc = nlp(sent)
    solvent_removal_flag = 0
    reason = 'default'
    innerflag = False
    checked_stability = set()
    checked_collapse = set()
    checked_words = []
    double_negative = False
    ### This is a list of negation words that can help us identify cases that are negated (e.g. "never stable")
    negation_words = ['no','not','never','fail','miss','missed','missing','lack','lacked','lacking','slow','decrease','decreased','decreases','decreasing',
                       'lose','loses','losing','loss','lost','without','absence','minimal','poor','diminish','diminishing','diminishes','diminished',
                       'prevent','prevented','preventing','prevents','difficulty','difficult','mitigate','mitigates','mitigating','mitigated',
                       'challenging','challenges','challenged','challenge','reduce','reduces','reducing','reduced']

    ### It is hard to determine the step that corresponds to solvent removal stability WITHIN a TGA sentence. Thus we will skip cases 
    ### where that may happen.
    TG_curves = ['TGA', 'TG', 'thermogravimetric', 'thermal analysis', 'thermal gravimetric','thermal-gravimetric','thermal-gravimetric',
                'thermal‐gravimetric','gravimetric', 'weight loss']
    for sentence in doc.sentences:
        negation = False
        TGA_flag = False
        if any([word in sentence.text for word in TG_curves]):
            TGA_flag = True
            if ('weight' in collapse) or ('weight' in stable):
                # These are cases where we know the TGA result confidently.
                TGA_flag = False
        ### Here, we have performed dependency parsing and we are unwrapping the dependencies.
        dependencies = sentence.dependencies
        for dependency in dependencies:
            check1, check2 = False, False # Make sure both elements of dependency can be parsed
            if isinstance(dependency[0].text,str):
                check1 = True
                word1 = dependency[0].text
            if isinstance(dependency[2].text,str):
                check2 = True
                word2 = dependency[2].text 
            if not (check1 and check2):
                # Sometimes the dependency is a root word, and thus not tied to anything
                continue
            if (((word1 == 'stability') or (word1 == 'stable') or (word2 == 'stability') or (word2 == 'stable')) and
                ((word1 == 'water') or (word2 == 'water'))):
                # We find that the sentence is talking about water stability --> beyond the simple phrases we identified before.
                solvent_removal_flag = -100
                reason = 'water stability'
                break
            elif (((word1 == 'stability') or (word1 == 'stable') or (word2 == 'stability') or (word2 == 'stable')) and
                ((word1 == 'air') or (word2 == 'air'))):
                # We find that the sentence is talking about air stability --> beyond the simple phrases we identified before.
                solvent_removal_flag = -100
                reason = 'air stability'
                break
            elif ((('degrad' in word1) or ('degrad' in word2)) and ((word1 == 'solvent') or (word2 == 'solvent') or (word1 == 'guest') or (word2 == 'guest'))):
                # We find that the sentence is talking about solvent molecule stability (e.g. DMF degradation).
                solvent_removal_flag = -100
                reason = 'solvent molecule stability'
                break
            elif ((('collaps' in word1) or ('collaps' in word2)) and ((word1 == 'above') or (word2 == 'above') or (word1 == 'at') or (word2 == 'at') or (word1 == 'after') or (word2 == 'after'))):
                # Multistep procedure identified --> cannot parse these and identify things well.
                solvent_removal_flag = 0
                reason = 'TGA sentence, rule based parsing will fail for multistep procedure'
                break
            elif (((('loss' in word1) or ('loss' in word2) or ('lose' in word1) or ('lose' in word2) or ('lost' in word1) or ('lost' in word2)) and (('molecule' in word1) or ('molecule' in word2)))):
                # Skip the dependency that is indicating loss of solvent.
                solvent_removal_flag = 0
                reason = 'solvent stability confounding loss keywords'
                continue 
            if ((word1.strip() in negation_words) or (word2.strip() in negation_words)) and not ((word1.strip() in negation_words) and (word2.strip() in negation_words)):
                if (len(collapse) > 0 and len(stable) > 0):
                    # If both stable and collapse keywords, we rely on stability negation (e.g. not stable, not crystalline...)
                    if ((any([val in word1.strip() for val in stable]) and any([word1.strip().startswith(val) for val in stable])) or 
                        (any([val in word2.strip() for val in stable]) and any([word2.strip().startswith(val) for val in stable]))):
                        solvent_removal_flag = -1 # this tells us the word was initially stable, but then was negated (e.g. not crystalline).
                        reason = 'negation with stable kw'
                        checked_stability |= set([val for val in stable if val in word2.strip()])
                        checked_words.append(word1)
                        checked_words.append(word2)
                else:
                    if ((any([val in word1.strip() for val in stable]) and any([word1.strip().startswith(val) for val in stable])) or 
                        (any([val in word2.strip() for val in stable]) and any([word2.strip().startswith(val) for val in stable]))):
                        solvent_removal_flag = -1 # this tells us the word was initially stable, but then was negated (e.g. not crystalline).
                        reason = 'negation with stable kw'
                        checked_stability |= set([val for val in stable if val in word2.strip()])
                        checked_words.append(word1)
                        checked_words.append(word2)
                    elif ((any([val in word1.strip() for val in collapse]) and any([word1.strip().startswith(val) for val in collapse])) or 
                        (any([val in word2.strip() for val in collapse]) and any([word2.strip().startswith(val) for val in collapse]))):
                        solvent_removal_flag = 1 # this tells us the word was initially unstable, but then was negated (e.g. not collapsed)
                        reason = 'negation with collapse kw'
                        checked_collapse |= set([val for val in collapse if val in word2.strip()])
                        checked_words.append(word1)
                        checked_words.append(word2)
            elif ((word1.strip() in negation_words) and (word2.strip() in negation_words)):
                double_negative = True
        if double_negative:
            if len(list(checked_stability))>0:
                if (['los' in val for val in list(checked_words) if isinstance(val, str)].count(True)>1):
                    solvent_removal_flag = -1
                    reason = 'two loss keywords, single negation with stability kw'
                else:
                    solvent_removal_flag = 1
                    reason = 'double negation with stability kw'
            elif len(list(checked_collapse))>0:
                print(checked_collapse)
                if (['los' in val for val in list(checked_words) if isinstance(val, str)].count(True)>1):
                    solvent_removal_flag = 1
                    reason = 'two loss keywords, single negation with collapse kw'
                else:
                    solvent_removal_flag = -1
                    reason = 'double negation with collapse kw'
            else:
                solvent_removal_flag = 0
                reason = 'unclear double negative'
    if not TGA_flag:
        if solvent_removal_flag == 0:
            if len(collapse) > 0 and len(stable) > 0:
                # By default, will keep solvent_removal_flag of 0, hard to deconvolute
                reason = 'has both collapse and stability kw'
            elif len(collapse) > 0 and len(stable) == 0:
                # Assign something as unstable
                solvent_removal_flag = -1
                reason = 'has collapse kw'
            elif len(stable) > 0 and len(collapse) == 0:
                # Assign something as stable
                solvent_removal_flag = 1
                reason = 'has stable kw'
    else:
        if reason == 'default':
            solvent_removal_flag = 0
            reason = 'TGA sentence, rule based parsing will fail for multistep procedure'
    return solvent_removal_flag, reason

for journal in os.listdir(basedir):
    if not os.path.isdir(basedir+journal):
        continue
    for article in os.listdir(basedir+'/'+journal):
        if 'html' not in article:
            continue
        print(article)
        print('--------')
        f = open(basedir+journal+'/'+article, 'rb')
        doc = Document.from_file(f)
        # Use Chem Data Extractor to tokenize the sections
        cwt = ChemWordTokenizer()
        header_counter = 0
        sent_counter = 0
        length_counter = 0
        header_list = []
        flag = False
        title = False
        intro_found = False
        search_results = []
        current_heading = 'nothing'
        for elem in doc.elements:
            # First store the title
            if isinstance(elem,chemdataextractor.doc.text.Title):
                title = elem.text
                continue
            # Look for the headers that may not be relevant (E.g. supporting info, supplementary info, citations, etc.)
            if isinstance(elem,chemdataextractor.doc.text.Heading):
                if (('reference' in elem.text.lower()) or ('crossref' in elem.text.lower()) or ('citation' in elem.text.lower()) or 
                    ('citing' in elem.text.lower()) or ('supporting' in elem.text.lower()) or ('supplementary' in elem.text.lower())):
                    flag = True
                else:
                    flag = False
                if flag:
                    # If those sections are identified, skip that part for now.
                    continue
                # Check to see if an introduction exists
                if 'intro' in elem.text.lower():
                    intro_found = True
                    current_heading = 'intro'
                    for i, sentence in enumerate(elem.sentences):
                        print(sentence)
                        sent_counter += 1
                        length_counter += len(sentence.text)
                else: 
                    current_heading = 'nothing'
                header_list.append(elem.text)
                header_counter += 1
            elif isinstance(elem,chemdataextractor.doc.text.Paragraph) and (current_heading=='intro'):
                for i, sentence in enumerate(elem.sentences):
                    sent_counter += 1
                    length_counter += len(sentence.text)
            elif isinstance(elem,chemdataextractor.doc.text.Paragraph) and (current_heading=='nothing'):
                if ('ARTICLE SECTIONS' in elem.text) or (elem.text == 'Jump To'):
                    continue
                checked = set()
                for sentence in elem.sentences:
                    # For each sentence, first check for false positive phrases.
                    linker_bool = organic_linker_phrase(sentence.text)
                    water_bool = water_phrase(sentence.text)
                    air_bool = air_phrase(sentence.text)
                    fake = check_fake_kw(sentence.text)
                    if (linker_bool or water_bool or fake or air_bool):
                        # skip if any of the phrases are false positives
                        continue

                    ### Next, check for the keywords we would care about 
                    solvent_check = [val for val in single_word_solvent if val in sentence.text]
                    collapse_check = [val for val in single_word_collapse if val in sentence.text]
                    stable_check = [val for val in single_word_stable if val in sentence.text]

                    if (len(solvent_check) == 1) and ('loss' in solvent_check):
                        # Make sure solvent identification isnt only "loss of crystallinity" because of the word "loss"
                        fake_loss = check_fake_loss(sentence.text)
                        if fake_loss:
                            # If it is, skip...
                            continue
                    if any(['stable' in val for val in stable_check]):
                        for val in stable_check:
                            # remove substring matches for stable that don't mean what we want.
                            if 'unstable' in val:
                                stable_check.remove(val)
                            if 'instable' in val:
                                stable_check.remove(val)
                            if 'adjustable' in val:
                                stable_check.remove(val)
                    if any(['stability' in val for val in stable_check]):
                        for val in stable_check:
                            # remove substring matches for stable that don't mean what we want.
                            if 'instability' in val:
                                stable_check.remove(val)
                    if any(['activat' in val for val in solvent_check]):
                        for val in solvent_check:
                            # remove substring matches for stable that don't mean what we want.
                            if 'deactivat' in val:
                                solvent_check.remove(val)
                    if (len(collapse_check) == 0) and (len(stable_check) == 0):
                        # If we do not have any words that are stable keywords or collapse keywords, continue. We cannot determine stability.
                        sent_counter += 1
                        length_counter += len(sentence.text)
                        continue
                    elif (len(solvent_check)>0) and ((len(collapse_check)>0) or (len(stable_check)>0)):
                        # Case where both solvent and stability info present. We can attempt to assign a label here.
                        solvent = True
                        stability = True
                        # For the cases that look promising, do the dependency parsing to analyze everything.
                        solvent_removal_flag, reason = check_dependencies_given_solvent(sentence.text, collapse_check, stable_check)
                        if solvent_removal_flag == -100:
                            solvent = False
                            stability = True
                            solvent_removal_flag = 0
                            solvent_check = []
                    elif (len(solvent_check)==0) and ((len(collapse_check)>0) or (len(stable_check)>0)):
                        # Case where only stability info present.
                        solvent_removal_flag = 0
                        solvent = False
                        stability = True
                        reason = 'no solvent kw'
                    # Perform sentiment analysis with textblob and VADER to store the information for later use
                    sid = SentimentIntensityAnalyzer()
                    ss = sid.polarity_scores(sentence.text)
                    textblob_sentiment = TextBlob(sentence.text)
                    textblob_sentiment.correct()
                    textblob_dict = {'polarity':textblob_sentiment.sentiment.polarity, 'subjectivity':textblob_sentiment.sentiment.subjectivity}
                    search_results.append({'filename': journal+'/'+article,
                                                       'title': title,
                                                       'sentence': sentence.text, 
                                                       'sentence_counter': sent_counter,
                                                       'stability_keyword': stable_check,
                                                       'collapse_keyword': collapse_check,
                                                       'solvent_keyword': solvent_check,
                                                       'solvent_removal_flag': solvent_removal_flag,
                                                       'solvent_removal_flag_reason': reason,
                                                       'solvent': solvent,
                                                       'stability': stability,
                                                       'VADER': ss,
                                                       'TEXTBLOB':textblob_dict})
                    sent_counter += 1
                    length_counter += len(sentence.text)

                ### At this point, we have the text that we need to search through for matches.
        # Next, store the data in a dictionary. Each article has one dictionary tied to it.
        temp_dict = {'intro':intro_found,'num_sections':header_counter, 'sections':header_list,'num_sentences':sent_counter,'num_char':length_counter, 'search_results':search_results}
        solvent_keyword_dict[journal+'/'+article] = temp_dict

with open('collapse_keyword_searches.json','w') as g:
    json.dump(solvent_keyword_dict,g)
