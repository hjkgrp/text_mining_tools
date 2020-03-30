#!/usr/local/bin/python
# Class written by Aditya Nandy for Kulik Group
from articledownloader.articledownloader import ArticleDownloader
from pybliometrics.scopus import AbstractRetrieval
from bs4 import BeautifulSoup, NavigableString
from nltk import sent_tokenize
import re, os
import pandas as pd
"""
The article class takes in a DOI and a basepath for constructing a corpus.
The file structure of our corpus is the following:
    basepath
        |------10.1021
        |       |---paper1.html
        |       |---paper2.html
        |       |---paper3.html
        |       |---paper4.html
        |------10.1002 
        |       |---paper1.html
        |       |---paper2.html
        |------AnalyzedResults
                    |---query_1.pickle
                    |---doi_1.pickle

Each DOI prefix will make a single path where the file will be
downloaded. We sort this by publisher. After the article is
downloaded, then we can analyze the article by "containerizing" it 
and inspecting its parts. If the file is analyzed and stored, the 
Article class can be pickled for later use.

This class will assume that you are working with scraped HTML pages.
You can do this readily with nearly all files EXCEPT ACS.
We have been *EXPLICITLY* told by the library NOT to scrape 
ACS websites, which is in violation with ACS policy. If you need
lots of ACS papers (more than can be downloaded by hand), talk
to Aditya, who has signed the ACS license agreement and thus can
get you a zip of the XML files from ACS. It should be fine to 
download HTML versions of ACS articles for analysis. 

Once a corpus is constructed that consists of a set of papers,
we can do various types of analysis. Sentiment analysis may be
particularly relevant for looking at properties (can tell you)
whether or not the sentiment in a sentence is positive or 
negative. This class prepares the article for things like
sentiment analysis or utilities.

Currently, I have set up support for articles in nature, RSC,
and wiley journals. ACS journals require more explicit access.
"""


class Article:
    def __init__(self, doi, basepath, elsevier_key=False):
        self.doi = doi
        self.split_doi()
        self.basepath = basepath
        if self.basepath.strip()[-1] != '/':
            self.basepath = self.basepath.strip() + '/'
        self.check_dir()
        self.elsevier_key = elsevier_key
        self.download_article()

        # The class will stop at downloading the article
        # To do analysis, you would have to read the article
        # and break it apart into its pieces.

    def check_dir(self, append_to_basepath=False):
        if not os.path.exists(self.basepath):
            raise AssertionError('The basepath you specified for the corpus does\
                                 not exist. Please construct the empty directory.')
        else:
            if append_to_basepath:
                if not os.path.exists(self.basepath + append_to_basepath):
                    os.mkdir(self.basepath + append_to_basepath)

    def download_article(self):
        # We can use the article downloader developed by Elsa
        # Olivetti's group to build our corpus.
        prefix, rest, getter = self.split_doi()
        self.check_dir(prefix)
        if getter == None:
            print('We do not have support for this publisher at this time.')
            new_df = pd.DataFrame()
            doilist = []
            if os.path.exists(self.basepath+str(prefix)+\
                '/not_automatically_downloaded.csv'):
                not_auto_download = pd.read_csv(self.basepath+str(prefix)+\
                    '/not_automatically_downloaded.csv')
                doilist = not_auto_download['doi'].tolist()
            doilist.append(self.doi)
            doilist = list(set(doilist))
            new_df['doi'] = doilist
            new_df.to_csv(self.basepath+str(prefix)+\
                '/not_automatically_downloaded.csv',index=False)
        elif getter == 'acs':
            print('ACS pub. Not downloading, you can download manually!',self.doi)
            # We cannot automate ACS downloads, do NOT try to overcome this.
            new_df = pd.DataFrame()
            doilist = []
            if os.path.exists(self.basepath+str(prefix)+\
                '/not_automatically_downloaded.csv'):
                not_auto_download = pd.read_csv(self.basepath+str(prefix)+\
                    '/not_automatically_downloaded.csv')
                doilist = not_auto_download['doi'].tolist()
            doilist.append(self.doi)
            doilist = list(set(doilist))
            new_df['doi'] = doilist
            new_df.to_csv(self.basepath+str(prefix)+\
                '/not_automatically_downloaded.csv',index=False)
        else:
            print('ATTEMPTING DOWNLOAD!', self.doi)
            # The downloader will not overwrite existing downloads.
            if not os.path.exists(self.basepath+str(prefix) +
                             '/'+str(rest)+'.html'):
                downloader = ArticleDownloader(
                    str(self.elsevier_key), timeout_sec=150)
                temp_file = open(self.basepath+str(prefix) +
                                 '/'+str(rest)+'.html', 'wb')
                downloader.get_html_from_doi(self.doi, temp_file, getter)
                temp_file.close()

    def split_doi(self):
        prefix = str(self.doi.split('/', 1)[0]).strip()
        rest = str(self.doi.split('/', 1)[1]).strip()
        getter_dict = {'10.1039': 'rsc',
                       '10.1002': 'wiley',
                       '10.1038': 'nature',
                       '10.1126': 'aaas',
                       '10.1021': 'acs'}
        if prefix not in getter_dict.keys():
            getter = None
        else:
            getter = getter_dict[prefix]
        self.prefix = prefix
        self.getter = getter
        return prefix, rest, getter

    def read_paper(self):
        # Once a paper is downloaded, this method is used
        # 'read' the paper. This is necessary for later steps.
        def replace(fin_name, srcStr, desStr, fout_name):
            fin = open(fin_name, "r")
            fout = open(fout_name, "w")
            txt = fin.read()
            txtout = re.sub(srcStr, desStr, txt)
            fout.write(txtout)
            fin.close()
            fout.close()
        prefix, rest, getter = self.split_doi()
        if os.path.exists(self.basepath + str(prefix) +
                          '/'+str(rest)+'.html'):
            filename = (self.basepath + str(prefix) +
                        '/'+str(rest)+'.html')
            replace(filename, ">\d+\w*[</\w+>]*</a>",
                    "></\w+></a>", "temp.html")
            html_doc = open("temp.html", encoding="UTF-8")
            f = BeautifulSoup(html_doc, 'html.parser')
            self.f = f
            self.original_f = f  # This is the original soup doc. Do not touch.
            os.remove("temp.html")
        else:
            raise AssertionError('This paper does not exist at '+\
                                str(self.basepath) +str(prefix) + \
                                '/' + str(rest)+'.html. Download it first.')


    def populate_full_paper(self):
        # DO THIS LAST!
        full_paper = False
        full_paper_sentences = []
        # After a paper is loaded, this method
        # replaces the special characters in the paper
        # breaks apart the sentences, recombines them
        # with correct closure. It stores the full paper
        # and the paper in sentence form. This should be
        # done LAST as it alters the tree.
        for i in self.f(["script", "style", 'ol', 'ul', 'li', \
                         'table', 'a', 'noscript', 'option']):
            i.extract()
            for i in self.f(["div"]):
                # Get rid of citation tree which clutters the text.
                if (i.get('class') == ['citationInfo'] or \
                    i.get('class') == ['casRecord'] or \
                    i.get('class') == ['casContent'] or \
                    i.get('class') == ['casTitle'] or \
                    i.get('class') == ['casAuthors'] or \
                    i.get('class') == ['casAbstract']):
                    i.extract()
            for i in self.f(['sup', 'sub']):
                # Flattens subscripts, which are difficult to mine
                i.unwrap()
        full_paper = self.f.get_text()
        if full_paper:
            full_paper_subbed = self.clean_text(full_paper)
            full_paper_sentences = sent_tokenize(full_paper_subbed)
        self.full_paper = full_paper
        self.full_paper_sentences = full_paper_sentences
        return full_paper, full_paper_sentences

    def get_title(self):
        title = None
        meta_tags = self.f.find_all('meta')
        for i, tag in enumerate(meta_tags):
            if tag.has_attr('name'):
                if tag['name'].lower() in ['dc.title','citation_title']:
                    title = tag['content']
        self.title = title
        return title

    def get_authors(self):
        author_list = []
        meta_tags = self.f.find_all('meta')
        for i, tag in enumerate(meta_tags):
            if tag.has_attr('name'):
                if (tag['name'].lower() in ['citation_author', 'dc.creator']):
                    clean_text = self.clean_text(str(tag['content']))
                    author_list.append(clean_text)
        self.authors = list(set(author_list))
        return author_list

    def get_journal_name(self):
        journal_name = False
        meta_tags = self.f.find_all('meta')
        for i, tag in enumerate(meta_tags):
            if tag.has_attr('name'):
                if (tag['name'].lower() in ['citation_journal_title']):
                    journal_name = "_".join(tag['content'].lower().split())
                    break
        if journal_name == False:
            temp = self.f.find_all('title')
            if len(temp)>0:
                new_temp = temp[0].get_text().split('|')
                if len(new_temp) > 1:
                    journal = new_temp[1]
                    journal_name = "_".join(journal.lower().split())
        self.journal_name = journal_name
        return journal_name

    def get_publication_date(self):
        pub_year = False
        meta_tags = self.f.find_all('meta')
        for i, tag in enumerate(meta_tags):
            if tag.has_attr('name'):
                if tag['name'].lower() in ['citation_date', 'dc.date','citation_publication_date']:
                    pub_year = tag['content']
                    break
        self.publication_year = pub_year
        return pub_year

    def get_figure_captions(self):
        # Figure captions are stored in the order they appear
        caption_dict = {}
        counter = 0
        checked = set()
        if self.getter == 'rsc':
            figures = self.f.find_all('div', attrs={'class': 'image_table'})
            for figure in figures:
                temp = figure.get_text()
                if ('fig' in temp.lower()) and (temp.lower() not in checked):
                    checked.add(temp.lower())
                    counter += 1
                    caption_dict[counter] = temp
        if (self.getter in ['acs','nature','wiley']):
            figures = self.f.find_all('figure')
            for val in figures:
                temp = val.get_text()
                if ('fig' in temp.lower()) and (temp.lower() not in checked):
                    new_temp = temp.lower().replace('Open in figure viewerPowerPoint','')
                    clean_text = self.clean_text(new_temp)
                    checked.add(clean_text)
                    counter += 1
                    caption_dict[counter] = temp
        self.figure_captions = caption_dict
        return caption_dict

    def get_table_captions(self):
        caption_dict = {}
        counter = 0
        checked = set()
        if self.getter == 'rsc':
            tables = self.f.find_all('div', attrs={'class': 'table_caption'})
            for table in tables:
                temp = table.get_text()
                if ('table' in temp.lower()) and (temp.lower() not in checked):
                    checked.add(temp.lower())
                    counter += 1
                    caption_dict[counter] = temp
        if self.getter == 'acs':
            tables = self.f.find_all('div', attrs={'class': 'NLM_table-wrap'})
            for table in tables:
                temp = table.get_text()
                if ('table' in temp.lower()) and (temp.lower() not in checked):
                    checked.add(temp.lower())
                    counter += 1
                    caption_dict[counter] = temp
        if self.getter == 'wiley':
            tables = self.f.find_all('header',attrs={'class':'article-table-caption'})
            for table in tables:
                temp = table.get_text()
                if ('table' in temp.lower()) and (temp.lower() not in checked):
                    checked.add(temp.lower())
                    counter += 1
                    caption_dict[counter] = temp
        self.table_caption_dict = caption_dict
        return caption_dict

    def get_article_type(self):
        article_type = False
        article_type_list = []
        if self.getter == 'acs':
            article_type_list = self.f.find_all('meta', attrs={'name': 'dc.Type'})
        if self.getter == 'nature':
            article_type_list = self.f.find_all('meta', attrs={'name': 'citation_article_type'})
        if len(article_type_list)>0:
            article_type = article_type_list[0].get_text()
        self.article_type = article_type
        return article_type

    def get_section_names(self):
        section_name_dict = {}
        counter = 0
        checked = set()
        if self.getter == 'rsc':
            sections = self.f.find_all('span', attrs={'class': 'a_heading'})
            for section in sections:
                temp = section.get_text()
                if (temp.lower() not in checked):
                    checked.add(temp.lower())
                    counter += 1
                    section_name_dict[counter] = temp
        if self.getter == 'acs':
            sections = self.f.find_all(
                'div', attrs={'class': 'article_content-title'})
            for section in sections:
                temp = section.get_text()
                if (temp.lower() not in checked):
                    checked.add(temp.lower())
                    counter += 1
                    section_name_dict[counter] = temp
                    if 'references' in temp.lower():
                        break
        if self.getter == 'nature':
            sections = self.f.find_all('c-article-section__title')
            for section in sections:
                temp = section.get_text()
                if (temp.lower() not in checked):
                    checked.add(temp.lower())
                    counter += 1
                    section_name_dict[counter] = temp
                    if 'references' in temp.lower():
                        break
        if self.getter == 'wiley':
            for i in self.f(['h2']):
                temp = i.get('class')
                if temp != None and len(temp)>0:
                    if 'article-section__title' in temp:
                        counter += 1
                        section_name_dict[counter] = i.get_text()
        self.section_name_dict = section_name_dict
        return section_name_dict

    def get_section_text(self):
        # After the section names are obtained, we can split a paper apart
        # into its corresponding sections. This is helpful for doing section
        # specific sentiment analysis at a later time.
        section_text_dict = {}
        section_text_dict_sentences = {}

        def between(cur, end):
            # This function climbs down the tree.
            while cur and cur != end:
                if isinstance(cur, NavigableString):
                    text = cur.strip()
                    if len(text):
                        yield text
                cur = cur.next_element
        if len(self.section_name_dict) != 0:
            # Loop over everything between the sections
            if self.getter == 'acs':
                kw = 'div'
            elif self.getter == 'rsc':
                kw = 'span'
            elif self.getter == 'wiley':
                kw = 'h2'
            sections_to_enumerate = list(self.section_name_dict.values())
            for i, section in enumerate(sections_to_enumerate[:-1]):
                if self.getter in ['acs','rsc']:
                    section_text = ' '.join(text for text in between(self.f.find(kw,
                                                                        text=section),
                                                                 self.f.find(kw, 
                                                                    text=sections_to_enumerate[i+1])))
                elif self.getter in ['wiley']:
                    section_text = ' '.join(text for text in between(self.f.find(kw,
                                                                        text=section).parent,
                                                                 self.f.find(kw, 
                                                                    text=sections_to_enumerate[i+1])))
                section_text_dict[i] = {section: section_text}
                # Next, preprocess text and store as sentences
                subbed_section_text = self.clean_text(section_text)
                subbed_section_text = subbed_section_text.lower()
                sent_text = sent_tokenize(subbed_section_text)
                section_text_dict_sentences[i] = {section: sent_text}
        self.section_text_dict = section_text_dict
        self.section_text_dict_sentences = section_text_dict_sentences
        return section_text_dict, section_text_dict_sentences

    def get_abstract(self, from_scopus = False):
        abstract = False
        abstract_sentences = []
        if from_scopus:
            try:
                abstract_dict = AbstractRetrieval(self.doi)
            except:
                print('Failed to get abstract via SCOPUS API for '+str(self.doi))
                return abstract, abstract_sentences
            abstract = abstract_dict.description
        if not from_scopus:
            if self.getter == 'rsc':
                temp = self.f.find_all('p', attrs={'class': 'abstract'})
            if self.getter == 'acs':
                temp = self.f.find_all('p', attrs={'class': 'articleBody_abstractText'})
            if self.getter == 'nature':
                temp = self.f.find_all('meta', attrs={'name': 'description'})
                if len(temp)>0:
                    abstract = temp[0]['content']
            if self.getter == 'wiley':
                temp = self.f.find_all('div', attrs={'class': 'abstract-group'})
                if len(temp)>0:
                    abstract = temp[0].get_text().strip()
            if (self.getter in ['rsc','acs']):
                if (len(temp) > 0):
                    abstract = temp[0].get_text()
        if abstract:
            abstract_replaced = self.clean_text(abstract)
            abstract_replaced = abstract_replaced.lower()
            abstract_sentences = sent_tokenize(abstract_replaced)
        self.abstract = abstract
        self.abstract_sentences = abstract_sentences
        return abstract, abstract_sentences

    def get_cited_papers(self):
        citation_dict = {}
        counter = 0
        if self.getter == 'rsc':
            for i in self.f(['span']):
                temp = i.get('id')
                if temp != None and len(temp)>0:
                    if 'cit' in temp:
                        counter += 1
                        citation_dict[counter] = i.get_text()
        if self.getter == 'wiley':
            for i in self.f(['li']):
                temp = i.get('data-bib-id')
                if temp != None and len(temp)>0:
                    if 'bib' in temp:
                        counter += 1
                        citation_dict[counter] = i.get_text()
        if self.getter == 'acs':
            citations = self.f.find_all('div',attrs={'class':'citationInfo'})
            for citation in citations:
                counter += 1
                citation_dict[counter] = citation.get_text()
        if self.getter == 'nature':
            citations = self.f.find_all('li',attrs={'class':'c-article-references__item js-c-reading-companion-references-item'})
            for citation in citations:
                counter += 1
                citation_dict[counter] = citation.get_text()
        self.citation_dict = citation_dict
        return citation_dict

    def populate_metadata(self):
        # This populates important data after the paper is read
        self.read_paper()
        self.get_article_type()
        self.get_title()
        self.get_journal_name()
        self.get_authors()
        self.get_publication_date()
        self.get_abstract()
        self.get_cited_papers()

    def populate_paper_by_section(self):
        self.read_paper()
        self.get_section_names()
        self.get_section_text()

    def populate_figure_and_table_captions(self):
        self.read_paper()
        self.get_table_captions()
        self.get_figure_captions()

    def full_analysis(self, get_full_paper=True):
        self.populate_metadata()
        self.populate_paper_by_section()
        self.populate_figure_and_table_captions()
        self.read_table_data()
        if get_full_paper:
            self.populate_full_paper()

    def clean_text(self, text):
        # Currently, text is ridded of these characters,
        # which make mining difficult. You can add to this
        # list if other things are not enabling mining.
        output_text = re.sub(u'\xa0', " ",text)
        output_text = re.sub(u"\u2013", " ", output_text)
        output_text = re.sub(u"\u2009", " ", output_text)
        output_text = re.sub(u"\u2005", " ", output_text)
        output_text = re.sub(u"\u2014", " ", output_text)
        output_text = re.sub("\n", "", output_text)
        output_text = re.sub("&thinsp;", "", output_text)
        output_text = re.sub("&nbsp;", " ", output_text)
        output_text = re.sub("-", " ", output_text)
        return output_text

    def read_table_data(self):
        table_dict = {}
        counter = 0
        tables = self.f.find_all('table')
        if len(tables)>0 and tables != None:
            for table in tables:
                if (table == None) or (table.find_all('table') != []):
                    # Cannot handle nested tables.
                    continue
                counter += 1
                data = []
                table_head = table.thead
                if table_head == None:
                    # Could not identify tableheader, no use storing.
                    continue
                rows = table_head.find_all('tr')
                if rows == None or len(rows)==0:
                    # Not a real table.
                    continue
                rownum = 0
                rownow = 0
                for row in rows:
                    rownum += 1
                    if rownow < rownum:
                        data.append([]) 
                        rownow += 1
                    cols = row.find_all('th')
                    colnum = 0
                    colnow = len(data[rownum - 1])
                    for col in cols:
                        text = col.get_text().strip()
                        text = self.clean_text(text)
                        colnum += 1
                        if colnow < colnum:
                            data[rownum-1].append([])
                            colnow += 1
                        while data[rownum-1][colnum-1] != []:
                            colnum += 1
                            if colnow < colnum:
                                data[rownum-1].append([])
                                colnow += 1
                        colspannum = 1
                        if (col.attrs.get('colspan') != None) and (col.attrs.get('colspan') != ''):
                            colspannum = int(col['colspan'])
                        rowspannum = 1
                        if col.attrs.get('rowspan') != None:
                            rowspannum = int(col['rowspan'])
                        if rowspannum == 1 and colspannum == 1:
                            temp = colnum
                            while data[rownum-1][temp-1] != []:
                                temp += 1
                            data[rownum-1][temp-1].append(text)
                        if rowspannum == 1 and colspannum != 1:
                            for j in range(colspannum - 1):
                                data[rownum-1].append([])
                            colnow += colspannum - 1
                            for i in range(colspannum):
                                if data[rownum-1][colnum-1+i] == []:
                                    data[rownum-1][colnum-1+i].append(text)
                        if rowspannum != 1 and colspannum == 1:
                            if rownow < rownum + rowspannum-1:
                                for j in range(rownum + rowspannum - rownow - 1):
                                    data.append([]) 
                                    rownow += 1
                            for i in range(rowspannum):
                                if len(data[rownum+i-1]) < colnum + colspannum:
                                    for j in range(colnum + colspannum - len(data[rownum+i-1])):
                                        data[rownum+i-1].append([])
                                for j in range(colspannum):
                                    if data[rownum+i-1][colnum+j-1] == []:
                                        data[rownum+i-1][colnum+j-1].append(text)
                table_body = table.tbody
                if table_body != None:
                    rows = table_body.find_all('tr')
                    head_row_num = len(data)
                    if head_row_num == 0:
                        head_row_num = 1
                    rownum = len(data)
                    rownow = len(data)
                    for row in rows: 
                        rownum += 1 
                        if rownow < rownum:
                            data.append([])
                            rownow += 1 
                        cols = row.find_all('td')
                        colnum = 0 
                        colnow = len(data[rownum - 1]) 
                        for col in cols:
                            text = col.get_text().strip()
                            text = self.clean_text(text)
                            colnum += 1
                            if colnow < colnum:
                                data[rownum-1].append([]) 
                                colnow += 1 
                            while data[rownum-1][colnum-1] != []:
                                colnum += 1
                                if colnow < colnum: 
                                    data[rownum-1].append([])
                                    colnow += 1
                            colspannum = 1
                            if col.attrs.get('colspan') != None and col.attrs.get('colspan') != '':
                                colspannum = int(col['colspan'])
                            rowspannum = 1
                            if col.attrs.get('rowspan') != None:
                                rowspannum = int(col['rowspan'])
                            if rowspannum == 1 and colspannum == 1:
                                temp = colnum
                                while data[rownum-1][temp-1] != []:
                                    temp += 1
                                data[rownum-1][temp-1].append(text)
                            if rowspannum == 1 and colspannum != 1:
                                for j in range(colspannum - 1):
                                    data[rownum-1].append([])
                                colnow += colspannum - 1
                                for i in range(colspannum):
                                    if data[rownum-1][colnum-1+i] == []:
                                        data[rownum-1][colnum-1+i].append(text)
                            if rowspannum != 1 and colspannum == 1:
                                if rownow < rownum + rowspannum-1:
                                    for j in range(rownum + rowspannum - rownow - 1):
                                        data.append([])
                                        rownow += 1
                                for i in range(rowspannum):
                                    if len(data[rownum+i-1]) < colnum:
                                        for j in range(colnum - len(data[rownum+i-1])):
                                            data[rownum+i-1].append([])
                                    if data[rownum+i-1][colnum-1] == []:
                                        data[rownum+i-1][colnum-1].append(text) 
                            if rowspannum != 1 and colspannum != 1:
                                if rownow < rownum + rowspannum-1:
                                    for j in range(rownum + rowspannum - rownow - 1): 
                                        data.append([])
                                        rownow += 1
                                for i in range(rowspannum):
                                    if len(data[rownum+i-1]) < colnum + colspannum:
                                        for j in range(colnum + colspannum - len(data[rownum+i-1])):
                                            data[rownum+i-1].append([]) 
                                    for j in range(colspannum): 
                                        if data[rownum+i-1][colnum+j-1] == []:
                                            data[rownum+i-1][colnum+j-1].append(text)
                for i in range(len(data)):
                    for j in range(len(data[i])):
                        if data[i][j] == []:
                            data[i][j].append('')
                table_dict[counter] = data
        self.table_dict = table_dict
        return table_dict


