#!/usr/local/bin/python
# class written by Aditya Nandy for Kulik Group
from articledownloader.articledownloader import ArticleDownloader
from pybliometrics.scopus import ScopusSearch
from text_mining_tools.article import Article
from requests.utils import quote
from csv import reader
import itertools
import pandas as pd
import os
'''
The query class takes in a list of query words.
It then produces a set of "results," which are hits
on those query words. These results are the DOIs.

Each of these DOIs can then be instantiated as an
article class and downloaded to a custom path.

I have purposefully separated out the query class
from the article class. A given query class can 
help you keep track of the articles that are associated
with it. For instance, let's say I want to find articles
that have Mn(acac)3 in them (or are associated somehow).

I would instantiate a query class with Mn(acac)3 and 
other keywords that I thought would come up with the
relevant hits. The query class will then query the DOIs
associated with those keywords and then download the
articles that it can.

ACS articles cannot be automatically downloaded.
Doing so will block MIT's access, so I have purposefully
included protections against automated downloads for acs
publications.

The query class will then be saved as a pickle file,
which will contain keywords used during the search,
databases searched, and the DOIs that resulted (along)
with the custom path. From the query class, you can 
get all articles associated with a given set of keywords.

The query class hierarchically holds the article class
(in the same way a mol3D holds an atom3D). It is entirely
separable from the Article class. If you wanted to analyze
just a few papers, you could do that readily with some simple
Article classes.

The query class allows setting journal limitations 
(so that you do not query broadly, should you choose.) 
'''


class Query:
    def __init__(self, basepath, keywords, elsevier_key=None,
                 journal_limit=False, number_of_results=10000,
                 automate_download=False, analyze_downloaded=False):
        # basepath is an ABSOLUTE PATH where the corpus of papers
        #                will be stored.
        # keywords is a LIST of arguments that will be searched for.
        # elsevier_key is the key (STRING) for a scopus search.
        #               If you do not provide it,only crossref will
        #               be searched by default. Else, both elsevier
        #               and crossref will be queried.
        # journal_limit is a LIST of journal names that the search
        #                will be limited to.
        # number_of_results will limit the query to a certain number
        #               of maximum results. Default is 10K.
        # automate_download will automatically download and instantiate 
        #               article classes for all articles tied to a query.
        #               If this is not done, the query class can still
        #               be tied to all of its article classes with a 
        #               bound method (tie_articles_to_query).
        
        self.basepath = basepath
        self.check_dir()
        self.check_dir('AnalyzedResults/')
        if not isinstance(keywords,list):
          keywords = [keywords]
        self.keywords = keywords
        self.journal_limit = journal_limit
        self.elsevier_key = elsevier_key
        self.number_of_results = number_of_results
        self.query_results = None
        self.deduplicated_results = None
        self.execute_queries()
        if automate_download:
            reply = str(input('You are about to download '+\
                              str(number_of_results)+ \
                              'is this truly what you want? \
                              (recommended < 1000). \
                               (y/n): ')).lower().strip()
            if reply[0] == 'n':
                print('Will not download papers at this time.')
            elif reply[0] == 'y':
                print('OK, beginning download.')
                article_dict = {}
                print(self.deduplicated_results['doi'].values())
                for i, doi in enumerate(self.deduplicated_results['doi'].values()):
                    print('doi',doi)
                    this_article = Article(doi=doi, basepath=self.basepath, elsevier_key=self.elsevier_key)
                    article_dict[doi] = this_article
                self.article_dict = article_dict

    def execute_queries(self):
        downloader = ArticleDownloader(self.elsevier_key, timeout_sec=150)
        dois = []
        query_list = []
        journal_list = []
        issn_list = []
        found_by = []
        rows = self.number_of_results  # Number of results
        queries = self.prep_queries()
        print('THESE ARE THE QUERIES',queries)
        if (not self.journal_limit):
            issns = self.map_journal_to_ISSN(get_keys=True)
        else:
            issns = [self.map_journal_to_ISSN(
                journal=val) for val in self.journal_limit]
        self.issns = issns
        for query in queries:
            for issn in issns:
                print('Now querying ' + str(query) + ' in ' +
                      str(self.map_journal_to_ISSN(issn=issn)))
                prefix = self.issn_to_doi_prefix_mapper(issn)
                downloader_dois = downloader.get_dois_from_search(query,rows=rows, prefix=prefix, issn=issn)
                print(downloader_dois)
                found_by += ['crossref']*len(downloader_dois)
                if self.elsevier_key:
                    print('Elsevier key provided. Executing scopus query.')
                    s = ScopusSearch('KEY(' + str(query) + '), ISSN(' + str(issn) +
                                     ')', refresh=True)
                    df = pd.DataFrame(pd.DataFrame(s.results))
                    if df.shape[0] >= 1:
                        scopus_dois = df['doi'].tolist()
                        found_by += ['scopus']*len(scopus_dois)
                        downloader_dois += scopus_dois
                    dois += downloader_dois
                    query_list += [query]*len(downloader_dois)
                    journal_list += ([str(self.map_journal_to_ISSN(issn=issn))] *
                                     len(downloader_dois))
                    issn_list += [issn]*len(downloader_dois)
        merged = list(itertools.chain.from_iterable(dois))
        doi_list = pd.DataFrame()
        doi_list['doi'] = dois
        doi_list['journal'] = journal_list
        doi_list['issn'] = issn_list
        doi_list['query'] = query_list
        doi_list['api'] = found_by
        print(doi_list)
        print('----')
        dropped_doi_list = doi_list.drop_duplicates(subset=['doi'],keep='first')
        # The dataframe is compiled, and the dictionary form
        # is stored as a query_results attribute, along with
        # a deduplicated version.
        self.query_results = doi_list.to_dict()
        self.deduplicated_results = dropped_doi_list.to_dict()

    def tie_articles_to_query(self):
        article_dict = {}
        for i, doi in enumerate(self.deduplicated_results['doi'].values()):
            this_article = Article(doi, self.basepath, self.elsevier_key)
            article_dict[doi] = this_article
        self.article_dict = article_dict

    def prep_queries(self):
        queries = []
        # We need to prep all keywords to be API friendly.
        for keyword in self.keywords:
            query = quote(keyword)
            query = query.split()
            query = '+'.join(query)
            queries.append(query)
        self.queries = queries  # Bind queries to object to save later.
        return queries

    def check_dir(self, append_to_basepath = False):
        if not os.path.exists(self.basepath):
            raise AssertionError('The basepath you specified for the corpus does\
                                 not exist. Please construct the empty directory.')
        else:
            if append_to_basepath:
                if not os.path.exists(self.basepath.rstrip('/')+'/'+append_to_basepath):
                    os.mkdir(self.basepath.rstrip('/')+'/'+append_to_basepath)

    def map_journal_to_ISSN(self, journal=False, issn=False, get_keys=False):
        issn_dict = {'2155-5435': ['acs_catalysis',
                                   'acs_catal'],
                     '1520-4898': ['accounts_for_chemical_research',
                                   'acc_chem_res'],
                     '2574-0962': ['acs_applied_energy_materials',
                                   'acs_appl_energy_mater'],
                     '1944-8244': ['acs_applied_materials_and_interfaces',
                                   'acs_appl_mater_interfaces'],
                     '2574-0970': ['acs_applied_nano_materials',
                                   'acs_appl_nano_mater'],
                     '2374-7951': ['acs_central_science',
                                   'acs_cent_sci'],
                     '2156-8952': ['acs_combinatorial_science',
                                   'acs_comb_sci'],
                     '2380-8195': ['acs_energy_letters',
                                   'acs_energy_lett'],
                     '1936-0851': ['acs_nano'],
                     '2470-1343': ['acs_omega'],
                     '2168-0485': ['acs_sustainable_chemistry_and_engineering',
                                   'acs_sus_chem_eng'],
                     '0897-4756': ['chemistry_of_materials',
                                   'chem_mater'],
                     '1520-6890': ['chemical_reviews',
                                   'chem_rev'],
                     '1528-7483': ['crystal_growth_and_design',
                                   'cryst_growth_des'],
                     '0887-0624': ['energy_and_fuels',
                                   'energy_fuels'],
                     '1520-5851': ['environmenental_science_and_technology',
                                   'environ_sci_technol'],
                     '0888-5885': ['industrial_engineering_and_chemistry_research',
                                   'ind_eng_chem_res'],
                     '0020-1669': ['inorganic_chemistry',
                                   'inorg_chem'],
                     '0002-7863': ['journal_of_the_american_chemical_society',
                                   'j_am_chem_soc'],
                     '0021-9568': ['journal_of_chemical_engineering_Data',
                                   'j_chem_eng_data'],
                     '1089-5639': ['the_journal_of_physical_chemistry_a',
                                   'j_phys_chem_a'],
                     '1520-5207': ['the_journal_of_physical_chemistry_b',
                                   'j_phys_chem_b'],
                     '1932-7447': ['the_journal_of_physical_chemistry_c',
                                   'j_phys_chem_c'],
                     '1948-7185': ['the_journal_of_physical_chemistry_letters',
                                   'j_phys_chem_lett'],
                     '0743-7463': ['langmuir'],
                     '1530-6984': ['nano_letters'],
                     '0276-7333': ['organometallics'],
                     '1521-3773': ['angewandte_chemie_international_edition',
                                   'angew_chem_int_ed'],
                     '1521-3765': ['chemistry_a_european_journal',
                                   'chem_eur_j'],
                     '1521-4095': ['advanced_materials',
                                   'adv_mater'],
                     '2198-3844': ['advanced_science',
                                   'adv_sci'],
                     '1616-3028': ['advanced_functional_materials',
                                   'adv_funct_mater'],
                     '2041-6539': ['chemical_science',
                                   'chem_sci'],
                     '2044-4753': ['catalysis_science_and_technology',
                                   'cat_sci_tech'],
                     '1754-5706': ['energy_and_environmental_science',
                                   'energy_environ_sci'],
                     '1359-7345': ['chemical_communications',
                                   'chem_commun'],
                     '2045-2322': ['scientific_reports',
                                   'sci_rep'],
                     '1755-4349': ['nature_chemistry',
                                   'nat_chem'],
                     '1476-4660': ['nature_materials',
                                   'nat_mater'],
                     '2041-1723': ['nature_communications',
                                   'nat_commun'],
                     '1476-4687': ['nature'],
                     '1095-9203': ['science'],
                     '2375-2548': ['science_advances',
                                   'sci_adv']}
        if get_keys:
            # This is invoked if all possible journals are
            # to be queried. Returns the ISSNs of all possible
            # journals in the list.
            return issn_dict.keys()
        else:
            journal_to_ISSN_map = None
            if issn and journal:
                # This should never happen. It means info
                # already known. If both journal
                # and ISSN are handed in, returns ISSN value.
                journal_to_ISSN_map = issn

            elif issn and (not journal):
                # Takes in issn and returns full journal name
                journal_to_ISSN_map = issn_dict[issn][0]

            if journal and (not issn):
                # Takes in a journal name, makes sure it is not a
                # colloquial mapping, and then finds the corresponding
                # ISSN. Returns ISSN.
                journal_name = self.colloquial_mapping(journal)
                journal_name = journal_name.lower()
                journal_name_list = journal_name.split()
                journal_name_list = [val.strip('.')
                                     for val in journal_name_list]
                journal_name = '_'.join(journal_name_list)
                for i, val in enumerate(issn_dict.keys()):
                    temporary_name_list = issn_dict[val]
                    if journal_name in temporary_name_list:
                        journal_to_ISSN_map = val
                        break
            return journal_to_ISSN_map

    def colloquial_mapping(self, abbreviation):
        # This bound method turns a colloquialism into the real abbreviation.
        colloquial_dict = {'acr': 'acc_chem_res',
                           'angew': 'angew_chem_int_ed',
                           'cgd': 'cryst_growth_des',
                           'ic': 'inorg_chem',
                           'iecr': 'ind_eng_chem_res',
                           'jacs': 'j_am_chem_soc',
                           'jpca': 'j_phys_chem_a',
                           'jpcb': 'j_phys_chem_b',
                           'jpcc': 'j_phys_chem_c',
                           'jpcl': 'j_phys_chem_lett',
                           'nature_comm': 'nat_commun',
                           'nat_comm': 'nat_commun',
                           'nature_chem': 'nat_chem',
                           'nat_mat': 'nat_mater',
                           'nature_mat': 'nat_mater'}
        if abbreviation not in colloquial_dict.keys():
            mapping_to_return = abbreviation
        else:
            mapping_to_return = colloquial_dict[abbreviation]
        return mapping_to_return

    def issn_to_doi_prefix_mapper(self, issn):
        # This bound method takes an ISSN and gets the doi prefix.
        if issn in ['1521-3773', '1521-3765', '1521-4095',
                    '2198-3844', '1616-3028']:
            prefix = '10.1002'  # Wiley journals
        elif issn in ['2041-6539', '2044-4753', '1754-5706',
                      '1359-7345']:
            prefix = '10.1039'  # RSC journals
        elif issn in ['1755-4349', '2041-1723', '1476-4660',
                      '1476-4687', '2045-2322']:
            # Nature and subfamilies (Chem, Commun, Mater, Sci Rep)
            prefix = '10.1038'
        elif issn in ['1095-9203', '2375-2548']:
            prefix = '10.1126'  # Science family of journals
        else:
            prefix = '10.1021'  # ACS journals
        return prefix



