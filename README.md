# text_mining_tools
Tools to facilitate mining the scientific literature.

These tools contain two major classes:
1. A Query class, which collects a list of DOIs from queries made, which can be limited to specific journals.
2. An Article class, which allows operations to be performed on a given doi (from downloading to manipulations).

In addition to the classes, some tools reside in full_text_mine.py. This permits Vader sentiment analysis.

Currently, Wiley, ACS, RSC, Nature, and Science publications are supported by default. Only HTML/XML support is provided for mining.
Thus, while Science articles can be downloaded, they are not in HTML format and thus cannot be mined, unless you can convert them to HTML. Be careful about this! Turning PDFs into mineable data is not a simple task. To proceed with your install, follow the instructions:

```bash
# install dependencies with the following
pip3 install -r requirements.txt 
```
Next, install the package.
```bash
# install the package with the following
python setup.py develop
```

After finding your articledownloader install, you will have to replace two files, which are stored locally in this repo (under adjusted_article_downloader/). To do this, find where your article downloader install is. It is highly likely that it is at the following location:
```python
<anaconda-path>/envs/<conda-env-name>/lib/python3.6/site-packages/articledownloader/
``` 
Then, replace the two files under that path. We need to replace two scripts (articledownloader.py & scrapers.py), which will be under articledownloader/ in the path you located above. Copy the files from text_mining_tools/adjusted_article_downloader/ to replace the equivalent files in the path above.
```bash
cp text_mining_tools/adjusted_article_downloader/* <anaconda-path>/envs/<conda-env-name>/lib/python3.6/site-packages/articledownloader/
``` 
If you installed inside of a conda environment (recommended!),  <conda-env-name> represents the name of your conda environment, and <anaconda-path> represents where your anaconda install is. 
  
Note: The first time you install the NLTK package, you will need to manually install subpackages. This can easily
be done by opening up NLTK in a python terminal and doing the following imports.

```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
```
This will prompt you for the necessary downloads, after which you should be able to use the package freely. It may ask you to install the "punkt" package. Please follow the install recommendations that are spawned.

Currently, python 3.6 is recommended for this package.

*** Note: Please set aside a hard disk with plenty of space if you are planning automated downloads. ***

We recommend installing stanza additionally for dependency parsing. 
  
```bash
pip install stanza
```
The first time stanza is installed, by default, models will not be installed. We want this for our pipelines. Thus, we need to run the following:
```python
import stanza
stanza.download('en')
```  
  
If you choose to use pybliometrics to do abstract installs, you can install pybliometrics via pip.
  
```bash
pip install pybliometrics
```
  
You can then set up your Elsevier API Key using the following link: https://pybliometrics.readthedocs.io/en/stable/configuration.html, which would make abstract mining possible afterwards. The information for your Elsevier key will be stored in a config.ini file that is in a hidden folder (either .pybliometrics/ or .scopus/), that pybliometrics uses to automate abstract downloads.
