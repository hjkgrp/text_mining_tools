# text_mining_tools
Tools to facilitate mining the scientific literature.

These tools contain two major classes:
1. A Query class, which collects a list of DOIs from queries made, which can be limited to specific journals.
2. An Article class, which allows operations to be performed on a given doi (from downloading to manipulations).

Currently, Wiley, ACS, RSC, Nature, and Science publications are supported by default.

```python
# install dependencies with the following
pip3 install -r requirements.txt 
```
<strong>
After finding your articledownloader install, you will have to replace two files, which are stored locally in this repo (under adjusted_article_downloader/. To do this, find where your article downloader install is. Then, replace the two files under that path. They should be replacing two scripts (articledownloader.py & scrapers.py), which will be under .../articledownloader/. To sanity check, this folder should contain .../articledownloader/__init__.py.</strong>

Note: The first time you install the NLTK package, you will need to manually install subpackages. This can easily
be done by opening up NLTK in a python and doing the following imports.

```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
```
This will prompt you for the necessary downloads, after which you should be able to use the package freely.

Currently, python 3.6 is recommended for this package.
