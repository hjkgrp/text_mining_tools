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
After finding your articledownloader install, you will have to replace two files, which are stored locally in this repo.
