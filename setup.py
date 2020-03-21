import setuptools

setuptools.setup(name='text_mining_tools', 
		 version="0.0.1",
		 description=('simple module for scraping scientific papers'),
		 packages=setuptools.find_packages(),
		 install_requirements=['pandas','nltk','requests','articledownloader','pybliometrics'],
		 python_requires='>=3.6',
		 include_package_data=True)

