from setuptools import setup
import os


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(name='nlptasks',
      version='0.3.0',
      description=(
          "Boilerplate code to wrap different libs for NLP tasks."
      ),
      # long_description=read('README.md'),
      # long_description_content_type='text/markdown',
      url='http://github.com/ulf1/nlptasks',
      author='Ulf Hamster',
      author_email='554c46@gmail.com',
      license='Apache License 2.0',
      packages=['nlptasks'],
      install_requires=[
          'setuptools>=40.0.0',
          'torch>=1.1.0,<2',
          'tensorflow>=2.4.0,<3',
          'numpy>=1.18.0,<2',
          'scipy>=1.5.0,<2',
          'pandas>=1.1.0,<2',
          'spacy==2.3.*',
          'spacy-lookups-data==0.3.*',
          'stanza==1.1.*',
          'flair==0.7.*',
          'nltk==3.5',
          'SoMaJo==2.1.1',
          'SoMeWeTa==1.7.1',
          'pad-sequences>=0.5.*',
          'treesimi>=0.1.1'
      ],
      scripts=[
          'scripts/nlptasks_downloader.py'
      ],
      python_requires='>=3.6',
      zip_safe=False)
