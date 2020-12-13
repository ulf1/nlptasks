
from setuptools import setup
import m2r


setup(name='nlptasks',
      version='0.2.8',
      description=(
          "Boilerplate code to wrap different libs for NLP tasks."
      ),
      long_description=m2r.parse_from_file('README.md'),
      long_description_content_type='text/x-rst',
      url='http://github.com/ulf1/nlptasks',
      author='Ulf Hamster',
      author_email='554c46@gmail.com',
      license='Apache License 2.0',
      packages=['nlptasks'],
      install_requires=[
          'setuptools>=40.0.0',
          'm2r>=0.2.1',
          'torch>=1.1.0,<2',
          'tensorflow>=2.3.0,<3',
          'numpy>=1.18.0,<2',
          'scipy>=1.5.0,<2',
          'pandas>=1.1.0,<2',
          'spacy==2.3.*',
          'spacy-lookups-data==0.3.*',
          'stanza==1.1.*',
          'flair==0.6.*',
          'nltk==3.5',
          'SoMaJo==2.1.1',
          'SoMeWeTa==1.7.1',
          'pad-sequences>=0.5.*',
          'ray>=1.*'
      ],
      scripts=[
          'scripts/nlptasks_downloader.py'
      ],
      python_requires='>=3.6',
      zip_safe=False)

