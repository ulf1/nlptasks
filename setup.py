
from setuptools import setup
import os


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


with open("requirements.txt", "r") as fp:
    required = fp.read().splitlines()
    required = [line for line in required if len(line)>0]
    required = [line for line in required if line[0]!="#"]


setup(name='nlptasks',
      version='0.1.0',
      description=(
          "Boilerplate code to wrap different libs for NLP tasks."
      ),
      long_description=read('README.md'),
      long_description_content_type='text/markdown',
      url='http://github.com/ulf1/nlptasks',
      author='Ulf Hamster',
      author_email='554c46@gmail.com',
      license='MIT',
      packages=['nlptasks'],
      install_requires=required + [
          'setuptools>=40.0.0'],
      python_requires='>=3.8',
      zip_safe=False)
