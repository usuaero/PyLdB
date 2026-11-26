"""pyldb: A tool for the calculation of the perceived loudness of a pressure signature."""

from setuptools import setup

setup(name='pyldb',
      version='1.0.0',
      description='A tool for the calculation of the perceived loudness of a pressure signature.',
      url='NA',
      author='cbolander',
      author_email='christianrbolander@gmail.com',
      license='MIT',
      packages=['pyldb'],
      package_data={'pyldb': ['tables/*.csv']},
      python_requires='>=3.9',
      install_requires=['numpy>=2.0'],
      zip_safe=False)
