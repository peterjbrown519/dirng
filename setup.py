from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='dirng',
      version='0.0.1',
      author='Peter Brown',
      author_email='pjb519@york.ac.uk',
      description='Designing quantum-secure randomness expansion protocols',
      long_description = readme(),
      url='https://github.com/peterjbrown519/dirng',
      license='MIT',
      packages=['dirng'],
      install_requires=[
          'numpy',
	  'scipy',
	  'tabulate',
	  'ncpol2sdpa',
	  'sympy',
	  'tabulate',

      ],
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU License",
        "Operating System :: OS Independent",
	"Topic :: Scientific/Engineering :: Physics",
    ],)
