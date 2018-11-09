from setuptools import setup, find_packages

setup(name='dirng',
	  version='1.0.1',
	  author='Peter J. Brown',
	  author_email='pjb519@york.ac.uk',
	  description='Designing quantum-secure randomness expansion protocols',
	  long_description = 'For details see the github repo github.com/peterjbrown519/dirng',
	  url='https://github.com/peterjbrown519/dirng',
	  license='GNU',
	  packages= find_packages(),
	  package_dir={'dirng': 'dirng'},
	  package_data ={'dirng': ['etc/dirng_config.json']},
     	  include_package_data = True,
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
		"License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
		"Operating System :: OS Independent",
	"Topic :: Scientific/Engineering :: Physics",
	],)
