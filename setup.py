from setuptools import setup, find_packages
from setuptools.command.install import install
import pkg_resources
import os
import json



class CustomInstallCommand(install):
	"""Customized setuptools install command - allows you to pass solver path for defaulting."""
	user_options = install.user_options + [
	('solver=', None, 'Path to a solver.'),
	]
	def initialize_options(self):
		install.initialize_options(self)
		self.solver = ''

	def finalize_options(self):
		install.finalize_options(self)

	def run(self):
		manager = pkg_resources.ResourceManager()
		config_file = manager.resource_filename('dirng','etc/dirng_config.json')
		with open(config_file, 'r') as f:
			cfg = json.load(f)
		cfg["DEFAULT_SOLVER_PATH"] = self.solver
		with open(config_file, 'w') as f:
			json.dump(cfg, f)
		install.run(self)

setup(name='dirng',
	  version='1.0.0',
	  author='Peter J. Brown',
	  author_email='pjb519@york.ac.uk',
	  description='Designing quantum-secure randomness expansion protocols',
	  long_description = 'For details see the github repo github.com/peterjbrown519/dirng',
	  url='https://github.com/peterjbrown519/dirng',
	  cmdclass={
	  'install': CustomInstallCommand,
		},
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
