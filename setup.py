from setuptools import setup
from setuptools.command.install import install
import os


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
		install.run(self)
		with open(os.path.join(os.path.dirname(__file__), 'dirng', 'config.py'), 'w') as f:
			f.write('DEFAULT_SOLVER_PATH = \'' + str(self.solver) + '\'\n')
			f.write('SUPPORTED_SOLVERS = [\'sdpa\', \'sdpa_dd\', \'sdpa_qd\', \'sdpa_gmp\', \'sdpa.exe\']')

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
		"License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
		"Operating System :: OS Independent",
	"Topic :: Scientific/Engineering :: Physics",
	],)
