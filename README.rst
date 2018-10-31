####################
Overview
####################
A python3 package for developing device-independent randomness expansion protocols - based on work in [DIRNG]_. Package aims to provide a user-friendly method of constructing and analysing device-independent randomness expansion protocols. The semidefinite relaxations are performed using the ncpol2sdpa package [NCPOL]_ and the resulting SDPs are computed using the SDPA solver [SDPA]_. Many thanks go to the developers of these latter two projects.

Hoping in the future to extend this to other DI tools, developing some DIQI library, e.g. DIQKD, DIRA, steering...

Happy to hear from anyone who's interested in a similar project :)

.. [DIRNG] ARXIV num here
.. [NCPOL] `Peter Wittek. Algorithm 950: Ncpol2sdpa---Sparse Semidefinite Programming Relaxations for Polynomial Optimization Problems of Noncommuting Variables. ACM Transactions on Mathematical Software, 41(3), 21, 2015. DOI: 10.1145/2699464. arXiv:1308.6029.`
.. [SDPA] "A high-performance software package for semidefinite programs: SDPA 7," Makoto Yamashita, Katsuki Fujisawa, Kazuhide Nakata, Maho Nakata, Mituhiro Fukuda, Kazuhiro Kobayashi, and Kazushige Goto, Research Report B-460 Dept. of Mathematical and Computing Science, Tokyo Institute of Technology, Tokyo, Japan, September, 2010.



####################
Installation
####################

Dependencies
------------
- *SDPA solver*  
		For installation instructions and downloads please visit their sourceforge page_.
		
		**Important configuration note**
		
			The default precision with which sdpa writes its solutions to file is 3 decimal places. This has to be increased prior to usage otherwise errors will be prevalent in computations. This precision can be modified by changing the param.sdpa file, replacing 
			
				*%+8.3e     char\*  YPrint   (default %+8.3e,   NOPRINT skips printout)*
					
			with
				
					*%+8.8e     char\*  YPrint   (default %+8.3e,   NOPRINT skips printout).*
					
			Please also make sure that a copy of the mpdified param.sdpa is located within the same directory as the executable (if not there by default).

.. _page: http://sdpa.sourceforge.net/download.html

Installation from source
------------------------
1. Download repository
2. From inside the top dirng directory run

	python3 setup.py install
	
Installation with pip
---------------------
The package will also be hosted on pypi_. To install from here run

	pip3 install dirng

.. _pypi: https://pypi.org
	
#####
Usage
#####
Explicit examples are provided in */examples/*. Here we review the main structure of the package and look at a general script demonstrating its usage. Those looking for a quick start can skip straight to that script_. For full details on functionality please see the comments within the module files.

Classes
-------
There are three classes in dirng.

1.	**Game**
	
	*Description*
	
		A *Game* object represents a nonlocal game which will be played by the untrusted devices. 

	*Attributes*
		
		- name: Name of the game.
		- score: Expected score achieved by the devices.
		- delta: Width of confidence interval about the expected score.
		- matrix: Matrix representing the coefficients of the Bell-expression. Matrix is written in `CG-form`_.
		
**Still under construction**

.. _script: Hello

.. Footnote the cg-form and explain usage.
