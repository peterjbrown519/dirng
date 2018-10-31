####################
Overview
####################
A python package for developing device-independent randomness expansion protocols - based on work in [DIRNG]_. Package aims to provide a user-friendly method of constructing and analysing device-independent randomness expansion protocols. The semidefinite relaxations are performed using the ncpol2sdpa package [NCPOL]_ and the resulting SDPs are computed using the SDPA solver [SDPA]_. Many thanks go to the developers of these latter two projects.

Hoping in the future to extend this to other DI tools, developing some DIQI library, e.g. DIQKD, DIRA, steering...

Happy to hear from anyone who's interested in a similar project :)

.. [DIRNG] ARXIV num here
.. [NCPOL] Peter Wittek. Algorithm 950: Ncpol2sdpa---Sparse Semidefinite Programming Relaxations for Polynomial Optimization Problems of Noncommuting Variables. ACM Transactions on Mathematical Software, 41(3), 21, 2015. DOI: 10.1145/2699464. arXiv:1308.6029.
.. [SDPA] "A high-performance software package for semidefinite programs: SDPA 7," Makoto Yamashita, Katsuki Fujisawa, Kazuhide Nakata, Maho Nakata, Mituhiro Fukuda, Kazuhiro Kobayashi, and Kazushige Goto, Research Report B-460 Dept. of Mathematical and Computing Science, Tokyo Institute of Technology, Tokyo, Japan, September, 2010.



####################
Installation
####################
Dependencies
------------
- *SDPA solver*  
		For installation and downloads please visit here_.
		
		**IMPORTANT CONFIGURATION NOTE**
			The default write to file precision is 3 decimal places. This has to be increased prior to usage, otherwise errors will affect solutions. This precision can be modified by changing the param.sdpa file, replacing 
				%+8.3e     char*  YPrint   (default %+8.3e,   NOPRINT skips printout)
			with
				%+8.8e     char*  YPrint   (default %+8.3e,   NOPRINT skips printout).
		The param.sdpa file should be located within the same directory as the sdpa executable.
