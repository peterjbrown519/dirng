# dirng_protocols

####################
#     OVERVIEW     #
####################
A device-independent randomness expansion package developed alongside [1]. Package aims to provide a 
user-friendly method of constructing and analysing device-independent randomness expansion protocols.
The semidefinite relaxations are performed using the ncpol2sdpa package [2] and the resulting 
SDPs are to be computed using a solver from the SDPA family [3]. Many thanks go to the developers of
these two projects.

Hoping in the future to extend this to other DI tools, developing some DIQI library, e.g.
DIQKD, DIRA, steering...

Happy to hear from anyone who's interested in a similar project :)

[1] - In preparation.
[2] - Peter Wittek. Algorithm 950: Ncpol2sdpa---Sparse Semidefinite Programming Relaxations for Polynomial Optimization Problems of Noncommuting Variables. ACM Transactions on Mathematical Software, 41(3), 21, 2015. DOI: 10.1145/2699464. arXiv:1308.6029.
[3] - "A high-performance software package for semidefinite programs: SDPA 7," 
Makoto Yamashita, Katsuki Fujisawa, Kazuhide Nakata, Maho Nakata, Mituhiro Fukuda, Kazuhiro Kobayashi, and Kazushige Goto, 
Research Report B-460 Dept. of Mathematical and Computing Science, Tokyo Institute of Technology, Tokyo, Japan, September, 2010.



####################
#   Installation   #
####################

1: Dependencies - All dependencies are listed in the requirements.txt file. They can be installed manually
		  or via pip in the terminal. To install via pip, download the requirements.txt file and use
		  the command
						pip install -r /path/to/requirements.txt
		  
		  which should install all dependencies automatically.

2: SDPA solver - Selected solver from the family should be installed somewhere locally.
		 IMPORTANT CONFIG NOTE: The precision with which the solver writes its solution
		 to file should be increased before using it with the package. 
		 In later versions this can be done by modifying the param.sdpa file. For earlier
		 versions see the manuals. 

