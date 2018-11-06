####################
Overview
####################
A python3 package for developing device-independent randomness expansion protocols - based on work in [DIRNG]_. Package aims to provide a user-friendly method of constructing and analysing device-independent randomness expansion protocols. The semidefinite relaxations are performed using the ncpol2sdpa package [NCPOL]_ and the resulting SDPs are computed using the SDPA solver [SDPA]_. Many thanks go to the developers of these latter two projects.

Hoping in the future to extend this to other DI tools, developing some DIQI library, e.g. DIQKD, DIRA, steering...

Happy to hear from anyone who's interested in a similar project :)

.. [DIRNG] Peter J. Brown, Sammy Ragy and Roger Colbeck, "An adaptive framework for quantum-secure device-independent randomness expansion". arXiv:1810.13346_.
.. [NCPOL] Peter Wittek. Algorithm 950: Ncpol2sdpa---Sparse Semidefinite Programming Relaxations for Polynomial Optimization Problems of Noncommuting Variables. ACM Transactions on Mathematical Software, 41(3), 21, 2015. DOI: 10.1145/2699464. arXiv:1308.6029. Code available on gitlab_.
.. [SDPA] "A high-performance software package for semidefinite programs: SDPA 7," Makoto Yamashita, Katsuki Fujisawa, Kazuhide Nakata, Maho Nakata, Mituhiro Fukuda, Kazuhiro Kobayashi, and Kazushige Goto, Research Report B-460 Dept. of Mathematical and Computing Science, Tokyo Institute of Technology, Tokyo, Japan, September, 2010.

.. _arXiv:1810.13346: https://arxiv.org/abs/1810.13346
.. _gitlab: https://gitlab.com/peterwittek/ncpol2sdpa

####################
Installation
####################

Dependencies
================
	**SDP solver**

		Currently the only supported solvers are those from the sdpa family. They can be downloaded and extracted from their  sourceforge page_.

		**Important configuration note**

			The default precision with which sdpa writes its solutions to file is 3 decimal places. This has to be increased prior to usage otherwise errors will be prevalent in computations. This precision can be modified by changing the param.sdpa file that comes with the solver. In particular, replacing

				*%+8.3e     char\*  YPrint   (default %+8.3e,   NOPRINT skips printout)*

			with

				*%+8.8e     char\*  YPrint   (default %+8.3e,   NOPRINT skips printout),*

			will increase the precision.

.. _page: http://sdpa.sourceforge.net/download.html

Note: To automatically point the package to the solver, one can edit the dirng_config.json file replacing '/path/to/solver/' with the appropriate path. The location of the config file can be found by running \'*pip3 show dirng*\', it should be in the /etc/ subdirectory.  (**Windows users**: Paths must be specified using '\\\\' instead of '\\' in the path name.)

Installation with pip
=========================
The package will also be hosted on pypi_. To install from here run the command

.. code-block::

	pip3 install dirng

.. _pypi: https://pypi.org/project/dirng/

#####
Usage
#####
Explicit examples are provided in */examples/* directory. Here we review the main structure of the package and look at a general script demonstrating its usage. Those looking for a quick start can skip straight to that script. For full details on functionality please see the comments within the module's files.

Classes
-------
There are three classes in dirng.

1.	**Game**

	*Description*

		A Game object represents a nonlocal game which will be played by the untrusted devices.

	*Attributes*

		- **name**: Name of the game.
		- **score**: Expected score achieved by the untrusted devices.
		- **delta**: Width of confidence interval about the expected score.
		- **matrix**: Matrix representing the coefficients of the Bell-expression.

	*Usage*

		.. code-block::
			:linenos:

				from dirng import Game

				"""
				Let's create the CHSH game - 2 inputs / 2 outputs

				If p(a,b|x,y) is the distribution of the devices, then we write our
				Bell-expressions as \sum_{abxy} s_ab|xy p(a,b|x,y). The matrix of
				coefficients is then
								|  s00|00	s01|00	s00|01	s01|01	|
								|  s10|00	s11|00	s10|01	s11|01	|
								|  s00|10	s01|10	s00|11	s01|11	|
								|  s10|10	s11|10	s10|11	s11|11	|
				"""

				chsh_coefficients = [[ 0.25, 0.00, 0.25, 0.00],
						     		 [ 0.00, 0.25, 0.00, 0.25],
						     		 [ 0.25, 0.00, 0.00, 0.25],
						     		 [ 0.00, 0.25, 0.25, 0.00]]

				# Initialising the game object
				chsh = Game(name = 'chsh', score = 0.853, matrix = chsh_coefficients, delta = 0.001)

				# If the score was maybe a little ambitious, we can change it...
				chsh.score = 0.75

2.	**Devices**

	*Description*

		A pair of untrusted devices. They play nonlocal games and produce random numbers.

	*Attributes*

		- **name**: Name given to the devices.
		- **io_config**: The input output configuration of the devices. If m_i, n_j are the number of outputs for the i-th and j-th measurement of the 1st and 2nd device respectively. Then we write the io_config as [[m_1,m_2,...],[n_1,n_2,...]].
		- **generation_inputs**: Device inputs used during generation rounds.
		- **relaxation_level**: Level of NPA hierarchy used during computations.
		- **solver**: /path/to/the/solver/used/
		- **games**: A list of Game objects played by the device.

	*Usage*

		.. code-block::
			:linenos:

				# Continuing from before
				from dirng import Devices

				# We can initialise the device by passing it a settings dictionary.
				device_settings = {
					'name' : 'Mittens',
					'io_config' : [[2,2], [2,2]],
					'generation_inputs' : [0,0],
					'relaxation_level' : 2,
					'games' : [chsh],
					'solver' : '/path/to/a/solver/'
				}

				dev = Devices(device_settings)

				# As before these attributes can be changed after initialisation
				dev.generation_inputs = [1,1]
				dev.relaxation_level = 3

				# We can also add additional games if they are compatible with our devices alphabet.
				dev.games += another_game_object

				# The randomness can then be calculated by calling the hmin attribute
				randomness = dev.hmin
				print(randomness)

				# For a general view of the device we can also call print
				print(dev)

		If we want to change the scores of the games played by the device, we can set them all at once by

		.. code-block::
			:linenos:

			# Setting scores (and the delta values) for the two games that dev plays
			dev.score = [0.8, 0.7]
			dev.delta = [0.0001, 0.001]

			# Recompute the min-entropy
			print(dev.hmin)

		The games are ordered by the device alphabetically w.r.t. their names. So the list of scores should reflect that ordering.

		A useful function for calculating score vectors is distribution2Score()

		.. code-block::
			:linenos:

			# Suppose we have some distribution
			p = [[0.20, 0.30, 0.30, 0.20],
				[0.30, 0.20, 0.20, 0.30],
				[0.25, 0.25, 0.25, 0.25],
				[0.25, 0.25, 0.25, 0.25]]

			# We can calculate the expected score vector for a device pair by
			w = dev.distribution2Score(p)

			# We can then set that score like before
			dev.score = w

3.	**Protocol**

.. _arXiv:quant-ph/0306129: https://arxiv.org/abs/quant-ph/0306129

**Still under construction**

