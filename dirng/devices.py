import numpy as np
import ncpol2sdpa as ncp
from math import log
import copy as cpy
from tabulate import tabulate
from .device_methods import guessingProbabilityObjectiveFunction, distribution2Score
from .cg_methods import expression2CG, distribution2CG
from .games import Game
from pkg_resources import resource_filename, Requirement
import json
import os, warnings

class Devices:
	"""
	Devices Class

	Description:
			Object representing a pair of untrusted devices. Combined with a nonlocal
			game and score, the Devices object computes the guessing probability
			SDP to give associated bounds on Hmin. Can be initialised with a settings
			dictionary dictating the initial values of its various attributes. Otherwise
			attributes can be assigned in the usual manner

			Example:
				settings = {
								'name' 					: 		'device1',
								'io_config'				:		[[2,2],[2,2]],
								'generation_inputs'		: 		[0,0],
								'relaxation_level'		:		2,
								'games'					:		[list of Game objects],
								'solver'				:		'/path/to/solver/',
								'verbose'				:		0}

	Attributes:

			name					-			String labelling the devices.

			io_config				-			The input/output configuration specified in array form.
												E.g. If Alice has two inputs, one with 2 outputs and the
												other with 3 outputs, and Bob has 3 inputs with 1, 5 and
												6 outputs respectively then we write
												io_config = [[2,3],[1,5,6]].

			generation_inputs		-			Inputs used during generation rounds
												chosen from {0,1,...,|X|-1} and
												{0,1,...,|Y|-1}. [x_gen, y_gen].

			generation_output_size 	-			Number of outputs on the generation rounds.
			 									(Dictated by io_config).

			relaxation_level		-			Level of the NPA hierarchy to use with the sdp relaxation.

			solver					-			Path to the solver -- '/path/to/solver/'.

			verbose					-			shouty

			status					-			Solution status for the guessing probability program.
												If infeasible then gp set to 1.0 and hmin set to 0.0.

			gp						-			The guessing probability associated with the device pair.
			hmin					-			Min entropy -log(gp,2)

			games 					-			List storing the Game objects played by the device pair.
			score 					-			Score vector for the games.
			delta					-			Delta vector for the games.

	Hidden attributes:

			_sdp 			-		ncpol2sdpa sdp object
			_solver_exe		-		solver executable fed to ncpol2sdpa
			_solver_name	-		...
			_needs_relaxing -		flag to indicate that devices need an sdp relaxation
			_needs_resetting-		flag to indicate that sdp object needs resetting
			_io_config_set	-		flag to indicate that the io_config has been set
									a warning is given if it is set again.


	PUBLIC Methods:

			Name 			-		distribution2Score
			Arguments 		- 		distribution
			Description		-		Calculates the expected score vector for the devices
									should they play according to the distribution provided.
			Returns		 	-		score vector

			Name 			-		dualSolution
			Arguments 		- 		v - score vector
			Description		-		None
			Returns		 	-		The dual solution in the form [av, lv, v, _v, status]
											av - dual constant (from normalisation)
											lv - dual vector
											_v - v with the cgshift applied
											status - sdp status when computed

	PRIVATE Methods:

			Name			-		_computeGP
			Arguments		-		None
			Description		-		Solves the guessing probability program for the current state of
									the device using the devices' sdp relaxation object.
			Returns			-		hmin

			Name			-		_getDualVariables
			Arguments		-		None
			Returns			-		Returns the collection [av,lv,_v,v,status] which constitute a
									dual solution to the computed guessing probability.
									av 	   - alpha
									lv 	   - lambda vector
									v  	   - score vector parameterising the program without the
										 	 cg-translation applied.
									_v 	   - score vector parameterising the program with the
										 	 cg-translation applied.
									status - Solution status returned by solver. (Hopefully is 'optimal')

			Name			-		_relax
			Arguments		-		None
			Description		-		Creates the sdp relaxation associated with the current state of
									the device using the ncpol2sdpa package. Then solves the guessing
									probability program, extracting both the primal and dual solutions.
			Returns			-		--

			Name 			-		_reset
			Arguments		-		None
			Description		-		Processes any new constraints incurred by score changes.
			 						Recalibrates the sdp relaxation to the current state of the device.

	TODO:
		1 - Extend to more than 2 devices.
		2 - Update handling of sdp solution status flags when new ncpol2sdpa is released.
		3 - Additional solver support (Mosek)
	"""
	path = resource_filename(Requirement.parse('dirng'), 'dirng/etc/dirng_config.json')
	with open(path, 'r') as file:
		cfg = json.load(file)
	DEFAULT_SOLVER_PATH = cfg['DEFAULT_SOLVER_PATH']
	SUPPORTED_SOLVERS = cfg['SUPPORTED_SOLVERS']


	#Class constructor
	def __init__(self, settings = {}):
		# Default settings
		# Can be directly modified
		self._name = 'device1'
		self._io_config = [[2,2],[2,2]]
		self._generation_inputs = [0,0]
		self._relaxation_level = 2
		self._solver = None
		self._games = []
		self._score = []
		self._delta = []
		self._verbose = 0
		# Should not be able to be directly modified by user.
		self._sdp = None
		self._gp = None
		self._hmin = None
		self._status = None
		self._solver_name = None
		self._solver_exe = None

		# Keep track when new games are added or scores are modified
		self._needs_relaxing = True
		self._needs_resetting = False
		self._io_config_set = False

		# These are the dual variables that constitute fmin. They are stored within the
		# device - for copying purposes - but all setting / getting is handled by a protocol object.
		# We would only care for these once a protocol object had been initialised anyway.
		self._fmin_variables = []


		# We need to set io_config before we set games, otherwise cg conversion will not work.
		# Need to clean this up....
		if 'io_config' in settings:
			setattr(self, 'io_config', settings['io_config'])
		for setting, value in settings.items():
			if setting != 'io_config':
				setattr(self, setting, value)



	# Directly modifiable attributes
	# Name
	@property
	def name(self):
		return self._name

	@name.setter
	def name(self, value):
		if isinstance(value, str):
			self._name = value
		else:
			raise TypeError('Attribute \'name\' should be a string.')



	# io_config
	@property
	def io_config(self):
		return self._io_config

	@io_config.setter
	def io_config(self, value):
		self._io_config = value
		if self._io_config_set:
			warnings.warn('Changing the input/output configuration may render any existing games played by the device incompatible.', Warning)
		self._io_config_set = True


	# generation inputs
	@property
	def generation_inputs(self):
		return self._generation_inputs

	@generation_inputs.setter
	def generation_inputs(self, value):
		self._generation_inputs = value
		# This can almost certainly just be a reset, need to check though.
		self._needs_relaxing = True

	# generation output size
	@property
	def generation_output_size(self):
		return  np.array([self.io_config[k][self.generation_inputs[k]] for k in range(len(self.generation_inputs))])

	@generation_output_size.setter
	def generation_output_size(self, value):
		for index, val in enumerate(value):
			self._io_config[index][self.generation_inputs[index]] = val

	@property
	def num_inputs(self):
		"""
		Returns the number of inputs for each individual device
		"""
		return map(len, self.io_config)



	# relaxation_level
	@property
	def relaxation_level(self):
		return self._relaxation_level

	@relaxation_level.setter
	def relaxation_level(self, value):
		if isinstance(value, int):
			self._relaxation_level = value
		else:
			raise TypeError('Attribute \'relaxation_level\' should be an integer.')



	# solver
	@property
	def solver(self):
		return self._solver

	@solver.setter
	def solver(self, value):
		# First check if the path supplied is a directory
		if os.path.isdir(value):
			# Look for one of the supported solvers
			for solver in SUPPORTED_SOLVERS:
				new_path = os.path.join(value, solver)
				if os.path.exists(new_path):
					# Hooray we found it
					self._setSolverExe(new_path)
					return None

			raise OSError('Could not find solver \'{}\''.format(value))

		else:
			self._setSolverExe(value)

	def _setSolverExe(self,solver_path):
		directory, name = os.path.split(solver_path)
		if name == '':
			raise OSError('Could not find solver \'{}\''.format(solver_path))

		self._solver_name = name
		self._solver_exe = {'executable' : solver_path}
		if self._solver_name[:4] == 'sdpa' and os.path.exists(os.path.join(directory, 'param.sdpa')):
			self._solver_exe['paramsfile'] = os.path.join(directory, 'param.sdpa')
		elif not os.path.exists(os.path.join(directory, 'param.sdpa')):
			warnings.warn('Could not find param.sdpa file in solver\'s directory. Computation accuracy may be severly affected', Warning)



	# games
	@property
	def games(self):
		return sorted(self._games)

	@games.setter
	def games(self, value):
		# If it is a single game then list it
		if isinstance(value, Game):
			# Update the game object to include its cg attributes
			value._cgmatrix = expression2CG(self.io_config, value.matrix)
			value._cgshift = value._cgmatrix[0][0]
			self._games = [value]

		elif isinstance(value, list):
			self._games = []
			for val in value:
				val._cgmatrix = expression2CG(self.io_config, val.matrix)
				val._cgshift = val._cgmatrix[0][0]
				if isinstance(val, Game):
					self._games += [val]
		else:
			raise TypeError('The games attribute should be set using a single Game object, or a list of Game objects.')

		self._needs_relaxing = True
		# If we relax then we have no need to reset.
		if self._needs_resetting:
			self._needs_resetting = False



	# score
	@property
	def score(self):
		return [game.score for game in self.games]

	@score.setter
	def score(self, value):
		# If just a single score is given then set all games to that score.
		if isinstance(value, float) or isinstance(value, int):
			for game in self._games:
				game.score = value
		# Else if we have a list of scores then sort the games and set accordingly
		else:
			if len(value) != len(self.games):
				raise ValueError('Device has {} game(s) but the supplied score has {} component(s).'.format(len(self.games), len(value)))
			else:
				for index, game in enumerate(self.games):
					game.score = value[index]

		# If we aren't already flagged to relax then we should at least reset.
		if not self._needs_relaxing:
			self._needs_resetting = True

	# _cgshift property of the games played within the devices.
	@property
	def _cgshift(self):
		return np.array([game._cgshift for game in self.games])

	@property
	def _cgshiftscore(self):
		return self.score - self._cgshift



	# delta
	@property
	def delta(self):
		return [game.delta for game in self.games]

	@delta.setter
	def delta(self, value):
		# If just a single score is given then set all games to that score.
		if isinstance(value, float) or isinstance(value, int):
			for game in self._games:
				game.delta = value
		# Else if we have a list of scores then sort the games and set accordingly
		elif isinstance(value, list):
			if len(value) != len(self.games):
				raise ValueError('Device has {} game(s) but the supplied delta has {} component(s).'.format(len(self.games), len(value)))
			else:
				for index, game in enumerate(self.games):
					game.delta = value[index]
		else:
			raise TypeError('Delta should be set by either a float or a list of floats (ordered alphabetically w.r.t. the game names)')



	# verbosity
	@property
	def verbose(self):
		return self._verbose
	@verbose.setter
	def verbose(self, value):
		if isinstance(value, int) or isinstance(value, bool):
			self._verbose = value
		else:
			raise TypeError('Verbose should be set to True/False or some integer value.')



	# gp
	@property
	def gp(self):
		# Check the sdp is in a state to compute
		if self._needs_relaxing:
			self._relax()
			self._needs_relaxing = False
			self._needs_resetting = False
			self._computeGP()
		elif self._needs_resetting:
			self._reset()
			self._needs_resetting = False
			self._computeGP()

		return self._gp

	# Hmin
	@property
	def hmin(self):
		return -log(self.gp, 2)

	#Status
	@property
	def status(self):
		return self._status

	@property
	def fmin_variables(self):
		# If not set yet then set them
		if len(self._fmin_variables) < 1:
			self._fmin_variables = self.dualSolution()
		return self._fmin_variables

	# Custom deepcopy method
	# Needed to avoid problems with copying the _sdp relaxation; sympy's
	# caching system has some ongoing issues with deepcopy.
	# We modify deepcopy to just produce a shallow copy of _sdp
	# We also append 'COPY' to the device's name to indicate it is a copy.
	def __deepcopy__(self, memo):
		cls = self.__class__
		result = cls.__new__(cls)
		memo[id(self)] = result
		for k, v in self.__dict__.items():
			if k == '_sdp':
				setattr(result, k, cpy.copy(v))
			elif k == 'name':
				setattr(result, k, cpy.copy(v) + '-COPY')
			else:
				setattr(result, k, cpy.deepcopy(v, memo))
		return result

	def distribution2Score(self, distribution):
		"""
		Returns score vector for the given distribution
		"""
		return distribution2Score(self.games, distribution)

	def dualSolution(self, v = None):
		"""
		Given an OVG score v. Solve the gp program and return the dual solution.
		If no score is given then it defaults to the current score vector
		"""
		if v is None:
			# Return the dual solution from _sdp object
			dsol = self._getDualSolution()
		else:
			temp_device = cpy.deepcopy(self)
			temp_device.score = v
			dsol = temp_device._getDualSolution()
		return dsol[:]

	#########################################################
	#					   PRIVATE							#
	#########################################################

	def _computeGP(self):
		"""
		Solves the guessing probability program and sets the devices' gp, hmin and solution status accordingly.
		"""
		# Now solve
		self._sdp.solve(self._solver_name, self._solver_exe)
		self._status = cpy.copy(self._sdp.status)

		# TODO currently don't have in-depth status flags so we have to guess what 'unknown' means
		if self.status == 'unknown' and abs(self._sdp.primal - self._sdp.dual) < 0.001:
			self._status = 'optimal'

		if self.status == 'optimal':
			self._gp = - self._sdp.dual
		else:
			self._gp = 1.0
		return self.hmin

	def _getDualSolution(self):
		"""
		Extracts the dual solution to the guessing probability program from the devices' sdp object.
		Returns:
			av		-		Dual solution constant (alpha)
			lv		-		Dual solution vector   (lambda)
			v		-		Dual program parameterisation vector [without cg-translation] (nu)
			_v		-		Dual program parameterisation vector [with cg-translation]
			status	-		Solution status
		"""
		# Check the sdp is in a state to compute
		if self._needs_relaxing:
			self._relax()
			self._needs_relaxing = False
			self._needs_resetting = False
			self._computeGP()
		elif self._needs_resetting:
			self._reset()
			self._needs_resetting = False
			self._computeGP()

		if self.status == 'optimal':
			d_vars = self._sdp.y_mat[np.prod(self.generation_output_size):]
			d_vars = -np.array([d_vars[i][0][0]-d_vars[i+1][0][0] for i in range(0,len(d_vars),2)])
		else:
			d_vars = [1] + [0 for i in range(len(self.games))]

		return d_vars[0], np.array(d_vars[1:]), self.score, self.score - self._cgshift, self.status


	def _relax(self):
		"""
		Creates the sdp relaxation object from ncpol2sdpa.
		"""
		if self.solver == None:
			self.solver = DEFAULT_SOLVER_PATH
		self._eq_cons = []				# equality constraints
		self._proj_cons = {}			# projective constraints
		self._A_ops = []				# Alice's operators
		self._B_ops = []				# Bob's operators
		self._obj = 0					# Objective function
		self._obj_const = ''			# Extra objective normalisation constant
		self._sdp = None				# SDP object

		# Creating the operator constraints
		nrm = ''
		# Need as many decompositions as there are generating outcomes
		for k in range(np.prod(self.generation_output_size)):
			self._A_ops += [ncp.generate_measurements(self.io_config[0],'A' + str(k) + '_')]
			self._B_ops += [ncp.generate_measurements(self.io_config[1],'B' + str(k) + '_')]
			self._proj_cons.update(ncp.projective_measurement_constraints(self._A_ops[k],self._B_ops[k]))

			#Also building a normalisation string for next step
			nrm += '+' + str(k) + '[0,0]'

		# Adding the constraints
		# Normalisation constraint
		self._eq_cons.append(nrm + '-1')

		self._base_constraint_expressions = []
		# Create the game expressions
		for game in self.games:
			tmp_expr = 0
			for k in range(np.prod(self.generation_output_size)):
				tmp_expr += -ncp.define_objective_with_I(game._cgmatrix, self._A_ops[k], self._B_ops[k])

			self._base_constraint_expressions.append(tmp_expr)

		# Specify the scores for these expressions including any shifts
		for i, game in enumerate(self.games):
			#We must account for overshifting in the score coming from the decomposition
			self._eq_cons.append(self._base_constraint_expressions[i] - game.score - game._cgshift*(np.prod(self.generation_output_size)-1))


		self._obj, self._obj_const = guessingProbabilityObjectiveFunction(self.io_config, self.generation_inputs, self._A_ops, self._B_ops)

		# Initialising SDP
		ops = [ncp.flatten([self._A_ops[0],self._B_ops[0]]),
			   ncp.flatten([self._A_ops[1],self._B_ops[1]]),
			   ncp.flatten([self._A_ops[2],self._B_ops[2]]),
			   ncp.flatten([self._A_ops[3],self._B_ops[3]])]

		self._sdp = ncp.SdpRelaxation(ops, verbose=self.verbose, normalized=False)
		self._sdp.get_relaxation(level = self._relaxation_level,
								 momentequalities = self._eq_cons,
								 objective = self._obj,
								 substitutions = self._proj_cons,
								 extraobjexpr = self._obj_const)

	def _reset(self):
		"""
		Recalibrates the sdp relaxation to account for any changes in the game scores.
		"""
		nrm=''
		self._eq_cons=[]		# reset constraints holder

		#Add normalisation constraint
		for k in range(np.prod(self.generation_output_size)):
			nrm += '+' + str(k) + '[0,0]'
		self._eq_cons.append(nrm + '-1')

		#Add constraints from the score vector
		for i, game in enumerate(self.games):
			#We must account for overshifting in the score coming from the decomposition
			self._eq_cons.append(self._base_constraint_expressions[i] - game.score - game._cgshift*(np.prod(self.generation_output_size)-1))

		self._sdp.process_constraints(momentequalities=self._eq_cons)


	#############################################
	#		Printing the device to screen		#
	#############################################

	def __repr__(self):
		A_config_data = [('A' + str(k), str(out)) for k, out in enumerate(self.io_config[0])]
		B_config_data = [('B' + str(k), str(out)) for k, out in enumerate(self.io_config[0])]
		game_data = []
		for game in self.games:
			game_data.append([game.name, game.score, game.delta])
		string = '\r+--+--+--+--+--+--+--+--+--+--+'
		string += '\nI/O CONFIGURATION:\n'
		a_table = tabulate(A_config_data + B_config_data, headers = ['Measurement', 'Num-outcomes'])
		string += a_table
		string += '\n\nGAMES:\n'
		game_table = tabulate(game_data,headers=['Name','Score','Delta'])
		string += game_table

		if self._gp is not None:
			string+= '\n\nSDP RESULTS:\n'
			string += tabulate([['DIGP', 					self.gp],
								['Hmin',					self.hmin],
								['status',		self.status]
								],numalign='right',floatfmt=(None,'.3f','.3f'))
		string += '\n+--+--+--+--+--+--+--+--+--+--+'
		return string
