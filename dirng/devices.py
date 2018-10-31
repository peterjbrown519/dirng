import numpy as np
import ncpol2sdpa as ncp
from math import log, sqrt, ceil
import copy as cpy
from tabulate import tabulate
from .device_methods import guessingProbabilityObjectiveFunction
from .games import Game
import os



class Devices:
	"""
	Devices Class

	Description:
				Object representing a pair of untrusted devices. Combined with a nonlocal
				game and score, the UntrustedDevice object computes the guessing probability
				SDP to give associated bounds on Hmin. Should be initialised with a device_settings
				dictionary dictating the initial values of its various attributes.

				Example:
					settings = {
									'name' 					: 		'device1',
									'io_config'				:		[[2,2],[2,2]],
									'generation_inputs'		: 		[0,0],
									'relaxation_level'		:		2,
									'games'					:		[list of Game objects],
									'sdpa_path'				:		'/path/to/sdpa/solver/',
									'verbose'				:		0}

	Attributes:
				name				-			String labelling the devices.
				io_config			-			The input/output configuration specified in array form.
												E.g. If Alice has two inputs, one with 2 outputs and the other with 3 outputs, and
												Bob has 3 inputs with 1, 5 and 6 outputs respectively then we write
												io_config = [[2,3],[1,5,6]].
				generation_inputs	-			Inputs used during generation rounds -- chosen from {0,1,...,|X|-1} and {0,1,...,|Y|-1}. [x_gen, y_gen]
				relaxation_level	-			Level of the NPA hierarchy to use with the sdp relaxation.
				sdpa_path			-			Path to the sdpa solver folder -- '/path/to/sdpa_solver/'.
				verbose				-			Verbosity of ncpol2sdpa package

				_sdp				-			The sdp object from ncpol2sdpa.
				status				-			Solution status for the guessing probability program. If infeasible then gp set to 1.0 and
												hmin set to 0.0

				gp					-			The guessing probability associated with the device pair.
				hmin				-			Min entropy -log(gp,2)

				games 				-			Dictionary storing the nonlocal game objects played by the device pair.

				# DUAL SOLUTIONS
				av					- 			alpha associated with the device
				lv					-			lambda vector
				v					-			score vector
				_v					-			score vector with cg_shift

	PUBLIC Methods:
					Name			-		addGames
					Arguments		-		Takes any number of Game Objects
					Description		-		Adds an entry to the device's games dictionary for the given games and then
											resets the sdp-relaxation object to include the new constraints. If no sdp-relaxation
											has been created then it creates one.
					Returns			-		--

					Name			-		aIn (bIn)
					Arguments		-		None
					Description		-		None
					Returns			-		Number of inputs for Alice's (Bob's) device

					Name			-		computeHmin
					Arguments		-		None
					Description		-		Solves the guessing probability program for the current state of the device
											using the devices' sdp relaxation object.
					Returns			-		hmin

					Name 			-		delta
					Arguments 		- 		None
					Description		-		None
					Returns		 	-		The current delta vector for the nonlocal games played by the
											device pair. Vector is ordered alphabetically with respect to the
											game names.

					Name 			-		dualSolution
					Arguments 		- 		v - score vector
					Description		-		None
					Returns		 	-		The dual solution in the form
													[av, lv, v, _v, status]
											av - dual constant (from normalisation)
											lv - dual vector
											_v - v with the cgShift applied
											status - sdp status when computed

					Name 			-		genOutputSize
					Arguments 		- 		None
					Description		-		None
					Returns		 	-		The number of outputs associated with the generation inputs.

					Name			-		relax
					Arguments		-		None
					Description		-		Creates the sdp relaxation associated with the current state of the device
											using the ncpol2sdpa package. Then solves the guessing probability program,
											extracting both the primal and dual solutions.
					Returns			-		--

					Name 			-		score
					Arguments 		- 		None
					Description		-		None
					Returns		 	-		The current score vector for the nonlocal games played by the
											device pair. Vector is ordered alphabetically with respect to the
											game names.

					Name			-		setDelta
					Arguments		-		delta - float or vector (vector ordered alphabetically w.r.t. game names)
					Description		-		Sets the devices' delta vector.
											Note:
												1 - If float given then sets all delta values equal to float provided
												2 - If vector given then sets delta vector as specified.
					Returns			-		--

					Name			-		setScore
					Arguments		-		w - score vector (Ordered alphabetically w.r.t. game names)
					Description		-		Sets the devices' score vector.
					Returns			-		--

	PRIVATE Methods:
					Name			-		_cgShift
					Arguments		-		None
					Returns			-		A vector of the cg-translations for the various game scores.

					Name			-		_getDualVariables
					Arguments		-		None
					Returns			-		Returns the collection [av,lv,_v,v,status] which constitute a dual solution to the
											computed guessing probability.
											av - alpha
											lv - lambda vector
											v  - score vector parameterising the program without the cg-translation applied.
											_v - score vector parameterising the program with the cg-translation applied.
											status - Solution status returned by solver. (Hopefully is 'optimal')

					Name			-		_saveDualVariables
					Arguments		-		None
					Description		-		Sets the devices' dual solutions av, lv, v, _v to values dictated by current settings.

					Name 			-		_resetSDP
					Arguments		-		None
					Description		-		Processes any new constraints incurred by score changes. Recalibrates the sdp
											relaxation to the current state of the device.

	TODO:
		1 - Extend to more than 2 devices.
		2 - Update handling of sdp solution status flags when new ncpol2sdpa is released.
	"""
	#Class constructor
	def __init__(self, settings):
		# Device's name
		self.name = settings['name']

		# Device alphabet structure
		self.io_config = settings['io_config']

		# Inputs for generation
		self.generation_inputs = settings['generation_inputs']

		# Device's sdp relaxation
		self.relaxation_level = settings['relaxation_level']

		try:
			self.solver_name = 'sdpa'
			if settings['sdpa_path'] != '/':
				settings['sdpa_path'] += '/'
			self.solver_exe = {'executable' : settings['sdpa_path'] + 'sdpa',
							'paramsfile' : settings['sdpa_path'] + 'param.sdpa'}
		except ValueError:
			print('Please add the path to the sdpa solver in the settings dictionary: \'sdpa_path\'')

		if not os.path.exists(self.solver_exe['executable']):
			raise ValueError('Could not find sdpa solver - please check the \'sdpa_path\' setting.')

		# sdp relaxation
		self._sdp = None

		# Guessing probability and hmin
		self.gp = None
		self.hmin = None
		self.status = None

		if 'verbose' in settings:
			self.verbose = settings['verbose']
		else:
			self.verbose = 0

		#Games specified for device
		self.games = {}
		if 'games' in settings:
			self.addGames(*settings['games'])

	# Custom deepcopy method
	# Needed to avoid problems with copying the _sdp relaxation; sympy's
	# caching system has some ongoing issues with deepcopy.
	# We modify deepcopy to just produce a shallow copy of _sdp
	# We also append 'CPY' to the device's name to indicate it is a copy.
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

	#########################################################
	#					   PUBLIC							#
	#########################################################

	def addGames(self,*args):
		"""
		Adds specified nonlocal games objects to the devices' games dictionary.
		Resets the devices' sdp object to incorporate any new games as
		additional constraints.
		"""
		for game in args:
			self.games[game.name] = cpy.deepcopy(game)

		self.relax()

	def aIn(self):
		"""
		Returns number of inputs for Alice's device.
		"""
		return len(self.io_config[0])

	def bIn(self):
		"""
		Returns number of inputs for Bob's device.
		"""
		return len(self.io_config[1])

	def computeHmin(self):
		"""
		Solves the guessing probability program and sets the devices' gp, hmin and solution status accordingly.
		"""
		try:
			self._sdp.solve(self.solver_name,self.solver_exe)
			self.status = cpy.copy(self._sdp.status)
			# TODO currently don't have in-depth status flags so we have to guess what 'unknown' means
			if self.status == 'unknown' and abs(self._sdp.primal - self._sdp.dual) < 0.001:
				self.status = 'optimal'

			if self.status == 'optimal':
				self.gp = - self._sdp.dual
				self.hmin = -log(self.gp,2)
			else:
				self.gp = 1.0
				self.hmin = 0.0
			return self.hmin

		except RuntimeError:
			print('SDP failure - setting trivial solution.')
			self.gp = 1.0
			self.hmin = 0.0
			self.status = 'SDP FAILURE'
			return 0.0

	def delta(self):
		"""
		Returns the current delta vector for the devices' games.
		"""
		return np.array([self.games[name].delta for name in sorted(self.games)])

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
			temp_device.setScore(v)
			dsol = temp_device._getDualSolution()
		return dsol

	def genOutputSize(self):
		"""
		Returns the numbers of outputs associated with the generation input measurements.
		"""
		return np.array([self.io_config[k][self.generation_inputs[k]] for k in range(len(self.generation_inputs))])

	def relax(self):
		"""
		Creates the sdp relaxation object from ncpol2sdpa and then computes hmin.
		"""
		self._eq_cons = []				# equality constraints
		self._proj_cons = {}			# projective constraints
		self._A_ops = []				# Alice's operators
		self._B_ops = []				# Bob's operators
		self._obj = 0					# Objective function
		self._obj_const = ''			# Extra objective normalisation constant
		self._sdp = None				# SDP object

		# Creating the operator constraints
		nrm = ''
		# Need as many decompositions as outcomes
		for k in range(np.prod(self.genOutputSize())):
			self._A_ops += [ncp.generate_measurements(self.io_config[0],'A' + str(k) + '_')]
			self._B_ops += [ncp.generate_measurements(self.io_config[1],'B' + str(k) + '_')]
			self._proj_cons.update(ncp.projective_measurement_constraints(self._A_ops[k],self._B_ops[k]))

			#Also building a normalisation string for next step
			nrm += '+' + str(k) + '[0,0]'

		# Adding the constraints
		#Normalisation constraint
		self._eq_cons.append(nrm + '-1')

		self._base_constraint_expressions = []
		# Create the game expressions
		for name in sorted(self.games):
			tmp_expr = 0
			for k in range(np.prod(self.genOutputSize())):
				tmp_expr += -ncp.define_objective_with_I(self.games[name].matrix,self._A_ops[k],self._B_ops[k])

			self._base_constraint_expressions.append(tmp_expr)

		for i, name in enumerate(sorted(self.games)):
			#We must account for overshifting in the score coming from the decomposition
			self._eq_cons.append(self._base_constraint_expressions[i] - self.games[name].score - self.games[name].matrix[0][0]*(np.prod(self.genOutputSize())-1))


		self._obj, self._obj_const = guessingProbabilityObjectiveFunction(self.io_config, self.generation_inputs, self._A_ops, self._B_ops)

		# Initialising SDP
		ops = [ncp.flatten([self._A_ops[0],self._B_ops[0]]),
			   ncp.flatten([self._A_ops[1],self._B_ops[1]]),
			   ncp.flatten([self._A_ops[2],self._B_ops[2]]),
			   ncp.flatten([self._A_ops[3],self._B_ops[3]])]

		self._sdp = ncp.SdpRelaxation(ops,verbose=self.verbose, normalized=False)
		self._sdp.get_relaxation(level = self.relaxation_level,
								 momentequalities = self._eq_cons,
								 objective = self._obj,
								 substitutions = self._proj_cons,
								 extraobjexpr = self._obj_const)
		self.computeHmin()

	def score(self):
		"""
		Returns the current score vector for the devices.
		"""
		return np.array([self.games[name].score for name in sorted(self.games)])

	def scoreVectorFromDistribution(self, distribution):
		"""
		Returns score vector for the given distribution (cg-form).
		"""
		for name in sorted(self.games):
			score = [np.sum(distribution*self.games[name].matrix) for name in sorted(self.games)]

		return score

	def setDelta(self, delta):
		"""
		Sets the delta values for the game played by the device.
		Note:
		 	1 - A single float value can be presented and then each game's delta receives this value.
			2 - If specifying the entire vector then values should be ordered alphabetically with reference
				to the game names.
		"""
		# Construct appropriate delta vector
		if isinstance(delta, float):
			delta_vector = np.array([delta for _ in range(len(self.games))])
		else:
			delta_vector = delta[:]

		# Set each game's delta value.
		for ind, name in enumerate(sorted(self.games)):
			self.games[name].delta = delta_vector[ind]

	def setScore(self, w):
		"""
		Sets the score vector for the device.
		NOTE: The elements of the score vector should be ordered alphabetically
			  with respect to their individual game names.
		"""
		for ind, name in enumerate(sorted(self.games)):
			self.games[name].score = w[ind]

		if self._sdp is None:
			self.relax()
		else:
			self._resetSDP()

	#########################################################
	#					   PRIVATE							#
	#########################################################

	def _cgShift(self):
		"""
		Returns the cg-translation vector for the nonlocal game scores.
		"""
		return np.array([self.games[name].matrix[0][0] for name in sorted(self.games)])

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
		if self.status == 'optimal':
			d_vars = self._sdp.y_mat[np.prod(self.genOutputSize()):]
			d_vars = -np.array([d_vars[i][0][0]-d_vars[i+1][0][0] for i in range(0,len(d_vars),2)])
		else:
			d_vars = [1] + [0 for i in range(len(self.games))]

		return d_vars[0], np.array(d_vars[1:]), self.score(), self.score() - self._cgShift(), self.status


	def _resetSDP(self):
		"""
		Recalibrates the sdp relaxation to account for any changes in the game scores.
		"""
		nrm=''
		self._eq_cons=[]		# reset constraints holder

		#Add normalisation constraint
		for k in range(np.prod(self.genOutputSize())):
			nrm += '+' + str(k) + '[0,0]'
		self._eq_cons.append(nrm + '-1')

		#Add constraints from the score vector
		for i, name in enumerate(sorted(self.games)):
			#We must account for overshifting in the score coming from the decomposition
			self._eq_cons.append(self._base_constraint_expressions[i] - self.games[name].score - self.games[name].matrix[0][0]*(np.prod(self.genOutputSize())-1))

		self._sdp.process_constraints(momentequalities=self._eq_cons)
		self.computeHmin()


	#############################################
	#		Printing the device to screen		#
	#############################################

	def __repr__(self):
		A_config_data = [('A' + str(k), str(out)) for k, out in enumerate(self.io_config[0])]
		B_config_data = [('B' + str(k), str(out)) for k, out in enumerate(self.io_config[0])]
		game_data = []
		for name in sorted(self.games):
			game_data.append(self.games[name].getData()[:-1])
		string = '\r+--+--+--+--+--+--+--+--+--+--+'
		string += '\nI/O CONFIGURATION:\n'
		a_table = tabulate(A_config_data + B_config_data, headers = ['Measurement', 'Num-outcomes'])
		string += a_table
		string += '\n\nGAMES:\n'
		game_table = tabulate(game_data,headers=['Name','Score','Delta'])
		string += game_table

		if self.gp is not None:
			string+= '\n\nSDP RESULTS:\n'
			string += tabulate([['DIGP', 					self.gp],
								['Hmin',					self.hmin],
								['status',		self.status]
								],numalign='right',floatfmt=(None,'.3f','.3f'))
		string += '\n+--+--+--+--+--+--+--+--+--+--+'
		return string
