from math import exp
from tabulate import tabulate
from dirng.eat_methods import eatRate, eatRateGA, optimiseFminChoice


class Protocol:
	"""
	Protocol Class

	Description:
				Object representing a device-independent randomness expansion protocol.
				Stores the chosen protocol parameters and when combined with an untrusted device object,
				can be used to calculate the relevant entropy accumulation rates.

				Initialisation requires a parameters dictionary containing initial values for
				the protocol parameters. These values can be changed later.

				Example:
					parameters = {
									'number_of_rounds' 		: 		number of rounds in the protocol
									'test_probability'		: 		probability that any given round is a test round
									'eps_smooth'			:		smoothing epsilon for Hmin
									'eps_eat'				:		EAT epsilon (maximum probability with which the bound on the accumulated entropy could be wrong)
					}

	Attributes:
				n				-			number of rounds
				y				-			probability that any given round is a test round
				eps_s			-			Hmin smoothing epsilon
				eps_eat 		-			EAT epsilon
				fmin			-			Dictionary containing the dual solution corresponding to the min-tradeoff function choice.
											Key is the devices' name.

	PUBLIC Methods:
					Name 			-		completeness
					Arguments 		- 		device 		(Class: Devices)
					Description 	-		Calculates the completeness error of the protocol.
					Returns			-		completeness error

					Name 			-		setFmin
					Arguments 		- 		device	- Untrusted device object
											v 		- OVG score vector
					Description 	-		Updates the protocol's fmin dictionary. Setting the device.name entry
											to the dual solution to the GP program for the vector v.
					Returns			-		--

	"""
	def __init__(self, settings = {}):
		# Default settings
		self._n = 1e10
		self._y = 1e-2
		self._eps_smooth = 1e-8
		self._eps_eat = 1e-8

		# Process any settings passed
		for setting, value in settings.items():
			setattr(self, setting, value)

	@property
	def n(self):
		return self._n
	@n.setter
	def n(self, value):
		if isinstance(value, int):
			if value > 0:
				self._n = value
			else:
				raise ValueError('Number of rounds (n) should be positive.')
		else:
			# If can be cast to an integer then do so.
			try:
				if value > 0:
					self._n = int(value)
				else:
					raise ValueError('Number of rounds (n) should be positive.')
			except:
				raise TypeError('Number of rounds (n) should be an integer.')

	@property
	def y(self):
		return self._y
	@y.setter
	def y(self, value):
		if isinstance(value, float):
			if 0 < value and value <= 1:
				self._y = value
			else:
				raise ValueError('Test probability (y) takes values in (0,1].')
		else:
			raise TypeError('Test probability (y) should be a float.')

	@property
	def eps_smooth(self):
		return self._eps_smooth
	@eps_smooth.setter
	def eps_smooth(self, value):
		if isinstance(value, float):
			if 0 < value and value <= 1:
				self._eps_smooth = value
			else:
				raise ValueError('Smoothing epsilon (eps_smooth) takes values in (0,1].')
		else:
			raise TypeError('Smoothing epsilon (eps_smooth) should be a float.')

	@property
	def eps_eat(self):
		return self._eps_eat
	@eps_eat.setter
	def eps_eat(self, value):
		if isinstance(value, float):
			if 0 < value and value <= 1:
				self._eps_eat = value
			else:
				raise ValueError('EAT epsilon (eps_eat) takes values in (0,1].')
		else:
			raise TypeError('EAT epsilon (eps_eat) should be a float.')


	def completeness(self, device):
		"""
		Calculates the completeness error of the protocol for a specified device.
		"""
		total = 0
		d = device.delta
		w = device.score
		# This form of the Chernoff bound should have d <= w. If the components
		# don't satisfy this then we can trivially map to a score which would (1-w).
		# As all scores are in (0,1).
		# For this reason, we do not need to enforce the delta constraints within the
		# Game object. 
		for i in range(len(d)):
			if d[i] <= w[i]:
				total += 2*exp(-self.y*(d[i]**2)*self.n/(3 * w[i]))
			else:
				total += 2*exp(-self.y*(d[i]**2)*self.n/(3 * (1-w[i])))
		return min([total,1.0])

	def setFmin(self, device, v_choice = None):
		"""
		Sets parameters indicating min-tradeoff function choice
		"""
		# Check if v choice was specified
		if v_choice is None:
			v_choice = device.dualSolution()
		elif isinstance(v_choice[-1], float):
			# If only v was passed then get the rest of the solution
			v_choice = device.dualSolution(v_choice[:])

		device._fmin_variables = v_choice[:]

	# Functions from eat_methods
	def eatRate(self, device, v_choice=None, w=None):
		return eatRate(self, device, v_choice, w)
	def eatRateGA(self, device, v_choice = None, w = None, step_size = 0.01, min_step_size = 1.0e-4, tol = 1.0e-6, h=1.0e-6, verbose = 0, update=True):
		return eatRateGA(self, device, v_choice, w, step_size, min_step_size, tol, h, verbose, update)
	def optimiseFminChoice(self, device, v_choice = None, w = None,
							num_iterations = 0, jump_radius = 0.001,
							step_size = 0.005, min_step_size = 1.0e-5,
							tol = 1.0e-5, h=1.0e-5, verbose = 0, update = True):
		return optimiseFminChoice(self, device, v_choice, w,
								num_iterations, jump_radius,
								step_size, min_step_size,
								tol, h, verbose, update)

	# String produced when print is called
	def __repr__(self):
		print_string = "\n\nPROTOCOL SETTINGS:\n"
		print_string += tabulate([	["Number of rounds", 		self.n],
						  			["Test probability", 		self.y],
						  			['epsilon_s',				self.eps_smooth],
						  			['epsilon_eat',				self.eps_eat]
						  			],numalign='right')
		return print_string
