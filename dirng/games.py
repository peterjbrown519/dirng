import warnings
class Game:
	"""
	Game Class

	Description:
				Object representing a nonlocal game to be played on a PAIR of devices.

	Attributes:
				name				-			A string that acts as a label for the game

				matrix				-			A matrix with entries indicating the coefficients of the Bell-expression.
				 								E.g. some matrix

															|	s00|00  s01|00  s00|01  s01|01	|
													expression =	|	s10|00  s11|00  s10|01  s11|01	|
															|	s00|10  s01|10  s00|11  s01|11	|
															|	s10|10  s11|10  s10|11  s11|11	|

				score				-			The score achieved by the devices for the given game.
												Given the above expression this would be
												sum_{abxy} p(ab|xy) s(ab|xy)

				delta 				-			Statistical confidence associated with a score. I.e. it is assumed that the
												empirical score is contained within the interval [score-delta, score+delta]
												Moreover, the value of delta should be no larger than the score.
	"""
	def __init__(self, name = None, matrix = None, score = 0.0, delta = 0.0):
		self._name = name
		self._matrix = matrix
		self._score = score
		self._delta = delta

		# These elements will be processed by a device object
		self.__cgmatrix = None
		self.__cgshift = 0.0

	@property
	def name(self):
		return self._name

	@name.setter
	def name(self, value):
		self._name = value

	@property
	def score(self):
		return self._score

	@score.setter
	def score(self, value):
		if isinstance(value, float) or isinstance(value, int):
			if value < 0 or value > 1:
				warnings.warn('Score has been set outside of the unit interval. Functions like completeness error and eatRate may no longer be directly applicable - Hmin can still be computed. Consider rescaling the games matrix such that 0 <= score <= 1.')
			self._score = value
		else:
			raise TypeError('The game score should be set to float or integer values.')

	@property
	def delta(self):
		return self._delta

	@delta.setter
	def delta(self, value):
		if isinstance(value, float) or isinstance(value, int):
			if 0 <= value and value <= 1:
				self._delta = value
			else:
				raise ValueError('Delta takes values within the unit interval.')
		else:
			raise TypeError('Delta should be a float satisfying 0 <= score <= 1.')

	@property
	def matrix(self):
		return self._matrix

	@matrix.setter
	def matrix(self, value):
		self._matrix = value

	# We will be sorting list of games alphabetically by name. Need to implement relations
	def __lt__(self, other):
		return self.name < other.name
	def __eq__(self, other):
		return self.name == other.name
	# Hacky way of overloading +=
	def __iter__(self):
		return iter([self])
