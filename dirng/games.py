class Game:
	"""
	Game Class

	Description:
				Object representing a nonlocal game to be played on a PAIR of devices.

	Attributes:
				name				-			A string that acts as a label for the game
				matrix				-			A Collins-Gisin (cg) matrix representation of the scoring rule
												for the game.
												E.g. CHSH on two 2-input 2-output devices has cg representation
																	[[ 0.75,-0.50, 0.00]
																	 [-0.50, 0.50, 0.50]
																	 [ 0.00, 0.50,-0.50]]
												where the top-left element is the constant change in score acquired
												when converting to this representation, i.e. chsh = chsh_cg + 0.75
				score				-			The score achieved by the devices for the given game.
												###NOTE###	This score should be the original score for the nonlocal
												game and NOT the modified cg-score. E.g. for CHSH the score would be
												in the range [0.75, 0.5 + sqrt(2)/4] and NOT [0.0, sqrt(2)/4 - 0.25].
				delta 				-			Statistical confidence associated with a score. I.e. it is assumed that the
												empirical score is contained within the interval [score-delta, score+delta]
												Moreover, the value of delta should be no larger than the score.
	"""
	def __init__(self,name=None,matrix=None,score=0.0,delta=0.0):
		self.name = name
		self.matrix = matrix
		self.score = score
		self.delta = delta

	def __lt__(self, other):
		return self.name < other.name

	def __eq__(self, other):
		return self.name == other.name

	def getData(self):
		return self.name, self.score, self.delta, self.matrix
