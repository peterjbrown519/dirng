"""
Helper functions for defining the untrusted devices
"""
from .games import Game
from .cg_methods import expression2CG
import numpy as np
import ncpol2sdpa as ncp
import sympy as sym
import copy as cpy


def EBGames(io_config, distribution = None, delta = None):
	"""
	Returns a (reduced set of games) corresponding to the EB OVG.

	Inputs:
			device 			- 		untrusted device object
			distribution	-		probability distribution
			delta			-		delta vector (can just be float value)
	Returns:
			A list of Game objects corresponding to the OVG EB
	"""
	a_in, b_in = map(len, io_config)

	games = []
	# The joint distribution components first
	for x in range(len(io_config[0])):
		for y in range(len(io_config[1])):
			for a in range(io_config[0][x] - 1):
				for b in range(io_config[1][y] - 1):
					mat = np.zeros([sum(io_config[0]), sum(io_config[1])])
					mat[sum(io_config[0][:x]) + a][sum(io_config[1][:y]) + b] = 1.0
					games.append(Game(name='p(' + str(a) + str(b) + '|' + str(x) + str(y) + ')', matrix = mat))

	# Now Alice's marginals
	for x in range(len(io_config[0])):
		for a in range(io_config[0][x]-1):
			mat = np.zeros([sum(io_config[0]), sum(io_config[1])])
			for b in range(io_config[1][0]):
				mat[sum(io_config[0][:x]) + a][b] = 1.0
			games.append(Game(name='p(' + str(a) + '|' + str(x) + ')_A', matrix = mat))

	# Now Bob's marginals
	for y in range(len(io_config[1])):
		for b in range(io_config[1][y]-1):
			mat = np.zeros([sum(io_config[0]), sum(io_config[1])])
			for a in range(io_config[0][0]):
				mat[a][sum(io_config[1][:y]) + b] = 1.0
			games.append(Game(name='p(' + str(b) + '|' + str(y) + ')_B', matrix = mat))



	# Set score and delta
	for ind, game in enumerate(games):
		if distribution is not None:
			game.score = distribution2Score(game, distribution)

		if delta is not None:
			if isinstance(delta, float):
				game.delta = delta
			else:
				game.delta = delta[ind]

	return games

def distribution2Score(games, distribution):
	"""
	Inputs:
			games				-		Game object or list
			distribution		-		Distribution
	Returns:
			score				- 		sorted alphabetically w.r.t. game names if multiple
	"""
	if isinstance(games, Game):
		score = np.sum(np.array(distribution)*np.array(games.matrix))
	else:
		score = []
		for game in sorted(games):
			score.append(np.sum(np.array(distribution)*np.array(game.matrix)))
	return score

def guessingProbabilityObjectiveFunction(io_config, generation_inputs, A, B):
	"""
	Computes the objective function for the guessing probability program for a pair of devices.
	Requires passing the input-output configuration of the devices, the generation inputs and the
	ncpol2sdpa operators used with the relaxation (A and B).

	Returns the operator expression and any additional expressions necessary for the computation of
	the program.
	"""

	aGen, bGen = generation_inputs[0], generation_inputs[1]
	aOut, bOut = io_config[0][aGen], io_config[1][bGen]

	gp_objective_function = 0.0
	# Only the final term will have an additional constant.
	gp_additional_expression = "-" + str(aOut*bOut - 1) + "[0,0]"

	# We loop through all of the possible outputs for the distribution and add their cg reduced expressions
	# to the gp objective function.
	for a in range(aOut):
		for b in range(bOut):
			# Index telling us which decomposition we look at
			term_index = a*bOut + b

			# Use the cg-reduction method to get the corresponding expression
			# Minus 1.0 for maximisation
			term = np.zeros((aOut,bOut))
			term[a,b] = -1.0
			cg_term = expression2CG([[aOut],[bOut]], term)

			# Extract sympy expression
			current_expression = 0.0

			# First run through Bob's operators
			for index, value in enumerate(cg_term[0,1:]):
				current_expression += value*B[term_index][bGen][index]

			# Then Alice's
			for index, value in enumerate(cg_term[1:,0]):
				current_expression += value*A[term_index][aGen][index]

			# The their joint operators
			for i in range(aOut-1):
				for j in range(bOut-1):
					current_expression += cg_term[i+1,j+1]*A[term_index][aGen][i]*B[term_index][bGen][j]

			gp_objective_function += current_expression

	return gp_objective_function, gp_additional_expression
