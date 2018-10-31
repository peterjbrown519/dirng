"""
Helper functions for defining the untrusted devices
"""
from .games import Game
import numpy as np
import ncpol2sdpa as ncp
import sympy as sym
import copy as cpy

def cgExpressionReduction(io_config, expression):
	"""
	Inputs:
			expression - matrix representing the coefficients s_ab|xy of the Bell expression.
			 			 Eg: for io_config = [[2,2],[2,2]],

						 	|	s00|00  s01|00  s00|01  s01|01	|
			expression =	|	s10|00  s11|00  s10|01  s11|01	|
							|	s00|10  s01|10  s00|11  s01|11	|
							|	s10|10  s11|10  s10|11  s11|11	|

			io_config - input/output configuration

	Returns:
			CG-reduction of the expression matrix.
	"""

	#Adding one to each output results in all the operators being defined
	A = ncp.generate_measurements([int(a+1) for a in io_config[0]], 'A')
	B = ncp.generate_measurements([int(b+1) for b in io_config[1]], 'B')

	op_mat = np.outer(np.array(ncp.flatten(A)),
					  np.array(ncp.flatten(B)))

	#Obtain the overall expression
	expr = np.sum(expression*op_mat)

	#Removing the terms we don't want
	for i in range(len(A)):
		expr = expr.subs(A[i][-1],1-np.sum(A[i][:-1]))
	for j in range(len(B)):
		expr = expr.subs(B[j][-1],1-np.sum(B[j][:-1]))

	# Now we get our full expression
	expr = expr.expand()

	# Removing the additional operator
	for i in range(len(A)):
		A[i] = A[i][:-1]
	for i in range(len(B)):
		B[i] = B[i][:-1]

	A = np.array([1] + ncp.flatten(A))
	B = np.array([1] + ncp.flatten(B))


	# Construct the operator cg-matrix and then find relevant coefficients
	op_mat = np.outer(A,B)
	coefs = expr.as_coefficients_dict()
	for i in range(len(A)):
		for j in range(len(B)):
			if op_mat[i][j] in coefs:
				op_mat[i][j] = coefs[op_mat[i,j]]
			else:
				op_mat[i][j] = 0

	return op_mat

def EBGames(io_config, distribution = None, delta = None):
	"""
	Returns a list of games corresponding to the EB OVG.

	Inputs:
			device 			- 		untrusted device object
			distribution	-		probability distribution in CG-form
			delta			-		delta vector (can just be float value)
	Returns:
			A list of Game objects corresponding to the OVG EB
	"""
	a_in, b_in = list(map(len, io_config))
	a_ops = [''] + ['A' + str(k) for k in range(a_in)]
	b_ops = [''] + ['B' + str(k) for k in range(b_in)]

	empty_mat = [[0 for j in range(b_in+1)] for i in range(a_in+1)]
	games = []
	for i, a in enumerate(a_ops):
		for j, b in enumerate(b_ops):
			mat = cpy.deepcopy(empty_mat)
			mat[i][j] = 1
			games.append(Game(name=a+b, matrix = mat))

	games = sorted(games[1:])

	for ind, game in enumerate(games):
		if distribution is not None:
			game.score = getScoreFromDistribution(game, distribution)

		if delta is not None:
			if isinstance(delta, float):
				game.delta = delta
			else:
				game.delta = delta[ind]

	return games

def getScoreFromDistribution(game, distribution):
	"""
	Inputs:
			game		-		Game object
			dist		-		Distribution (cg-form)
	Returns:
			score
	"""
	score = np.sum(distribution*game.matrix)
	return score

def cgDistributionReduction(io_config, distribution):
	"""
	Takes in a matrix representing the full joint distribution and inputs-outputs
	conveyed by io_config and returns the Collins-Gisin representation of that matrix.
	"""
	aIn, bIn = len(io_config[0]), len(io_config[1])
	joint_block = np.zeros((aIn,bIn))
	a_marginals = np.zeros(aIn)
	b_marginals = np.zeros(bIn)

	# Computing the entries
	for x in range(aIn):
		x_shift = int(np.sum(io_config[0][:x]))
		a_marginals[x] = np.sum(distribution[x_shift][0:io_config[1][0]])
		for y in range(bIn):
			y_shift = int(np.sum(io_config[1][:y]))
			joint_block[x,y] = distribution[x_shift][y_shift]
			b_marginals[y] = np.sum(np.array(distribution)[0:io_config[0][0],y_shift])

	a_marginals = [1] + a_marginals.tolist()
	cg_distribution = np.hstack((np.array([a_marginals]).T,np.vstack((b_marginals, joint_block))))

	return cg_distribution

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
			cg_term = cgExpressionReduction([[aOut],[bOut]], term)

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

def lnspace(a,b,size=50):
	"""
	Creates an array of points in [a,b] for plotting with the stepsize decreasing as
	you approach b.

	Just applies an affine transform to numpy's geomspace in order to reverse the
	step size shifts if b < a.
	"""
	space = np.geomspace(1,10,size)
	return -(a-b)*space/9 + a + (a-b)/9
