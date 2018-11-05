"""
Functions for handling the cg_form of matrices
"""
import numpy as np
import ncpol2sdpa as ncp

def expression2CG(io_config, expression):
	"""
	Inputs:

			io_config - input/output configuration

			expression - matrix representing the coefficients s_ab|xy of the Bell expression.
			 			 Eg: for io_config = [[2,2],[2,2]],

						 	|	s00|00  s01|00  s00|01  s01|01	|
			expression =	|	s10|00  s11|00  s10|01  s11|01	|
							|	s00|10  s01|10  s00|11  s01|11	|
							|	s10|10  s11|10  s10|11  s11|11	|

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

def distribution2CG(io_config, distribution):
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
