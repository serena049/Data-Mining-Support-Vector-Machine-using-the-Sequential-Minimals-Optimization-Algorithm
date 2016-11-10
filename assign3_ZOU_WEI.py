#!/usr/bin/python 
# -*- coding: utf-8 -*-

#### Using Python 2.7
#### HW3
#### By Wei Zou


import numpy as np 
from numpy import genfromtxt
import matplotlib.pyplot as plt
import random
import sys




def ComputelinearK(X1,X2):
	K1 = np.dot(X1, np.transpose(X2))
	return K1


def ComputeGaussianK(X1,X2,spread):
	K2 = np.zeros((n,n))
	for i in xrange(n):
		for j in xrange(n):
			K2[i,j] = np.exp(-(np.linalg.norm(X1[i]-X2[j])**2/(2*spread)))
	return K2 


def ComputequadraticK(X1,X2):
	K3 = np.dot(X1**2, np.transpose(X2**2))
	return K3


def ComputeKernal(Kernaltype,X1,X2,spread):
	if (Kernaltype == 'linear'):
		K = ComputelinearK(X1,X2)
	elif (Kernaltype =='quadratic'):
		K= ComputequadraticK(X1,X2)
	elif (Kernaltype == 'Gaussian'):
		K = ComputeGaussianK(X1,X2,spread)
	else:
		print ('wrong kernal type!')

	return K


def ComputeDeltaE(K, Y,alpha,k1,k2):
	K1 = K[:,k1] # K(Xj, Xk1)
	K2 = K[:,k2] # K(Xj, Xk2)
	alpha = np.array(alpha)
	Y = np.array(Y)
	DeltaE = np.dot(alpha*Y,K1)-np.dot(alpha*Y,K2)+(Y[k2]-Y[k1])
	return DeltaE


def Computeb(X,Y,alpha,K):

	b = []
	for i in xrange(n):
		if (alpha[i]>0):
			sum_part = 0
			for j in xrange(n):
				if(alpha[j]>0):
					sum_part += alpha[j]*Y[j]*K[i,j] # equaltion 21.33
			bi = Y[i] - sum_part
			b.append(bi)
	b = sum(b)/float(len(b))
	return b


def Computew(X,Y,alpha):
	w= sum([alpha[i]*Y[i]*X[i] for i in xrange(len(Y)) if alpha[i]>0]) # equation 21.23
	return w



def Computeh(X,Y,Kernaltype,alpha,newX,spread):
	alpha = np.array(alpha)
	Y = np.array(Y)
	h = np.dot(alpha*Y, ComputeKernal(Kernaltype, X,newX,spread))
	return h




def pred_Y(b,h):
	y_pred = np.sign(h+b)  #equation 21.38
	return y_pred




def SMO(Kernaltype, X,Y, epsilon,C,spread):
	alpha = np.zeros(n)
	newalpha = np.zeros(n)
	tol = epsilon
	tryall = True
	alphachange = 100
	K = ComputeKernal(Kernaltype,X,X,spread)


	while(alphachange >eps):
		alpha_prev = alpha.copy()
		
		for j in xrange(n):
			if (tryall == False) and (alpha[j] -tol < 0 or alpha[j]+tol>C):
				continue

			list_to_use = range(n)
			list_to_use.remove(j)
			random.shuffle (list_to_use)
			

			for i in list_to_use:
			
				if (tryall == False) and (alpha[i] -tol < 0 or alpha[i]+tol>C):
					continue
				#print Kernaltype
				
				Kij = K[i,i]+K[j,j]-2*K[i,j]
					
				if (Kij==0):
					continue

				newalpha[j] = alpha[j]
				newalpha[i] = alpha[i]

				if(Y[i] != Y[j]):
					L = max(0, newalpha[j]-newalpha[i])
					H = min(C, C-newalpha[i]+newalpha[j])
				else:
					L = max(0,newalpha[i]+newalpha[j]-C)
					H = min(C, newalpha[i]+newalpha[j])


				if (L==H):
					continue
				deltaE = ComputeDeltaE(K,Y, alpha,i,j)

				alpha[j] = newalpha[j]+Y[j]*deltaE/Kij
			
				if (alpha[j]<L):
					alpha[j] = L
				elif (alpha[j]>H):
					alpha[j] = H

				alpha[i] = newalpha[i]+Y[i]*Y[j]*(newalpha[j]-alpha[j])

		tryall = False

		alphachange = np.linalg.norm(alpha-alpha_prev)




	b = Computeb(X,Y,alpha,K)
	w = Computew(X,Y,alpha)
	h = Computeh(X,Y,Kernaltype,alpha,X,spread)

	Y_pred = pred_Y(b,h)

	accuracy = sum(Y_pred == Y)/float(n)

	#now the print part
	print "The support vectors are:"
	for i in xrange(n):
		if(alpha[i]>0):
			print i, alpha[i] 
	print ""
	print "number of support vectors:", sum(alpha>0)
	print ""
	print "bias:",b
	print ""
	print "weights:",w
	print ""
	print "accuracy:",accuracy


if __name__ == "__main__":

	if (len(sys.argv) == 6):
		filename = sys.argv[1]
		C = float(sys.argv[2])
		eps = float(sys.argv[3])
		Kernaltype = sys.argv[4]
		spread = float(sys.argv[5])
	else:
		filename = sys.argv[1]
		C = float(sys.argv[2])
		eps = float(sys.argv[3])
		Kernaltype = sys.argv[4]
		spread = 0

	data = np.genfromtxt(filename, delimiter=',', dtype='float')
	c = np.shape(data)[1] # number of columns
	n = np.shape(data)[0] # number of obs
	Y = data[:,-1]
	X = data[:,0:-1]


	SMO(Kernaltype, X,Y,eps,C,spread)








