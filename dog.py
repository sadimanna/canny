import numpy as np
import math
#Oriented Odd Symmetric Gaussian Filter :: First Derivative of Gaussian
def deroGauss(w=5,s=1,angle=0):
	wlim = (w-1)/2
	y,x = np.meshgrid(np.arange(-wlim,wlim+1),np.arange(-wlim,wlim+1))
	G = np.exp(-np.sum((np.square(x),np.square(y)),axis=0)/(2*np.float64(s)**2))
	G = G/np.sum(G)
	dGdx = -np.multiply(x,G)/np.float64(s)**2
	dGdy = -np.multiply(y,G)/np.float64(s)**2

	angle = angle*math.pi/180 #converting to radians

	dog = math.cos(angle)*dGdx + math.sin(angle)*dGdy

	return dog
