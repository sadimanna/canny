import numpy as np
import cv2, time, math
from scipy.signal import convolve2d as conv2
from matplotlib import pyplot as plt
from bilateralfilt import bilatfilt
from dog import deroGauss
#...........................................................................................
def get_edges(I,sd):
	dim = I.shape
	Idog2d = np.zeros((nang,dim[0],dim[1]))
	for i in xrange(nang):
		dog2d = deroGauss(5,sd,angles[i])
		Idog2dtemp = abs(conv2(I,dog2d,mode='same',boundary='fill'))
		Idog2dtemp[Idog2dtemp<0]=0
		Idog2d[i,:,:] = Idog2dtemp
	return Idog2d
#...........................................................................................
def nonmaxsup(I,gradang):
	dim = I.shape
	Inms = np.zeros(dim)
	xshift = int(np.round(math.cos(gradang*np.pi/180)))
	yshift = int(np.round(math.sin(gradang*np.pi/180)))
	Ipad = np.pad(I,(1,),'constant',constant_values = (0,0))
	for r in xrange(1,dim[0]+1):
		for c in xrange(1,dim[1]+1):
			maggrad = [Ipad[r-xshift,c-yshift],Ipad[r,c],Ipad[r+xshift,c+yshift]]
			if Ipad[r,c] == np.max(maggrad):
				Inms[r-1,c-1] = Ipad[r,c]
	return Inms
#...........................................................................................
def calc_sigt(I,threshval):
	M,N = I.shape
	ulim = np.uint8(np.max(I))	
	N1 = np.count_nonzero(I>threshval)
	N2 = np.count_nonzero(I<=threshval)
	w1 = np.float64(N1)/(M*N)
	w2 = np.float64(N2)/(M*N)
	#print N1,N2,w1,w2
	try:
		u1 = np.sum(i*np.count_nonzero(np.multiply(I>i-0.5,I<=i+0.5))/N1 for i in range(threshval+1,ulim))
		u2 = np.sum(i*np.count_nonzero(np.multiply(I>i-0.5,I<=i+0.5))/N2 for i in range(threshval+1))
		uT = u1*w1+u2*w2
		sigt = w1*w2*(u1-u2)**2
		#print u1,u2,uT,sigt
	except:
		return 0
	return sigt
#...........................................................................................
def get_threshold(I):
	max_sigt = 0
	opt_t = 0
	ulim = np.uint8(np.max(I))
	print ulim
	for t in xrange(ulim+1):
		sigt = calc_sigt(I,t)
		#print t, sigt
		if sigt > max_sigt:
			max_sigt = sigt
			opt_t = t
	print 'optimal high threshold: ',opt_t
	return opt_t
	
#...........................................................................................
def threshold(I,uth):
	lth = uth/2.5
	Ith = np.zeros(I.shape)
	Ith[I>=uth] = 255
	Ith[I<lth] = 0
	Ith[np.multiply(I>=lth, I<uth)] = 100
	return Ith
#...........................................................................................
def hysteresis(I):
	r,c = I.shape
	#xshift = int(np.round(math.cos(gradang*np.pi/180)))
	#yshift = int(np.round(math.sin(gradang*np.pi/180)))
	Ipad = np.pad(I,(1,),'edge')
	c255 = np.count_nonzero(Ipad==255)
	imgchange = True
	for i in xrange(1,r+1):
		for j in xrange(1,c+1):
			if Ipad[i,j] == 100:
				#if Ipad[i-xshift,j+yshift]==255 or Ipad[i+xshift,j-yshift]==255: 
				if np.count_nonzero(Ipad[r-1:r+1,c-1:c+1]==255)>0:
					Ipad[i,j] = 255
				else:
					Ipad[i,j] = 0
	Ih = Ipad[1:r+1,1:c+1]
	return Ih
#...........................................................................................
#Reading the image
img = cv2.imread('108073.jpg')
gimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
dim = img.shape
#...........................................................................................
#Bilateral filtering
print 'Bilateral filtering...\n'
gimg = bilatfilt(gimg,5,3,10)
print 'after bilat: ',np.max(gimg),'\n'
#...........................................................................................
stime = time.time()
angles = [0,45,90,135]
nang = len(angles)
#...........................................................................................
#Gradient of Image
print 'Calculating Gradient...\n'
img_edges = get_edges(gimg,2)
print 'after gradient: ',np.max(img_edges),'\n'
#...........................................................................................
#Non-max suppression
print 'Suppressing Non-maximas...\n'
for n in xrange(nang):
	img_edges[n,:,:] = nonmaxsup(img_edges[n,:,:],angles[n])
print 'after nms: ', np.max(img_edges)
img_edge = np.max(img_edges,axis=0)
lim = np.uint8(np.max(img_edge))
plt.imshow(img_edge)
plt.show()
#...........................................................................................
#Converting to uint8
#img_edges_uint8 = np.uint8(img_edges)
#...........................................................................................
#Thresholding
print 'Calculating Threshold...\n'
th = get_threshold(gimg)
the = get_threshold(img_edge)
#...........................................................................................
print '\nThresholding...\n'
img_edge = threshold(img_edge, the*0.25)
#cv2.imshow('afterthe',img_edge)
#...........................................................................................
#Hysteresis
print 'Applying Hysteresis...\n'
#for i in xrange(nang):
img_edge = nonmaxsup(hysteresis(img_edge),90)
#...........................................................................................
#img_edge = np.max(img_edges,axis=0)
#...........................................................................................
#OpenCV Canny Function
img_canny = cv2.Canny(np.uint8(gimg),th/3,th)
cv2.imshow('Uncanny',img_edge)
cv2.imshow('Canny',img_canny)
print 'Time taken :: ', str(time.time()-stime)+' seconds...'
cv2.waitKey(0)
