import numpy as np
import scipy.special as ss

# adapted from Matlab code by C. Maetzler, June 2002
# http://arrc.ou.edu/~rockee/NRA_2007_website/Mie-scattering-Matlab.pdf
# and from Robin Wordsworth for EPS237 Planetary Radiation and Climate

# from complex index of refraction (m = m_r + i*m_i), size parameter
# (x = 2*pi*a*m_medium/lambda), and number of terms in expansion (nmax),
# calculate the associated Mie coefficients an, bn
def mie_coeff(m, x, nmax):

	#create n array
	n = np.arange(1,nmax+1)
	nu = n + 0.5
	y = m*x
	m2 = m*m

	#calculate Bessel function quantities
	bx = ss.jv(nu,x)*(0.5*np.pi/x)**0.5
	by = ss.jv(nu,y)*(0.5*np.pi/y)**0.5
	yx = ss.yv(nu,x)*(0.5*np.pi/x)**0.5

	hx = bx + 1j*yx
	b1x = np.zeros(nmax,dtype=complex)
	b1x[0] = np.sin(x)/x
	b1x[1:nmax] = bx[0:nmax-1]
	b1y = np.zeros(nmax,dtype=complex)
	b1y[0] = np.sin(y)/y
	b1y[1:nmax] = by[0:nmax-1]
	y1x = np.zeros(nmax,dtype=complex)
	y1x[0] = -np.cos(x)/x
	y1x[1:nmax] = yx[0:nmax-1]
	h1x = b1x + 1j*y1x
	ax  = x*b1x - n*bx
	ay  = y*b1y - n*by
	ahx = x*h1x - n*hx

	#finally the coefficients themselves
	an  = (m2*by*ax - bx*ay)/(m2*by*ahx - hx*ay)
	bn  = (by*ax - bx*ay)/(by*ahx - hx*ay)

	return an, bn

#calculate Mie scattering efficiency vs. size parameter
def mie_scatter(m_r,m_i,x0=None,xparams=None,vary_lambda=True,nmax=100,nx=500):
	# control x manually
	if x0!=None:
		x = np.array([x0])
		nx = 1
	elif xparams!=None and vary_lambda:
		a = xparams[0]
		m_medium = xparams[1]
		lambda_min = xparams[2]
		lambda_max = xparams[3]
		x_min = 2*np.pi*a*m_medium/lambda_max
		x_max = 2*np.pi*a*m_medium/lambda_min

		# size parameter array
		x  = np.logspace(np.log10(x_min),np.log10(x_max),nx)
	elif xparams!=None:
		a_min = xparams[0]
		a_max = xparams[1]
		m_medium = xparams[2]
		lam = xparams[3]
		x_min = 2*np.pi*a_min*m_medium/lam
		x_max = 2*np.pi*a_max*m_medium/lam
		# size parameter array
		x  = np.logspace(np.log10(x_min),np.log10(x_max),nx)

	# otherwise just take range where Mie theory is accurate
	else:
		x = np.logspace(-2,2,nx)

	# number of terms in polynomial expansion
	n    = np.arange(1,nmax+1)
	m    = complex(m_r,m_i)
	y    = m*x

	# set up empty scattering and extinction efficiency arrays
	Qs   = np.zeros(nx)
	Qe   = np.zeros(nx)
	for i in range(nx): # loop over Mie size parameter
		a, b   = mie_coeff(m,x[i],nmax)
		# scattering and extinction efficiencies
		Qs[i] = np.sum((2*n+1)*(abs(a)**2 + abs(b)**2))
		Qe[i] = np.sum((2*n+1)*(a + b).real)

	Qs = (2/x**2)*Qs
	Qe = (2/x**2)*Qe

	return x, Qs, Qe

# Rayleigh scattering
def Rayleigh(x,m_r):
	return (8*x**4./3.)*(m_r**2 - 1)**2/(m_r**2 + 2)**2

# approx scattering effeciency in no absorbtion regime (m_i=0)
def vandehulst_approx(x,m_r):
	rho = 2*x*(m_r-1)
	return 2 - 4.*np.sin(rho)/rho + 4./rho**2*(1-np.cos(rho))
