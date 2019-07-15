import numpy as np
import scipy.special as ss

################################################################
# Mie theory calculations of scattering and extinction efficiencies
# for homogeneous spherical particles
# adapted from Matlab code by Robin Wordsworth for EPS237 Planetary Radiation and Climate
# which was adapted from Matlab code by C. Maetzler, June 2002
# http://www.atmo.arizona.edu/students/courselinks/spring08/atmo336s1/courses/spring09/atmo656b/maetzler_mie_v2.pdf
################################################################

def mie_coeff(m, x, nmax):
	'''
	calculate Mie coefficients a_n, b_n
	inputs:
		* m [complex] - index of refraction
		* x [] - size parameter, x = 2*pi*r*m_medium/lambda
		* nmax [] - number of terms in expansion
	outputs:
		* an [] - Mie coefficients
		* bn [] - Mie coefficients
	'''

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

def mie_scatter(m_r,m_i,x0=None,xparams=None,vary_lambda=True,nmax=100,nx=500):
	'''
	calculate Mie scattering and absorption efficiencies vs. size parameter
	inputs:
		* m_r [] - real component of index of refraction
		* m_i [] - imaginary component of index of refraction
		* x0 [] - optional, for scattering efficiencies for one size parameter x0
		* xparams [] - optional, used to calculated size parameters
		if want to vary particle radius or wavelength of indicident light;
		varying radius:
			xparams[0] - particle radius [length unit, same as wavelengths below]
			xparams[1] - index of refraction of particle []
			xparams[2] -  indicident wavelength minumum [length unit, same as radius above]
			xparams[3] - indicident wavelength maximum  [length unit, same as radius above]
		varying wavelength:
			xparams[0] - minimum particle radius [length unit, same as wavelength below]
			xparams[1] - maximum particle radius [length unit, same as wavelength below]
			xparams[2] -  index of refraction of particle []
			xparams[3] - indicident wavelength  [length unit, same as radii above]
		* vary_lambda [boolean] - is lambda varied in x? used to distinguish how xparams is used
		* nmax [int] - number of terms in polynomial expansion
		* nx [int] - number of size parameters to calculate efficiencies for
	outputs:
		* x [] - size parameter
		* Qs []  - scattering efficiency
		* Qe [] - extinction efficiency
	'''
	# control x manually
	if x0!=None:
		x = np.array([x0])
		nx = 1
	# control x from xparams, varying lambda, constant particle radius
	elif xparams!=None and vary_lambda:
		a = xparams[0]
		m_medium = xparams[1]
		lambda_min = xparams[2]
		lambda_max = xparams[3]
		x_min = 2*np.pi*a*m_medium/lambda_max
		x_max = 2*np.pi*a*m_medium/lambda_min
		# size parameter array
		x  = np.logspace(np.log10(x_min),np.log10(x_max),nx)
	# control x from xparams, varying particle radius, constant lambda
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

def Rayleigh(x,m_r):
	'''
	calculate Rayeleigh scattering
	inputs:
		* x [] - size parameter
		* m_r [] - real component of index of refraction
	output:
		* [] - scattering efficiency
	'''
	return (8*x**4./3.)*(m_r**2 - 1)**2/(m_r**2 + 2)**2
