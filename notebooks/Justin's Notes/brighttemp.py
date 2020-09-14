import numpy as np

def brighttemp(sb, beam, freq, RJ=False): #sb is in Jy/beam
    # unpack axis
    bmaj, bmin = beam #in arcseconds^2

    #Constants
    boltzc = 1.380649e-16 # in erg/K
    sol = 2.99792e10 #in cm/s

    global SI_area
    #Calculate beam area in SI and CGS
    SI_area = np.pi*bmaj*bmin/(4*np.log(2)) # in arcsec^2 units
    sr_area = SI_area*((np.pi/180)/3600)**2 #in sr; 2.35e-11 str = 1 arcsec^2

    #Convert to ergs
    toerg = sb*1e-23/sr_area # in erg Hz^-1 s^-1 cm^-2 units

    #Solve for Temperature
    T_b = (toerg * sol**2)/(2*freq**2*boltzc) # in Kelvin
    return(T_b)
