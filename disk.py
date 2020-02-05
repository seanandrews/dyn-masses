import os
import sys
import yaml
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import integrate 
from scipy.interpolate import interp1d


class disk:

    # constants
    msun = 1.989e33
    AU = sc.au * 1e2
    mu = 2.37
    m_p = sc.m_p * 1e3
    kB = sc.k * 1e7
    G = sc.G * 1e3

    # fixed values
    min_dens = 1e0  # minimum gas density in [H2/cm**3]
    max_dens = 1e20 # maximum gas density in [H2/cm**3]
    min_temp = 5e0  # minimum temperature in [K]
    max_temp = 5e2  # maximum temperature in [K]


    def __init__(self, modelname, grid, writestruct=True):

        # load parameters
        conf = open(modelname + ".yaml")
        config = yaml.load(conf, Loader=yaml.FullLoader)
        self.hostpars = config["host_params"]
        self.diskpars = config["disk_params"]
        self.setup = config["setup"]
        conf.close()

        # stellar properties
        self.mstar = self.hostpars["M_star"] * self.msun

        # grid properties (spherical coordinate system)
        self.rvals, self.tvals = grid.r_centers, grid.t_centers
        self.nr, self.nt = grid.nr, grid.nt
        self.rr, self.tt = np.meshgrid(self.rvals, self.tvals)

        # corresponding cylindrical quantities
        self.rcyl = self.rr * np.sin(self.tt)
        self.zcyl = self.rr * np.cos(self.tt)


        # compute gas temperature structure
        self.T_args = self.diskpars["temperature"]["arguments"]
        self.temp = self.temperature(**self.T_args)
        self.temp = np.clip(self.temp, self.min_temp, self.max_temp)

        # compute density structure
        self.sig_args = self.diskpars["gas_surface_density"]["arguments"]
        self.sigg = self.sigma_gas(**self.sig_args)
        self.rho_args = {**self.sig_args, **self.T_args, 
                         **self.diskpars["abundance"]["arguments"]}
        self.rhogas, self.nmol = self.density_gas(**self.rho_args)

        # compute velocity structure
        self.vel_args = self.diskpars["rotation"]["arguments"]
        self.vel = self.velocity(**self.vel_args)
        self.vturb_args = self.diskpars["turbulence"]["arguments"]
        self.dvturb = self.vturb(**self.vturb_args)


        if writestruct:
            # open files
            temp_inp = open(modelname+'/gas_temperature.inp', 'w')
            rhog_inp = open(modelname+'/gas_density.inp', 'w')
            mname = self.setup["molecule"]
            nmol_inp = open(modelname+'/numberdens_'+mname+'.inp', 'w')
            vel_inp  = open(modelname+'/gas_velocity.inp', 'w')
            turb_inp = open(modelname+'/microturbulence.inp', 'w')

            # file headers
            temp_inp.write('1\n%d\n' % (self.nr * self.nt))
            rhog_inp.write('1\n%d\n' % (self.nr * self.nt))
            nmol_inp.write('1\n%d\n' % (self.nr * self.nt))
            vel_inp.write('1\n%d\n' % (self.nr * self.nt))
            turb_inp.write('1\n%d\n' % (self.nr * self.nt))

            # populate files
            for j in range(self.nt):
                for i in range(self.nr):
                    temp_inp.write('%.6e\n' % self.temp[j,i])
                    rhog_inp.write('%.6e\n' % self.rhogas[j,i])
                    nmol_inp.write('%.6e\n' % self.nmol[j,i])
                    vel_inp.write('0 0 %.6e\n' % self.vel[j,i])
                    turb_inp.write('%.6e\n' % self.dvturb[j,i])

            # close files
            temp_inp.close()
            rhog_inp.close()
            nmol_inp.close()
            vel_inp.close()
            turb_inp.close()



    ### Temperature Structure.
    def temperature(self, r=None, z=None, **args):

        # Dartois et al. 2003 (type II)
        if (self.diskpars["temperature"]["type"] == 'dartois'):
            try:
                r0 = args["r0_T"] * self.AU
                T0mid, qmid = args["T0mid"], args["qmid"]
                T0atm, qatm = args.pop("T0atm", T0mid), args.pop("qatm", qmid)
                delta, ZqHp = args.pop("delta", 2.0), args.pop("ZqHp", 4.0)
            except KeyError:
                raise ValueError("Specify at least `r0_T`, `T0mid`, `qmid`.")
            r = self.rcyl if r is None else r
            z = self.zcyl if z is None else z
            Tmid = self.powerlaw(r, T0mid, qmid, r0)
            Tatm = self.powerlaw(r, T0atm, qatm, r0)
            zatm = ZqHp * self.scaleheight(r=r, T=Tmid)
            T = np.cos(np.pi * z / 2.0 / zatm)**(2 * delta)
            T = np.where(abs(z) < zatm, (Tmid - Tatm) * T, 0.0)
            return T + Tatm

        # vertically isothermal
        if (self.diskpars["temperature"]["type"] == 'isoz'):
            try:
                r0 = args["r0_T"] * self.AU
                T0mid, qmid = args["T0mid"], args["qmid"]
            except KeyError:
                raise ValueError("Specify at least `rT0`, `T0mid`, `qmid`.")
            r = self.rcyl if r is None else r
            return self.powerlaw(r, T0mid, qmid, r0)


    def scaleheight(self, r=None, T=None):
        T = self.temperature if T is None else T
        r = self.rcyl if r is None else r
        return self.soundspeed(T=T) / np.sqrt(self.G * self.mstar / r**3)


    def soundspeed(self, T=None):
        T = self.temperature if T is None else T
        return np.sqrt(self.kB * T / self.mu / self.m_p)



    # Density Structure.

    def sigma_gas(self, r=None, **args):
    
        # Similarity-solution
        if self.diskpars["gas_surface_density"]["type"] == 'self_similar':
            try:
                Rc, sig0, p1 = args["Rc"] * self.AU, args["sig0"], args["p1"]
                p2 = args.pop("p2", 2.-p1)
            except KeyError:
                raise ValueError("Specify at least `Rc`, `sig0`, `pg1`.")
            r = self.rvals if r is None else r
            return self.powerlaw(r, sig0, -p1, Rc) * np.exp(-(r / Rc)**p2)

        # Power-law
        if self.diskpars["gas_surface_density"]["type"] == 'powerlaw':
            try:
                Rc, sig0, p1 = args["Rc"] * self.AU, args["sig0"], args["p1"]
                p2 = args.pop("p2", 10.)
            except KeyError:
                raise ValueError("Specify at least `Rc`, `sig0`, `pg1`.")
            r = self.rvals if r is None else r
            sig_in  = self.powerlaw(r, sig0, -p1, Rc)
            sig_out = self.powerlaw(r, sig0, -p2, Rc)
            sig = np.where(abs(r) < Rc, sig_in, sig_out)
            return sig


    def sigma_dust(self, r=None, **args):

        # Similarity-solution
        if self.diskpars["dust_surface_density"]["type"] == 'self_similar':
            try:
                Rc, sig0, p1 = args["Rc"] * self.AU, args["sig0"], args["p1"]
                p2 = args.pop("p2", 2.-p1)
            except KeyError:
                raise ValueError("Specify at least `Rc`, `sig0`, `p1`.")
            r = self.rvals if r is None else r
            return self.powerlaw(r, sig0, -p1, Rc) * np.exp(-(r / Rc)**p2)

        # Power-law
        if self.diskpars["dust_surface_density"]["type"] == 'powerlaw':
            try:
                Rc, sig0, p1 = args["Rc"] * self.AU, args["sig0"], args["p1"]
                p2 = args.pop("p2", 10.)
            except KeyError:
                raise ValueError("Specify at least `Rc`, `sig0`, `p1`.")
            r = self.rvals if r is None else r
            sig_in  = self.powerlaw(r, sig0, -p1, Rc)
            sig_out = self.powerlaw(r, sig0, -p2, Rc)
            sig = np.where(abs(r) < Rc, sig_in, sig_out)
            return sig

        
    def density_gas(self, r=None, z=None, **args):

        try:
            xmol = args["xmol"]
        except KeyError:
            print("Specify at least `xmol`.")
        depl = args.pop("depletion", 1e-8)

        # default scenario is to cycle through spherical grid
        if r is None:
            rho_gas = np.zeros((self.nt, self.nr))
            nmol = np.zeros((self.nt, self.nr))
            for i in range(self.nr):
                for j in range(self.nt):

                    # cylindrical coordinates
                    r, z = self.rcyl[j,i], self.zcyl[j,i]

                    # define a special z grid for integration (zg)
                    zmin, zmax, nz = 0.1, 5.*r, 1024
                    zg = np.logspace(np.log10(zmin), np.log10(zmax + zmin), nz) 
                    zg -= zmin

                    # if z >= zmax, return the minimum density
                    if (z >= zmax): 
                        rho_gas[j,i] = self.min_dens * self.m_p * self.mu
                        abund = xmol * depl
                        nmol[j,i] = abund * rho_gas[j,i] / self.m_p / self.mu
                    else:
                        # vertical temperature profile
                        Tz = self.temperature(r, zg, **args)

                        # vertical temperature gradient
                        dT = np.diff(np.log(Tz))
                        dz = np.diff(zg)
                        dlnTdz = np.append(dT, dT[-1]) / np.append(dz, dz[-1])
                
                        # vertical gravity
                        gz = self.G * self.mstar * zg / np.hypot(r, zg)**3
                        gz /= self.soundspeed(T=Tz)**2

                        # vertical density gradient
                        dlnpdz = -dlnTdz - gz

                        # numerical integration
                        lnp = integrate.cumtrapz(dlnpdz, zg, initial=0)
                        rho0 = np.exp(lnp)

                        # normalize
                        rho = 0.5 * rho0 * self.sigma_gas(r=r, **args)
                        rho /= integrate.trapz(rho0, zg)

                        # interpolator to go back to the original gridpoint
                        f = interp1d(zg, rho) 

                        # gas density at specified height
                        min_rho = self.min_dens * self.m_p * self.mu
                        rho_gas[j,i] = np.max([f(z), min_rho])

                        """ Molecular (number) densities """
                        if self.diskpars["abundance"]["type"] == 'chemical':

                            # find the index of the nearest z cell
                            ix = np.argmin(np.abs(zg-z))

                            # integrate density profile *down* to that height
                            sig_ix = integrate.trapz(rho[ix:], zg[ix:])
                            NH2_ix = sig_ix / self.m_p / self.mu

                            # note the column density for photodissociation
                            Npd = 10.**(args.pop("logNpd", 21.11))

                            # note the freezeout temperature
                            Tfreeze = args.pop("tfreeze", 21.0)

                            # compute the molecular abundance
                            if ((NH2_ix >= Npd) & 
                                (self.temperature(r, z, **args) >= Tfreeze)):
                                abund = xmol
                            else: abund = xmol * depl

                        if self.diskpars["abundance"]["type"] == 'layer':

                            # identify the layer heights and radial bounds
                            zrmin = args.pop("zrmin", 0.0)
                            zrmax = args.pop("zrmax", 1.0)
                            rmin = args.pop("rmin", self.rcyl.min()) * self.AU
                            rmax = args.pop("rmax", self.rcyl.max()) * self.AU

                            # compute abundance
                            if ((r > rmin) & (r <= rmax) & 
                                (z/r > zrmin) & (z/r <= zrmax)):
                                abund = xmol
                            else: abund = xmol * depl

                        # molecular number density
                        nmol[j,i] = rho_gas[j,i] * abund / self.m_p / self.mu

        return rho_gas, nmol



    # Dynamical functions.

    def velocity(self, r=None, z=None, **args):

        # Keplerian rotation 
        if self.diskpars["rotation"]["type"] == 'keplerian':

            # bulk rotation
            vkep2 = self.G * self.mstar * self.rcyl**2
            if args.pop("height", True):
                vkep2 /= self.rr**3
            else:
                vkep2 /= self.rcyl**3

            # radial pressure contribution (presumes you've already calculated
	    # density and temperature structures)
            if args.pop("pressure", False):
                # pressure and (cylindrical) radial gradient
                P = self.rhogas * self.kB * self.temp / self.m_p / self.mu
                dPdr = np.gradient(P, self.rvals, axis=1) * np.sin(self.tt) + \
                       np.gradient(P, self.tvals, axis=0) * np.cos(self.tt) / \
                       self.rr
                vprs2 = self.rr * np.sin(self.tt) * dPdr / self.rhogas

            else: vprs2 = 0.0

            # self-gravity
            vgrv2 = 0.0

            # return the combined velocity field
            return np.sqrt(vkep2 + vprs2 + vgrv2)


    def vturb(self, r=None, z=None, **args):
        try:
            xi = args["xi"]
        except KeyError:
            raise ValueError("Specify at least `xi`.")
        
        return self.soundspeed(T=self.temp) * xi

        

    # Analytical functions.

    @staticmethod
    def powerlaw(x, y0, q, x0=1.0):
        """ Simple powerlaw function. """
        return y0 * (x / x0) ** q
