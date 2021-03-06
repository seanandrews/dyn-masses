import os
import sys
import yaml
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

class plotutils:

    # constants
    msun = 1.989e33
    AU = sc.au * 1e2
    mu = 2.37
    m_p = sc.m_p * 1e3
    kB = sc.k * 1e7
    G = sc.G * 1e3


    def __init__(self, mname, struct=None, cyl=False):

        self.mname = mname

        # load parameter file
        conf = open(mname + ".yaml")
        config = yaml.load(conf, Loader=yaml.FullLoader)
        self.setup = config["setup"]
        self.diskpars = config["disk_params"]
        conf.close()

        # if no structure is passed, extract it from files
        if struct is None:
            # spatial grid
            _ = np.loadtxt(mname+'/amr_grid.inp', skiprows=5, max_rows=1)
            nr, nt = np.int(_[0]), np.int(_[1])
            Rw = np.loadtxt(mname+'/amr_grid.inp', skiprows=6, max_rows=nr+1)
            Tw = np.loadtxt(mname+'/amr_grid.inp', skiprows=nr+7, max_rows=nt+1)
            self.Rgrid = 0.5*(Rw[:-1] + Rw[1:])
            self.Tgrid = 0.5*(Tw[:-1] + Tw[1:])

            if self.setup["incl_dust"]:
                # dust densities
                ndust = int(np.loadtxt(mname+'/dust_density.inp',
                            skiprows=2, max_rows=1))
                rhod_in = np.loadtxt(mname+'/dust_density.inp', skiprows=3)
                if (ndust > 1):
                    rhod_a = np.reshape(rhod_in, (ndust, nt, nr))
                    self.rhod = np.sum(rhod_a, axis=0)
                    #dind, all_acm = np.loadtxt('opacs/dsharp_sizeindex.txt').T
                    #acm = all_acm[:ndust]
                    #self.rhod = np.trapz(rhod_a, acm, axis=0)
                else: self.rhod = np.reshape(rhod_in, (nt, nr))

                _ = self.plot_dustdens()
                _.savefig(mname+'/dustdens.png')

                # radial profiles
                self.dr, self.dsig, self.dH = np.loadtxt(mname + \
                                                         '/dust_profiles.txt', \
                                                         skiprows=1).T
                _ = self.plot_sigd()
                _.savefig(mname+'/sigma_d.png')
                _ = self.plot_Hdust()
                _.savefig(mname+'/Hdust.png')

                # temperatures
                T_in = np.loadtxt(mname+'/dust_temperature.dat', skiprows=3)
                if (ndust > 1):
                    temp_a = np.reshape(T_in, (ndust, nt, nr))
                    self.temp = np.sum(temp_a * rhod_a, axis=0) / \
                                np.sum(rhod_a, axis=0)
                else: self.temp = np.reshape(T_in, (nt, nr))

            else:
                T_in = np.loadtxt(mname+'/gas_temperature.inp', skiprows=2)
                self.temp = np.reshape(T_in, (nt, nr))
                
            _ = self.plot_temp()
            _.savefig(mname+'/temp.png')


            if self.setup["incl_lines"]:
                # gas densities
                rhog_in = np.loadtxt(mname+'/gas_density.inp', skiprows=2)
                self.rhog = np.reshape(rhog_in, (nt, nr))
                _ = self.plot_gasdens()
                _.savefig(mname+'/gasdens.png')

                # molecular volume densities
                mol = self.setup["molecule"]
                nmol_in = np.loadtxt(mname+'/numberdens_'+mol+'.inp', 
                                     skiprows=2)
                self.nmol = np.reshape(nmol_in, (nt, nr))
                _ = self.plot_nmol()
                _.savefig(mname+'/nmol.png')

                # gas velocities
                vel_in = np.loadtxt(mname+'/gas_velocity.inp', skiprows=2)
                self.vel = np.reshape(vel_in[:,2], (nt, nr))
                _ = self.plot_vel()
                _.savefig(mname+'/vel.png')

                # microturbulence
                vturb_in = np.loadtxt(mname+'/microturbulence.inp', skiprows=2)
                self.dvturb = np.reshape(vturb_in, (nt, nr))
                _ = self.plot_vturb()
                _.savefig(mname+'/vturb.png')

                # radial profiles
                self.gr, self.gsig, self.gH = np.loadtxt(mname + \
                                                         '/gas_profiles.txt', \
                                                         skiprows=1).T
                _ = self.plot_sigg()
                _.savefig(mname+'/sigma_g.png')
                _ = self.plot_Hp()
                _.savefig(mname+'/Hp.png')


            if (self.setup["incl_lines"] and self.setup["incl_dust"]):
                _ = self.plot_sigboth()
                _.savefig(mname+'/sigma_dg.png')
                _ = self.plot_Hboth()
                _.savefig(mname+'/Hboth.png')




        # otherwise, just use the passed structure
        else:
            print('using struct')

            # if you don't have a model directory, make one
            if not os.path.exists(mname): os.mkdir(mname)

            # spatial axes
            if cyl:
                self.Rgrid = struct.rvals
                self.Zgrid = struct.zvals
                fsuf = '_cyl'
            else:
                self.Rgrid = struct.rvals
                self.Tgrid = struct.tvals
                fsuf = ''

            # temperatures
            self.temp = struct.temp
            _ = self.plot_temp(cyl=cyl)
            _.savefig(mname+'/temp'+fsuf+'.png')

            if self.setup["incl_lines"]:
                self.rhog = struct.rhogas
                _ = self.plot_gasdens(cyl=cyl)
                _.savefig(mname+'/gasdens'+fsuf+'.png')

                self.nmol = struct.nmol
                _ = self.plot_nmol(cyl=cyl)
                _.savefig(mname+'/nmol'+fsuf+'.png')

                self.vel = struct.vel
                _ = self.plot_vel(cyl=cyl)
                _.savefig(mname+'/vel'+fsuf+'.png')

                self.dvturb = struct.dvturb
                _ = self.plot_vturb(cyl=cyl)
                _.savefig(mname+'/vturb'+fsuf+'.png')

            if self.setup["incl_dust"]:
                self.rhod = struct.rhodust
                _ = self.plot_dustdens(cyl=cyl)
                _.savefig(mname+'/dustdens'+fsuf+'.png')


    @staticmethod
    def _grab_axes(fig):
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        return fig, ax

    @staticmethod
    def _gentrify_structure_ax(ax, cyl=False):
        ax.set_xlim([2., 500])
        ax.set_xscale('log')
        if cyl:
            ax.set_ylim([0.2, 500])
            ax.set_yscale('log')
            ax.set_xlabel(r"$r$ [au]")
            ax.set_ylabel(r"$z$ [au]")
        else:
            ax.set_ylim([0.0, 0.7])
            ax.set_xlabel("$R$ [au]")
            ax.set_ylabel("$\pi$/2 - $\Theta$")


    def plot_sigg(self, fig=None):
        fig, ax = self._grab_axes(fig)
        ax.plot(self.gr, self.gsig, color='gray', lw=3)
        ax.set_xlim([2., 500.])
        ax.set_xscale('log')
        ax.set_ylim([0.8*np.min(self.gsig), 1.2*np.max(self.gsig)])
        ax.set_yscale('log')
        ax.set_xlabel("$R$ [au]")
        ax.set_ylabel(r"$\Sigma_{\rm gas}\,\,[{\rm g / cm}^2]$")
        return fig

    def plot_Hp(self, fig=None):
        fig, ax = self._grab_axes(fig)
        ax.plot(self.gr, self.gH, color='gray', lw=3)
        ax.set_xlim([2., 500.])
        ax.set_xscale('log')
        ax.set_ylim([0.8*np.min(self.gH), 1.2*np.max(self.gH)])
        ax.set_yscale('log')
        ax.set_xlabel("$R$ [au]")
        ax.set_ylabel(r"$H_p\,\,[{\rm au}]$")
        return fig

    def plot_sigd(self, fig=None):
        fig, ax = self._grab_axes(fig)
        ax.plot(self.dr, self.dsig, 'C1', lw=3)
        ax.set_xlim([2., 500.])
        ax.set_xscale('log')
        ax.set_ylim([0.8*np.min(self.dsig), 1.2*np.max(self.dsig)])
        ax.set_yscale('log')
        ax.set_xlabel("$R$ [au]")
        ax.set_ylabel(r"$\Sigma_{\rm dust}\,\,[{\rm g / cm}^2]$")
        return fig


    def plot_Hdust(self, fig=None):
        fig, ax = self._grab_axes(fig)
        ax.plot(self.dr, self.dH, color='C1', lw=3)
        ax.set_xlim([2., 500.])
        ax.set_xscale('log')
        ax.set_ylim([0.8*np.min(self.dH), 1.2*np.max(self.dH)])
        ax.set_yscale('log')
        ax.set_xlabel("$R$ [au]")
        ax.set_ylabel(r"$H_{\rm dust}\,\,[{\rm au}]$")
        return fig


    def plot_sigboth(self, fig=None):
        fig, ax = self._grab_axes(fig)
        ax.plot(self.gr, self.gsig, color='gray', lw=3)
        ax.plot(self.dr, self.dsig, color='C1', lw=3)
        ax.set_xlim([2., 500.])
        ax.set_xscale('log')
        ax.set_ylim([0.8*np.min(self.dsig), 1.2*np.max(self.gsig)])
        ax.set_yscale('log')
        ax.set_xlabel("$R$ [au]")
        ax.set_ylabel(r"$\Sigma\,\,[{\rm g / cm}^2]$")
        return fig


    def plot_Hboth(self, fig=None):
        fig, ax = self._grab_axes(fig)
        ax.plot(self.gr, self.gH, color='gray', lw=3)
        ax.plot(self.dr, self.dH, color='C1', lw=3)
        ax.set_xlim([2., 500.])
        ax.set_xscale('log')
        ax.set_ylim([0.8*np.min(self.dH), 1.2*np.max(self.gH)])
        ax.set_yscale('log')
        ax.set_xlabel("$R$ [au]")
        ax.set_ylabel(r"$H\,\,[{\rm au}]$")
        return fig


    def plot_temp(self, fig=None, contourf_kwargs=None, cyl=False):
        fig, ax = self._grab_axes(fig)
        if cyl:
            xx, yy = self.Rgrid / self.AU, self.Zgrid / self.AU
            yyy, toplot = yy, self.temp
        else:
            xx, yy = self.Rgrid / self.AU, 0.5*np.pi - self.Tgrid[::-1]
            yyy = np.concatenate([-yy[::-1], yy])
            toplot = np.vstack([self.temp, self.temp[::-1]])

        contourf_kwargs = {} if contourf_kwargs is None else contourf_kwargs
        levels = np.linspace(5, 100, 50)
        cmap = contourf_kwargs.pop("cmap", "plasma")
        im = ax.contourf(xx, yyy, toplot, 
                         levels=levels, cmap=cmap, **contourf_kwargs)

        if cyl:
            ax.plot(xx, 0.1 * xx, ':k')
            ax.plot(xx, 0.2 * xx, ':k')
            ax.plot(xx, 0.3 * xx, ':k')
            gr, gs, gH = np.loadtxt(self.mname+'/gas_profiles.txt',skiprows=1).T
            ax.plot(gr, gH, '--', color='gray', lw=3)
            ax.contour(xx, yyy, toplot, levels=[20.])
        cax = make_axes_locatable(ax)
        cax = cax.append_axes("right", size="3%", pad="1.5%")
        cb = plt.colorbar(im, cax=cax, ticks=np.arange(0, 300, 20))
        cb.set_label(r"$T\,\,[{\rm K}]$", rotation=270, labelpad=15)

        self._gentrify_structure_ax(ax, cyl=cyl)
        return fig


    def plot_gasdens(self, fig=None, contourf_kwargs=None, cyl=False):
        fig, ax = self._grab_axes(fig)
        if cyl:
            xx, yy = self.Rgrid / self.AU, self.Zgrid / self.AU
            yyy, toplot = yy, np.log10(self.rhog / self.m_p / self.mu)
        else:
            xx, yy = self.Rgrid / self.AU, 0.5*np.pi - self.Tgrid[::-1]
            yyy = np.concatenate([-yy[::-1], yy])
            toplot = np.log10(np.vstack([self.rhog, self.rhog[::-1]]) / \
                              self.m_p / self.mu)

        contourf_kwargs = {} if contourf_kwargs is None else contourf_kwargs
        levels = np.linspace(0, 15, 50)
        cmap = contourf_kwargs.pop("cmap", "bone_r")
        im = ax.contourf(xx, yyy, toplot, 
                         levels=levels, cmap=cmap, **contourf_kwargs)

        cax = make_axes_locatable(ax)
        cax = cax.append_axes("right", size="3%", pad="1.5%")
        cb = plt.colorbar(im, cax=cax, ticks=np.arange(-30, 30, 2))
        cb.set_label(r"$\log_{10}(n_{\rm H_2}\,\,[{\rm cm^{-3}}])$",
                     rotation=270, labelpad=15)

        self._gentrify_structure_ax(ax, cyl=cyl)
        return fig


    def plot_nmol(self, fig=None, contourf_kwargs=None, cyl=False):
        fig, ax = self._grab_axes(fig)
        if cyl:
            xx, yy = self.Rgrid / self.AU, self.Zgrid / self.AU
            yyy, toplot = yy, np.log10(self.nmol)
        else:
            xx, yy = self.Rgrid / self.AU, 0.5*np.pi - self.Tgrid[::-1]
            yyy = np.concatenate([-yy[::-1], yy])
            toplot = np.log10(np.vstack([self.nmol, self.nmol[::-1]]))

        contourf_kwargs = {} if contourf_kwargs is None else contourf_kwargs
        levels = np.linspace(0, 15, 50)
        levels += np.log10(self.diskpars["abundance"]["arguments"]["xmol"])
        cmap = contourf_kwargs.pop("cmap", "afmhot_r")
        im = ax.contourf(xx, yyy, toplot, 
                         levels=levels, cmap=cmap, **contourf_kwargs)
        cax = make_axes_locatable(ax)
        cax = cax.append_axes("right", size="3%", pad="1.5%")
        cb = plt.colorbar(im, cax=cax, ticks=np.arange(-30, 30, 2))
        cb.set_label(r"$\log_{10}(n_{\rm " + self.setup["molecule"].upper() + \
                     r"}\,\,[{\rm cm^{-3}}])$", rotation=270, labelpad=15)

        self._gentrify_structure_ax(ax, cyl=cyl)
        return fig


    def plot_vel(self, fig=None, contourf_kwargs=None, cyl=False):
        fig, ax = self._grab_axes(fig)
        if cyl:
            xx, yy = self.Rgrid / self.AU, self.Zgrid / self.AU
            yyy, toplot = yy, np.log10(self.vel / 1e5)
        else:
            xx, yy = self.Rgrid / self.AU, 0.5*np.pi - self.Tgrid[::-1]
            yyy = np.concatenate([-yy[::-1], yy])
            toplot = np.log10(np.vstack([self.vel, self.vel[::-1]]) / 1e5)

        contourf_kwargs = {} if contourf_kwargs is None else contourf_kwargs
        levels = np.linspace(0, 1.5, 50)
        cmap = contourf_kwargs.pop("cmap", "viridis")
        im = ax.contourf(xx, yyy, toplot, 
                         levels=levels, cmap=cmap, **contourf_kwargs)

        cax = make_axes_locatable(ax)
        cax = cax.append_axes("right", size="3%", pad="1.5%")
        cb = plt.colorbar(im, cax=cax, ticks=np.arange(0, 2, 0.5))
        cb.set_label(r"$\log_{10}(v_{\phi})\,\,[{\rm km/s}])$",
                     rotation=270, labelpad=15)

        self._gentrify_structure_ax(ax, cyl=cyl)
        return fig


    def plot_vturb(self, fig=None, contourf_kwargs=None, cyl=False):
        fig, ax = self._grab_axes(fig)
        if cyl:
            xx, yy = self.Rgrid / self.AU, self.Zgrid / self.AU
            yyy, toplot = yy, np.log10(self.dvturb / 1e5) 
        else:
            xx, yy = self.Rgrid / self.AU, 0.5*np.pi - self.Tgrid[::-1]
            yyy = np.concatenate([-yy[::-1], yy])
            toplot = np.log10(np.vstack([self.dvturb, self.dvturb[::-1]]) / 1e5)

        contourf_kwargs = {} if contourf_kwargs is None else contourf_kwargs
        levels = np.linspace(-3, 0, 50)
        cmap = contourf_kwargs.pop("cmap", "cool")
        im = ax.contourf(xx, yyy, toplot, 
                         levels=levels, cmap=cmap, **contourf_kwargs)

        cax = make_axes_locatable(ax)
        cax = cax.append_axes("right", size="3%", pad="1.5%")
        cb = plt.colorbar(im, cax=cax, ticks=np.arange(-3, 0, 0.5))
        cb.set_label(r"$\log_{10}(\delta v_{\rm turb})\,\,[{\rm km/s}])$",
                     rotation=270, labelpad=15)

        self._gentrify_structure_ax(ax, cyl=cyl)
        return fig


    def plot_dustdens(self, fig=None, contourf_kwargs=None, cyl=False):
        fig, ax = self._grab_axes(fig)
        if cyl:
            xx, yy = self.Rgrid / self.AU, self.Zgrid / self.AU
            yyy, toplot = yy, np.log10(self.rhod)
        else:
            xx, yy = self.Rgrid / self.AU, 0.5*np.pi - self.Tgrid[::-1]
            yyy = np.concatenate([-yy[::-1], yy])
            toplot = np.log10(np.vstack([self.rhod, self.rhod[::-1]]))

        contourf_kwargs = {} if contourf_kwargs is None else contourf_kwargs
        levels = np.linspace(-20, -10, 50)
        cmap = contourf_kwargs.pop("cmap", "gist_heat_r")
        im = ax.contourf(xx, yyy, toplot,
                         levels=levels, cmap=cmap, **contourf_kwargs)

        cax = make_axes_locatable(ax)
        cax = cax.append_axes("right", size="3%", pad="1.5%")
        cb = plt.colorbar(im, cax=cax, ticks=np.arange(-30, 0, 2))
        cb.set_label(r"$\log_{10}(\rho_{\rm dust}\,\,[{\rm g/cm^{-3}}])$",
                     rotation=270, labelpad=15)

        self._gentrify_structure_ax(ax, cyl=cyl)
        return fig

