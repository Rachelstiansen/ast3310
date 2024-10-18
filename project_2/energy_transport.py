""""
How to run code: press run and all the plots will be saved in the same folder.

Note that since the class interpolates the opacity values everytime it is called, it takes a little
time to run. I included a progress bar to make it easier to see when the code is finished running. The progress bar starts for 
each time the class is called, and it is called several times. If I had had more time, I would have made it so that it only 
interpolates once, and it would hopefully have run faster. 

Please pour yourself a cup of coffee while the code does its magic :)

(Also note that you will get errors in the terminal when the loop stops to early, I did not have the time to silence those.)
"""

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
import scipy.constants as sc
from scipy.interpolate import RectBivariateSpline
from tqdm import tqdm

from energy_production import Energy_production

plt.rcParams.update({"font.size": 14}) # For all the plots to have text size 14

class Energy_transportation:
    """
    The class simulates the energy production in a star by using a Runge-Kutta fourth order integration method to solve the 
    coupled differential equation system.
    """
    def __init__(self, r0, rho0, T0, L0, m0, P0, sanity_check=None):

        self.r_0 = r0          
        self.rho_0 = rho0       
        self.T_0 = T0           
        self.L_0 = L0           
        self.m_0 = m0  
        self.P_0 = P0         

        # Solar values to get the units right:
        self.L_sun = 3.846e26   # [W]
        self.M_sun = 1.989e30   # [kg]
        self.R_sun = 6.96e8     # [m]
        self.rho_sun = 1.408e3  # [kg m^-3]

        # Updating the input values to physical units:
        self.r_0 = r0 * self.R_sun              # [solar radius]
        self.m_0 = m0 * self.M_sun              # [solar mass]
        self.L_0 = L0 * self.L_sun              # [solar luminosity]
        self.rho_0 = rho0 * self.rho_sun        # [average solar density]

        # Element abundances:
        self.X = 0.7; self.Y_3He = 1e-10; self.Y = 0.29
        self.Z_7Li = 1e-7; self.Z_7Be = 1e-7; self.Z_14N = 1e-11

        if sanity_check == True:
            self.mu = 0.6 # To get correct values when comparing to values in example 5.1
        else:
            self.mu = 1 / (2 * self.X + (3/4) * self.Y + self.Y_3He + (4/7) * self.Z_7Li + (5/7) * self.Z_7Be + (4/7) * self.Z_14N)
            print(f"\nCalculated value of mean molecular weight: {self.mu:.3f}\n ")

        self.k_B = 1.3806e-23           # Boltzmann's constant      [m^2 kg/(s^2 K)]
        self.m_u = sc.m_u               # atomic mass unit          [kg]
        self.sigma = 5.6704e-8          # Stefan-Boltzmanns const.  [W m^-2 K^-4]
        self.a = 4 * self.sigma / sc.c  # radiation density constant
        self.G = sc.G                   # gravitational const.      [N m^2 kg^-2]
        self.alpha_l_m = 1              # assuming to be = 1 
        self.nabla_ad = 2/5             # adiabatic temperature gradient
        self.c_P = 5 * self.k_B / (2 * self.mu * self.m_u) # specific heat capacity at constant pressure [J K^-1 m^-1]

        # Initializing empty lists in order to save values of the parameters for plotting later.
        self.V = np.zeros(4) # To store values of r, P, L, T
        self.m = []; self.r = []; self.P = []; self.T = []; self.rho = []
        self.kappa = []; self.nabla_stable_list = []; self.nabla_star_list = []
        self.dT = []; self.F_con = []; self.F_rad = []; self.epsilon = []; 
        self.PPI = []; self.PPII = []; self.PPIII = []; self.CNO = []
        
    def opacity(self, T, rho, sanity_check=None):
        """
        Function that reads and interpolates the opacity data from opacity.txt. If sanity_check=True the function prints
        a nicely formatted table of the values calculated compared to the ones listed in the project decription and the
        relative errors between them.
        """
        log_R = np.loadtxt("opacity.txt", max_rows=1, usecols=range(1, 20))     # [g cm^-3 K^-3]
        log_T = np.loadtxt("opacity.txt", skiprows=1, usecols=(0))              # [K]
        log_kappa = np.loadtxt("opacity.txt", skiprows=1, usecols=range(1, 20)) # [cm^2 g^-1]

        interp_spline = RectBivariateSpline(log_T, log_R, log_kappa)

        # Converting input parameters from SI to cgs:
        rho_cgs = rho * 1e-3                                    # [g cm^-3]
        log_R_cgs = np.log10(rho_cgs / (T / 1e6)**3)            # [g cm^-3 K^-3]
        log_T_cgs = np.log10(T)                                 # [K]
        log_kappa_cgs = interp_spline.ev(log_T_cgs, log_R_cgs)  # [cm^2 g^-1]

        # Converting from cgs to SI:
        kappa_SI = 10**(log_kappa_cgs) * 0.1
       
        if sanity_check == True:

            T_sanity = np.array([3.75, 3.755, 3.755, 3.755, 3.755, 3.770, 3.780, 3.795, 3.770, 3.775, 3.780, 3.795, 3.8])
            R_sanity = - np.array([6, 5.95, 5.8, 5.7, 5.55, 5.95, 5.95, 5.95, 5.8, 5.75, 5.7, 5.55, 5.5])
            opacity_sanity_cgs = - np.array([1.55, 1.51, 1.57, 1.61, 1.67, 1.33, 1.2, 1.02, 1.39, 1.35, 1.31, 1.16, 1.11])
            opacity_sanity_SI = np.array([2.84, 3.11, 2.68, 2.46, 2.12, 4.70, 6.25, 9.45, 4.05, 4.43, 4.94, 6.89, 7.69]) * 1e-3

            opacity_calc_cgs = interp_spline.ev(T_sanity, R_sanity)
            opacity_calc_SI = 10**(opacity_calc_cgs) * 0.1

            rel_error_cgs = abs((opacity_calc_cgs - opacity_sanity_cgs) / (opacity_calc_cgs)) 
            rel_error_SI = abs((opacity_calc_SI - opacity_sanity_SI) / (opacity_calc_SI)) 

            if np.any(rel_error_cgs) > 0.05 or np.any(rel_error_SI) > 0.05:
                print("Warning: relative error larger than 5% \n")

            # For printing nicely formatted table
            headers = ["log_10 T", "log_10 R \n(cgs)", "log_10 \nkappa (cgs)", "log_10 kappa \ncalc (cgs)","kappa \n(SI)", 
                       "kappa \ncalc (SI)", "relative \nerror % (cgs)", "relative \nerror % (SI)"]
            table = zip(T_sanity, R_sanity, opacity_sanity_cgs, opacity_calc_cgs, opacity_sanity_SI, opacity_calc_SI, rel_error_cgs, rel_error_SI)
            print(f"Sanity check for interpolation of opacity")
            print(tabulate(table, headers=headers, tablefmt="fancy_grid"), "\n")

        return kappa_SI
    
    def calc_pressure(self, rho, T):
        """
        Function that calculates the pressure in the Sun for a given density rho
        and temperature T.
        """
        P_rad = self.a / 3 * T**4
        P_gas = rho * self.k_B * T / (self.mu * self.m_u)
        P_tot = P_rad + P_gas

        return P_tot

    def calc_density(self, P, T):
        """
        Function that calculates the density for a given pressure P and temperature T
        """
        P_rad = self.a / 3 * T**4
        rho = ((P - P_rad) * self.mu * self.m_u) / (self.k_B * T)

        return rho

    def g(self, m, r):
        """"
        Calculates the gravitational acceleration g for a given mass m and radius r
        """
        return self.G * m / r**2
    
    def U(self, T, rho, m, r, kappa):
        """
        Calculates the composite variable U
        """
        return 64 * self.sigma * T**3 / (3 * kappa * rho**2 * self.c_P) * np.sqrt(self.H_P(T, m, r) / self.g(m, r))

    def H_P(self, T, m, r):
        """
        Calculates the pressure scale height H_P
        """
        return self.k_B * T / (self.mu * self.m_u * self.g(m, r))
    
    def l_m(self, T, m, r):
        """
        Calculates the mixing length l_m
        """
        return self.alpha_l_m * self.H_P(T, m, r)

    def geometric_factor(self, r):
        """
        Calculates the geometric factor used to find Xi
        """
        S = 4 * np.pi * r**2 # Surface area of spherical parcel
        Q = np.pi * r**2     # Surface normal to the velocity
        d = 2 * r            # Diameter of sphere
        
        return S / (Q * d)

    def K(self, T, rho, m, r, kappa):
        """
        Function for calculating the composite parameter K, which is used to calculate xi.
        """
        return self.U(T, rho, m, r, kappa) * self.geometric_factor(r) / self.l_m(T, m, r)

    def Lum(self, T, rho, include_chains=False):
        """
        Function for retrieving the luminosity and energy from the different branches which is calculated in the class
        Energy_production which was used in project 1. If include_chains=True the function returns the total energy and the
        energy from the different branches.
        """
        Sun = Energy_production(T, rho)
        
        if include_chains == False:
            return Sun.energy_produced(verbose=False)[1]     # Luminosity
        
        if include_chains == True:
            epsilon = Sun.energy_produced(verbose=False)[1]
            PPI = Sun.energy_produced(verbose=False)[2]     # Energy from PPI branch
            PPII = Sun.energy_produced(verbose=False)[3]    # Energy from PPII branch
            PPIII = Sun.energy_produced(verbose=False)[4]   # Energy from PPIII branch
            CNO = Sun.energy_produced(verbose=False)[5]     # Energy from CNO cycle

            return epsilon, PPI, PPII, PPIII, CNO

    def nabla_stable(self, T, r, rho, m, L, kappa):
        """
        Function for calculating the temperature gradient we use when the energy is only
        being transported by radiation.
        """
        return 3 * L * kappa * rho * self.H_P(T, m, r) / (64 * np.pi * r**2 * self.sigma * T**4)

    def Xi(self, T, rho, m, r, nabla_stable, kappa):
        """
        Function for calculating xi which is used in the expression for nabla^*. We do this by finding the roots of the 
        polynomial and returning only the real root we get as this is the only valid solution.
        """
        R = self.U(T, rho, m, r, kappa) / self.l_m(T, m, r)**2
        
        a = 1 / R
        b = 1
        c = self.K(T, rho, m, r, kappa)
        d = - (nabla_stable - self.nabla_ad)
        coeff = np.array([a, b, c, d])

        roots = np.roots(coeff)
        real_idx = np.where(roots.imag == 0)[0][0]
        Xi = np.real(roots[real_idx])
        return Xi

    def v(self, T, m, r, rho, nabla_stable, kappa):
        """
        Function for calculating the velocity of the gas parcel, used to compare with values in example 5.1
        """
        return self.l_m(T, m, r) * np.sqrt(self.g(m, r) / self.H_P(T, m, r)) / 2 * self.Xi(T, rho, m, r, nabla_stable, kappa)

    def Flux(self, rho, T, m, r, nabla_stable, kappa, nabla_star, con=None):
        """
        Function for calculating the convective and radiative energy flux in the star. When
        calling the function we can choose con=True for the function to return the convective energy flux,
        otherwise the function returns the radiative energy flux.
        """
        F_con = rho * self.c_P * T * np.sqrt(self.g(m, r)) * self.H_P(T, m, r)**(-3/2) * (self.l_m(T, m, r) / 2)**2 * self.Xi(T, rho, m, r, nabla_stable, kappa)**3
        F_rad = 16 * self.sigma * T**4 * nabla_star / (3 * kappa * rho * self.H_P(T, m, r))

        if con == True:
            return F_con
        else:
            return F_rad

    def move_inwards(self, p, integrate=True, include_chains=False):
        """
        Function for integrating the system of coupled differential equation using a fourth order
        Runge-Kutta integration method. If integrate=True the function returns only m and V so we can use these 
        as input-parameters in the function eqs(). If include_chains=True then the function returns all the parameters
        we need for plotting in addition to the total energy epsilon and the energy from the PP-brances, CNO cycle.
        """
        P0 = self.P_0 * self.calc_pressure(self.rho_0, self.T_0)
        eqs_0 = np.array([self.r_0, P0, self.L_0, self.T_0]) # initial values of r, P, L, T   
        
        N = int(4e3)

        V = np.zeros((4, N + 1)); m = np.zeros(N + 1) # initializing V which includes r, P, L, T and m
        V[:, 0] = eqs_0; m[0] = self.m_0  # setting initial values

        # Initializing arrays to store values of all the parameters we want to plot later
        r = np.zeros(N + 1); P = np.zeros(N + 1); L = np.zeros(N + 1); T = np.zeros(N + 1)
        rho = np.zeros(N + 1); kappa = np.zeros(N + 1); nabla_stable = np.zeros(N + 1)
        nabla_star = np.zeros(N + 1); F_con = np.zeros(N + 1); F_rad = np.zeros(N + 1)

        epsilon = np.zeros(N + 1); PP1 = np.zeros(N + 1); PP2 = np.zeros(N + 1)
        PP3 = np.zeros(N + 1); CNO = np.zeros(N + 1)
        
        for i in tqdm(range(N)):
            # Implementing variable step length:
            dV_list = self.eqs(m[i], V[:, i], loop=True)
            dm = - np.min(V[:, i] * p / np.abs(dV_list)) 

            if include_chains != None:
                # When we want to also store values of total energy epsilon and the energy from the PP-brances, CNO cycle
                _, r[i], P[i], L[i], T[i], rho[i], kappa[i], nabla_stable[i], nabla_star[i], F_con[i], F_rad[i], epsilon[i], PP1[i], PP2[i], PP3[i], CNO[i] = self.eqs(m[i], V[:, i], loop=False, include_chains=True)
            else:
                _, r[i], P[i], L[i], T[i], rho[i], kappa[i], nabla_stable[i], nabla_star[i], F_con[i], F_rad[i] = self.eqs(m[i], V[:, i], loop=False)
            
            # Implementing RK4:
            k1 = dm * self.eqs(m[i], V[:, i], loop=True)
            k2 = dm * self.eqs(m[i] + 0.5 * dm, V[:, i] + 0.5 * k1, loop=True)
            k3 = dm * self.eqs(m[i] + 0.5 * dm, V[:, i] + 0.5 * k2, loop=True)
            k4 = dm * self.eqs(m[i] + dm, V[:, i] + k3, loop=True)

            V[:, i + 1] = V[:, i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            m[i + 1] = m[i] + dm


            if m[i + 1] < 0 or np.any(V[:, i + 1]) < 0:
                print("One of the parameters M, R, P, L, T became negative")

                break
        
        # Saving arrays of the parameter values up to the point where the loop stopped, if it stopped
        self.m = m[:i]; self.V = V[:, :i]
        
        self.r = r[:i]; self.P = P[:i]; self.L = L[:i]; self.T = T[:i]
        self.rho = rho[:i]; self.kappa = kappa[:i]; self.nabla_stable_list = nabla_stable[:i]
        self.nabla_star_list = nabla_star[:i]; self.F_con = F_con[:i]; self.F_rad = F_rad[:i]
        self.epsilon = epsilon[:i]; self.PPI = PP1[:i]; self.PPII = PP2[:i]
        self.PPIII = PP3[:i]; self.CNO = CNO[:i]

        if integrate == True:
            return self.m, self.V
        else:
            if include_chains == True:
                return self.m, self.r, self.P, self.L, self.T, self.rho, self.kappa, self.nabla_stable_list, self.nabla_star_list, self.F_con, self.F_rad, self.epsilon, self.PPI, self.PPII, self.PPIII, self.CNO
            else:
                return self.m, self.r, self.P, self.L, self.T, self.rho, self.kappa, self.nabla_stable_list, self.nabla_star_list, self.F_con, self.F_rad
            
    
    def eqs(self, m, eqs, loop=False, include_chains=False, sanity_check=None):
        """
        Function used to calculate all the equations needed to solve the coupled differential system of equations. If loop=True the 
        function returns the 4 differential equations such that they can be solved in the integration loop in the function move_inwards(). 
        If include_chains=True then the function returns all the parameters we need for plotting in addition to the total energy epsilon 
        and the energy from the PP-brances, CNO cycle. If sanity_check=True other initial values are defined to compare vlaues with the ones 
        listed in example 5.1. The calculated values are printed when the function is called.
        """
        if sanity_check == True:
            T = 0.9e6                       # [K]
            rho = 55.9                      # [kg m^-3]
            r = 0.84 * self.R_sun 
            m = 0.99 * self.M_sun
            L = self.L_sun
            kappa = 3.98                    # [m^2 kg^-1]
            P = self.calc_pressure(rho, T)

        else:
            r, P, L, T = eqs
            rho = self.calc_density(P, T)
            kappa = self.opacity(T, rho)

        nabla_stable = self.nabla_stable(T, r, rho, m, L, kappa)

        # The coupled differential equations:
        dr = 1 / (4 * np.pi * r**2 * rho)
        dP = - self.G * m / (4 * np.pi * r**4)
        dL = self.Lum(T, rho)

        if nabla_stable > self.nabla_ad:
            # Star is convectively unstable: convection
            nabla_star = self.Xi(T, rho, m, r, nabla_stable, kappa)**2 + self.Xi(T, rho, m, r, nabla_stable, kappa) * self.K(T, rho, m, r, kappa) + self.nabla_ad
            dT = nabla_star * T * dP / P
            F_con = self.Flux(rho, T, m, r, nabla_stable, kappa, nabla_star, con=True)
        else:
            # Star is convectively stable: radiation
            dT = - 3 * kappa * L / (256 * np.pi**2 * self.sigma * r**4 * T**3) # For only radiative transport
            nabla_star = nabla_stable
            F_con = 0
        
        F_rad = self.Flux(rho, T, m, r, nabla_stable, kappa, nabla_star)

        if sanity_check:
            print("Sanity check: checking values from example 5.1: \n")
            print("nabla_stable = ", nabla_stable, "with relative error ", abs(nabla_stable - 3.26) / 3.26 * 100, "%")
            print("Xi = ", self.Xi(T, rho, m, r, nabla_stable, kappa), "with relative error ", abs(self.Xi(T, rho, m, r, nabla_stable, kappa) - 1.173e-3) / 1.173e-3 * 100, "%")
            print("U = ", self.U(T, rho, m, r, kappa), " m^2, with relative error ", abs(self.U(T, rho, m, r, kappa) - 5.94e5) / 5.94e5 * 100, "%")
            print("H_P = ", self.H_P(T, m, r), " m, with relative error ", abs(self.H_P(T, m, r) - 32.4e6) / 32.4e6 * 100, "%")
            print("nabla_star =", nabla_star, "with relative error ", abs(nabla_star - 0.4) / 0.4 * 100, "%")
            print("v = ", self.v(T, m, r, rho, nabla_stable, kappa), " m/s with relative error ", abs(self.v(T, m, r, rho, nabla_stable, kappa) - 65.5) / 65.5 * 100, "%")
            print("F_con/(F_con + F_rad) = ", F_con / (F_con + F_rad), "with relative error ", abs(F_con / (F_con + F_rad) - 0.88) / 0.88 * 100, "%")
            print("F_rad / (F_con + F_rad) = ", F_rad / (F_con + F_rad), "with relative error ", abs( F_rad / (F_con + F_rad) - 0.12) / 0.12 * 100, "%")
        
        if loop == True:
            return np.array([dr, dP, dL, dT])
        
        if loop == False:
            if include_chains == True:
                epsilon, PP1, PP2, PP3, CNO = self.Lum(T, rho, include_chains=True)
                return m, r, P, L, T, rho, kappa, nabla_stable, nabla_star, F_con, F_rad, epsilon, PP1, PP2, PP3, CNO
            else:
                return m, r, P, L, T, rho, kappa, nabla_stable, nabla_star, F_con, F_rad

    def sanity_check(self):
        """
        Function which when called, prints the calculated sanity check values.
        """
        m, V = self.move_inwards(0.01)
        self.eqs(m, V, sanity_check=True)

    def cross_section(self, R, L, F_C, sanity=False, savefig=False, title=None, savename=None):
        """
        This is the same function as given in the project description, with only some 
        minor changes. The function plots the cross section of a star given an array of
        the radius R, luminosity L and convective flux F_C.
        """
        r_range    = 1.2            
        core_limit = 0.995 * np.max(L)

        star_zone = np.int32(np.where(L>core_limit,0.5,-0.5) * np.where(F_C>0,3,1) + 2)
        colors=['b','c','yellow','r']

        # Initialize the figure
        plt.figure(figsize=(6,6))
        fig = plt.gcf()
        ax  = plt.gca()
        ax.set_xlim(-r_range, r_range)
        ax.set_ylim(-r_range, r_range)
        ax.set_aspect('equal')

        star_zone_prev = -1
        for k in range(0,len(R)-1):
            if star_zone[k]!=star_zone_prev: # only plot a new *filled* circle if a new star_zone
                star_zone_prev = star_zone[k]
                circle = plt.Circle((0,0),R[k]/self.R_sun,fc=colors[star_zone[k]],fill=True,ec=None)
                ax.add_artist(circle)

        circle_white=plt.Circle((0,0),R[-1]/self.R_sun,fc='white',fill=True,lw=0)
        ax.add_artist(circle_white)

        # create legends
        circle_red    = plt.Circle((2*r_range, 2*r_range), 0.1*r_range, color=colors[3], fill=True)
        circle_yellow = plt.Circle((2*r_range, 2*r_range), 0.1*r_range, color=colors[2], fill=True)
        circle_cyan   = plt.Circle((2*r_range, 2*r_range), 0.1*r_range, color=colors[1], fill=True)
        circle_blue   = plt.Circle((2*r_range, 2*r_range), 0.1*r_range, color=colors[0], fill=True)

        ax.legend([circle_red,circle_yellow,circle_cyan,circle_blue],\
                ['Convection outside core','Radiation outside core','Radiation inside core','Convection inside core']\
                , fontsize=13)
        plt.xlabel(r'$R/R_\odot$', fontsize=13)
        plt.ylabel(r'$R/R_\odot$', fontsize=13)
        plt.title('Cross section of star', fontsize=15)

        if title != None:
            plt.title(title)

        fig.tight_layout()

        if savefig:
            if sanity:
                fig.savefig('sanity_cross_section_2D.png', dpi=300)
            else:
                if savename != None:
                    fig.savefig('cross_section_'+ savename +'.png', dpi=300)
                else: 
                    fig.savefig("cross_section.png")


    def varying_cross_section(self, R, L, F_C, sup_title, titles, savefig=False):
        """
        Function which creates a subplot of two cross sections next to each other. This is used to compare
        cross-sections when increasing and decreasing the initial values of parameters.
        """
        r_range = 1.2
        core_limits = [0.995 * np.max(l_vals) for l_vals in L]

        fig, axs = plt.subplots(1, 2, figsize=(12,6))
        fig.suptitle(sup_title)

        for i in range(2):
            r_vals = R[i]
            l_vals = L[i]
            fc_vals = F_C[i]

            star_zone = np.int32(np.where(l_vals > core_limits[i], 0.5, -0.5) * np.where(fc_vals > 0, 3, 1) + 2)
            colors=['b', 'c', 'yellow', 'r']

            axs[i].set_xlim(-r_range, r_range)
            axs[i].set_ylim(-r_range, r_range)
            axs[i].set_aspect('equal')

            star_zone_prev = -1
            for k in range(len(r_vals)-1):
                if star_zone[k] != star_zone_prev:
                    star_zone_prev = star_zone[k]
                    circle = plt.Circle((0,0), r_vals[k]/self.R_sun, fc=colors[star_zone[k]], fill=True, ec=None)
                    axs[i].add_artist(circle)

            circle_white = plt.Circle((0,0), r_vals[-1]/self.R_sun, fc='white', fill=True, lw=0)
            axs[i].add_artist(circle_white)

            circle_red    = plt.Circle((2*r_range, 2*r_range), 0.1*r_range, color=colors[3], fill=True)
            circle_yellow = plt.Circle((2*r_range, 2*r_range), 0.1*r_range, color=colors[2], fill=True)
            circle_cyan   = plt.Circle((2*r_range, 2*r_range), 0.1*r_range, color=colors[1], fill=True)
            circle_blue   = plt.Circle((2*r_range, 2*r_range), 0.1*r_range, color=colors[0], fill=True)

            axs[i].legend([circle_red,circle_yellow,circle_cyan,circle_blue],\
                ['Convection outside core', 'Radiation outside core', 'Radiation inside core', 'Convection inside core'], fontsize=13)

            axs[i].set(xlabel=r'$R/R_\odot$', ylabel=r'$R/R_\odot$')
            axs[i].set_title(titles[i])
        
        fig.tight_layout()

        if savefig == True:
            fig.savefig(sup_title + ".png")


    def plot_gradients(self, R, nabla_stable, nabla_star):
        """
        Function for plotting the three gradients nabla_stable,
        nabla_star and nabla_ad used for sanity check.
        """
        plt.figure(figsize=(8, 5))
        plt.title("Temperature gradients for (p=0.01)")
        plt.plot(R/self.R_sun, nabla_stable, label=r"$\nabla_{stable}$")
        plt.plot(R/self.R_sun, nabla_star, label=r"$\nabla^*$")
        plt.plot(R/self.R_sun, self.nabla_ad * np.ones_like(R), label=r"$\nabla_{ad}$")
        plt.xlabel(r"R/R$_{\odot}$")
        plt.ylabel(r"$\nabla$")
        plt.yscale("log")
        plt.legend()
        plt.savefig("sanity_gradients.png")


# initial parameters for sanity check:
R0 = 1
rho0 = 1.42e-7
T0 = 5770
L0 = 1
M0 = 1
P0 = 1

# ------------------ Varying params, plotting cross sections and finding best model: --------------------------


def variate_init():
    """
    Due to .the lack of time I decided to make savefigs for cross_sections one at a time to at 
    least get something to discuss and continue, I realize that this could have been done much better, and more efficient. 
    This function takes a while to run as it calls the class several times.
    """
    R0_inc = R0 * 1.2 # we do not change this as much because the plot is set to have a certain size
    simulate = Energy_transportation(R0_inc, rho0, T0, L0, M0, P0)
    M, R_inc, P, L_inc, T, rho, kappa, nabla_stable, nabla_star, F_con_inc, F_rad = simulate.move_inwards(0.01, integrate=False)
    
    R0_dec = R0 * 0.7
    simulate = Energy_transportation(R0_dec, rho0, T0, L0, M0, P0)
    M, R_dec, P, L_dec, T, rho, kappa, nabla_stable, nabla_star, F_con_dec, F_rad = simulate.move_inwards(0.01, integrate=False)
    
    simulate.varying_cross_section([R_inc, R_dec], [L_inc, L_dec], [F_con_inc, F_con_dec], sup_title=r"Varying initial radius", titles=[r"Increasing $R_0$, $R_0 = 1.2 R_{\odot}$", r"Decreasing $R_0$, $R_0 = 0.7 R_{\odot}$"], savefig=True)

    T0_inc = T0 * 10
    simulate = Energy_transportation(R0, rho0, T0_inc, L0, M0, P0)
    M, R_inc, P, L_inc, T, rho, kappa, nabla_stable, nabla_star, F_con_inc, F_rad = simulate.move_inwards(0.01, integrate=False)
   
    T0_dec = T0 * 0.1
    simulate = Energy_transportation(R0, rho0, T0_dec, L0, M0, P0)
    M, R_dec, P, L_dec, T, rho, kappa, nabla_stable, nabla_star, F_con_dec, F_rad = simulate.move_inwards(0.01, integrate=False)
    
    simulate.varying_cross_section([R_inc, R_dec], [L_inc, L_dec], [F_con_inc, F_con_dec], sup_title=r"Varying initial temperature", titles=[r"Increasing $T_0$, $T_0 = 10 \cdot 5770 K$ = " f"{T0_inc} K", r"Decreasing $T_0$, $T_0 = 0.1 \cdot 5770 K$ = " f"{T0_dec:.0f} K"], savefig=True)

    rho0_inc = rho0 * 200
    simulate = Energy_transportation(R0, rho0_inc, T0, L0, M0, P0)
    M, R_inc, P, L_inc, T, rho, kappa, nabla_stable, nabla_star, F_con_inc, F_rad = simulate.move_inwards(0.01, integrate=False)
    
    rho0_dec = rho0 * 0.02
    simulate = Energy_transportation(R0, rho0_dec, T0, L0, M0, P0)
    M, R_dec, P, L_dec, T, rho, kappa, nabla_stable, nabla_star, F_con_dec, F_rad = simulate.move_inwards(0.01, integrate=False) 
    
    simulate.varying_cross_section([R_inc, R_dec], [L_inc, L_dec], [F_con_inc, F_con_dec], sup_title=r"Varying initial density", titles=[r"Increasing $\rho_0$, $\rho_0 = 200 \bar\rho_{\odot}$ = " f"{rho0_inc}" + r"kgm$^{-3}$", r"Decreasing $\rho_0$, $\rho_0 = 0.02 \bar\rho_{\odot}$ = " f"{rho0_dec}" + r"kgm$^{-3}$"], savefig=True)

    L0_inc = L0 * 10
    simulate = Energy_transportation(R0, rho0, T0, L0_inc, M0, P0)
    M, R_inc, P, L_inc, T, rho, kappa, nabla_stable, nabla_star, F_con_inc, F_rad = simulate.move_inwards(0.01, integrate=False)
    
    L0_dec = L0 * 0.1
    simulate = Energy_transportation(R0, rho0, T0, L0_dec, M0, P0)
    M, R_dec, P, L_dec, T, rho, kappa, nabla_stable, nabla_star, F_con_dec, F_rad = simulate.move_inwards(0.01, integrate=False)
    
    simulate.varying_cross_section([R_inc, R_dec], [L_inc, L_dec], [F_con_inc, F_con_dec], sup_title=r"Varying initial luminosity", titles=[r"Increasing $L_0$, $L_0 = 10 L_{\odot}$", r"Decreasing $L_0$, $L_0 = 0.1 L_{\odot}$"], savefig=True)

    P0_inc = P0 * 200
    simulate = Energy_transportation(R0, rho0, T0, L0, M0, P0_inc)
    M, R_inc, P, L_inc, T, rho, kappa, nabla_stable, nabla_star, F_con_inc, F_rad = simulate.move_inwards(0.01, integrate=False)
    
    P0_dec = P0 * 0.02
    simulate = Energy_transportation(R0, rho0, T0, L0, M0, P0_dec)
    M, R_dec, P, L_dec, T, rho, kappa, nabla_stable, nabla_star, F_con_dec, F_rad = simulate.move_inwards(0.01, integrate=False)
    
    simulate.varying_cross_section([R_inc, R_dec], [L_inc, L_dec], [F_con_inc, F_con_dec], sup_title=r"Varying initial pressure", titles=[r"Increasing $P_0$, $200 \cdot P_0 $", r"Decreasing $P_0$, $0.02 P_0$"], savefig=True)


def final_model():
    """
    Function for plotting all the parameters and quantitites from the final model.
    """
    R0_final = 1.1 * R0; rho0_final = 1.3e-5; L0_final = 1.5 * L0
    simulate = Energy_transportation(R0_final, rho0_final, T0, L0_final, M0, P0)
    M, R, P, L, T, rho, kappa, nabla_stable, nabla_star, F_con, F_rad, epsilon, PP1, PP2, PP3, CNO = simulate.move_inwards(0.01, integrate=False, include_chains=True)
    simulate.cross_section(R, L, F_con, savefig=True, title=f"Final model", savename="final_model")

    core_idx = np.where(L < 0.995 * L[0])[0][0]
    R_core = R[core_idx]
    outer_con_range = R[np.logical_and(nabla_stable > 2/5, R > R_core)] # nabla_ad = 2/5
    outer_con_width = (outer_con_range[0] - outer_con_range[-1]) / R[0]
    
    print(f"Values for final model: \n")
    print(f"L going to {100 * L[-1] / L[0]} % of L0")
    print(f"M going to {100 * M[-1] / M[0]} % of M0")
    print(f"R going to {100 * R[-1] / R[0]} % of R0")

    print(f"R_core reaching out to {100 * R_core / R[0]} % of R0")
    print(f"Surface convection zone width is {outer_con_width} of R0")

    # Plotting the main parameters:
    plt.figure(figsize=(7,5))
    
    plt.plot(R/R[0], P, label="P")
    plt.plot(R/R[0], rho/rho[0], label=r"$\rho/ \rho_{0}$")
    plt.xlabel(r"r/r$_{0}$")
    plt.ylabel(r"P [Pa] and $\rho/\rho_{0}$")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("final_P_rho.png")

    fig, ax = plt.subplots(2, figsize=(7, 9))
    
    ax[0].plot(R/R[0], T, label="T")
    ax[0].set_xlabel(r"r/r$_{0}$")
    ax[0].set_ylabel("Temperature, T [K]")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(R/R[0], M/M[0], label=r"m/m$_{0}$")
    ax[1].plot(R/R[0], L/L[0], label=r"L/L$_{0}$")
    ax[1].set_xlabel(r"r/r$_{0}$")
    ax[1].set_ylabel(r"m/m$_{0}$ and L/L$_{0}$")
    ax[1].legend()
    ax[1].grid()

    fig.tight_layout()
    plt.savefig("final_model_params.png")
    
    # Plotting the fractions of energy being transported:
    plt.figure()
    plt.title("Energy transportation by radiation and convection")
    F_con_frac = F_con / (F_con + F_rad)
    F_rad_frac = F_rad / (F_con + F_rad)
    plt.plot(R/R[0], F_con_frac, label=r"$F_{con}$")
    plt.plot(R/R[0], F_rad_frac, label=r"$F_{rad}$")
    plt.xlabel(r"r/r$_{0}$")
    plt.ylabel("Fraction of energy transported")
    plt.legend()
    plt.grid()
    plt.savefig("final_model_flux_frac.png")

    # Plotting gradients:
    nabla_ad = 2/5
    plt.figure()
    plt.title("Temperature gradients")
    plt.plot(R/R[0], nabla_stable, linestyle="--", color="g", label=r"$\nabla_{stable}$")
    plt.plot(R/R[0], nabla_ad * np.ones_like(R), color="b", label=r"$\nabla_{ad}$")
    plt.plot(R/R[0], nabla_star, linestyle="--", color="orange", label=r"$\nabla^*$")
    plt.xlabel(r"r/r$_{0}$")
    plt.ylabel(r"$\nabla$")
    plt.legend()
    plt.grid()
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("final_model_gradients.png")

    plt.figure()
    plt.title("Zoom in of temperature gradients")
    plt.plot(R/R[0], nabla_stable, linestyle="--", color="g", label=r"$\nabla_{stable}$")
    plt.plot(R/R[0], nabla_ad * np.ones_like(R), color="b",  label=r"$\nabla_{ad}$")
    plt.plot(R/R[0], nabla_star, linestyle="--", color="orange", label=r"$\nabla^*$")
    plt.xlabel(r"r/r$_{0}$")
    plt.ylabel(r"$\nabla$")
    plt.xlim(0.7, 1.02)
    plt.ylim(0.39, 0.45)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("grad_zoom.png")

    # Plotting relative energy production
    total_E = PP1 + PP2 + PP3 + CNO
    fig = plt.figure()

    ax1 = fig.add_subplot()
    # ax2 = ax1.twiny() # To compare with temperature plot i proj 1

    fig.suptitle("Relative energy production")
    ax1.plot(R/R[0], CNO / total_E, linestyle="--", label="CNO")
    ax1.plot(R/R[0], PP1 / total_E, label="PPI")
    ax1.plot(R/R[0], PP2 / total_E, label="PPII")
    ax1.plot(R/R[0], PP3 / total_E, label="PPIII")
    ax1.plot(R/R[0], epsilon/np.max(epsilon), label=r"$\varepsilon(r)/\varepsilon_{max}$")
    plt.xlabel(r"r/r$_{0}$")
    ax1.set_ylabel("Relative energy")
    ax1.legend()
    ax1.grid()
    fig.tight_layout()

    plt.savefig("relative_energy.png")

    # Relative energy as a function of temperature instead:
    plt.figure()
    plt.title("Relative energy production")
    plt.plot(T, CNO / total_E, linestyle="--", label="CNO")
    plt.plot(T, PP1 / total_E, label="PPI")
    plt.plot(T, PP2 / total_E, label="PPII")
    plt.plot(T, PP3 / total_E, label="PPIII")
    plt.plot(T, epsilon/np.max(epsilon), label=r"$\varepsilon(r)/\varepsilon_{max}$")
    plt.xlabel("T [K]")
    plt.ylabel("Relative energy")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    
    plt.savefig("rel_E_temp.png")


variate_init()
final_model()

# Calling class and function to compare calculated values with opacity table and with values in example 5.1:
sanity = Energy_transportation(R0, rho0, T0, L0, M0, P0, sanity_check=True) # sanity_check=True give correct values for mu for next call

# Note that the fourth element from the bottom of the relative error in SI units is slightly above 5% such that a warning is printed.
sanity.opacity(T0, rho0, sanity_check=True) 
sanity.sanity_check()

"""
But to get the same sanity gradient plot we need the calculated mu value so we need to call the function again.
Note that the terminal will give an error because the loop stopped too early since the initial values in the sanity check
does not let r reach 0. The gradient plot gets saved."""
sanity = Energy_transportation(R0, rho0, T0, L0, M0, P0)
M, R, P, L, T, rho, kappa, nabla_stable, nabla_star, F_con, F_rad = sanity.move_inwards(0.01, integrate=False)
sanity.plot_gradients(R, nabla_stable, nabla_star)
sanity.cross_section(R, L, F_con, savefig=True, title="Cross section sanity check", sanity=True)


