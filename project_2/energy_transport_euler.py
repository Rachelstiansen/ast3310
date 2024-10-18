import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
import scipy.constants as sc
from scipy.interpolate import RectBivariateSpline
from tqdm import tqdm

from energy_production import Energy_production

class Energy_transportation:
    
    def __init__(self, r0, rho0, T0, L0, m0, sanity_check=None):

        self.r_0 = r0           # [solar mass]
        self.rho_0 = rho0       
        self.T_0 = T0           
        self.L_0 = L0           
        self.m_0 = m0           

        self.L_sun = 3.846e26   # [W]
        self.M_sun = 1.989e30   # [kg]
        self.R_sun = 6.96e8     # [m]
        self.rho_sun = 1.408e3  # [kg m^-3]

        # To get inputs correct
        self.r_0 = r0 * self.R_sun
        self.m_0 = m0 * self.M_sun
        self.L_0 = L0 * self.L_sun
        self.rho_0 = rho0 * self.rho_sun

        # Element abundances:
        self.X = 0.7; self.Y_3He = 1e-10; self.Y = 0.29
        self.Z_7Li = 1e-7; self.Z_7Be = 1e-7; self.Z_14N = 1e-11

        if sanity_check == True:
            self.mu = 0.6
        else:
            self.mu = 1 / (2 * self.X + (3/4) * self.Y + self.Y_3He + (4/7) * self.Z_7Li + (5/7) * self.Z_7Be + (4/7) * self.Z_14N)

        self.k_B = 1.3806e-23           # Boltzmann's constant      [m^2 kg/(s^2 K)]
        self.m_u = sc.m_u               # atomic mass unit          [kg]
        self.sigma = 5.6704e-8          # Stefan-Boltzmanns const.  [W m^-2 K^-4]
        self.a = 4 * self.sigma / sc.c  # radiation density constant
        self.G = sc.G                   # gravitational const. [N m^2 kg^-2]
        self.alpha_l_m = 1      
        self.nabla_ad = 2/5
        self.c_P = 5 * self.k_B / (2 * self.mu * self.m_u)

        #self.r = []; self.rho = []; self.T = []; self.L = []; self.m = []
        #self.P = []; 
    
    def opacity(self, T, rho, sanity_check=None):
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
            rel_error_SI = abs((opacity_calc_SI - opacity_sanity_SI) / (opacity_calc_SI)) # just above 5% ....

            headers = ["log_10 T", "log_10 R \n(cgs)", "log_10 \nkappa (cgs)", "log_10 kappa \ncalc (cgs)","kappa \n(SI)", 
                       "kappa \ncalc (SI)", "relative \nerror % (cgs)", "relative \nerror % (SI)"]
            table = zip(T_sanity, R_sanity, opacity_sanity_cgs, opacity_calc_cgs, opacity_sanity_SI, opacity_calc_SI, rel_error_cgs, rel_error_SI)
            print(f"Sanity check for interpolation of opacity")
            print(tabulate(table, headers=headers, tablefmt="fancy_grid"), "\n")

        return kappa_SI
    
    def calc_pressure(self, rho, T):
        # should use mu=0.6?
        P_rad = self.a / 3 * T**4
        P_gas = rho * self.k_B * T / (self.mu * self.m_u)
        P_tot = P_rad + P_gas

        return P_tot

    def calc_density(self, P, T):
        # should use mu=0.6?
        P_rad = self.a / 3 * T**4
        rho = ((P - P_rad) * self.mu * self.m_u) / (self.k_B * T)

        return rho

    def g(self, m, r):
        """"
        Calculates the gravitational acceleration g
        """
        return self.G * m / r**2
    
    def U(self, T, rho, m, r, kappa):
        """
        Calculates U
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
        S = 4 * np.pi * r**2 # Surface area of spherical parcel
        Q = np.pi * r**2     # Surface normal to the velocity
        d = 2 * r            # Diameter of sphere
        
        return S / (Q * d)

    def K(self, T, rho, m, r, kappa):
        return self.U(T, rho, m, r, kappa) * self.geometric_factor(r) / self.l_m(T, m, r)

    def L(self, T, rho):
        Sun = Energy_production(T, rho)
        return Sun.energy_produced(verbose=False)[1]     # total energy

    def nabla_stable(self, T, r, rho, m):
        return 3 * self.L(T, rho) * self.opacity(T, rho) * rho * self.H_P(T, m, r) / (64 * np.pi * r**2 * self.sigma * T**4)

    def Xi(self, T, rho, m, r, nabla_stable, kappa):
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
    
    def nabla_p(self):
        ... # rearrange the qubic equation to find this

    def v(self, T, m, r, rho, nabla_stable, kappa):
        return self.l_m(T, m, r) * np.sqrt(self.g(m, r) / self.H_P(T, m, r)) / 2 * self.Xi(T, rho, m, r, nabla_stable, kappa)

    def Flux(self, rho, T, m, r, nabla_stable, kappa, nabla_star, con=None):
        # print(self.g(m, r), m, r)
        F_con = rho * self.c_P * T * np.sqrt(self.g(m, r)) * self.H_P(T, m, r)**(-3/2) * (self.l_m(T, m, r) / 2)**2 * self.Xi(T, rho, m, r, nabla_stable, kappa)**3
        F_rad = 16 * self.sigma * T**4 * nabla_star / (3 * kappa * rho * self.H_P(T, m, r))

        if con == True:
            return F_con
        else:
            return F_rad


    def euler_integrate(self, p):
        # Trying euler method
        eqs_0 = np.array([self.r_0, self.calc_pressure(self.rho_0, self.T_0), self.L_0, self.T_0])    
        
        N = int(4e3)

        V = np.zeros((4, N + 1))
        V[:, 0] = eqs_0
        m = np.zeros(N + 1)
        m[0] = self.m_0

        for i in tqdm(range(N)):

            dV_list = self.eqs(m[i], V[:, i])
            dm = - np.min(V[:, i] * p / np.abs(dV_list))

            V[:, i + 1] = V[:, i] + self.eqs(m[i], V[:, i]) * dm
            m[i + 1] = m[i] + dm

            if abs(m[i] - m[i-1]) < 1e15:
                print("Step length dm converged to zero")

            if m[i + 1] < 0 or np.any(V[:, i + 1]) < 0:
                print("Mass or something is negative")
                print(V[:, i], m[i])
                print(V[:, i + 1], m[i + 1])
                break
            
        return m, V
    
    def eqs(self, m, eqs, sanity_check=None):

        if sanity_check == True:
            T = 0.9e6 # [K]
            rho = 55.9 # [kg m^-3]
            r = 0.84 * self.R_sun 
            m = 0.99 * self.M_sun
            L = self.L_sun
            kappa = 3.98 # [m^2 kg^-1]
            P = self.calc_pressure(rho, T)
            nabla_stable = 3.26 # should not seet this one, but calc.

        else:
            r, P, L, T = eqs
            rho = self.calc_density(P, T)
            kappa = self.opacity(T, rho)
            nabla_stable = self.nabla_stable(T, r, rho, m)


        dr = 1 / (4 * np.pi * r**2 * rho)
        dP = - self.G * m / (4 * np.pi * r**4)
        dL = self.L(T, rho) # self.L(1.57e7, 1.62e5)

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
            print("nabla_stable = ", nabla_stable)
            print("Xi = ", self.Xi(T, rho, m, r, nabla_stable, kappa)) # right order, but could be better
            print("U = ", self.U(T, rho, m, r, kappa))# right order, but not number
            print("H_P = ", self.H_P(T, m, r)) # seems not to far from what we expect
            print("nabla_star =", nabla_star)
            print("v = ", self.v(T, m, r, rho, nabla_stable, kappa))
            print("F_con/(F_con + F_rad) = ", F_con / (F_con + F_rad))
            print("F_rad / (F_con + F_rad) = ", F_rad / (F_con + F_rad))

        return np.array([dr, dP, dL, dT])
    
    def sanity_check(self):
        m, V = self.euler_integrate(0.01)
        self.eqs(m, V, sanity_check=True)
    
    def simulate_star(self):
        m, V = self.euler_integrate(0.01)
        r, P, L, T = V



# sanity_check=True gives correct value for mu
sanity = Energy_transportation(1, 1.42e-7, 5770, 1, 1, sanity_check=True)
simulate = Energy_transportation(1, 1.42e-7, 5770, 1, 1)
# sanity.sanity_check()
simulate.simulate_star()


# test.opacity(10, 10, sanity_check=True)

"""
Questions: 
- should we check for example 5.2 also?
- delta in the velocity? -> find it from exercise 5.6)

Next: implement nabla star for unstable.

To do: check that all the inequalities hold, try for a RK4 solver method
- change rho, T, r, m = variables

Problem: dm gets too big so that m gets negative and then i get sqrt(neg num) in g....
"""