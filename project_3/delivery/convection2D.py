"""
This code makes 2D simulations of the convective zone in a star. When running the code you will get the options to
make videos and take snapshots in time. It takes my computer about 3 minutes to make one 300 sec video. 
Please enjoy a good cup of coffee in the meantime, or start reading the report :)
"""

import FVis3 as FVis
import matplotlib.pyplot as plt
import numpy as np
import astropy.constants as ac


class TwoDconvection:
    """
    Class for solving the three equations of hydrodynamics in order to simulate the 
    convective zone in a star. 
    """
    def __init__(self, amplitude, sigmas, perturbation=None):
        """
        Defining variables and constants:
        """
        # Perturbation:
        self.A = amplitude                  # Amplitude of Gaussian perturbation
        self.sigma_x, self.sigma_y = sigmas # Standard deviations in Gaussian perturbation
        self.perturb = perturbation         # If this is not None the perturbation is added

        self.p = 0.1 # Used for CourantFriedrichs-Lewy (CFL) condition

        # Describing the box:
        self.N_y = 100      # Number of points in y-direction
        self.N_x = 300      # Number of points in x-direction
        self.y_height = 4e6 # Height of box in y-direction [m]
        self.x_width = 12e6 # Width of box in x-direction [m]
        
        self.dy = self.y_height / self.N_y
        self.dx = self.x_width / self.N_x

        self.y = np.linspace(0, self.y_height, self.N_y)
        self.x = np.linspace(0, self.x_width, self.N_x)

        self.mu = 0.61      # Mean molecular weight when assuming ideal gas

        self.G = ac.G.value         # Gravitational constant [m^3 kg^-1 s^-2]
        self.M_Sun = ac.M_sun.value # Solar mass [kg]
        self.R_Sun = ac.R_sun.value # Nominal solar radius [m]
        self.m_u = ac.u.value       # Atomic mass unit [kg]
        self.k_B = ac.k_B.value     # the Boltzmann constant [J/K]

        self.g = self.G * self.M_Sun / self.R_Sun**2 # Constant gravitational acceleration [m/s^2]
        self.nabla = 2/5 + 0.0001 # We add 0.0001 to always have convection: nabla > nabla_ad
        self.gamma = 5/3          # Adiabatic constant

        self.u = np.zeros((self.N_y, self.N_x))     # Initialize array for velocity in x-direction
        self.w = np.zeros((self.N_y, self.N_x))     # Initialize array for velocity in y-direction

        self.T = np.zeros((self.N_y, self.N_x))     # Initialize array for temperature
        self.P = np.zeros((self.N_y, self.N_x))     # Initialize array for pressure

        self.e = np.zeros((self.N_y, self.N_x))     # Initialize array for internal energy
        self.rho = np.zeros((self.N_y, self.N_x))   # Initialize array for density

        self.gauss = np.zeros((self.N_y, self.N_x)) # Initialize array for Gaussian perturbation

        # Initial values at the top (in the photosphere)
        self.w0 = 0         # Initial velocity in y direction
        self.T0 = 5778      # Temperature at solar photosphere [K]
        self.P0 = 1.8e4     # Pressure at solar photosphere [Pa]

        # Empty arrays for derivatives:
        self.drho_dt = np.zeros((self.N_y, self.N_x)); self.drhou_dt = np.zeros((self.N_y, self.N_x))
        self.drhow_dt = np.zeros((self.N_y, self.N_x)); self.de_dt = np.zeros((self.N_y, self.N_x))

    def gaussian(self, x, y):
        """
        Calculates the gaussian perturbation and updates the globbal variable self.gauss
        """
        mean_x = self.x_width / 2; mean_y = self.y_height / 2
        self.gauss = self.A * np.exp(-((x - mean_x)**2 / (2 * self.sigma_x**2) + (y - mean_y)**2 / (2 * self.sigma_y**2)))

    def initialise(self):
        """
        Initializes temperature, pressure, density and internal energy. Adds the Gaussian perturbation
        if an instance of the class is created with perturbation as an input-parameter.
        """
        R0 = self.y_height 

        xx, yy = np.meshgrid(self.x, self.y)

        self.T = self.T0 - (yy - R0) * self.mu * self.g * self.m_u * self.nabla / self.k_B
        self.P = self.P0 * (self.T / self.T0)**(1/self.nabla)

        self.gaussian(xx, yy)

        if self.perturb:
            # Adds the Gaussian perturbation
            self.T += self.gauss

        self.e = self.P / (self.gamma - 1)
        self.rho = self.mu * self.m_u * self.P / (self.k_B * self.T)


    def timestep(self):
        """
        Calculates the timestep
        """
        # Setting error handling to ignore division by zero from stationary points in the velocity field
        np.seterr(invalid="ignore", divide="ignore")
        
        # Relative change of position per time step:
        rel_y = np.max(np.abs(self.w[1:-1] / self.dy))
        rel_x = np.max(np.abs(self.u[1:-1] / self.dx))

        # Relative change per time step for the primary variables:
        rel_rho = np.max(np.abs(self.drho_dt[1:-1] / self.rho[1:-1]))
        rel_e = np.max(np.abs(self.de_dt[1:-1] / self.e[1:-1]))
        rel_rhou = np.max(np.abs(self.drhou_dt[1:-1] / (self.rho[1:-1] * self.u[1:-1])))
        rel_rhow = np.max(np.abs(self.drhow_dt[1:-1] / (self.rho[1:-1] * self.w[1:-1])))

        rel_vals = np.array([rel_y, rel_x, rel_rho, rel_e, rel_rhou, rel_rhow])
        
        # Removing NaNs and infs:
        rel_vals = rel_vals = rel_vals[np.logical_and(np.isinf(rel_vals) != True, np.isnan(rel_vals) != True)]
        
        # The largest relative change per time step for any of the quantities:
        delta = np.nanmax(rel_vals)

        dt = self.p / delta  # [s] Courant Friedrichs-Lewy (CFL) condition

        if dt < 1e-2:
            # If dt is too low, it is set to 1e-2 to reduce instability
            dt = 1e-2
        
        elif dt > 0.1:
            # If dt is too high, it is set to 0.1, also to reduce instability
            dt = 0.1

        return dt

    def boundary_conditions(self):
        """
        Sets boundary conditions for energy, density and velocity
        """
        # Vertical boundary:

        # Vertical velocity component is set to 0 at the upper and lower boundary:
        self.w[-1, :] = self.w0; self.w[0, :] = self.w0 

        # Horizontal velocity component is set to 0 at the boundary: 
        self.u[-1, :] = (4 * self.u[-2, :] - self.u[-3, :]) / 3 # Lower bound
        self.u[0, :] = (4 * self.u[1, :] - self.u[2, :]) / 3    # Upper bound

        # Lower bound:
        self.e[-1, :] = (4 * self.e[-2, :] -  self.e[-3, :]) / (3 + 2 * self.dy * self.mu * self.m_u * self.g / (self.k_B * self.T[-1, :]))
        # Upper bound:
        self.e[0, :] = (4 * self.e[1, :] - self.e[2, :]) / (3 - 2 * self.dy * self.mu * self.m_u * self.g / (self.k_B * self.T[0, :]))

        self.P[0, :] = (self.gamma - 1) * self.e[0, :]
        self.P[-1, :] = (self.gamma - 1) * self.e[-1, :]

        # Lower bound:
        self.rho[-1, :] = self.mu * self.m_u * (self.gamma - 1) * self.e[-1, :] / (self.k_B * self.T[-1, :])
        # Upper bound:
        self.rho[0, :] = self.mu * self.m_u * (self.gamma - 1) * self.e[0, :] / (self.k_B * self.T[0, :])


    def central_x(self, phi):
        """
        Central difference scheme in x-direction
        phi = some variable/function ex. T, P
        """
        phi_right = np.roll(phi, -1, axis=1)  
        phi_left = np.roll(phi, 1, axis=1)

        return (phi_right - phi_left) / (2 * self.dx)

    def central_y(self, phi):
        """
        Central difference scheme in y-direction
        """
        phi_up = np.roll(phi, -1, axis=0)  
        phi_down = np.roll(phi, 1, axis=0)

        return (phi_up - phi_down) / (2 * self.dy)

    def upwind_x(self, phi, v):
        """
        Upwind difference scheme in x-direction
        """
        phi_right = np.roll(phi, -1, axis=1)
        phi_left = np.roll(phi, 1, axis=1)

        v_pos = (phi - phi_left) / self.dx
        v_neg = (phi_right - phi) / self.dx

        return np.where(v >= 0, v_pos, v_neg)

    def upwind_y(self, phi, v):
        """
        Upwind difference scheme in y-direction
        """
        phi_up = np.roll(phi, -1, axis=0)
        phi_down = np.roll(phi, 1, axis=0)

        v_pos = (phi - phi_down) / self.dy
        v_neg = (phi_up - phi) / self.dy

        return np.where(v >= 0, v_pos, v_neg)
    
    def calc_equilibrium(self):
        """
        Calculates how close the system is to hydrostatic equilibrium
        """
        dP_dy = self.central_y(self.P)
        print(f"avg(|dP/dy - rho g|) = {np.average(np.abs(dP_dy - self.rho * self.g))}")

    def hydro_solver(self):
        """
        Hydrodynamic equations solver
        """
        # Continuity equation:
        du_dx = self.central_x(self.u)
        drho_dx = self.upwind_x(self.rho, self.u)
        dw_dy = self.central_y(self.w)
        drho_dy = self.upwind_y(self.rho, self.w)
        self.drho_dt[1:-1] = - self.rho[1:-1] * du_dx[1:-1] - self.u[1:-1] * drho_dx[1:-1] - self.rho[1:-1] * dw_dy[1:-1] - self.w[1:-1] * drho_dy[1:-1]
        
        # Momentum equation, x-direction:
        du_dx = self.upwind_x(self.u, self.u)
        drhou_dx = self.upwind_x(self.rho * self.u, self.u)
        dw_dy = self.central_y(self.w)
        drhou_dy = self.upwind_y(self.rho * self.u, self.w)
        dP_dx = self.central_x(self.P)
        self.drhou_dt[1:-1] = - self.rho[1:-1] * self.u[1:-1] * du_dx[1:-1] - self.u[1:-1] * drhou_dx[1:-1] - self.rho[1:-1] * self.u[1:-1] * dw_dy[1:-1] - self.w[1:-1] * drhou_dy[1:-1] - dP_dx[1:-1]

        # Momentum equation, y-direction:
        du_dx = self.central_x(self.u)
        drhow_dx = self.upwind_x(self.rho * self.w, self.u)
        dw_dy = self.upwind_y(self.w, self.w)
        drhow_dy = self.upwind_y(self.rho * self.w, self.w)
        dP_dy = self.central_y(self.P)
        self.drhow_dt[1:-1] = - self.rho[1:-1] * self.w[1:-1] * du_dx[1:-1] - self.u[1:-1] * drhow_dx[1:-1] - self.rho[1:-1] * self.w[1:-1] * dw_dy[1:-1] - self.w[1:-1] * drhow_dy[1:-1]- dP_dy[1:-1] - self.rho[1:-1] * self.g

        # Internal energy equation
        du_dx = self.central_x(self.u)
        de_dx = self.upwind_x(self.e, self.u)
        dw_dy = self.central_y(self.w)
        de_dy = self.upwind_y(self.e, self.w)
        self.de_dt[1:-1] = - self.e[1:-1] * du_dx[1:-1] - self.u[1:-1] * de_dx[1:-1] - self.e[1:-1] * dw_dy[1:-1] - self.w[1:-1] * de_dy[1:-1] - self.P[1:-1] * du_dx[1:-1] - self.P[1:-1] * dw_dy[1:-1]

        dt = self.timestep() # Updating time step

        # Updating primary variables
        rho_next = self.rho[1:-1] + self.drho_dt[1:-1] * dt
        e_next = self.e[1:-1] + self.de_dt[1:-1] * dt

        u_next = (self.rho[1:-1] * self.u[1:-1] + self.drhou_dt[1:-1] * dt) / rho_next
        w_next = (self.rho[1:-1] * self.w[1:-1] + self.drhow_dt[1:-1] * dt) / rho_next

        self.rho[1:-1] = rho_next; self.e[1:-1] = e_next; self.w[1:-1] = w_next; self.u[1:-1] = u_next

        # Setting boundary conditions:
        self.boundary_conditions()

        # Updating T & P:
        self.P[:] = (self.gamma - 1) * self.e
        self.T[:] = self.P * self.mu * self.m_u / (self.rho * self.k_B)

        return dt
    

amplitude = 0.9 * 5778          # Amplitude of pertubation
sigma_x = 1e6; sigma_y = 1e6    # Standard deviations in perturbation


# ----------------------------- SANITY CHECK -------------------------------------
run_sanity = input("Do you want to make sanity videos? y/n \n")

if run_sanity == "y":
    sanity = TwoDconvection(amplitude, (sigma_x, sigma_y))
    sanity.initialise()
    
    vis = FVis.FluidVisualiser(fontsize=18)
    vis.save_data(60, sanity.hydro_solver, u=sanity.u, w=sanity.w, T=sanity.T, rho=sanity.rho, e=sanity.e, P=sanity.P, folder="Sanity_check")

    # Checking if all parameters are 0:
    vis.animate_2D("T", video_fps=5, units={"Lx":"Mm", "Lz":"Mm"}, extent=[0, 12, 0, 4], showQuiver=True, video_name="sanity_check_T", save=True)
    vis.animate_2D("P", video_fps=5, units={"Lx":"Mm", "Lz":"Mm"}, extent=[0, 12, 0, 4], showQuiver=True, video_name="sanity_check_P", save=True)
    vis.animate_2D("rho", video_fps=5, units={"Lx":"Mm", "Lz":"Mm"}, extent=[0, 12, 0, 4], showQuiver=True, video_name="sanity_check_rho", save=True)
    vis.animate_2D("e", video_fps=5, units={"Lx":"Mm", "Lz":"Mm"}, extent=[0, 12, 0, 4], showQuiver=True, video_name="sanity_check_e", save=True)
    vis.animate_2D("u", video_fps=5, units={"Lx":"Mm", "Lz":"Mm"}, extent=[0, 12, 0, 4], showQuiver=True, video_name="sanity_check_u", save=True)

    # Note that w does not stay 0, but has a much lower value than when the perturbation is introduced:
    vis.animate_2D("w", video_fps=5, units={"Lx":"Mm", "Lz":"Mm"}, extent=[0, 12, 0, 4], showQuiver=True, video_name="sanity_check_w", save=True)

    print_equilibrium = input("Do you want to print avg(|dP/dy - rho g|) in order to see \n how close to hydrostatic equilibrium the system is? y/n \n")
    
    if print_equilibrium == "y":
        sanity.calc_equilibrium()

# ------------------------- ADDING PERTURBATION -----------------------------------
run_perturbation = input("Do you want to make perturbation videos? y/n \n")

if run_perturbation == "y":

    param = input("Which parameter do you wish to make video of? T/w/energy flux \n")
    create_snapshots = input("Do you wish to take snapshots? y/n \n")
    plot_avg_flux = input("Do you wish to plot the time evolution of the average vertical energy flux? y/n \n")
    
    solver = TwoDconvection(amplitude, (sigma_x, sigma_y), perturbation=True)
    solver.initialise()

    snapshot_times = [0, 75, 120, 170, 240]
    
    vis = FVis.FluidVisualiser(fontsize=18)
    vis.save_data(300, solver.hydro_solver, sim_fps=1, u=solver.u, w=solver.w, T=solver.T, rho=solver.rho, e=solver.e, P=solver.P, folder="Convection")

    if plot_avg_flux == "y":
        vis.plot_avg("ew", showTrendline=True, units={"Lx":"Mm", "Lz":"Mm"})
    
    if param == "T":
        if create_snapshots == "y":
            vis.animate_2D("T", units={"Lx":"Mm", "Lz":"Mm"}, snapshots=snapshot_times, quiverscale=1.2, video_name="convection_T_300", extent=[0, 12, 0, 4], showQuiver=True)
        vis.animate_2D("T", units={"Lx":"Mm", "Lz":"Mm"}, video_name="convection_T_300", quiverscale=1.2, extent=[0, 12, 0, 4], showQuiver=True, save=True)
    
    elif param == "w":
        """
        I added this one to see that even though w does not stay 0 in hydrostatic equilibrium,
        the vertical velocity still looks physical.
        """
        if create_snapshots == "y":
            vis.animate_2D("w", units={"Lx":"Mm", "Lz":"Mm"}, snapshots=snapshot_times, quiverscale=1.2, video_name="convection_w_300", extent=[0, 12, 0, 4], showQuiver=True)
        vis.animate_2D("w", units={"Lx":"Mm", "Lz":"Mm"}, video_name="convection_w_300", quiverscale=1.2, extent=[0, 12, 0, 4], showQuiver=True, save=True)
    
    elif param == "energy flux":
        if create_snapshots == "y":
            vis.animate_energyflux(units={"Lx":"Mm", "Lz":"Mm"}, snapshots=snapshot_times, video_name="energy_flux_300", extent=[0, 12, 0, 4])
        vis.animate_energyflux(units={"Lx":"Mm", "Lz":"Mm"}, video_name="energy_flux_300", extent=[0, 12, 0, 4], save=True)
    
    elif param != "T" or param != "w" or param != "energy flux":
        print("You did not state which parameter you wanted to simulate.")
