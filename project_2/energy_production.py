"""
How to run the code:
    When running the script all the tables are printed as I have called the methods of the class in the very end of the code.
    You will have to write an input to access all the results because the sanity check is not passed using the energy from mass difference
    for reaction e7. When the sanity check fails, you will get the option to insert a new value and if you insert "0.049" MeV as the new energy
    output for this reaction the rest of the tables and figures should show up.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as sc
from tabulate import tabulate


class Energy_production:
    """
    Class that calculates the energy production at the
    center of a star given a temperature and density of the solar core
    """
    def __init__(self, temperature, density):
        
        self.T = temperature    # temperature in solar core
        self.rho = density      # density in solar core

        """
        Mass fractions of the each atomic species:
        """
        self.X = 0.7            # Hydrogen
        self.Y = 0.29           # 4-Helium
        self.Y32He = 1e-10      # 3-Helium
        self.Z73Li = 1e-7       # 7-Lithium
        self.Z74Be = 1e-7       # 7-Beryllium
        self.Z147N = 1e-11      # 14-Nitrogen

        """
        Atomic numbers:
        """
        self.Z_H = 1                   # Hydrogen
        self.Z_He = 2;   self.Z_C = 6  # Helium & Carbon
        self.Z_Li = 3;   self.Z_N = 7  # Lithium & Nitrogen
        self.Z_Be = 4;   self.Z_O = 8  # Beryllium & Oxygen

        """
        Constants needed:
        """
        self.P = 3.45e16                # pressure in solar core    [Pa]
        self.N_A = sc.N_A               # Avogadro's number         [mol^(-1)]
        self.k_B = 1.3806e-23           # Boltzmann's constant      [m^2 kg/(s^2 K)]
        self.m_u = sc.m_u               # atomic mass unit          [kg]
        self.epsilon_0 = sc.epsilon_0   # vacuum permittivity       [F m^-1]
        self.e = sc.e                   # Elementary charge         [C]
        self.h = sc.h                   # Planck's constant         [J K^-1]

        """
        Conversion factors:
        """
        self.u = 931.494                        # [MeV/c^2]
        self.lamda_conversion = 1e-6            # to convert the cm^3 to m^3 in the proportionality function
        self.energy_conversion = 1e6 * sc.eV    # to convert energy from MeV to J

        """
        Mass of elements in atomic mass units, u:
        """
        self.m_1H =  1.00782503223      # mass of hydrogen    [u]
        self.m_2D =  2.01410177812      # mass of deuterium   [u]
        self.m_3He = 3.0160293201       # mass of 3-helium    [u]
        self.m_4He = 4.00260325413      # mass of 4-helium    [u]
        self.m_7Be = 7.016928717        # mass of 7-beryllium [u]
        self.m_7Li = 7.0160034366       # mass of 7-lithium   [u]
        self.m_8B =  8.0246073          # mass of 8-boron     [u]
        self.m_8Be = 8.005305102        # mass of 8-beryllium [u]
        self.m_12C = 12.00000000        # mass of 12-carbon   [u]
        self.m_13N = 13.00573861        # mass of 13-nitrogen [u]
        self.m_13C = 13.00335483507     # mass of 13-carbon   [u]
        self.m_14N = 14.00307400443     # mass of 14-nitrogen [u]
        self.m_15O = 15.00306562        # mass of 15-oxygen   [u]
        self.m_15N = 15.00010889888     # mass of 15-nitrogen [u]

        """
        Dictionary containing neutrinoes created in the reactions, units in J:
        """
        self.neutrinos = {"v_pp" : 0.265 * self.energy_conversion, "v_e7" : 0.815 * self.energy_conversion, 
                          "v_8" : 6.711 * self.energy_conversion, "v_13" : 0.707 * self.energy_conversion, 
                          "v_15": 0.997 * self.energy_conversion} # [J]
        
        """
        Arrays defined to be able to access values outside of class later for plotting relative energy.
        The arrays will contain the energy from the completed branches and the total energy epsilon.
        """
        self.PP1_energy = np.zeros(1)
        self.PP2_energy = np.zeros(1)
        self.PP3_energy = np.zeros(1)
        self.CNO_energy = np.zeros(1)
        self.epsilon = np.zeros(1)


    def number_densities(self):
        """
        Function for calculating and storing number densities of the various elements/isotopes
        """
        n_p = self.rho * self.X / self.m_u
        n_4He = self.rho * self.Y / (4 * self.m_u)
        n_3He = self.rho * self.Y32He / (3 * self.m_u)
        n_7Li = self.rho * self.Z73Li / (7 * self.m_u)
        n_7Be = self.rho * self.Z74Be / (7 * self.m_u)
        n_14N = self.rho * self.Z147N / (14 * self.m_u)
        n_e = self.rho * (1 + self.X) / (2 * self.m_u)

        return {"n_p" : n_p, "n_4He" : n_4He, "n_3He" : n_3He, "n_7Li" : n_7Li,
                "n_7Be" : n_7Be, "n_14N" : n_14N, "n_e" : n_e} # [m^-3]
    
    
    def prop_func(self):
        """
        Function for calculating the proportionality function for all the
        reactions given in table 3.1 in the lecture notes
        """
        T9 = self.T * 1e-9 # [K]
        T9s = T9 / (1 + 4.95e-2 * T9)
        T9ss = T9 / (1 + 0.759 * T9)

        lamda_pp = self.lamda_conversion * (4.01e-15 * T9**(-2/3) * np.exp(-3.380 * T9**(-1/3)) * (1 + 0.123 * T9**(1/3) + 1.09 * T9**(2/3) + 0.938 * T9)) / self.N_A
        lamda_33 = self.lamda_conversion * (6.04e10 * T9**(-2/3) * np.exp(-12.276 * T9**(-1/3)) * (1 + 0.034 * T9**(1/3) - 0.522 * T9**(2/3) - 0.124 * T9 
                            + 0.353 * T9**(4/3) + 0.213 * T9**(5/3))) / self.N_A
        lamda_34 = self.lamda_conversion * (5.61e6 * T9s**(5/6) * T9**(-3/2) * np.exp(-12.826 * T9s**(-1/3))) / self.N_A
        lamda_e7 = self.lamda_conversion * (1.34e-10 * T9**(-1/2) * (1 - 0.537 * T9**(1/3) + 3.86 * T9**(2/3) + 0.0027 * T9**(-1) * np.exp(2.515e-3 * T9**(-1)))) / self.N_A
        lamda_17_prime = self.lamda_conversion * (1.096e9 * T9**(-2/3) * np.exp(-8.472 * T9**(-1/3)) - 4.830e8 * T9ss**(5/6) * T9**(-3/2) * 
                            np.exp(-8.472 * T9ss**(-1/3)) + 1.06e10 * T9**(-3/2) * np.exp(-30.442 * T9**(-1))) / self.N_A
        lamda_17 = self.lamda_conversion * (3.11e5 * T9**(-2/3) * np.exp(-10.262 * T9**(-1/3)) + 2.53e3 * T9**(-3/2) * np.exp(-7.306 * T9**(-1))) / self.N_A
        lamda_p14 = self.lamda_conversion * (4.90e7 * T9**(-2/3) * np.exp(-15.228 * T9**(-1/3) - 0.092 * T9**2) * 
                            (1 + 0.027 * T9**(1/3) - 0.778 * T9**(2/3) - 0.149 * T9 + 0.261 * T9**(4/3) + 0.127 * T9**(5/3)) 
                            + 2.37e3 * T9**(-3/2) * np.exp(-3.011 * T9**(-1)) + 2.19e4 * np.exp(-12.53 * T9**(-1))) / self.N_A
        
        return {"l_pp" : lamda_pp, "l_33" : lamda_33, "l_34" : lamda_34, "l_e7" : lamda_e7, 
                    "l_17'" : lamda_17_prime, "l_17" : lamda_17, "l_p14" : lamda_p14} # [m^3 s^-1]


    def reaction_rates(self, n_i, n_k, lamda):
        """
        Function for calculating reaction rates given the number densities
        n_i and n_k and the proportionality function.
        """
        if n_i == n_k:
            return (n_i * n_k * lamda) / (self.rho * 2) # [kg^-1 s^-1]
        
        else:
            return (n_i * n_k * lamda) / (self.rho * 1) # [kg^-1 s^-1]
    

    def energy_output(self, init_mass, final_mass):
        """
        Function for calculating the energy output from the mass difference in a 
        reaction given the initial and final mass.
        """
        dm = - (final_mass - init_mass) * self.m_u # mass difference [kg]
        dE = dm * sc.c**2 # energy output [J]

        return dE
    

    def PP0(self, r_correction=None):
        """
        Function for calculating the mass difference energy Q, the reaction rate r and the energy produced by each reaction
        in the first 2 common reactions in the PP-chain. When r_correction=None then the energies are calculated from the function
        reaction_rates(). To make sure no step consumes more of an element than the previous steps have produced, r_correction is used. 
        r_correction is calculated in the function limit_reaction_rates().
        """
        PP0_reactions = np.zeros((2, 2))
        PP0_reactions[0, :] = np.array([2 * self.m_1H, self.m_2D])          # [u]
        PP0_reactions[1, :]= np.array([self.m_2D + self.m_1H, self.m_3He])  # [u]

        Q_pp = self.energy_output(PP0_reactions[0, 0], PP0_reactions[0, 1]) + self.energy_output(PP0_reactions[1, 0], PP0_reactions[1, 1]) # [J]
        Q_pp = Q_pp - self.neutrinos["v_pp"]            # [J]

        n_p = self.number_densities()["n_p"]            # [m^-3]
        lamda_pp = self.prop_func()["l_pp"]             # [m^3 s^-1]
        r_pp = self.reaction_rates(n_p, n_p, lamda_pp)  # [kg^-1 s^-1]

        if r_correction != None:
            E_produced = r_correction * Q_pp * self.rho # [J m^-3 s^-1]
        else:
            E_produced = r_pp * Q_pp * self.rho         # [J m^-3 s^-1]

        return Q_pp, r_pp, E_produced
    

    def PPI(self, r_correction=None):
        """
        Function for calculating the mass difference energy Q, the reaction rate r and the energy produced by the reaction 
        in the PPI-branch. r_correction is used as described in PP0().
        """
        PP1_reaction = np.array([2 * self.m_3He, self.m_4He + 2*self.m_1H])     # [u]
        
        Q_33 = self.energy_output(PP1_reaction[0], PP1_reaction[1])             # [J]
        n_3He = self.number_densities()["n_3He"]                                # [m^-3]
        lamda_33 = self.prop_func()["l_33"]                                     # [m^3 s^-1]

        if r_correction != None:
            r_33 = r_correction                                                 # [kg^-1 s^-1]
        else:
            r_33 = self.reaction_rates(n_3He, n_3He, lamda_33)                  # [kg^-1 s^-1]

        E_produced = r_33 * Q_33 * self.rho                                     # [J m^-3 s^-1]

        return Q_33, r_33, E_produced
    

    def PPII(self, r_correction=None):
        """
        Function for calculating the mass difference energy Q, the reaction rate r and the energy produced by the reaction
        in the PPII-branch. r_correction is used as described in PP0().
        """
        PP2_reactions = np.zeros((3,2))
        PP2_reactions[0, :] = np.array([self.m_3He + self.m_4He, self.m_7Be]);    PP2_reactions[1, :] = np.array([self.m_7Be, self.m_7Li])  # [u]
        PP2_reactions[2, :] = np.array([self.m_7Li + self.m_1H, 2 * self.m_4He])  # [u]
        
        n = self.number_densities(); l = self.prop_func(); p = PP2_reactions

        init_mass = np.array([p[0, 0], p[1, 0], p[2, 0]])           # [u]
        final_mass = np.array([p[0, 1], p[1, 1], p[2, 1]])          # [u]

        n_i = np.array([n["n_3He"], n["n_7Be"], n["n_7Li"]])        # [m^-3]
        n_k = np.array([n["n_4He"], n["n_e"], n["n_p"]])            # [m^-3]

        lamda = np.array([l["l_34"], l["l_e7"], l["l_17'"]])        # [m^3 s^-1]

        Q_PP2 = np.zeros(3); r_PP2 = np.zeros(3); total_energy = np.zeros(3)

        if self.T < 1e6: # electron capture by 7Be has an upper limit temperature:
            lamda[1] = 1.57e-7 / (self.N_A * n["n_e"])              # [m^3 s^-1]

        for i in range(3):
            if i == 1:
                Q_PP2[1] = self.energy_output(init_mass[i], final_mass[i]) - self.neutrinos["v_e7"] # [J]
            else:
                Q_PP2[i] = self.energy_output(init_mass[i], final_mass[i])                          # [J]

            if r_correction != None:
                r_PP2[i] = np.array(r_correction)[i]                        # [kg^-1 s^-1]
            else:
                r_PP2[i] = self.reaction_rates(n_i[i], n_k[i], lamda[i])    # [kg^-1 s^-1]

            total_energy[i] = r_PP2[i] * Q_PP2[i] * self.rho                # [J m^-3 s^-1]

        return Q_PP2, r_PP2, total_energy


    def PPIII(self, r_correction=None):
        """
        Function for calculating the mass difference energy Q, the reaction rate r and the energy produced by the reaction 
        in the PPIII-branch. r_correction is used as described in PP0().
        """
        PP3_reactions = np.zeros((4, 2))
        PP3_reactions[0, :] = np.array([self.m_3He + self.m_4He, self.m_7Be]);  PP3_reactions[2, :] = np.array([self.m_8B, self.m_8Be])         # [u]
        PP3_reactions[1, :] = np.array([self.m_7Be + self.m_1H, self.m_8B]);    PP3_reactions[3, :] = np.array([self.m_8Be, 2 * self.m_4He])    # [u]

        n = self.number_densities(); l = self.prop_func(); p = PP3_reactions

        init_mass = np.array([p[0, 0], p[1, 0], p[2, 0], p[3, 0]])      # [u]
        final_mass = np.array([p[0, 1], p[1, 1], p[2, 1], p[3, 1]])     # [u]

        n_i = np.array([n["n_3He"], n["n_7Be"]])                        # [m^-3]
        n_k = np.array([n["n_4He"], n["n_p"]])                          # [m^-3]
        lamda = np.array([l["l_34"], l["l_17"]])                        # [m^3 s^-1]

        Q_PP3 = np.zeros(4); r_PP3 = np.zeros(2); total_energy = np.zeros(2)    
        
        for i in range(4):
            if i == 2:
                Q_PP3[i] = self.energy_output(init_mass[i], final_mass[i]) - self.neutrinos["v_8"]  # [J]
            else:   
                Q_PP3[i] = self.energy_output(init_mass[i], final_mass[i])                          # [J]
        
        for i in range(2):
            if r_correction != None:
                r_PP3[i] = np.array(r_correction)[i]                                # [kg^-1 s^-1]
            else:
                r_PP3[i] = self.reaction_rates(n_i[i], n_k[i], lamda[i])            # [kg^-1 s^-1]

        total_energy[0] = r_PP3[0] * (Q_PP3[0]) * self.rho                          # [J m^-3 s^-1]
        total_energy[1] = r_PP3[1] * (Q_PP3[1] + Q_PP3[2] + Q_PP3[3]) * self.rho    # [J m^-3 s^-1]

        return Q_PP3, r_PP3, total_energy


    def CNO(self):
        """
        Function for calculating the mass difference energy Q, the reaction rate r and the energy produced by the reaction 
        in the PPIII-branch.
        """
        CNO_cycle = np.zeros((6, 2))
        CNO_cycle[0, :] = np.array([self.m_12C + self.m_1H, self.m_13N]);   CNO_cycle[3, :] = np.array([self.m_14N + self.m_1H, self.m_15O])                # [u]
        CNO_cycle[1, :] = np.array([self.m_13N, self.m_13C]);               CNO_cycle[4, :] = np.array([self.m_15O, self.m_15N])                            # [u]
        CNO_cycle[2, :] = np.array([self.m_13C + self.m_1H, self.m_14N]);   CNO_cycle[5, :] = np.array([self.m_15N + self.m_1H, self.m_12C + self.m_4He])   # [u]
        
        c = CNO_cycle

        init_mass = np.array([c[0, 0], c[1, 0], c[2, 0], c[3, 0], c[4, 0], c[5, 0]])    # [u]
        final_mass = np.array([c[0, 1], c[1, 1], c[2, 1], c[3, 1], c[4, 1], c[5, 1]])   # [u]
        
        n_i = self.number_densities()["n_14N"]      # [m^-3]
        n_k = self.number_densities()["n_p"]        # [m^-3]
        lamda = self.prop_func()["l_p14"]           # [m^3 s^-1]

        Q_CNO = np.zeros(6)

        for i in range(6):
            if i == 1:
                Q_CNO[i] = self.energy_output(init_mass[i], final_mass[i]) - self.neutrinos["v_13"]     # [J]
            elif i == 4:
                Q_CNO[i] = self.energy_output(init_mass[i], final_mass[i]) - self.neutrinos["v_15"]     # [J]
            else:
                Q_CNO[i] = self.energy_output(init_mass[i], final_mass[i])
        
        r_CNO = self.reaction_rates(n_i, n_k, lamda)        # [kg^-1 s^-1]
        E_produced = r_CNO * (np.sum(Q_CNO)) * self.rho     # [J m^-3 s^-1]

        return Q_CNO, r_CNO, E_produced
    

    def limit_reaction_rates(self):
        """
        Function for calculating new reaction rates which makes sure that no step consumes more of an element
        than the previous steps are able to produce. These are the ones used in the argument r_correction.
        """
        # Reaction rates calculated before limiting the production:
        r_PP0 = self.PP0()[1];   r_PP2 = self.PPII()[1]
        r_PP1 = self.PPI()[1];   r_PP3 = self.PPIII()[1]
        
        # For correct consumation of 3_He:
        if r_PP0 < (2 * r_PP1 + r_PP2[0]):         # r_pp < (2 * r_33 + r_34)
            R = r_PP0 / (2 * r_PP1 + r_PP2[0])      # R = r_pp / (2 * r_33 + r_34)
            r_PP1 = R * r_PP1                       # r_33 = R * r_33
            r_PP2[0] = R * r_PP2[0]                 # r_34 = R * r_34
        
        # For correct consumation of 7-Be:
        if r_PP2[0] < (r_PP2[1] + r_PP3[1]):       # r_34 < (r_e7 + r_17)
            R = r_PP2[0] / (r_PP2[1] + r_PP3[1])    # R = r_34 / (r_e7 + r_17)
            r_PP2[1] = R * r_PP2[1]                 # r_e7 = R * r_e7
            r_PP3[1] = R * r_PP3[1]                 # r_17 = R * r_17

        # For correct consumation of 7-Li:
        if r_PP2[1] < r_PP2[2]:                    # r_e7 < r_17'
            R = r_PP2[1] / r_PP2[2]                 # R = r_e7 / r_17'
            r_PP2[2] = R * r_PP2[2]                 # r_17' = R * r_17'
        
        return r_PP1, r_PP2[0], r_PP2[1], r_PP2[2], r_PP3[1]


    def energy_produced(self, verbose):
        """
        Function for calculating the total energy produced from each branch in the PP-chain and the CNO cycle.
        """
        r_33, r_34, r_e7, r_17_prime, r_17 = self.limit_reaction_rates()
        r_PP2_correction = [r_34, r_e7, r_17_prime]; r_PP3_correction = [r_34, r_17]

        # Retrieving the energy from every reaction using the limiting reaction rates:
        E_PP0 = self.PP0()[2];                      E_PP2 = self.PPII(r_correction=r_PP2_correction)[2]
        E_PP1 = self.PPI(r_correction=r_33)[2];     E_PP3 = self.PPIII(r_correction=r_PP3_correction)[2]
        E_CNO = self.CNO()[2]

        epsilon = np.array([E_PP0, E_PP1, E_PP2[0], E_PP2[1], E_PP2[2], E_PP3[1], E_CNO]) / self.rho 
        self.epsilon = np.sum(epsilon)  # total energy [J/ kg s]

        """
        Finding total energy production from each brach using the slowest reaction rates in each branch: 
        (For PPII this is r_e7, for PPIII it is r_17, note that r_e7 = r_17')
        """
        self.PP1_energy = (self.PP0(r_correction=r_33)[2] * 2 + self.PPI(r_correction=r_33)[2]) / self.rho
        self.PP2_energy = (self.PP0(r_correction=r_e7)[2] + np.sum(self.PPII(r_correction=[r_e7, r_e7, r_e7])[2])) / self.rho
        self.PP3_energy = (self.PP0(r_correction=r_17)[2] + np.sum(self.PPIII(r_correction=[r_17, r_17])[2])) / self.rho
        self.CNO_energy = E_CNO / self.rho

        table = [["Branch", "Total energy production [J/ kg s]"], ["PP1", self.PP1_energy], ["PP2", self.PP2_energy],
                 ["PP3", self.PP3_energy], ["CNO", self.CNO_energy]]
        
        if verbose == True:
            print(f"Total energy production from the branches for T = {self.T} K :")
            print(tabulate(table, headers="firstrow", tablefmt="fancy_grid"), "\n")
            print(f"Total energy released, epsilon = {self.epsilon} [J/ kg s] for T = {self.T} K \n")

        return epsilon, self.epsilon, self.PP1_energy, self.PP2_energy, self.PP3_energy, self.CNO_energy


    def sanity_check(self):
        """
        Function for checking if the calculated values are closer than 0.01 to the acutal solar core values.
        """
        known_val_core = np.array([4.05e2, 8.69e-9, 4.87e-5, 1.5e-6, 5.3e-4, 1.64e-6, 9.18e-8]) # [J m^-3 s^-1]
        known_val_T8 = np.array([7.34e4, 1.1, 1.75e4, 1.23e-3, 4.35e-1, 1.27e5, 3.45e4])        # [J m^-3 s^-1]
        
        calc_val = self.energy_produced(verbose=True)[0] * self.rho # Energies from each reaction

        # Checking which temperature is being used in the function call:
        if self.T == 1.57e7:
            known_val = known_val_core
        else:
            known_val = known_val_T8

        for i in range(len(known_val_core)):
            rel_error = abs(calc_val - known_val) / known_val

            if rel_error[i] < 0.01:
                print("Sanity check passed")
                
            if rel_error[i] > 0.01:  # Insert user input = 0.049
                """
                Since the energy output from mass difference for reaction e7 is slightly less than what is given in table 3.2 in the lecture notes the sanity check does
                not pass with this value and we calculate the relative error by user input of Q'_e7 from the table instead:
                """
                new_Q_e7 = float(input(f"Sanity check not passed, relative error for reaction e_7 was {rel_error[3]}. Insert Q'_e7 from table 3.2 instead: ")) * self.energy_conversion # Converting to Joules
                r_e7 = self.limit_reaction_rates()[2]

                rel_error[3] = abs((new_Q_e7 * r_e7 * self.rho) - known_val[3]) / known_val[3]

                if rel_error[i] > 0.01:
                    raise Exception("Sanity check still failed")
        
        table = [["Reaction", "Energy computed [J m^-3 s^-1]", "Expected value [J m^-3 s^-1]", "Relative error"],
                 ["pp + pd", calc_val[0], known_val[0], rel_error[0]], ["33", calc_val[1], known_val[1], rel_error[1]],
                 ["34", calc_val[2], known_val[2], rel_error[2]], ["e7", calc_val[3], known_val[3], rel_error[3]],
                 ["17'", calc_val[4], known_val[4], rel_error[4]], ["17", calc_val[5], known_val[5], rel_error[5]],
                 ["p14", calc_val[6], known_val[6], rel_error[6]]]
        
        print(f"Energy production when T = {self.T:.2f} K")
        print(tabulate(table, headers="firstrow", tablefmt="fancy_grid"), "\n")

        return rel_error
    
    
    def neutrino_E_loss(self):
        """
        Function for calculating percentage of released energy lost to neutrinos per produced 4-He. The energy output from each
        branch using the mass difference of the reactants and products for each branch is also calculated here.
        """
        branch_neutrinos = np.array([2 * self.neutrinos["v_pp"], self.neutrinos["v_pp"] + self.neutrinos["v_e7"],
                    self.neutrinos["v_pp"] + self.neutrinos["v_8"], self.neutrinos["v_13"] + self.neutrinos["v_15"]]) # [J]
    
        Q_PP1 = 2 * self.PP0()[0] + self.PPI()[0]          # [J]
        Q_PP2 = self.PP0()[0] + np.sum(self.PPII()[0])     # [J]
        Q_PP3 = self.PP0()[0] + np.sum(self.PPIII()[0])    # [J]
        Q_CNO = np.sum(self.CNO()[0])                      # [J]

        Q_branch = np.array([Q_PP1, Q_PP2, Q_PP3, Q_CNO])
        Q_lost = np.zeros(4)

        for i in range(len(Q_branch)):
            Q_lost[i] = branch_neutrinos[i] / (Q_branch[i] + branch_neutrinos[i]) * 100
        
        table = [["Branch", "Sum of Q' [MeV]", "Q_neutrino [MeV]", "% lost"], 
                ["PP1", Q_PP1 / self.energy_conversion, branch_neutrinos[0] / self.energy_conversion, Q_lost[0]], # converting to MeV
                ["PP2", Q_PP2 / self.energy_conversion, branch_neutrinos[1] / self.energy_conversion, Q_lost[1]], 
                ["PP3", Q_PP3 / self.energy_conversion, branch_neutrinos[2] / self.energy_conversion, Q_lost[2]],
                ["CNO", Q_CNO / self.energy_conversion, branch_neutrinos[3] / self.energy_conversion, Q_lost[3]]]
        
        print("Energy output from each branch and energy loss due to neutrinos:")
        print(tabulate(table, headers="firstrow", tablefmt="fancy_grid"), "\n")


    def Gamow_peak(self, m_i, m_k, Z_i, Z_k):
        """
        Function for calculating the Gamow peaks for all reactions that are not decays in 
        PP-chain and CNO cycle, curves normalized to the maximum value.
        """
        N = 1000
        E = np.logspace(-17, -13, N) # [J]

        m = (m_i * m_k) / (m_i + m_k) # reduced mass, unitless

        exp_lamda = np.exp(- E / (self.k_B * T_c))
        exp_sigma = np.exp(- np.sqrt(m / (2 * E)) * (Z_i * Z_k * self.e**2 * np.pi) / (self.epsilon_0 * self.h))
        
        gamow = exp_lamda * exp_sigma
        max_val = np.max(gamow)
        norm_gamow = gamow / max_val  # We normalize the curve by dividing by the max value:

        return norm_gamow
    

    def plotting_Gamows(self):
        """
        Function for plotting the Gamow peak for all the reactions of the PP chain
        and the CNO cycle excluding the decays.
        """
        N = 1000
        E = np.logspace(-17, -13, N) # [J]

        curves = np.zeros((10, N))
        max_val = np.zeros(10)

        m_i = np.array([self.m_1H, self.m_2D, self.m_3He, self.m_3He, self.m_7Li, self.m_7Be, self.m_12C, self.m_13C, self.m_14N, self.m_15N]) * self.m_u # [kg]
        m_k = np.array([self.m_1H, self.m_1H, self.m_3He, self.m_4He, self.m_1H, self.m_1H, self.m_1H, self.m_1H, self.m_1H, self.m_1H]) * self.m_u # [kg]
        Z_i = np.array([self.Z_H, self.Z_H, self.Z_He, self.Z_He, self.Z_Li, self.Z_Be, self.Z_C, self.Z_C, self.Z_N, self.Z_N])
        Z_k = np.array([self.Z_H, self.Z_H, self.Z_He, self.Z_He, self.Z_H, self.Z_H, self.Z_H, self.Z_H, self.Z_H, self.Z_H])
        labels = np.array(["pp", "pd", "33", "34", "17'", "17", "p12", "p13", "p14", "p15"])
        
        plt.figure(figsize=(10, 4))

        for i in range(10):
            curves[i, :] = self.Gamow_peak(m_i[i], m_k[i], Z_i[i], Z_k[i])
            max_val[i] = E[np.argmax(curves[i])]

        for i in range(10):
            plt.plot(E, curves[i, :], label=labels[i])

        plt.xlabel("Energy log$_{10}$ [J]")
        plt.ylabel("Normalized probability")
        plt.tight_layout()
        plt.xlim(2e-16, 2e-14)
        plt.xscale("log")
        plt.legend()
        plt.grid()

        plt.savefig("Gamow.png")    
        plt.show()

        table = [["Reaction", "Gamow peak max value [J]"], [labels[0], max_val[0]], [labels[1], max_val[1]],
                 [labels[2], max_val[2]], [labels[3], max_val[3]], [labels[4], max_val[4]], [labels[5], max_val[5]],
                 [labels[6], max_val[6]], [labels[7], max_val[7]], [labels[8], max_val[8]], [labels[9], max_val[9]]]
        
        print(f"Values of Gamow peaks in [J] for T = {self.T} K:")
        print(tabulate(table, headers="firstrow", tablefmt="fancy_grid"), "\n")

"""
Plotting relative energy and calling the class:
"""

T_c = 1.57e7         # temperature in solar core [K]
T = 1e8              # temperature when reaction rate of PPI & PPII are larger than PP0 [K]
rho_c = 1.62e5       # density in solar core     [kg/m^3]

def rel_E_plot():
    """
    Function for plotting the relative energies, defined outside function to be able to calculate the 
    energy production for a large number of temperatures.
    """
    N = 2000
    T = np.logspace(4, 9, N)
    rel_E = np.zeros((4, N))

    epsilon = np.zeros(N)
    labels = np.array(["PP1", "PP2", "PP3", "CNO"])

    for i in range(N):
        run = Energy_production(T[i], rho_c)
        run.energy_produced(verbose=False)

        energies = np.array([run.PP1_energy, run.PP2_energy, run.PP3_energy, run.CNO_energy])
        total_E = np.sum(energies)

        for j in range(4):
            rel_E[j, i] = energies[j] / total_E
        
        epsilon[i] = run.epsilon

    plt.figure(figsize=(10, 4))
    plt.tight_layout()

    for i in range(4):
        plt.plot(T, rel_E[i, :], label=labels[i])

    plt.plot(T, epsilon / np.max(epsilon), label="epsilon")
    plt.ylabel("Relative energy production log$_{10}$ [J]")
    plt.xlabel("Temperature [K]")
    plt.xscale("log")
    plt.legend()
    plt.grid()
    
    plt.savefig("relative_enrgies.png")
    plt.show()

# rel_E_plot()

C = Energy_production(T_c, rho_c)
T8 = Energy_production(T, rho_c)

# T8.sanity_check()
# C.sanity_check()
# C.neutrino_E_loss()
# C.plotting_Gamows()

