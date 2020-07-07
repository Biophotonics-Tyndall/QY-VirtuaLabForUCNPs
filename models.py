import numpy as np
from sympy import init_session, solve, lambdify, Symbol, symbols, Eq
from sympy.physics.mechanics import dynamicsymbols
from tqdm import tqdm


constants = {
    "h": {
        "about": "plank constant",
        "value": 6.626e-34,
        "unit": "m^2 Kg/s = J.s"
    },
    "nu": {
        "about": "frequency of excitation light (975nm)",
        "value": 307692307692307.7,
        "unit": "1/s"
    },
     "nu2": {
        "about": "frequency of excitation light (800nm)",
        "value": 375000000000000.0,
        "unit": "1/s"
    }
}

# Yb Levels: Na, Nb
    # 2F7/2 2F5/2
Na, Nb, Nd = dynamicsymbols('N_a N_b N_d')
N = Symbol('N')
Nd0 = Symbol('N_d0')
# Tm Levels: N0, N1, N2, N3, N4
#           3H6, 3F4, 3H5, 3H4, 3F3 
N0, N1, N2, N3, N4 = dynamicsymbols('N_0 N_1 N_2 N_3 N_4')
t = Symbol('t')
# Constants
    # h -> plank constant
    # nu -> Excitation wavelength
    # Rij -> Decay rates
    # sigma_ij -> cross-section for absorption or emission
    # Gamma_ij -> Decay rate from i to j
    # W_etu -> ETU rate by Foster resonant energy trasfer 
    # W_cr -> reverse process cross-relaxation
h, nu, I, We, Wc, rho, rhob, phib, etha, kappa, phi= symbols('h nu I W_e W_c rho rho_b phi_b etha kappa phi')
sigmaab, sigma02, sigma10, sigma13 = symbols('sigma_ab sigma_01, sigma_10, sigma_13')
W0, W1, W2, W3, Wd = symbols('W_0 W_1 W_2 W_3 W_d')
Rba, RbaTot, R10, R20 = symbols('R_ba R_baT R_10 R_20')

# General equations
RbaTot = Rba + W0 * N0 + W1 * N1 + Wd * Nd
NaRate = Eq(Na.diff(t), (RbaTot) * Nb - sigmaab * rho / (h * nu) * Na)
NbRate = Eq(Nb.diff(t), - NaRate.rhs)
# Nd number of energy distributors available
NdRate = Eq(Nd.diff(t), Nd0 - Wd * Nd * Nb)

N0Rate = Eq(N0.diff(t), R20 * N2 - W0 * Nb * N0 + R10 * N1)
N1Rate = Eq(N1.diff(t), W0 * Nb * N0 - W1 * Nb * N1 - R10 * N1)
N2Rate = Eq(N2.diff(t), W1 * Nb * N1 - R20 * N2)




# def Models(sharedSample, model):
#     # self._sharedSample = sharedSample
#     self.model = model
#     if model == 'std':
#         sharedSample.model = StdModel(sharedSample)
#     elif model == 'dynLT':
#         sharedSample.model = DynLifeTime(sharedSample)


class StdModel():

    def __init__(self, sharedSample):
        self._sharedSample = sharedSample
        self.name = 'std'
        props = self._sharedSample.props
        # Equations
        null_vars = {'W_0': props['w0'], 'W_1': props['w1'], 'W_2': props['w2'], 'W_d': props['wd']}
        null_vars = {k: v for k, v in null_vars.items() if v == 0}

        RbaTot = Rba
        NaRate = Eq(Na.diff(t), (RbaTot) * Nb - sigmaab * rho / (h * nu) * Na).subs(null_vars)
        self.NaRate = Eq(Na.diff(t), (RbaTot) * Nb - sigmaab * rho / (h * nu) * Na).subs(null_vars)
        self.NbRate = Eq(Nb.diff(t), - NaRate.rhs).subs(null_vars)

        self.N0Rate = N0Rate.subs(null_vars)
        self.N1Rate = N1Rate.subs(null_vars)
        self.N2Rate = N2Rate.subs(null_vars)

    def rules(self):
        display(
            self.NaRate, self.NbRate,
            self.N0Rate, self.N1Rate, self.N2Rate
        )

    def builduplevels(self, n):
        "Build up the energy levels"

        for key in self._sharedSample.absorber.init_conds.keys():
            setattr(self, key, np.zeros(n, dtype=np.float64))
        for key in self._sharedSample.emitter.init_conds.keys():
            setattr(self, key, np.zeros(n, dtype=np.float64))

        # Initial conds
        self.Na[0] = self._sharedSample.absorber.init_conds['Na']['value']; self.Nb[0] = self._sharedSample.absorber.init_conds['Nb']['value']
        self.Nd[0] = self._sharedSample.absorber.init_conds['Nd']['value']

        self.N0[0] = self._sharedSample.emitter.init_conds['N0']['value']; self.N1[0] = self._sharedSample.emitter.init_conds['N1']['value']
        self.N2[0] = self._sharedSample.emitter.init_conds['N2']['value']
             

    def populatelevels(self, laserpower):

        n = len(laserpower)
        self.builduplevels(n)
        self.RbaTot = np.zeros(n, dtype=np.float64) 
        self.RbaTot[:] = 1 / self._sharedSample.props['taub']  

        p = laserpower.power * self._sharedSample.props['s'] / (constants['h']['value'] * constants['nu']['value'])
        t = laserpower.t
        N = self.Na[0] + self.Nb[0]

        for k in tqdm(range(n - 1), desc='Shining light 🚨'):
            dt = t[k+1] - t[k]
            self.Nb[k+1] = dt * (p[k+1] * self.Na[k] - self.RbaTot[k] * self.Nb[k]) + self.Nb[k]
            self.Na[k+1] = N - self.Nb[k+1]
        
            self.N1[k+1] = dt * (- 1/self._sharedSample.props['tau1'] * self.N1[k] + self._sharedSample.props['w0'] * self.N0[k] * self.Nb[k+1] - self._sharedSample.props['w1'] * self.N1[k] * self.Nb[k+1]) + self.N1[k]
            self.N0[k+1] = self.N0[k] + dt * (1/self._sharedSample.props['tau1'] * self.N1[k] - self._sharedSample.props['w0'] * self.N0[k] * self.Nb[k+1] + 1/self._sharedSample.props['tau2'] * self.N2[k])
            self.N2[k+1] = self.N2[k] + dt* (self._sharedSample.props['w1'] * self.N1[k] * self.Nb[k+1] - 1/self._sharedSample.props['tau2'] * self.N2[k])


class DynLifeTime():

    def __init__(self, sharedSample):
        self._sharedSample = sharedSample
        self.name = 'dynLT'
        props = self._sharedSample.props
        # Equations
        null_vars = {'W_0': props['w0'], 'W_1': props['w1'], 'W_2': props['w2'], 'W_d': props['wd']}
        null_vars = {k: v for k, v in null_vars.items() if v == 0}

        self.NaRate = NaRate.subs(null_vars)
        self.NbRate = NbRate.subs(null_vars)

        self.N0Rate = N0Rate.subs(null_vars)
        self.N1Rate = N1Rate.subs(null_vars)
        self.N2Rate = N2Rate.subs(null_vars)

    def rules(self):
        display(
            self.NaRate, self.NbRate,
            self.N0Rate, self.N1Rate, self.N2Rate
        )

    def builduplevels(self, n):
        "Build up the energy levels"

        for key in self._sharedSample.absorber.init_conds.keys():
            setattr(self, key, np.zeros(n, dtype=np.float64))
        for key in self._sharedSample.emitter.init_conds.keys():
            setattr(self, key, np.zeros(n, dtype=np.float64))

        # Initial conds
        self.Na[0] = self._sharedSample.absorber.init_conds['Na']['value']; self.Nb[0] = self._sharedSample.absorber.init_conds['Nb']['value']
        self.Nd[:] = self._sharedSample.absorber.init_conds['Nd']['value']

        self.N0[0] = self._sharedSample.emitter.init_conds['N0']['value']; self.N1[0] = self._sharedSample.emitter.init_conds['N1']['value']
        self.N2[0] = self._sharedSample.emitter.init_conds['N2']['value']
       

    def populatelevels(self, laserpower):

        n = len(laserpower)
        self.builduplevels(n)

        self.RbaTot = np.zeros(n, dtype=np.float64) 
        self.RbaTot[0] = 1 / self._sharedSample.props['taub'] + self._sharedSample.props['w0'] * self.N0[0] + self._sharedSample.props['w1'] * self.N1[0] + self._sharedSample.props['wd'] * self.Nd[0]

        p = laserpower.power * self._sharedSample.props['s'] / (constants['h']['value'] * constants['nu']['value'])
        t = laserpower.t
        N = self.Na[0] + self.Nb[0]

        for k in tqdm(range(n - 1), desc='Shining light 🚨'):
            dt = t[k+1] - t[k]
            self.Nb[k+1] = dt * (p[k+1] * self.Na[k] - self.RbaTot[k] * self.Nb[k]) + self.Nb[k]
            self.Na[k+1] = N - self.Nb[k+1]
        
            self.N1[k+1] = dt * (- 1/self._sharedSample.props['tau1'] * self.N1[k] + self._sharedSample.props['w0'] * self.N0[k] * self.Nb[k+1] - self._sharedSample.props['w1'] * self.N1[k] * self.Nb[k+1]) + self.N1[k]
            self.N0[k+1] = self.N0[k] + dt * (1/self._sharedSample.props['tau1'] * self.N1[k] - self._sharedSample.props['w0'] * self.N0[k] * self.Nb[k+1] + 1/self._sharedSample.props['tau2'] * self.N2[k])
            self.N2[k+1] = self.N2[k] + dt* (self._sharedSample.props['w1'] * self.N1[k] * self.Nb[k+1] - 1/self._sharedSample.props['tau2'] * self.N2[k])   
            
            self.RbaTot[k+1] = 1/self._sharedSample.props['taub'] + self._sharedSample.props['w0'] * self.N0[k+1] + self._sharedSample.props['w1'] * self.N1[k+1] + self._sharedSample.props['wd'] * self.Nd[k+1]


# class DynLTDeffects():

#     def __init__(self, sharedSample):
#         self._sharedSample = sharedSample
#         self.name = 'dynLT'
#         props = self._sharedSample.props
#         # Equations
#         null_vars = {'W_0': props['w0'], 'W_1': props['w1'], 'W_2': props['w2'], 'W_d': props['wd']}
#         null_vars = {k: v for k, v in null_vars.items() if v == 0}

#         self.NaRate = NaRate.subs(null_vars)
#         self.NbRate = NbRate.subs(null_vars)
#         # self.NdRate = NdRate.subs(null_vars)

#         self.N0Rate = N0Rate.subs(null_vars)
#         self.N1Rate = N1Rate.subs(null_vars)
#         self.N2Rate = N2Rate.subs(null_vars)

#     def rules(self):
#         display(
#             self.NaRate, self.NbRate, self.NdRate,
#             self.N0Rate, self.N1Rate, self.N2Rate
#         )

#     def builduplevels(self, n):
#         "Build up the energy levels"

#         for key in self._sharedSample.absorber.init_conds.keys():
#             setattr(self, key, np.zeros(n, dtype=np.float64))
#         for key in self._sharedSample.emitter.init_conds.keys():
#             setattr(self, key, np.zeros(n, dtype=np.float64))

#         # Initial conds
#         self.Na[0] = self._sharedSample.absorber.init_conds['Na']['value']; self.Nb[0] = self._sharedSample.absorber.init_conds['Nb']['value']
#         self.Nd[0] = self._sharedSample.absorber.init_conds['Nd']['value']

#         self.N0[0] = self._sharedSample.emitter.init_conds['N0']['value']; self.N1[0] = self._sharedSample.emitter.init_conds['N1']['value']
#         self.N2[0] = self._sharedSample.emitter.init_conds['N2']['value']
       

#     def evolve(self, laserpower):

#         n = len(laserpower)
#         self.builduplevels(n)

#         self.RbaTot = np.zeros(n, dtype=np.float64) 
#         self.RbaTot[0] = 1 / self._sharedSample.props['taub'] + self._sharedSample.props['w0'] * self.N0[0] + self._sharedSample.props['w1'] * self.N1[0] + self._sharedSample.props['wd'] * self.Nd[0]

#         p = laserpower.power * self._sharedSample.props['s'] / (constants['h']['value'] * constants['nu']['value'])
#         t = laserpower.t
#         N = self.Na[0] + self.Nb[0]

#         for k in tqdm(range(n - 1), desc='Shining light 🚨'):
#             dt = t[k+1] - t[k]
#             self.Nb[k+1] = dt * (p[k+1] * self.Na[k] - self.RbaTot[k] * self.Nb[k]) + self.Nb[k]
#             self.Na[k+1] = N - self.Nb[k+1]
        
#             self.N1[k+1] = dt * (- 1/self._sharedSample.props['tau1'] * self.N1[k] + self._sharedSample.props['w0'] * self.N0[k] * self.Nb[k+1] - self._sharedSample.props['w1'] * self.N1[k] * self.Nb[k+1]) + self.N1[k]
#             self.N0[k+1] = self.N0[k] + dt * (1/self._sharedSample.props['tau1'] * self.N1[k] - self._sharedSample.props['w0'] * self.N0[k] * self.Nb[k+1] + 1/self._sharedSample.props['tau2'] * self.N2[k])
#             self.N2[k+1] = self.N2[k] + dt* (self._sharedSample.props['w1'] * self.N1[k] * self.Nb[k+1] - 1/self._sharedSample.props['tau2'] * self.N2[k])   
            
#             self.RbaTot[k+1] = 1/self._sharedSample.props['taub'] + self._sharedSample.props['w0'] * self.N0[k+1] + self._sharedSample.props['w1'] * self.N1[k+1] + self._sharedSample.props['wd'] * self.Nd[k+1]
