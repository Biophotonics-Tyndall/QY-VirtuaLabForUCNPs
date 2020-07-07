import numpy as np
from sympy import init_session, solve, lambdify, Symbol, symbols, Eq, log, Derivative
from sympy.physics.mechanics import dynamicsymbols
# from scipy.optimize import curve_fit
import matplotlib.pyplot as pl
import pandas as pd
from time import sleep
# from tqdm import tqdm
import json
from tinydb import TinyDB, Query
from models import *
# init_session()

with open('inputtemplate.json', 'r') as json_file:
    inputtemplate = json.load(json_file)

def constantdetails(description, unit, axislabel, marker='o', value=None):
    details = {'about': description, 'unit': unit, 'axislabel': axislabel, 'marker': marker}
    if value:
        details['value'] = value
    
    return(details)


class ClassPyIt:
    '''Make an instanced object from dictionary
    dObj = ClassPyIt({'a': 1}
    dObj.a
    1
    '''
    def __init__(self, dicty):
        for k, v in dicty.items():
            setattr(self, str(k).strip(), v)


class Laser():
    _powerSweep = pd.DataFrame({'t': np.array([]), 'power': np.array([])}, )
    
    def __init__(self, laser_setup):
        for k, v in laser_setup.items():
            setattr(self, k, v)
    
        self.number_of_points_per_step = int(self.dt_per_step / self.dt_per_data_point)

    def rampdetails(self):
        print(f"""
            Power range: {self.range} w/cm^2
            in {self.mode} mode
            {self.number_of_power_points} power steps
            {self.number_of_points_per_step} points per step acquired every
            {self.dt_per_data_point} s
        """)

    def run(self):
        ''' Creates a power ramping.
            Power in W/cm^2
            timeInterval = total time per power step
            pointsPerInterval = number of data points per timeInterval
            mode: log or linear
        '''
        # total number of data points
        nt = self.number_of_points_per_step * self.number_of_power_points

        dt = self.dt_per_data_point
        self._powerSweep.t = np.linspace(0, (nt - 1) * dt, nt) + self.t0
        
        if self.mode == 'linear':
            self._powerSweep.power = np.sort(self.number_of_points_per_step * list(np.linspace(*self.range, self.number_of_power_points)))
        elif self.mode == 'log':
            self._powerSweep.power = np.sort(self.number_of_points_per_step * list(np.logspace(np.log10(self.range[0]), np.log10(self.range[1]), self.number_of_power_points)))
        else:
            raise ValueError('Invalid mode! Select either linear or log mode')

        self._powerSweep.power = self._powerSweep.power[::-1] if self.range[0] > self.range[1] else self._powerSweep.power
        
        print(f'{len(self._powerSweep.power)} data points')
        print(f'{dt=} s')


    def power(self):
        return(
            self._powerSweep
        )


    def profile(self):
        """Plot ramping profile"""

        fig, ax100 = pl.subplots()
        ax100.plot(self._powerSweep.t, self._powerSweep.power, 'red')
        ax100.legend(['Step power'])
        ax100.set_xlabel(r'$t (s)$')
        ax100.set_ylabel(r'$\rho (W/cm^2)$')
        pl.show()


class Sample():

    def __init__(self, sample_params):
       
        absorber = sample_params.get('absorber').copy()
        emitter = sample_params.get('emitter').copy()

        self.absorber = ClassPyIt(absorber)
        self.emitter = ClassPyIt(emitter)

         # General props
        self.props = {
            's': self.absorber.props['sigma']['value'],
            'wd': self.absorber.props['Wd']['value'],
            'taub': self.absorber.props['taub']['value'],
            'w0': self.emitter.props['W0']['value'],
            'w1': self.emitter.props['W1']['value'],
            'w2': self.emitter.props['W2']['value'],
            'tau1': self.emitter.props['tau1']['value'],
            'tau2': self.emitter.props['tau2']['value']
        }
        if sample_params['model'] == 'std':
            self.model = StdModel(self)
        elif sample_params['model'] == 'dynLT':
            self.model = DynLifeTime(self)

            

class SampleOld():

    def __init__(self, sample_init_conds, model='dynLT'):
        self.model = model
        self.model = Models(self, model)
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
        h, nu, I, We, Wc, rho, rhob, phib, etha = symbols('h nu I W_e W_c rho rho_b phi_b etha')
        sigmaab, sigma02, sigma10, sigma13 = symbols('sigma_ab sigma_01, sigma_10, sigma_13')
        W0, W1, W2, W3, Wd = symbols('W_0 W_1 W_2 W_3 W_d')
        Rba, RbaTot, R10, R20 = symbols('R_ba R_baT R_10 R_20')
        
        absorber = sample_init_conds.get('absorber').copy()
        emitter = sample_init_conds.get('emitter').copy()

        self.absorber = ClassPyIt(absorber)
        self.emitter = ClassPyIt(emitter)

         # General props
        self.props = {
            's': self.absorber.props['sigma']['value'],
            'wd': self.absorber.props['Wd']['value'],
            'taub': self.absorber.props['taub']['value'],
            'w0': self.emitter.props['W0']['value'],
            'w1': self.emitter.props['W1']['value'],
            'w2': self.emitter.props['W2']['value'],
            'tau1': self.emitter.props['tau1']['value'],
            'tau2': self.emitter.props['tau2']['value']
        }

        # Simplify eqs
        null_vars = {'W_0': self.props['w0'], 'W_1': self.props['w1'], 'W_2': self.props['w2'], 'W_d': self.props['wd']}
        null_vars = {k: v for k, v in null_vars.items() if v == 0}
        
        # select model
        if self.model == 'std':
            RbaTot = Rba
        elif self.model == 'dynLT':
            RbaTot = Rba + W0 * N0 + W1 * N1 + Wd * Nd

        self.absorber.NaRate = Eq(Na.diff(t), (RbaTot) * Nb - sigmaab * rho / (h * nu) * Na).subs(null_vars)
        self.absorber.NbRate = Eq(Nb.diff(t), - self.absorber.NaRate.rhs).subs(null_vars)
        # Nd number of energy distributors available
        self.absorber.NdRate = Eq(Nd.diff(t), Nd0 - Wd * Nd * Nb).subs(null_vars)

        self.emitter.N0Rate = Eq(N0.diff(t), R20 * N2 - W0 * Nb * N0 + R10 * N1).subs(null_vars)
        self.emitter.N1Rate = Eq(N1.diff(t), W0 * Nb * N0 - W1 * Nb * N1 - R10 * N1).subs(null_vars)
        self.emitter.N2Rate = Eq(N2.diff(t), W1 * Nb * N1 - R20 * N2).subs(null_vars)
        
    def rules(self):
        display(
            self.absorber.NaRate, self.absorber.NbRate, self.absorber.NdRate,
            self.emitter.N0Rate, self.emitter.N1Rate, self.emitter.N2Rate
        )
    
    def builduplevels(self, n):
        "Build up the energy levels"

        for key in self.absorber.init_conds.keys():
            setattr(self.absorber, key, np.zeros(n, dtype=np.float64))
        for key in self.emitter.init_conds.keys():
            setattr(self.emitter, key, np.zeros(n, dtype=np.float64))

        self.absorber.RbaTot = np.zeros(n, dtype=np.float64)

        # Initial conds
        self.absorber.Na[0] = self.absorber.init_conds['Na']['value']; self.absorber.Nb[0] = self.absorber.init_conds['Nb']['value']
        self.absorber.Nd[0] = self.absorber.init_conds['Nd']['value']

        self.emitter.N0[0] = self.emitter.init_conds['N0']['value']; self.emitter.N1[0] = self.emitter.init_conds['N1']['value']
        self.emitter.N2[0] = self.emitter.init_conds['N2']['value']
        if self.model == 'dynLT':
            self.absorber.RbaTot[0] = 1 / self.props['taub'] + self.props['w0'] * self.emitter.N0[0] + self.props['w1'] * self.emitter.N1[0] + self.props['wd'] * self.absorber.Nd[0]
        elif self.model == 'std':
            self.absorber.RbaTot[:] = 1 / self.props['taub']
            

    def populatelevels(self, laserpower):

        n = len(laserpower)
        self.builduplevels(n)
        
        p = laserpower.power * self.absorber.props['sigma']['value'] / (constants['h']['value'] * constants['nu']['value'])
        t = laserpower.t
        N = self.absorber.Na[0] + self.absorber.Nb[0]

        for k in tqdm(range(n - 1), desc='Shining light 🚨'):
            dt = t[k+1] - t[k]
            self.absorber.Nb[k+1] = dt * (p[k+1] * self.absorber.Na[k] - self.absorber.RbaTot[k] * self.absorber.Nb[k]) + self.absorber.Nb[k]
            self.absorber.Na[k+1] = N - self.absorber.Nb[k+1]
        
            self.emitter.N1[k+1] = dt * (- 1/self.props['tau1'] * self.emitter.N1[k] + self.props['w0'] * self.emitter.N0[k] * self.absorber.Nb[k+1] - self.props['w1'] * self.emitter.N1[k] * self.absorber.Nb[k+1]) + self.emitter.N1[k]
            self.emitter.N0[k+1] = self.emitter.N0[k] + dt * (1/self.props['tau1'] * self.emitter.N1[k] - self.props['w0'] * self.emitter.N0[k] * self.absorber.Nb[k+1] + 1/self.props['tau2'] * self.emitter.N2[k])
            self.emitter.N2[k+1] = self.emitter.N2[k] + dt* (self.props['w1'] * self.emitter.N1[k] * self.absorber.Nb[k+1] - 1/self.props['tau2'] * self.emitter.N2[k])
            
            if self.model == 'dynLT':
                self.absorber.RbaTot[k+1] = 1/self.props['taub'] + self.props['w0'] * self.emitter.N0[k+1] + self.props['w1'] * self.emitter.N1[k+1] + self.props['wd'] * self.absorber.Nd[k+1]
            elif self.model == 'std':
                pass


class Experiment():
    _metadata = {}
    _data = pd.DataFrame()
    _steadydata = pd.DataFrame()

    def __init__(self, label, laser_setup, sample):
        self.label = label
        self.laser = Laser(laser_setup)
        self.sample = Sample(sample)

        self._metadata = {
            "comment": "",
            "label": label,
            "laser_setup": laser_setup,
            "sample": sample
        }

    def run(self):
        self.laser.run()
        sleep(0.5)
        self.sample.model.populatelevels(self.laser.power())

    def transformdata(self, columns='all'):
        """columns = [Na, Nb, Nd, N0, N1, N2, RbaTot]"""

        cols = ['Na', 'Nb', 'Nd', 'N0', 'N1', 'N2', 'RbaTot']
        columns = cols if columns == 'all' else columns

        self._data = pd.concat([
            self.laser.power(), 
            pd.DataFrame(np.array([getattr(self.sample.model, col) for col in columns]).T, columns=columns)
        ], axis=1)
        
        self.getsteadydata()
        self._metadata['dataColumns'] = {}
        self._metadata['dataColumns']['t'] = constantdetails('time', 's', "$t$", 'o')
        self._metadata['dataColumns']['power'] = constantdetails('power density in W/cm^2', 'W/cm^2', "$\\rho$", 'o')
        self._metadata['dataColumns']['Na'] = constantdetails('Population of Yb ground state', 'cm^{-3}', "$N$", 'p')
        self._metadata['dataColumns']['Nb'] = constantdetails('Population of Yb excited state', 'cm^{-3}', '$N$', 'h')
        self._metadata['dataColumns']['Nd'] = constantdetails('Population of energy distributors', 'cm^{-3}', '$N$', '*')
        self._metadata['dataColumns']['N0'] = constantdetails('Population of Tm ground state', 'cm^{-3}', '$N$', 'v')
        self._metadata['dataColumns']['N1'] = constantdetails('Population of Tm 1st excited state', 'cm^{-3}', '$N$', '<')
        self._metadata['dataColumns']['N2'] = constantdetails('Population of Tm 2nd excited state', 'cm^{-3}', '$N$', '>')
        self._metadata['dataColumns']['RbaTot'] = constantdetails('Relative total decay rate b -> a', '1/s', '$R_{b, tot}$')

        # clean memory
        for col in (set(columns) - set(cols)):
            delattr(self.sample, col)
    
    def getsteadydata(self):
        n = int(self._metadata['laser_setup']['dt_per_step'] / self._metadata['laser_setup']['dt_per_data_point'])
        self._steadydata = self._data[n - 1::n].copy()
        

    def data(self, steady=False):
        return(self._steadydata if steady else self._data)

    def save(self, expname, exp_details):
        
        outputFile = f'output/{expname}.csv'
        outputMetadata = f'output/{expname}.json'

        self._data.to_csv(outputFile)

        self._metadata['comments'] = exp_details
        self._metadata['dataset_shape'] = self._data.shape
        self._metadata['dataFile'] = f'{expname}.csv'

        # db.insert(self._metadata)

        with open(outputMetadata, 'w') as f:
            f.write(json.dumps(metaData, indent=4)) 


class Analysis():

    def __init__(self, sharedlab):
        Ii, Ni, h, nui, Ri, Nap, rho, sigmaba = symbols('I_i N_i h nu_i R_i N_ap rho sigma_ab')
        self._sharedlab = sharedlab
        self._I = Eq(Ii, Ni * Ri * h * nui)
        self._kappa = Eq(kappa, rho * Derivative(log(Ni), rho))
        self._AbsPhotons = Eq(Nap, rho * Ni / (h * nui))
        self._QuantumY = Eq(phi, Ni * Ri / Nap)


    def emission(self, dueto, explist=''):
        '''dueto = N1, N2, N3
        '''
        display(self._I)
        if explist:
            explist = explist if type(explist) == list else [explist]
        else: explist = [self._sharedlab._current_exp]

        j = dueto[-1]
        nu = constants[f'nu{j}']['value']
        h = constants['h']['value']
        col = f'I{j}'

        for exp in explist:
            R = 1 / self._sharedlab.experiments(exp).sample.props[f'tau{j}']
            self._sharedlab.experiments(exp)._data[col] = self._sharedlab.experiments(exp)._data[dueto] * R * nu * h
            self._sharedlab.experiments(exp).getsteadydata()

            self._sharedlab.experiments(exp)._metadata['dataColumns'][col] = constantdetails(
                description=f'Emission intendsity due to level {j}',
                unit='J/s',
                axislabel='$I$'
            )

    def absorption(self, explist):
        '''Calculates absorption
        '''
        display(self._AbsPhotons)
        if explist:
            explist = explist if type(explist) == list else [explist]
        else: explist = [self._sharedlab._current_exp]

        nu = constants['nu']['value']
        h = constants['h']['value']
        col = 'absphotons'

        for exp in explist:
            sigma = self._sharedlab.experiments(exp).sample.props['s']
            data = self._sharedlab.experiments(exp).data()
            data[col] = data['Na'] * data['power'] * sigma / (h * nu)
            self._sharedlab.experiments(exp).getsteadydata()

            self._sharedlab.experiments(exp)._metadata['dataColumns'][col] = constantdetails(
                description=f'Number of photons absorbed during excitation of the absorber',
                unit='cm^{-3}',
                axislabel='$N_{a,ph}$',
            )

    def quantumyield(self, dueto, explist=''):
        '''Quantum Yield of the upconversion process 
        due to the emission level: dueto = N1, N2, N3
        '''
        display(self._QuantumY)
        if explist:
            explist = explist if type(explist) == list else [explist]
        else: explist = [self._sharedlab._current_exp]

        j = dueto[-1]
        col = f'quantyield{j}'
        nu = constants['nu']['value']
        h = constants['h']['value']

        for exp in explist:
            R = 1 / self._sharedlab.experiments(exp).sample.props[f'tau{j}']
            sigma = self._sharedlab.experiments(exp).sample.props['s']
            data = self._sharedlab.experiments(exp).data()
            data[col] = (data[dueto] * R) / (data['Na'] * data['power'] * sigma / (h * nu))
            self._sharedlab.experiments(exp).getsteadydata()

            self._sharedlab.experiments(exp)._metadata['dataColumns'][col] = constantdetails(
                description=f'Quantum Yield of upconversion due to {dueto}',
                unit='adm',
                axislabel='$\\phi$',
            )

    def kappa(self, dueto, explist=''):
        '''emission intensity slope
        dueto = N1, N2, N3
        '''
        display(self._kappa)
        if explist:
            explist = explist if type(explist) == list else [explist]
        else: explist = [self._sharedlab._current_exp]
        j = dueto[-1]

        for exp in explist:
            laser_setup = self._sharedlab.experiments(exp)._metadata['laser_setup']
            data = self._sharedlab.experiments(exp).data()
            stepsize = round(laser_setup['dt_per_step'] / laser_setup['dt_per_data_point'])
            # Instance kappa
            data[f'kappa{j}'] = 0.0
            # Get log of Nj
            NjLog = np.log(data[dueto])
            
            data[f'kappa{j}'] = data['power'] * (NjLog - NjLog.shift(stepsize)) / (data['power'] - data['power'].shift(stepsize))

            self._sharedlab.experiments(exp).getsteadydata()

            self._sharedlab.experiments(exp)._metadata['dataColumns'][f'kappa{j}'] = constantdetails(
                description=f'Slope of level {j} population',
                unit='adm',
                axislabel='$\\kappa$',
            )


class Lab():
    _experiments = dict()
    _current_exp = str()
    _markers = []
    _colours = []

    def __init__(self):
        self.calculate = Analysis(self)
        # self._markers = ['p', 'h', '*', 'v', '<', '>', 'o']
        # self._colours = ['#01328C', '#02734B', '#8A21C2', '#C70E4F', '#D1A24B', '#597BBA', '#825707']
        self._colours = ['tab:blue', 'tab:red', 'tab:orange', 'tab:green', 'tab:purple', 'tab:cyan', 'darkred']

    def newexp(self, label, laser_setup, sample):
        self._current_exp = label
        self._experiments[label] = Experiment(
            label,
            laser_setup,
            sample
        )
        self._experiments[label].run()
        self._experiments[label].transformdata()
        self._experiments[self._current_exp].color = self._colours[len(self._experiments) % len(self._colours)]
        
    
    def loadexp(self, expname):
        '''Loads data and metadata'''

        path = 'output/'
        dataPath = path + expname + '.csv'
        metadataPath = path + expname + '.json'
        
        with open(metadataPath, 'r') as json_file:
            metadata = json.load(json_file)

        self._current_exp = metadata['label']
        self._experiments[self._current_exp] = Experiment(
            metadata['label'],
            metadata['laser_setup'],
            metadata['sample']
        )
        self._experiments[self._current_exp]._metadata['dataColumns'] = metadata['dataColumns']
        self._experiments[self._current_exp]._data = pd.read_csv(dataPath, index_col=0)
        
        markers = ['o', 'o', 'p', 'h', '*', 'v', '<', '>', 'o']
        cols = self._experiments[self._current_exp]._data.columns

        for k in range(len(cols)):
            self._experiments[self._current_exp]._metadata['dataColumns'][cols[k]]['marker'] = markers[k] if k < len(markers) else 'o'

        self._experiments[self._current_exp].getsteadydata()
        self._experiments[self._current_exp].color = self._colours[len(self._experiments)]

        print(f"Experiment {metadata['label']} successfully loaded")
    
    def experiments(self, label=''):

        label = label if label else self._current_exp
        return(self._experiments[label])

    def plot(self, x, ylist, ylist2=[], sampling=10, frm=0, to=-1, exp='', steady=False):

        if exp:
            exp = exp if type(exp) == list else [exp]
        else: exp = [self._current_exp]

        if ylist2:
            fig, (ax1, ax2) = pl.subplots(2, sharex=True)
            axes = (ax1, ax2)
            ylist2 = ylist2 if type(ylist2) == list else [ylist2]
            y2label = ' '.join([
                self._experiments[exp[0]]._metadata['dataColumns'][ylist2[0]]['axislabel'], 
                f"(${self._experiments[exp[0]]._metadata['dataColumns'][ylist2[0]]['unit']}$)"
            ]) if ylist2[0] in self._experiments[exp[0]]._metadata['dataColumns'].keys() else ''
            
            ax2.set_ylabel(y2label)
        else:
            fig, ax1 = pl.subplots()
            axes = ax1
        ylist = ylist if type(ylist) == list else [ylist]
        ylabel = ' '.join([
            self._experiments[exp[0]]._metadata['dataColumns'][ylist[0]]['axislabel'], 
            f"(${self._experiments[exp[0]]._metadata['dataColumns'][ylist[0]]['unit']}$)"
        ]) if ylist[0] in self._experiments[exp[0]]._metadata['dataColumns'].keys() else ''

        for e in exp:
            # Select dataframe
            df = self._experiments[e]._steadydata.reset_index(drop=True) if steady else self._experiments[e]._data
            to = len(df) if to == -1 else to
            xlabel = ' '.join([self._experiments[e]._metadata['dataColumns'][x]['axislabel'], 
                f"(${self._experiments[e]._metadata['dataColumns'][x]['unit']}$)"])

            for y in ylist:
                if y in self._experiments[e]._metadata['dataColumns'].keys():
                    m = self._experiments[e]._metadata['dataColumns'][y]['marker']
                else: m = 'o'

                c = self._experiments[e].color

                ax1.loglog(df.loc[frm:to:sampling, x], df.loc[frm:to:sampling, y], marker=m, color=c, label=f'${y}-{e}$')
            
            if ylist2:
                for y2 in ylist2:
                    if y2 in self._experiments[e]._metadata['dataColumns'].keys():
                        m = self._experiments[e]._metadata['dataColumns'][y2]['marker']
                    else: m = 'o'

                    ax2.loglog(df.loc[frm:to:sampling, x], df.loc[frm:to:sampling, y2], marker=m, color=c, label=f'${y2}-{e}$')
                    ax2.legend()
        
        ax1.legend()
        ax1.set_ylabel(ylabel)
        pl.xlabel(xlabel)
        pl.show()
        return(axes)
    

    # def plot(self, x, ylist, sampling=10, frm=0, to=-1, exp=''):

    #     fig, (ax1, ax2) = pl.subplots(2, sharex=True)
    #     if exp:
    #         exp = exp if type(exp) == list else [exp]
    #     else: exp = [self._current_exp]

    #     ylist = ylist if type(ylist) == list else [ylist]
        
    #     for e in exp:
    #         t = self._experiments[e]._data.t
    #         ax2.semilogx(t[frm:to:sampling], self._experiments[e]._data.power[frm:to:sampling], 'rv')
    #         for y in ylist:
    #             y = self._experiments[e]._data[y]
    #             ax1.semilogx(t[frm:to:sampling], y[frm:to:sampling], 'o', label=r'$N_b$')
        
    #     ax1.legend()
    #     fig.suptitle(r'Population of the states')
    #     ax1.set_ylabel('N')
    #     ax2.set_ylabel(r'$\rho (W/cm^2)$')
    #     pl.xlabel('$t$')
    #     pl.show()

    # def calculate(self, param):

    #     if param == 'intensity':
    # }





