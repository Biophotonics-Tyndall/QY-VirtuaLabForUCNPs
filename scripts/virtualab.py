import numpy as np
from sympy import init_session, solve, lambdify, Symbol, symbols, Eq, log, Derivative, latex
from sympy.physics.mechanics import dynamicsymbols
from IPython.display import Math
import matplotlib.pyplot as pl
import pandas as pd
from scipy import constants
from time import sleep
from tqdm import tqdm
import json
from glob import glob


class Laser:
    
    def __init__(self, mode, wavelength):
        self.mode = mode
        self.wavelength = wavelength
        self.samplingRate = None
        
        if mode in ('linear', 'log'):
            self.range = ()
            self.samplesPerStep = None
        elif mode=='pulse':
            self.basePower = 0
            self.powerAtPeak = None
            self.period = None
            self.periodOn = None # delta t of duty cycle
            self.nCycles = 1
            # self.cycleEnd = 'top' # or bottom
        else:
            raise ValueError('Invalid mode! Select either linear, log or pulse mode')

        self._power = None
        
    def poweravg(self):
        if self.mode=='pulse':
            return(self.powerAtPeak * self.periodOn / self.period)
        else:
            print('Laser is not in pulse mode...')

    def dutycycle(self):
        if self.mode=='pulse':
            return(self.periodOn / self.period)
        else:
            print('Laser is not in pulse mode...')

    def run(self):
        ''' Creates a power ramping.
            Power in W/cm^2
            timeInterval = total time per power step
            pointsPerInterval = number of data points per timeInterval
            mode: log, linear or pulse
        '''
        self._power = pd.DataFrame(columns=['t', 'power'])
                
        if self.mode == 'linear':
            self._power['power'] = np.sort(
                self.samplesPerStep * list(np.linspace(*self.range)))
            self._power['power'] = self._power['power'].values[::-
                                                               1] if self.range[0] > self.range[1] else self._power['power']
        elif self.mode == 'log':
            self._power['power'] = np.sort(self.samplesPerStep * list(np.logspace(
                np.log10(self.range[0]), np.log10(self.range[1]), self.range[1])))
            self._power['power'] = self._power['power'].values[::-
                                                                1] if self.range[0] > self.range[1] else self._power['power']
        elif self.mode == 'pulse':
            nSamplesPerPeriod = int(self.samplingRate * self.period)
            nSamplesOn = int(self.samplingRate * self.periodOn)
            nSamplesOff = nSamplesPerPeriod - nSamplesOn
            self._power['power'] = self.basePower + np.array(self.nCycles * (nSamplesOff * [0] + nSamplesOn * [self.powerAtPeak])) 


        nt = len(self._power['power'])
        self._power['t'] = np.linspace(0, nt / self.samplingRate, nt)
        # print(f"{len(self._power['power'])} data points")

    def profile(self):
        """Plot time dependent power profile"""

        fig, ax = pl.subplots()
        ax.plot(self._power['t'], self._power['power'], 'red')
        ax.legend(['Step power'])
        ax.set_xlabel(r'$t (s)$')
        ax.set_ylabel(r'$\rho (W/cm^2)$')
        pl.show()


class Sample:
    
    def __init__(self, name):
        self.name = name
        self.props = {}
        self.energyStates = [] # List of symbols
        self.rateEquations = [] # List of rate equations for each state
        self.latexRateEqs = [] # list of rate equations in LaTeX syntax 
        self.prevStates = {} # dict with previous states values
        self.energyStatesPop = [] # List of np.arrays with population of the states
        self.boundaryConds = {} # boundary conditions for each energy level
        self.data = None # Data frame

    def model(self, states):
        """Decouple states list of tuples
        Args:
            model list(tuples): (Energy state symbol, rate equation, initial condition)    
        """
        self.energyStates = []
        self.rateEquations = []
        self.prevStates = {}
        for state in states:
            self.energyStates.append(state[0])
            self.rateEquations.append(state[1])
            self.latexRateEqs.append(latex(state[1]))
            self.prevStates.update({state[0]: state[2]})

    def showequations(self):
        for eq in self.latexRateEqs:
            display(Math(eq))

    def builduplevels(self, n):
        """Build a list of numpy arrays of n dimension with initial condition set"""
        self.energyStatesPop = []
        for i in range(len(self.energyStates)):
            # update model with values of the constants
            self.rateEquations[i] = self.rateEquations[i].subs(self.props)
            
            self.energyStatesPop.append(np.zeros(n, dtype=np.float64))
            self.energyStatesPop[i][0] = self.prevStates[list(self.prevStates.keys())[i]]
    
    def populatelevels(self, laserPower):
        """Calculate each energy level according to the lasers sweep array"""
        
        t = laserPower['t']
        n = len(laserPower)
        p = laserPower['power']
        self.builduplevels(n)
        currentStates = self.prevStates

        keep = True
        for k in tqdm(range(1, n), desc='Laser on 🚨'):
            dt = t[k] - t[k - 1]
                        
            for lv in range(len(self.energyStates)):
                NRate = float(self.rateEquations[lv].subs(self.prevStates).subs('rho', p[k]).args[1])
                self.energyStatesPop[lv][k] = dt * NRate + self.energyStatesPop[lv][k - 1] 
                # assert self.boundaryConds[self.energyStates[lv]][0] <= self.energyStatesPop[lv][k], f"{self.energyStates[lv]} = {self.energyStatesPop[lv][k]} below boundary"
                if not (self.boundaryConds[self.energyStates[lv]][0] <= self.energyStatesPop[lv][k] <= self.boundaryConds[self.energyStates[lv]][1]):
                    print(f"{self.energyStates[lv]} = {self.energyStatesPop[lv][k]} outside boundary")
                    keep = False
                    break
                # assert self.energyStatesPop[lv][k] <= self.boundaryConds[self.energyStates[lv]][1], f"{self.energyStates[lv]} = {self.energyStatesPop[lv][k]} above boundary"
                
                currentStates.update({self.energyStates[lv]: self.energyStatesPop[lv][k]})
            self.prevStates = currentStates
            if not keep: break
            
        self.data = pd.DataFrame({str(lv)[:3]: values for lv, values in zip(self.energyStates, self.energyStatesPop)})
        self.data = pd.concat([laserPower.loc[:k], self.data.loc[:k]], axis=1)

    def showlevels(self, x='t', levels=None):
        """Show the energy levels as a function of time of power"""
        levels = tuple(levels or self.data.columns[2:])

        fig, ax = pl.subplots()
        for lv in levels:
            lv = str(lv)
            ax.plot(self.data[x], self.data[lv], '-s', ms=3, label=lv)
        ax.set_xlabel('time $(s)$' if x == 't' else 'Power density $(W/cm^2)$')
        ax.set_ylabel('Pop. density $(cm^{-3}$)')
        pl.legend()

    # # calculate them on the jupyter notebook
    # def luminescence(self, fromLvl, emittedWavelength, R):
    #     nu =  constants.c / emittedWavelength
    #     self.data[f'L_{fromLvl}'] = self.data[f'N_{fromLvl}(t)'] * R / (constants.h * nu)

    # def numberOfPhotonsEmitted(self, fromLvl, R):
    #     self.data[f'n_photons_{fromLvl}_emitted'] = self.data[f'N_{fromLvl}(t)'] * R
    
    # def numberOfPhotonsAbsorbed(self):
    #     self.data['n_photons_absorbed'] = (self.props['N_sens'] - self.data['N_b(t)']) * self.props['sigma_ab'] * self.data['power'] \
    #         / (self.props['h'] * self.props['nu'])
    

class Experiment:
    
    def __init__(self):
        self.label = None
        self.sample = None
        self.laser = None

    def new(self, laser, sample):
        """Create new experiment"""
        self.laser = laser
        self.sample = sample
        self.label = sample.name
        self.laser.run()
        sleep(0.5)
        self.sample.populatelevels(self.laser._power)

    def load(self, expName):
        """Loads data from csv and metadata from json files"""
        savedFiles = glob('..\\output\\*.csv')
        dataFile = f'..\\output\\{expName}.csv'
        metadataFile = f'..\\output\\{expName}_metadata.json'

        if (dataFile in savedFiles):
            with open(metadataFile, 'r') as f:
                metadata = json.load(f)

            self.sample = Sample(name=metadata['exp_name'])
            self.sample.data = pd.read_csv(dataFile)
            self.sample.props.update(metadata['sample_props'])
            self.sample.boundaryConds.update(metadata['boundary_conds'])
            self.sample.energyStates = list(metadata['boundary_conds'].keys())
            self.sample.latexRateEqs = metadata['rate_equations']
            self.label = metadata['exp_name']

            self.laser = Laser(mode=metadata['laser_props']['mode'], wavelength=metadata['laser_props']['wavelength'])
            self.laser.__dict__.update(metadata['laser_props'])
            self.laser._power = self.sample.data[['t', 'power']]
        else:
            print(f"404: {dataFile} not found!")

    def save(self, fileName):
        "Saves data and metadata in separated files"

        fileName = fileName or self.label.replace(' ', '_')
        savedFiles = glob('..\\output\\*.csv')
        outputFile = f'..\\output\\{fileName}.csv'
        metadataFile = f'..\\output\\{fileName}_metadata.json'
        
        write = True
        if outputFile in savedFiles:
            print(f"{outputFile} already exists!")
            overwrite = input('Overwrite data? (y/n): ')
            write = write if overwrite.lower()=='y' else False
        
        if write:
            metadata = {}
            metadata['exp_name'] = self.label
            metadata['data_file'] = f"{fileName}.csv"
            metadata['laser_props'] = { k:v for k,v in vars(self.laser).items() if not k.startswith('_') }
            metadata['sample_props'] = {str(k): v for k,v in self.sample.props.items()}
            metadata['boundary_conds'] = {str(k): v for k,v in self.sample.boundaryConds.items()}
            metadata['rate_equations'] = self.sample.latexRateEqs

            with open(metadataFile, 'w') as outFile:
                json.dump(metadata, outFile)

            self.sample.data.to_csv(outputFile, index=False)            
            print(f'Data saved: {outputFile}!')

    def plot(self, x, yList=None, axis=None, label=None):

        yList = tuple(yList or self.sample.data.columns[2:])

        if axis is None:
            _, axs = pl.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        else: axs = axis

        if label:
            res = self
            for prop in label.split('.'):
                res = vars(res)[prop]
            label = f"{label.split('.')[-1]}:{res}"
        else: label = self.label

        for y in yList:
            axs[0].plot(self.sample.data[x], self.sample.data[y], '-s', ms=3, label=label + ' ' + y)
        axs[0].legend()
        axs[0].set_ylabel('Pop. density ($cm^{-3}$)')

        timeLabel = 'time $(s)$'
        powerLabel = 'Power density $(W/cm^2)$'
        if x=='t':
            yAxis = 'power'
            xLabel = timeLabel
            yLabel = powerLabel
        else:
            yAxis = 't'
            xLabel = powerLabel
            yLabel = timeLabel
            
        axs[1].plot(self.laser._power[x], self.laser._power[yAxis], '-', label=label)
        
        axs[1].set_ylabel(yLabel)
        axs[1].set_xlabel(xLabel)

        axs[1].legend()

        return(axs)


class Analysis:

    def __init__(self):
        self.reset()

    def reset(self):
        self.expList = []
        self.details = pd.DataFrame(columns=['label', 'P_avg', 'P_peak', 'T', 'Delta_t'])
    
    def addexperiment(self, name):
        exp = Experiment()
        exp.load(name)
        self.expList.append(exp)
        exp.label = f"E{len(self.details)}"
        expDetails = pd.DataFrame({
            'label': exp.label,
            'P_avg': exp.laser.poweravg(),
            'P_peak': exp.laser.powerAtPeak,
            'T': exp.laser.period,
            'Delta_t': exp.laser.periodOn
        }, index=[0])
        self.details = pd.concat([self.details, expDetails], ignore_index=True)
     
    def plot(self, x, yList=None, label=None):
        axs = self.expList[0].plot(x, yList, label=label)
        for exp in self.expList[1:]:
            exp.plot(x, yList, axs, label)


