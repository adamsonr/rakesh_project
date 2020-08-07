# -*- coding: utf-8 -*-
"""
Monte Carlo simulation of proprioception experiment to determine 
JND and Weber fraction from test outcomes.

On each trial of the experiment, the participant is presented with a random reference 
force level via visual and tactile feeback.  He is then asked to reproduce the 
force level in the absence of one or both of the feedback types.  This produces a sequence
of target force levels and actual force levels that are analyzed to determine the JND.

The error between actual and target force is assumed gaussian distributed with a 
standard deviation sig
"""

from numpy.random import normal 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq, minimize_scalar
from scipy.special import erf
import pandas as pd
''' 
Data format (from email from Rakesh)
csv file format: participantID_date_trialNum.csv (i.e. RG_080620_1.csv), with internal data structure:
participantID, trial_number, condition_code, distanceFromCrossHairLocation, forceValue, stiffnessValue \n
sample: RG, 99, 1, 400, 5, 10
units: initials/deidentified ID, trial number, condition being tested (0 = standard, 1 = visual only, 2 = tactile only, 3 = no feedback), distance in pixels, force in N, stiffness in N/m
'''

def MakeFakeDataFile(participantID, date, trialNum, target_force, force_std, 
                     force_offset, Ntrials, use_file_dialog=False):
    '''
    Generates a data output file for a simulated experiment
    
    Parameters
    -------------------
    participantID: string 
        target_force Target force (N)
    date: string 
        target_force Target force (N)
    trialNum: string 
        target_force Target force (N)
        
    force_std: float64
        Standard deviation of force estimation distribution (N)
    force_offset: float64
        Systematic offset error in force estimation
    Ntrials: int
        Number of trials to generate
    use_file_dialog: bool
        If True, a file dialog box will open.  If False file will write to the 
        current working directory without a prompt.
        
    Returns
    ------------------------
    fname: str
        Name of the file that data was saved to
        '''
    default_fname="%s_%s_%04d.csv" % (participantID, date, trialNum)
    
    #Genearte simulated data.  
    #First dimension is treatment (visual removed, tactile removed, both removed)
    #Second dim is column, 0: target force, 1: estimated force
    #Third is trial number
    
    d={'Target force': np.zeros(Ntrials),'Response force': np.zeros(Ntrials)}
    #Run the experiment for one condition
    for n in range(Ntrials):
        #Define target for for the current trial
        
        
        #Estimated force for the current trial
        est_force=normal(loc=(target_force+force_offset),scale=force_std)    
        d['Target force'][n]=target_force
        d['Response force'][n]=est_force
    try:
        import easygui
        fname=easygui.filesavebox(default=default_fname)
    except ModuleNotFoundError:
        print('Easygui module not installed.  Run "pip install easygui" from the command prompt and rerun this script.')
        raise ModuleNotFoundError
    
    df=pd.DataFrame.from_dict(d)
    df.to_csv(fname, index=False)
    return(fname)
    
def AnalyzeFile(filename,nbins=10,enable_plots=False):
    '''
    Generates a data output file for a simulated experiment
    
    Parameters
    -------------------
    filename: string
        Name of the file to load
    nbins: int (default: 10) 
        Number of histogram bins used in analysis
    
    Returns
    -------------------
    jnd: float64
        Estimated just noticeable difference
    weber_fraction: float64
        Estimated Weber fraction
    pse: float64 
        Estimated point of subjective equality
    ut: float64
        Estimated lower (75%) threshold 
    lt: float64
        Estimated lower (25%) threshold 
    '''     
    
    target,resp=np.loadtxt(filename,delimiter=',',unpack=True, skiprows=1)  
    
    h,bin_edges=np.histogram(resp,bins=nbins)
    
    bin_centers=(bin_edges[:-1]+bin_edges[1:])/2
    
    if enable_plots:
        plt.figure()
        plt.plot(bin_centers,np.cumsum(h),marker='s',linestyle='none')
        plt.ylabel("Number of responses")
        plt.xlabel("Force")
    
    #Fit data to an error function
    def erf_err(p):
        amp,scale,offset=p
        out=amp*erf(scale*(bin_centers-offset))+sum(h)/2
        return(out-np.cumsum(h))
        
    pguess=np.sum(h)/2,0.4,bin_centers[np.int32(nbins/2)]
    sol=leastsq(erf_err,pguess)    
    pfit=sol[0]
    xfit=np.linspace(bin_centers[0],bin_centers[-1],1000)
    amp,scale,offset=pfit
    yfit=amp*erf(scale*(xfit-offset))+sum(h)/2
    if enable_plots:
        plt.plot(xfit,yfit)
    
    
    #Identify the upper and lower thresholds
    myerf=lambda x: (erf(scale*(x-offset))+1)/2
    lt=minimize_scalar(lambda x: (myerf(x)-0.25)**2,bounds=[bin_centers[0],bin_centers[-1]],method='bounded')['x']  # lower 25% threshold
    ut=minimize_scalar(lambda x: (myerf(x)-0.75)**2,bounds=[bin_centers[0],bin_centers[-1]],method='bounded')['x']  # upper 75% threshold 
    pse=minimize_scalar(lambda x: (myerf(x)-0.5)**2,bounds=[bin_centers[0],bin_centers[-1]],method='bounded')['x']  # point of subjective equality
    jnd=(ut-lt)/2  # just noticeable difference
    weber_fraction=jnd/pse # weber fraction
    
    # Report results
    print("The JND is %0.2f" % jnd)
    print("The Weber fraction is %0.4f" % (weber_fraction,))
    print("The PSE is %0.2f" % (pse,))
    print("The upper threshold is %0.2f" % (ut,))
    print("The lower threshold is %0.2f" % (lt,))
    
    return(jnd,weber_fraction, pse, ut, lt)

if __name__=='__main__':
    #Demo the script
    fname = MakeFakeDataFile(participantID = "0001", 
                             date = "20200805", 
                             trialNum = 1, 
                             target_force = 10, 
                             force_std = 1, 
                             force_offset =0, 
                             Ntrials = 100 ,
                             use_file_dialog=True)
    
    jnd,weber_fraction, pse, ut, lt = AnalyzeFile(fname, enable_plots=True)
    