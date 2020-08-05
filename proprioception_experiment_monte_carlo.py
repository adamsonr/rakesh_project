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

from numpy.random import normal, uniform, seed
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq, minimize_scalar
from scipy.special import erf
from scipy.integrate import cumtrapz

#seed(1)

#Define experimental parameters 
force_std=1  # Standard deviation of force estimation distribution
Ntrials=100  # Number of trials
nbins=10  # Number of histogram bins used in analysis
force_offset=0  # Systematic offset error in force estimation
target_force=10 # Target force (N)

#Data coming out of the simulated trials.  
#First dimension is treatment (visual removed, tactile removed, both removed)
#Second dim is column, 0: target force, 1: estimated force
#Third is trial number
trial_data=np.zeros([1,2,Ntrials])

#Run the experiment for one condition
for n in range(Ntrials):
    #Define target for for the current trial
    
    
    #Estimated force for the current trial
    est_force=normal(loc=(target_force+force_offset),scale=force_std)    
    trial_data[0,0,n]=target_force
    trial_data[0,1,n]=est_force

est=trial_data[0,1,:]

h,bin_edges=np.histogram(est,bins=nbins)

bin_centers=(bin_edges[:-1]+bin_edges[1:])/2

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
plt.plot(xfit,yfit)


#Identify the upper and lower thresholds
myerf=lambda x: (erf(scale*(x-offset))+1)/2
LT=minimize_scalar(lambda x: (myerf(x)-0.25)**2,bounds=[bin_centers[0],bin_centers[-1]],method='bounded')['x']  # lower 25% threshold
UT=minimize_scalar(lambda x: (myerf(x)-0.75)**2,bounds=[bin_centers[0],bin_centers[-1]],method='bounded')['x']  # upper 75% threshold 
pse=minimize_scalar(lambda x: (myerf(x)-0.5)**2,bounds=[bin_centers[0],bin_centers[-1]],method='bounded')['x']  # point of subjective equality
jnd=(UT-LT)/2  # just noticeable difference
wf=jnd/pse # weber fraction

# Report results
print("The JND is %0.2f" % jnd)
print("The Weber fraction is %0.4f" % (jnd/pse,))
print("The PSE is %0.2f" % (pse,))
print("The upper threshold is %0.2f" % (UT,))
print("The lower threshold is %0.2f" % (LT,))

