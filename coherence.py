import matplotlib.pyplot as plt
from spyglass.common import get_electrode_indices
from typing import List 
import numpy as np
import pandas as pd
from ripple_detection.detectors import Kay_ripple_detector


import math


from spyglass.common import (RawPosition, HeadDir, Speed, LinPos, StateScriptFile, VideoFile,
                                  DataAcquisitionDevice, CameraDevice, Probe,
                                  DIOEvents,
                                  ElectrodeGroup, Electrode, Raw, SampleCount,
                                  LFPSelection, LFP, LFPBandSelection, LFPBand,
                                  FirFilter,
                                  IntervalList,                
                                  Nwbfile, AnalysisNwbfile, NwbfileKachery, AnalysisNwbfileKachery,
                                  get_electrode_indices)
from spyglass.common.common_position import (PositionInfoParameters,IntervalPositionInfo)
from spyglass.common.common_interval import interval_list_intersect
from spectral_connectivity import Multitaper
from spectral_connectivity import Connectivity
from spectral_connectivity import multitaper_connectivity

from spectral_connectivity.statistics import fisher_z_transform
from spectral_connectivity.statistics import coherence_bias

from plot_lfp import (get_x_y_timestamp_list)

def multitaper_and_connectivity(signal,sampling_frequency,time_halfbandwidth_product,time_window_duration,time_window_step,start_time):
    #this function takes in signal, which is in the format (n_data_point, n_trials, n_signals)
    #and some other package related variables- 
    #for my purpose, i have been using time_halfbandwidth_product=1.5
    #time_window_duration = .5
    #time_window_step= None
    #start time = -1* time before. 
    #it will return a time variable relative to the time start, the coherence magnitude, power, and a number of parameters which are useful in other contexts (trials, num tapers, z transform for CI realted info etc) 
    
    m = Multitaper(signal,
                   sampling_frequency=sampling_frequency,
                   time_halfbandwidth_product=time_halfbandwidth_product,
                   time_window_duration=time_window_duration,
                   time_window_step=time_window_step,
                   start_time=start_time)
    c = Connectivity(fourier_coefficients=m.fft(),
                     frequencies=m.frequencies,
                     time=m.time)
    time = m.time
    freq=c.frequencies,
    mag=c.coherence_magnitude
    power = c.power
    n_trials = m.n_trials
    coherency = c.coherency()[:, (c.frequencies == 8), 0, 1].squeeze()
    bias1 = coherence_bias(c.n_observations)
    num_tapers = m.n_tapers

    z_transform= fisher_z_transform(coherency, bias1)
    return(time, freq, mag, power,n_trials, z_transform, num_tapers)


def get_signal_from_trialtimes(trial_times, time_b4,time_after, electrodes, lfp_eseries, lfp_timestamps, lfp_data):
    #this will take a list of indices for multiple trials. for example, a list of all the segment switch times across trials. 
    #also takes the electrode IDs (for hpc, electrode 8 = tetrode 2 wire 4).
    #time b4 and time after are the number of seconds you want around the trial time, and it will mask the lfp data 
    #then, for all the trials for all electrodes, it will return signal which is the format necessary for the coherence function above. 
    
    sampling_rate = 1000
    num_sec = time_b4+time_after
    n_trials = len(trial_times)
    n_signals= len(electrodes)
    n_time_samples = (num_sec*sampling_rate)


    # inititalize "signal" which will contain alllll of trials of the data 
    signal = np.zeros((int(n_time_samples), n_trials, n_signals))
    
    for ix in range(len(trial_times)):
        time_at_center = trial_times[ix]
        time_start = np.array(time_at_center - time_b4)
        time_end =  np.array(time_at_center + time_after)
                
        x_elect_lfp_eg1, y_elect_lfp_eg1 = get_x_y_timestamp_list(time_start, time_end, electrodes, lfp_eseries, lfp_timestamps, lfp_data)
           
        if len(y_elect_lfp_eg1[0]) < n_time_samples:
            num_zeros = n_time_samples - len(y_elect_lfp_eg1[0])
            y_elect_lfp_eg1[:] = np.pad(y_elect_lfp_eg1[:], (0, int(num_zeros)), 'constant')
        if len(y_elect_lfp_eg1[0]) > n_time_samples:
            num_diff =len(y_elect_lfp_eg1[0])- n_time_samples 
            for elec in range(len(electrodes)):
                y_elect_lfp_eg1[elec] = y_elect_lfp_eg1[elec][0:int(-num_diff)]
        
        for elec in range(len(electrodes)):    
            signal[:,ix,elec]= (y_elect_lfp_eg1[elec])
    return(signal)

                               

def get_CI(z_transform, num_trials, num_tapers):
#this function takes the z transformed data, plus the number of trials and tapers from the coherence calcualtion, and returns the pos and negative 95% confidence intervals. 
    ci_95_pos=[]
    ci_95_neg=[]
    for ix in range(len(z_transform)):
        z_ixed = z_transform[ix]
        ci_95_pos.append(z_ixed + 1.96 * math.sqrt(1 / (2 * (num_trials * num_tapers) -2)))
        ci_95_neg.append(z_ixed - 1.96 * math.sqrt(1 / (2 * (num_trials * num_tapers) -2)))
        # ci_95_pos.append(1.96 * math.sqrt(1 / (2 * (num_trials * num_tapers) -2)))
        # ci_95_neg.append(-1* 1.96 * math.sqrt(1 / (2 * (num_trials * num_tapers) -2)))
    return(ci_95_pos, ci_95_neg)



    