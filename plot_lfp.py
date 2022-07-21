import matplotlib.pyplot as plt
from spyglass.common import get_electrode_indices
from typing import List 
import numpy as np
import pandas as pd
from ripple_detection.detectors import Kay_ripple_detector



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



def get_electrode_ids(nwb_file_name, which_elects, ref):
    #takes nwb file name(str), which_elects = every ___th electrode. For example, if you set it to be 4, it will grab every 4th electrode. 
    #set ref = True if you want to make sure the references are added to the list.
    #returns array of electrode ids that can be used to populate lfp table.  
    
    electrode_ids = (Electrode() & {'nwb_file_name' : nwb_file_name}).fetch('electrode_id')
    lfp_electrode_ids = electrode_ids[range(0, len(electrode_ids),which_elects)]
    print(f'original electrode ids, minus ref = {lfp_electrode_ids}')
    if ref == True:
        references = np.unique((Electrode() & {'nwb_file_name' : nwb_file_name}).fetch('original_reference_electrode'))
        print(f'added references = {references}')
        lfp_electrode_ids = np.append(lfp_electrode_ids,references) #if ref not in lfp_electrode_ids else None
        lfp_electrode_ids = np.unique(lfp_electrode_ids)    
        print(f'lfp electrode ids, plus references = {lfp_electrode_ids}')
    else:
        print('didnt add references')
    return(lfp_electrode_ids)


def remove_dead_tet_chans(nwb_file_name, lfp_electrode_ids, perm_dead_chan):
    #this function takes nwb file, current list of electrode ids, and perm_dead_channels which is the channel number above which all tetrode channels should be dead. 
    #removes the permanantly dead channels from electrode id list, doesnt add anything in their place. 
    tet_dead = ((Electrode() & {'nwb_file_name' : nwb_file_name} & {'probe_type' : 'tetrode_12.5'} & {'bad_channel' : 'True'}).fetch('electrode_id'))
    dont_exist = tet_dead[tet_dead>=perm_dead_chan]
    print(f'removing {dont_exist} because their electrode id was greater than/equal to {perm_dead_chan} and thus shouldnt exist on the hybrid animals')
    lfp_electrode_ids= np.setdiff1d(lfp_electrode_ids,dont_exist) 
    return(lfp_electrode_ids)


def check_for_dead_probes(nwb_file_name, lfp_electrode_ids):
    #this function checks for dead probe channels. 
    #if there are, it will print them, and then the rest of the code should be modified to handle them.
    probe_type = np.unique((Electrode() & {'nwb_file_name' : nwb_file_name}).fetch('probe_type'))[0]
    print(f'probe type = {probe_type}')
    probe_dead = (Electrode() & {'nwb_file_name' : nwb_file_name} & {'probe_type' : probe_type} & {'bad_channel' : 'True'}).fetch('electrode_id')
    if probe_dead.size==0:
        print(f'no dead probe channels from Electrode table for {nwb_file_name}')
    else:
        print('found dead probe channels, check out the code to remove them!')
           # #for the probes, (and check to make sure theyre still probe_type [0], check if there are dead channels, and is so, remove them and find the next one thats not dead to add. 


            #uncomment and test if there are dead probe channels, or make a test case later 

        # for ix in range(len(probe_dead)):
        #     #check the list of dead tetrode channels 
        #     if probe_dead[ix] in lfp_electrode_ids:
        #         #if one of the dead chanels is on the current list of lfp electrodes, print it out
        #         print(probe_dead[ix])
        #         #then remove it from the list
        #         lfp_electrode_ids = lfp_electrode_ids.remove(probe_dead[ix])
        #         #check if tet_dead[ix]+1 is dead. if not, add it to lfp electrodes. 
        #         next_elec = probe_dead[ix]+1 
        #         if next_elec in probe_dead:
        #             #if next_elec is also dead, print that out/
        #             print(f'{next_elec} is dead too')
        #         else: 
        #             #if its not dead, append it to lfp_electrode_ids. 
        #             lfp_electrodes_ids = lfp_electrode_ids.append(next_elec) 
        #     else:
        #         print('none!')
        # lfp_electrode_ids = np.unique(lfp_electrode_ids)


def checking_other_dead_tets(nwb_file_name,lfp_electrode_ids): 
      #this cell will check if there are any dead tetrodes in the lfp_electrodes list. if so, it will remove them, and add the next electrode to the list if it isnt dead. returns modified list of electrode ids, at this point with no dead channels included. 
    tet_dead = ((Electrode() & {'nwb_file_name' : nwb_file_name} & {'probe_type' :'tetrode_12.5'} & {'bad_channel' : 'True'}) - {'electrode_group_name': '24'} - {'electrode_group_name': '25'} - {'electrode_group_name': '26'}-  {'electrode_group_name': '27'} - {'electrode_group_name': '28'} - {'electrode_group_name': '29'} - {'electrode_group_name': '30'} - {'electrode_group_name': '31'}).fetch('electrode_id')

    for ix in range(len(tet_dead)):
        #check the list of dead tetrode channels 
        if tet_dead[ix] in lfp_electrode_ids:
            #if one of the dead channels is on the current list of lfp electrodes f=to load lfp for, print it out
            print(f' channel {tet_dead[ix]} has been found on lfp_electrode_ids list')
            #then remove it from the list of electrode ids
            lfp_electrode_ids = lfp_electrode_ids.remove(tet_dead[ix])
            print(f'removed {tet_dead[ix]}')    
            #check if tet_dead[ix]+1 is dead. if not, add it to lfp electrodes. 
            next_elec = tet_dead[ix]+1 
            if next_elec in tet_dead:
                #if next_elec is also dead, print that out/
                print(f'{next_elec} is dead too')
            else: 
                #if its not dead, append it to lfp_electrode_ids. 
                lfp_electrodes_ids = lfp_electrode_ids.append(next_elec) 
                print(f' channel {next_elect} has been added to electrode ids')
        else:
            print(f'channel {tet_dead[ix]} was not found on lfp_electrode_ids, was not removed, and no channels added')
    lfp_electrode_ids = np.unique(lfp_electrode_ids)
    return(lfp_electrode_ids)

def get_timestamps_and_data(nwb_file_name, filter_type: List=None, data_type: List=None):
    #this function will return the eseries, timestamps, and data for whichever type of data is indicated in data_type. 
    #give it nwb_file name, filter type (because after ripple filtering, will be two in LFPBand.)
    #data_type is a list of strings. can include ['raw','theta','lpf']
    if data_type is not None:
        if 'theta' in data_type:
            theta_eseries = (LFPBand() & {'nwb_file_name' : nwb_file_name} & {'filter_name' : filter_type[0]}).fetch_nwb()[0]['filtered_data']
            theta_timestamps = theta_eseries.timestamps
            theta_data = theta_eseries.data
            return(theta_timestamps,theta_data,theta_eseries)
        if 'raw' in data_type:
            orig_eseries = (Raw() & {'nwb_file_name' : nwb_file_name}).fetch_nwb()[0]['raw']
            raw_timestamps = orig_eseries.timestamps
            raw_data = orig_eseries.data
            return(raw_timestamps,raw_data,orig_eseries)
        if 'lfp' in data_type:
            lfp_eseries = (LFP() & {'nwb_file_name' : nwb_file_name}).fetch_nwb()[0]['lfp']
            lfp_timestamps = lfp_eseries.timestamps
            lfp_data = lfp_eseries.data
            return(lfp_timestamps,lfp_data,lfp_eseries)
        if 'ripple' in data_type:
            ripple_eseries = (LFPBand() & {'nwb_file_name' : nwb_file_name} & {'filter_name' : filter_type[0]}).fetch_nwb()[0]['filtered_data']
            ripple_timestamps = ripple_eseries.timestamps
            ripple_data = ripple_eseries.data
            return(ripple_timestamps,ripple_data,ripple_eseries)

def get_speed(position_info,epoch,time_from_start, time_interval_s):
    # position_info= key['position_info'] 
    # epoch= key['epoch']
    # time_from_start =key['time_from_start']
    # time_interval_s = key['time_interval_s']
    
    position_start_timestamp = (position_info.index>epoch[0][0] +time_from_start)
    position_end_timestamps = (position_info.index<epoch[0][0]+time_from_start + time_interval_s)
    position_x = position_info.index[position_start_timestamp & position_end_timestamps]
    position_y = position_info.head_speed[position_start_timestamp & position_end_timestamps]
    
    return(position_x,position_y)
        
        
        

    
def get_x_y_list(time_from_start, time_interval_s,epoch, electrode_id, eseries, timestamps, data):
    x_elect=[]
    y_elect=[]
    for ix in range(len(electrode_id)):
            #get electrode id from list
        electrode_id_ix = electrode_id[ix] #confusing ebcause this isnt an index, its an lectrode id 
            #get the index for that electrode id using the helper function 
        index = get_electrode_indices(eseries,[electrode_id_ix])
            # make the mask using the times 
        start_timestamp = timestamps> epoch[0][0] +time_from_start
        end_timestamp = timestamps<(epoch[0][0]+time_interval_s+time_from_start)
            #apply the mask to the timestamps and data passed in. 
        x = timestamps[start_timestamp & end_timestamp]
        y = data[(start_timestamp & end_timestamp),index[0]]
            #append a list that will be the return of the funciton, that one will iterate through in the notebook to use. 
        x_elect.append(x)
        y_elect.append(y)
    return (x_elect, y_elect)      
        
    
def simple_plot(x,y,title,xlab,ylab):
    f=plt.figure()
    plt.plot(x,y)
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    # return(f)        

def plot_overlay(x0,y0,lab0,color0,x1,y1,lab1,color1,title,xlab,ylab, offset= None, x2=None,y2=None,lab2=None,color2=None, x3=None,y3=None,lab3=None,color3=None,x4=None,y4=None,lab4=None,color4=None): 
    plt.figure(figsize=(40,15))
    plt.plot(x0,y0,label=lab0,color=color0)
    plt.plot(x1,y1+offset,label=lab1,color=color1,alpha=.5)
    if x2 is not None:
        plt.plot(x2,y2-offset,label=lab2,color=color2)
    if x3 is not None:
        plt.plot(x3,y3+2*offset,label = lab3,color=color3)
    if x4 is not None:
        plt.plot(x4,y4+3*offset,label = lab4,color=color4)
    plt.legend(fontsize=25)
    plt.title(title,fontsize=25)
    plt.xlabel(xlab,fontsize=25)
    plt.ylabel(ylab,fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)


    
    
# def plot_overlay_test(x0,y0,lab0,color0,x1,y1,lab1,color1,title,xlab,ylab,offset=None, **args):
#     plt.figure(figsize=(40,12))
#     plt.plot(x0,y0,label=lab0,color=color0)
#     plt.plot(x1,y1+offset,label=lab1,color=color1,alpha=.5)
#     if x2 is not None:
#         plt.plot(x2,y2-offset,label=lab2,color=color2)
#     if x3 is not None:
#         plt.plot(x3,y3+2*offset,label = lab3,color=color3)
#     if x4 is not None:
#         plt.plot(x4,y4+3*offset,label = lab4,color=color4)    
#     plt.legend()
#     plt.title(title,fontsize=25)
#     plt.xlabel(xlab,fontsize=25)
#     plt.ylabel(ylab,fontsize=25)
#     plt.xticks(fontsize=25)
#     plt.yticks(fontsize=25)    
    
# def plot_speed_versus_theta(x_pos, y_pos, theta_x, theta_y, title,xlab1,ylab1,xlab2,ylab2): 
#     fig, ax = plt.subplots(2, 1, figsize=(25, 7))
#     ax[0].plot(x_pos, y_pos)
#     ax[0].set_xlabel(xlab1, fontsize=18)
#     ax[0].set_ylabel(ylab1, fontsize=18)
#     ax[0].set_title(title, fontsize=28)

#     ax[1].plot(theta_x, theta_y)
#     ax[1].set_xlabel(xlab2, fontsize=18)
#     ax[1].set_ylabel(ylab2, fontsize=18)

def plot_overlay_with_pos(lfp_eseries, orig_eseries, theta_eseries, ripple_eseries, position_info, nwb_file_name, time_from_start, time_interval_s, epoch, electrode_id,offset):
    #tplot of all four data types and then speed on a lower subplot. just pass in the eseries for each signal, position info (which has the speed data) , the nwb file name for labeling, the time from start/time in seconds desired, the electrode channel(s) as a list, and an offset in uV if desired. 
    
    #load timestamps and data
    lfp_timestamps= lfp_eseries.timestamps
    lfp_data= lfp_eseries.data
    raw_timestamps = orig_eseries.timestamps
    raw_data = orig_eseries.data
    theta_timestamps = theta_eseries.timestamps
    theta_data= theta_eseries.data
    ripple_timestamps = ripple_eseries.timestamps
    ripple_data = ripple_eseries.data 
    #
    for ix in range(len(electrode_id)):
        electrode_id_ix = electrode_id[ix]
        x_elect_lfp, y_elect_lfp =get_x_y_list(time_from_start, time_interval_s,epoch, [electrode_id_ix], lfp_eseries, lfp_timestamps, lfp_data)
        x_elect_raw, y_elect_raw =get_x_y_list(time_from_start, time_interval_s,epoch, [electrode_id_ix], orig_eseries, raw_timestamps, raw_data)
        x_elect_theta, y_elect_theta =get_x_y_list(time_from_start, time_interval_s,epoch, [electrode_id_ix], theta_eseries, theta_timestamps, theta_data)
        x_elect_ripple, y_elect_ripple =get_x_y_list(time_from_start, time_interval_s,epoch, [electrode_id_ix], ripple_eseries, ripple_timestamps, ripple_data)
        x_pos,y_pos = get_speed(position_info,epoch,time_from_start,time_interval_s)

        fig, ax = plt.subplots(2, 1, figsize=(40, 20))

        ax[0].plot(x_elect_lfp[0], y_elect_lfp[0],label='lfp',color= 'blue')
        ax[0].plot(x_elect_raw[0], y_elect_raw[0]+offset,label='raw',color='black')
        ax[0].plot(x_elect_theta[0], y_elect_theta[0]-offset,label='theta',color='purple')
        ax[0].plot(x_elect_ripple[0], y_elect_ripple[0]-2*offset,label='ripple',color='pink')
        ax[0].set_xlabel('time(s)', fontsize=28)
        ax[0].set_ylabel('amplitude (AD units', fontsize=28)
        ax[0].set_title(f'all data types,{nwb_file_name}, time = {time_from_start} : {time_from_start+time_interval_s}\n electrode_id = {electrode_id_ix}, tetrode = {(electrode_id_ix/4)+1}'  , fontsize=28)
        ax[0].legend(['lfp', 'raw','theta','ripple'], fontsize=28)
        # ax[0].set_xticklabels(x_elect_lfp[0][:10, fontsize=28)
        ax[1].plot(x_pos,y_pos)
        ax[1].set_xlabel('time(s)', fontsize=28)

 







    
#This function- pass in a dictionary of all listed variables. 
#nwb file, time to start/length of segment relative to run 1 times. 
#a single electrode id, and the data corresponding to the type (can take in raw, lfp or theta filtered in the dict)
#returns timestamps and data masked by those timestamps, for plotting. 
# def get_x_y(key):
    
#     #get the stuff out of the dict 
#     time_from_start = key['time_from_start'] 
#     time_interval= key['time_interval']
#     electrode_id = key['electrode_id'] 
#     timestamps = key['timestamps']
#     data = key['data']
#     eseries = key['eseries']
#     epoch = key['epoch']
    
#     #find electrode index using helper function 
#     index = get_electrode_indices(eseries,[electrode_id])
    
#     #create the mask using the times
#     start_timestamp = timestamps>epoch[0][0] +time_from_start
#     end_timestamp = (timestamps<epoch[0][0]+time_interval+time_from_start)
    
#     #apply the mask to the data and timestamps passed in
#     x = timestamps[start_timestamp & end_timestamp]
#     y = data[(start_timestamp & end_timestamp),index[0]]
    
#     return(x,y)

#This function is the same as the last, but instead handles a list of electrodes_ids and returns a list of the data from those electrodes, masked by the timestamps. 
#if the data is over the same time, the x_elect is unnecessary (theyre all the same), but only slightly more inneficient so fix later. 

###########uncomment here future em when trying to fix things!

# def get_x_y_list(key):
#     #get the important stuff out of the dictionary
#     time_from_start = key['time_from_start'] 
#     time_interval= key['time_interval']
#     electrode_id = key['electrode_id'] 
#     timestamps = key['timestamps']
#     data = key['data']
#     eseries = key['eseries']
#     epoch= key['epoch']
   
#     x_elect=[]
#     y_elect=[]
#     for ix in range(len(electrode_id)):
        
#         #get electrode id from list
#         electrode_id_ix = electrode_id[ix]
#         #get the index for that electrode id using the helper function 
#         index = get_electrode_indices(eseries,[electrode_id_ix])     
#         #make the mask using the times 
#         start_timestamp = timestamps>epoch[0][0] +time_from_start
#         end_timestamp = (timestamps<epoch[0][0]+time_interval+time_from_start)
#         #apply the mask to the timestamps and data passed in. 
#         x = timestamps[start_timestamp & end_timestamp]
#         y = data[(start_timestamp & end_timestamp),index[0]]
#         #append a list that will be the return of the funciton, that one will iterate through in the notebook to use. 
#         x_elect.append(x)
#         y_elect.append(y)
#     return (x_elect, y_elect)






# def get_x_y_list(time_from_start, time_interval_s,epoch, electrode_id, eseries, timestamps, data):
#     x_elect=[]
#     y_elect=[]
#     for ix in range(len(electrode_id)):
#             #get electrode id from list
#         electrode_id_ix = electrode_id[ix]
#             #get the index for that electrode id using the helper function 
#         index = get_electrode_indices(eseries,[electrode_id_ix])
#             # make the mask using the times 
#         start_timestamp = timestamps> epoch[0][0] +time_from_start
#         end_timestamp = timestamps<(epoch[0][0]+time_interval_s+time_from_start)
#             #apply the mask to the timestamps and data passed in. 
#         x = timestamps[start_timestamp & end_timestamp]
#         y = data[(start_timestamp & end_timestamp),index[0]]
#             #append a list that will be the return of the funciton, that one will iterate through in the notebook to use. 
#         x_elect.append(x)
#         y_elect.append(y)
#     return (x_elect, y_elect)

# if data_type is not None:
    #     if 'raw' in data_type:
    #         timestamps, data, eseries = get_timestamps_and_data(nwb_file_name, filter_type = None, data_type = ['raw'])
    #     if 'lfp' in data_type:
    #         timestamps, data, eseries = get_timestamps_and_data(nwb_file_name, filter_type = None, data_type = ['lfp'])    
    #     if 'theta' in data_type:
    #         timestamps, data, eseries = get_timestamps_and_data(nwb_file_name, filter_type, data_type = ['theta']) 










# #THis was my original plan for having control over plotting lfp vs raw vs theta. If I only want to do 1/2 of the three. 
# #Was close to getting this, but making seperate subplots got complicted (passing through two functions. 
# #Gave up on it! important. 

# def plot_many(key, plot_type: List=None, ax=None):
#     if plot_type is not None:
#         if 'theta' in plot_type:
#             plot_theta(key)
#         if 'lfp' in plot_type:
#             plot_lfp(key)
#         if 'raw' in plot_type:
#             plot_raw(key) 


# #THis failed, i think, because i put the indexing and the plotting in the same function, making it more inflexible.         
# def plot_theta(key):     
#     nwb_file_name = key['nwb_file_name']
#     time_from_start = key['time_from_start'] 
#     time_interval= key['time_interval']
#     electrode_id = key['electrode_id'] 
#     theta_timestamps = key['theta_timestamps']
#     theta_data = key['theta_data']
#     theta_eseries = key['theta_eseries']
#     epoch = key['epoch']
#     sampling_rate_lfp = key['sampling_rate_lfp']
#     # print(nwb_file_name,time_from_start,time_interval_s)
#     theta_index = get_electrode_indices(theta_eseries,[electrode_id])   
    
#     theta_start_timestamp = theta_timestamps>epoch[0][0] +time_from_start
#     theta_end_timestamp = (theta_timestamps<epoch[0][0]+time_interval+time_from_start)
#     theta_x = theta_timestamps[theta_start_timestamp & theta_end_timestamp]
#     theta_y = theta_data[(theta_start_timestamp & theta_end_timestamp),theta_index[0]]
    
#     # # axarr = f.add_subplot(1,1,1) # here is where you add the subplot to f
#     ax= plt.plot(theta_x, theta_y)
#     # # plt.set_xlim(min(time), max(time))
#     # # plt.set_xlabel(xlab)
#     # # plt.set_ylabel(ylab)
#     # # plt.grid(show_grid)
#     # # plt.title(title, size=16)
#     return(ax)
    
# def plot_lfp(nwb_file_name, time_from_start, time_interval, electrode_id, lfp_timestamps, lfp_data, lfp_eseries, epoch, sampling_rate_lfp):
#     lfp_index = get_electrode_indices(lfp_eseries,[electrode_id])     
#     lfp_start_timestamp = lfp_timestamps>epoch[0][0] +time_from_start
#     lfp_end_timestamp = (lfp_timestamps<epoch[0][0]+time_interval+time_from_start)
#     lfp_x = lfp_timestamps[lfp_start_timestamp & lfp_end_timestamp]
#     lfp_y = lfp_data[(lfp_start_timestamp & lfp_end_timestamp),lfp_index[0]]
#     ax1 = plt.plot(lfp_x, lfp_y)
#     return(ax1)

# def plot_raw(key):
#     nwb_file_name = key['nwb_file_name']
#     time_from_start = key['time_from_start'] 
#     time_interval= key['time_interval']
#     electrode_id = key['electrode_id'] 
#     raw_timestamps = key['raw_timestamps']
#     raw_data = key['raw_data']
#     orig_eseries = key['orig_eseries']
#    epoch = key['epoch']
#     sampling_rate_orig = key['sampling_rate_orig']
    
    
    
#     raw_index = get_electrode_indices(orig_eseries,[electrode_id])     
#     raw_start_timestamp = raw_timestamps>epoch[0][0] +time_from_start
#     raw_end_timestamp = (raw_timestamps<epoch[0][0]+time_interval+time_from_start)
#     raw_x = raw_timestamps[raw_start_timestamp & raw_end_timestamp]
#     raw_y = raw_data[(raw_start_timestamp & raw_end_timestamp),raw_index[0]]
#     ax2 = plt.plot(raw_x, raw_y)
#     return(ax2)



#maybe breaking it into smaller peices is better? 

#This function- pass in a dictionary of all listed variables. 
#nwb file, time to start/length of segment relative to run 1 times. 
#a single electrode id, and the data corresponding to the type (can take in raw, lfp or theta filtered in the dict)
#returns timestamps and data masked by those timestamps, for plotting. 

# def plot_overlay(x0,y0,lab0,color0,x1,y1,lab1,color1,title,xlab,ylab, offset= None, x2=None,y2=None,lab2=None,color2=None, x3=None,y3=None,lab3=None,color3=None): 
#     plt.figure(figsize=(40,12))
#     plt.plot(x0,y0,label=lab0,color=color0)
#     plt.plot(x1,y1+offset,label=lab1,color=color1,alpha=.5)
#     if x2 is not None:
#         plt.plot(x2,y2-offset,label=lab2,color=color2)
#     if x3 is not None:
#         plt.plot(x3,y3+2*offset,label = lab3,color=color3)
#     plt.legend()
#     plt.title(title,fontsize=25)
#     plt.xlabel(xlab,fontsize=25)
#     plt.ylabel(ylab,fontsize=25)
#     plt.xticks(fontsize=25)
#     plt.yticks(fontsize=25)


    
    
# def plot_overlay_test(x0,y0,lab0,color0,x1,y1,lab1,color1,title,xlab,ylab,offset=None, **args):
#     plt.figure(figsize=(40,12))
#     plt.plot(x0,y0,label=lab0,color=color0)
#     plt.plot(x1,y1+offset,label=lab1,color=color1,alpha=.5)
#     if x2 is not None:
#         plt.plot(x2,y2-offset,label=lab2,color=color2)
#     if x3 is not None:
#         plt.plot(x3,y3+2*offset,label = lab3,color=color3)
#     plt.legend()
#     plt.title(title,fontsize=25)
#     plt.xlabel(xlab,fontsize=25)
#     plt.ylabel(ylab,fontsize=25)
#     plt.xticks(fontsize=25)
#     plt.yticks(fontsize=25)    
    
def plot_speed_versus_theta(x_speed, y_speed, theta_x, theta_y, title,xlab1,ylab1,xlab2,ylab2): 
    #this makes the specific plot of speed versus theta given the speed x and y and theta x and y. 
    fig, ax = plt.subplots(2, 1, figsize=(25, 7))
    ax[0].plot(x_speed, y_speed)
    ax[0].set_xlabel(xlab1, fontsize=18)
    ax[0].set_ylabel(ylab1, fontsize=18)
    ax[0].set_title(title, fontsize=28)

    ax[1].plot(theta_x, theta_y)
    ax[1].set_xlabel(xlab2, fontsize=18)
    ax[1].set_ylabel(ylab2, fontsize=18)

# #want to make a plot function that takes in times, electrode ids, gets the data (get_x_y), then then plots that data. 

# def plot_rips_x_elecs(ripple_times_df, ripple_ix, 

def get_x_y_timestamp_list(start_timestamp, end_timestamp, electrode_id, eseries, data_timestamps, data):
    #timestamps should be in the form of np arrays. just floats break it. 
    
    #this function does the same thing as get x_y_list but takes a specific timestamp instead, and should be condensed with the other function tomorrow
    x_elect=[]
    y_elect=[]
    for ix in range(len(electrode_id)):
            #get electrode id from list
        electrode_id_ix = electrode_id[ix]
            #get the index for that electrode id using the helper function 
        index = get_electrode_indices(eseries,[electrode_id_ix])
    #         # make the mask using the times 
    #         #apply the mask to the timestamps and data passed in. 
       
        mask_start_timestamp = data_timestamps>start_timestamp
        mask_end_timestamp = data_timestamps<end_timestamp
        x = data_timestamps[mask_start_timestamp & mask_end_timestamp]
        y = data[(mask_start_timestamp & mask_end_timestamp),index[0]]
#             #append a list that will be the return of the funciton, that one will iterate through in the notebook to use. 
        x_elect.append(x)
        y_elect.append(y)
    return (x_elect, y_elect)  

def get_speed_timestamp(start_timestamp, end_timestamp, position_info):
    #this function does the same thing as get x_y_list but takes a specific timestamp instead, and should be condensed with the other function tomorrow

        
    position_start_timestamp = (position_info.index>start_timestamp)
    position_end_timestamps = (position_info.index<end_timestamp)
    position_x = position_info.index[position_start_timestamp & position_end_timestamps]
    position_y = position_info.head_speed[position_start_timestamp & position_end_timestamps]
    
    return(position_x,position_y)



def get_pos_timestamp(start_timestamp, end_timestamp, position_info):
    #this function does the same thing as get x_y_list but takes a specific timestamp instead, and should be condensed with the other function tomorrow

    position_start_timestamp = (position_info.index>start_timestamp)
    position_end_timestamps = (position_info.index<end_timestamp)
    position_x = position_info.head_position_x[position_start_timestamp & position_end_timestamps]
    position_y = position_info.head_position_y[position_start_timestamp & position_end_timestamps]

    return(position_x,position_y)


def find_overlapping_times(data_list, interval_list=None):
    #this function will take datasets with different times(data_list), and will mask it based on interval_list
    #
    if interval_list is None:
        raise NotImplementedError
    
    else:
        ind_list = [None]*len(data_list)
        for ndx, data in enumerate(data_list):
            ind = np.full(data.size, False)
            for interval in interval_list:
                start_idx = np.where(data >= interval[0])[0][0]
                end_idx = np.where(data <= interval[1])[0][-1]
                ind[start_idx:end_idx+1] = True
            ind_list[ndx] = ind
    return ind_list


def ripple_detector(nwb_file_name, pos_interval, tetrode_ind): 
    lfp_sampling_rate = (LFPBand & {'nwb_file_name' : nwb_file_name,
                                   'filter_name' : 'Ripple 150-250 Hz'}).fetch('lfp_band_sampling_rate')

    # Get animal speed upsampled to LFP sampling rate
    lfp_pos_df = (IntervalPositionInfo & {'nwb_file_name' : nwb_file_name,
                                          'interval_list_name' : pos_interval,
                                          'position_info_param_name' : 'default_lfp'}).fetch1_dataframe()

    # Head speed in cm/s - because that is a param of the ripple detection detection 
    head_speed = np.asarray(lfp_pos_df.head_speed)

    # Position timestamps in s
    pos_time = lfp_pos_df.index

    # Get ripple-filtered LFP
    lfp_ripple_object = (LFPBand & {'nwb_file_name' : nwb_file_name,
                                    'filter_name' : 'Ripple 150-250 Hz'}).fetch_nwb()[0]

    # Ripple-filtered LFP in AD units
    # No need to convert to volts because ripple detection occurs on standardized data
    lfp_ripple = np.asarray( lfp_ripple_object['filtered_data'].data, dtype='double' )

    lfp_time = lfp_ripple_object['filtered_data'].timestamps[:]

    lfp_interval_list = (IntervalList & {'nwb_file_name' : nwb_file_name,
                                         'interval_list_name' : 'lfp valid times'}).fetch1('valid_times')
    position_interval_list = (IntervalList & {'nwb_file_name' : nwb_file_name,
                                              'interval_list_name' : pos_interval}).fetch1('valid_times')

    overlap_interval_list = interval_list_intersect(lfp_interval_list, position_interval_list)

    pos_ind, lfp_ind = find_overlapping_times([pos_time, lfp_time], overlap_interval_list)
#     ##############
    position_df = pd.DataFrame(pos_time[pos_ind],index =pos_time[pos_ind] )
    position_df['speed']= head_speed[pos_ind]

    # position_df.time
    position_df

    new_index = pd.Index(np.unique(np.concatenate(
                    (position_df.time, lfp_time[lfp_ind]))), name='time1')

    # new_index

    speed_df = (position_df.reindex(index=new_index)
                   .interpolate(method='linear')
                   .reindex(index=lfp_time[lfp_ind]))
    # speed_df
#     ##########
#     # lfp_electrode_ids = lfp_ripple_object['filtered_data'].electrodes[:].index
#     # electrode_groups = [(Electrode() & {'nwb_file_name' : nwb_file_name,
#     #                                     'electrode_id' : lfp_electrode}).fetch1('electrode_group_name')
#     #                      for lfp_electrode in lfp_electrode_ids]

#     # tetrode_ind = np.asarray( [int(group_name) <= 23 for group_name in electrode_groups] )
#     # tetrode_ind[[4, 5, 17]] = False
#     # tetrode_ind


    ripple_data_ind = np.ix_(lfp_ind, tetrode_ind)
    ripple_data_ind
    ########
    ripple_times_df = Kay_ripple_detector(lfp_time[lfp_ind],
                                          lfp_ripple[ripple_data_ind],
                                          np.array(speed_df.speed.to_list()),
                                          lfp_sampling_rate,
                                          speed_threshold=4.0,
                                          minimum_duration=0.015,
                                          zscore_threshold=2.0,
                                          smoothing_sigma=0.004,
                                          close_ripple_threshold=0.0)
    return(ripple_times_df)