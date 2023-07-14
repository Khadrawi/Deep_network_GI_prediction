import os
import numpy as np
from pandas import read_csv
from scipy.signal import decimate
from scipy.interpolate import interp1d
import math

class data:
    # This class is used to read the preprocessed data and segment it into chunks of data to train on
    def __init__(self, dir_name, subjects = 'all', num_threshold=7200000) -> None:
        # num_threshold: make all data have same length of samples before preprocessing
        self.num_threshold = num_threshold
        self.data, self.used_subjects_names, self.channels, self.data_length = self.read_files(dir_name, subjects)
        
    def read_files(self, dname, subjects):

        # num_subjects -> how many subjects to use. Values are either 'all or a list of names to use ['58-28', '68-21', ...]
        if subjects=='all':
            filenames =  os.listdir(dname)
        else:
            filenames = subjects

        used_subjects =  filenames
        data = {fname:data_block() for fname in filenames}
        for filename in filenames:
            subfname = os.listdir(os.path.join(dname, filename))
            # read baseline data
            baseline_id = [ i for i in range(len(subfname)) if 'base' in subfname[i] ][0] 
            baseline_file = os.path.join(dname, filename, subfname[baseline_id])
            # read feeding data
            feeding_id = [ i for i in range(len(subfname)) if 'feed' in subfname[i] ][0]
            feeding_file = os.path.join(dname, filename, subfname[feeding_id])
            
            data[filename].open_files(baseline_file, feeding_file, self.num_threshold)
            channels= data[filename].channels
            data_length = data[filename].length

        return data, used_subjects, channels, data_length

    def segment(self, segment_length):
        # Used to cut each 1 hour signal into 1 minute chunks
        self.number_of_segments = math.floor(self.data_length/segment_length) # in which the basline/feeding data was divided
        self.segments = [ i*segment_length for i in range(self.number_of_segments)]
        self.segment_length = segment_length
        for i in self.used_subjects_names:
            # segment datablock
            self.data[i].segment_data(self.segments)

    def prepare_sets(self, used_segments):
        # used_segments = [x,y] Specify which segments to choose out of 60, data[x:y]; use this to remove first 15 
        # minutes from baseline and feeding, choose used_segments from [0:60] NOTE second number not inculded
        used_number_of_segments = used_segments[1] - used_segments[0]
        # Putting each subject on top of each other, each subject has feeding then baseline on top of each other
        self.subject_labels_dict = {key:i for i, key in enumerate(self.data.keys())}
        subject_labels = np.repeat(np.array(list(self.subject_labels_dict.keys())), used_number_of_segments, axis=0) # Putting feeding and baseline on top of each other
        subject_labels = np.tile(subject_labels, 2) # Putting feeding and baseline on top of each other
        # Adding baseline followed by feeding
        # Baseline label = 0, Feeding lable = 1
        tmp_label = np.repeat(np.array([0,1])[:,np.newaxis], used_number_of_segments, axis=0)
        labels = np.repeat(tmp_label, len(self.used_subjects_names), axis=0)
        data = np.zeros((len(labels), len(self.channels)-1, self.segment_length))
        # NOTE!!! time channel removed
        for i, subject in enumerate(self.used_subjects_names):
            data[2*i*used_number_of_segments:(2*i+1)*used_number_of_segments,:,:] = np.swapaxes(self.data[subject].baseline[used_segments[0]:used_segments[1],:,1:], 1,2)# remove time channel "1:" 
            data[(2*i+1)*used_number_of_segments:(2*(i+1))*used_number_of_segments,:,:] = np.swapaxes(self.data[subject].feeding[used_segments[0]:used_segments[1],:,1:], 1,2)
        return data, labels, subject_labels 

class data_block:
    # Contains all data of any subject
    def __init__(self) -> None:
        
        self.baseline = None
        self.feeding = None
        self.channels = None
        self.length = None
        
    def open_files(self, baseline_file, feeding_file, num_threshold, using_channels=None):
        # open feeding and basline files
        self.baseline, self.channels, self.length = self.read(baseline_file)
        self.feeding, _, _ = self.read(feeding_file)

    def read(self, file_dir):
        # read csv file
        frame = read_csv(file_dir)
        vals = frame.values[:720000] # corresponds to 1 hour of data @200hz
        channels = frame.columns
        length = vals.shape[0]
        return vals, channels, length
    
    def segment_data(self, segments):
        # segment baseline and feeding
        self.num_segments = len(segments)
        self.baseline = np.array(np.split(self.baseline, segments[1:], axis=0))
        self.feeding = np.array(np.split(self.feeding, segments[1:], axis=0))
