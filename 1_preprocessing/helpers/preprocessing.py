from scipy.signal import iirfilter, filtfilt, butter, savgol_filter, decimate
import numpy as np
import pandas as pd
from itertools import product
import missingno as msno
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# this class contains preprocessing helpers
#############################################

class Preprocessing:

    def downsample_with_anti_aliasing(arr, factor, ftype):
        """
        this function downsamples an arry after applying an anti-aliasing filter
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.decimate.html
        :param arr: array
        :param factor: downsample factor
        :param ftype: filter type, 'iir' or 'fir'
        """
        print("Original length =", len(arr))
        arr_new = decimate(x=arr, q=factor, ftype=ftype)
        print("New length =", len(arr_new))
        return arr_new

    def highpass_butter_filter(arr, order, sample_freq, cutoff_freq):
        """
        highpass butterworth filter
        :param arr: array
        :param order: filter order
        :param freq: cutoff frequency
        """
        # build filter (e.g., order 2, and 0.05 Hz or 3 cpm)
        b, a = butter(order, cutoff_freq, btype='highpass', analog=False, fs=sample_freq, output = "ba")
        # apply filter
        new_arr = filtfilt(b, a, arr)
        return new_arr

    def lowpass_butter_filter(arr, order, sample_freq, cutoff_freq):
        """
        highpass butterworth filter
        :param arr: array
        :param order: filter order
        :param freq: cutoff frequency
        """
        # build filter
        b, a = butter(order, cutoff_freq, btype='lowpass', analog=False, fs=sample_freq, output = "ba")
        # apply filter
        new_arr = filtfilt(b, a, arr)
        return new_arr
    
    def bandpass_butter_filter(arr, order, sample_freq, lowcut, highcut):
        """
        highpass butterworth filter
        :param arr: array
        :param order: filter order
        :param freq: cutoff frequency
        """
        # build filter
        b, a = butter(order, [lowcut, highcut], btype='bandpass', analog=False, fs=sample_freq, output = "ba")
        # apply filter
        new_arr = filtfilt(b, a, arr)
        return new_arr
    
    
    def savitzky_golay_filter(arr, win_len, order):
        # Auckland uses window = 1.7 sec and polynomial order of 9
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html
        new_arr = savgol_filter(x=arr, 
                                window_length=win_len, 
                                polyorder=order,
                                mode='interp')
        return new_arr
    
    def percent_nans_zeros_tolist(df):
        """
        make lists of zeros and nan % of df
        :param df: df
        """
        # get nans
        nans_lst = df.isna().sum().to_list()   
        for i in range(len(nans_lst)):
            nans_lst[i] = round((nans_lst[i] / len(df)) * 100, 5)

        # get zeros
        zeros_lst = (df == 0).sum().to_list()
        for i in range(len(zeros_lst)):
            zeros_lst[i] = round((zeros_lst[i] / len(df)) * 100, 5)

        return nans_lst, zeros_lst

    def remove_dropped_packets_zerofill(arr, thres=100000):
        """
        remove dropped packets and replace with nans
        :param thres: threshold in microvolts, pts
        """
        arr_copy = arr.copy(deep=True)

        # identify artifact locations
        mask_artifacts = ((arr >= thres) | (arr <= -thres))

        # replace artifacts with nan
        arr_copy[mask_artifacts] = 0
        df = pd.DataFrame(arr_copy)
        # data = pd.Series(signal_copy)

        # store signal w/o artifact windows
        arr_new = df.iloc[:, 0]

        # print the % of observations removed in the process
        removed_pct = mask_artifacts.sum()/len(arr_new)
        removed_pct = round(removed_pct * 100, 2)
        # print('Percentage removed: ' + str(removed_pct) + ' %')

        return df, removed_pct
    
    def artifacts_removal(arr, thres=2000, pts_pre=10, pts_post=10, interpolate=True, fill_frontend=False):
        """
        remove artifacts and interpolate points
        :param thres: threshold in microvolts, pts
        :param pts_pre: number of points to remove before threshold crossing
        :param pts_post: number of points to remove after threshold crossing
        :param interpoate: True or False
        """
        arr_copy = arr.copy(deep=True)

        # identify artifact locations
        mask_artifacts = ((arr >= thres) | (arr <= -thres))

        # replace artifacts with nan
        arr_copy[mask_artifacts] = np.nan
        df = pd.DataFrame(arr_copy)
        # data = pd.Series(signal_copy)     

        # remove pts before artifact
        for x in range(0, pts_pre):
            df[1] = arr_copy.shift(periods=-(x + 1), fill_value=0)
            mask = df[1].isnull() # <— create a Boolean mask
            df.iloc[:, 0][mask] = np.nan # <— replace with NaN based on mask
            df = df.drop([1], axis=1) # <— drop the shifted column

        # remove pts after artifact
        for x in range(0, pts_post):
            df[1] = arr_copy.shift(periods=(x + 1), fill_value=0)
            mask = df[1].isnull() # <— create a Boolean mask
            df.iloc[:, 0][mask] = np.nan # <— replace with NaN based on mask
            df = df.drop([1], axis=1)

        # store signal w/o artifact windows
        arr_new = df.iloc[:, 0]
        
        if interpolate:
            df = arr_new.interpolate(method='linear')
            print('w/ Linear interpolation')
        else:
            print('w/o interpolation')
        
        # use this for case where NaNs on the front end of the df need to be filled 
        if fill_frontend:
            df = df.fillna(method='backfill')
            print('w/ backfill on front end')
        else:
            print('w/o backfill on front end')

        # print the % of observations removed in the process
        removed_pct = np.sum(arr_new.isna())/len(arr_new)
        removed_pct = round(removed_pct * 100, 2)
        print('Percentage removed: ' + str(removed_pct) + ' %')

        return df, removed_pct