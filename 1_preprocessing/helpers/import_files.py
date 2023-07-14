from rich import print
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from os.path import exists
import pickle

# this class contains file import helpers
#############################################

class Import():

    def open_pickle(file_path):
        """
        This function opens a pickle file
        :param file_path: path to file
        """
        # open pickle
        f = open(file_path, 'rb')
        df = pickle.load(f)
        f.close()
        print(df)
        return df
    
    def store_pickle(obj, file_path):
        """
        This function stores a pickle file
        :param file_path: path to file
        """
        # store pickle
        f = open(file_path, 'wb')
        pickle.dump(obj, f)
        f.close()
    
    def csv_import_to_pandas(file_path, df_name, print_df=False):
        """
        This function imports a csv file to pandas
        :param file_path: path to file
        :param df_name: name of final df
        """
        df_x = pd.read_csv(file_path)
        if print_df:
            print(df_x)
            # print sampling rate
            print('Sampling rate (Hz), assuming that time values are seconds:', int(len(df_x)/df_x.Time[-1:]))
        return df_x

