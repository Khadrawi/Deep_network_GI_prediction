from rich import print
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# this class contains plotting helpers
#############################################

class Plotting():
        
    def spect(arr, sample_freq, y_max_freq, plot_title):
        """
        :param arr: input array
        :param sample_freq: sampling freq
        """
        plt.figure(figsize=(15,5)) 
        Pxx, freqs, bins, im = plt.specgram(x=arr, 
                                        Fs=sample_freq,
                                        NFFT=sample_freq*60,
                                        noverlap=sample_freq*30,
                                        # NFFT=2048*50,
                                        # noverlap=2048*40,
                                        scale_by_freq=True,
                                        mode='magnitude', # {'default', 'psd', 'magnitude', 'angle', 'phase'}
                                        scale='linear', # {'default', 'linear', 'dB'}
                                        cmap='coolwarm') # choose a color map

        y_min = 0
        y_max = y_max_freq
        plt.title(plot_title)
        plt.ylim(y_min, y_max)
        plt.tick_params(axis='both', which='major', labelsize=18)
        plt.ylabel('Freq (Hz)', size = 24, color='red')
        plt.twinx()
        plt.ylim(y_min, y_max*60)
        plt.ylabel('cpm', size = 24, color='red')
        plt.tick_params(axis='both', which='major', labelsize=18)
        plt.show()
            
    def psd(x, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None, detrend='constant', 
        show=True, plot_save=False, plot_name='test.svg', ax=None, scales='linear', xlim=None, units='V', title='', 
        tot_pwr_lst=[0.05, 0.25], brad_pwr_lst=[0.05, 0.127], norm_pwr_lst=[0.127, 0.19], tach_pwr_lst=[0.19, 0.25]):

        """Estimate power spectral density characteristcs using Welch's method.
        And, I added calculation of power in 
        * total range, [0.05, 0.25], 3 to 15 cpm
        * brady, [0.05, 0.127], # 3 to 7.6 cpm
        * normo, [0.127, 0.19], 7.6 to 11.6 cpm
        * tachy, [0.19, 0.25], 11.6 to 15 cpm

        based on:
        'Marcos Duarte, https://github.com/demotu/BMC'
        'tnorm.py v.1 2013/09/16'
        This function is just a wrap of the scipy.signal.welch function with
        estimation of some frequency characteristcs and a plot. For completeness,
        most of the help from scipy.signal.welch function is pasted here.
        Welch's method [1]_ computes an estimate of the power spectral density
        by dividing the data into overlapping segments, computing a modified
        periodogram for each segment and averaging the periodograms.
        Parameters
        ----------
        x : array_like
            Time series of measurement values
        fs : float, optional
            Sampling frequency of the `x` time series in units of Hz. Defaults
            to 1.0.
        window : str or tuple or array_like, optional
            Desired window to use. See `get_window` for a list of windows and
            required parameters. If `window` is array_like it will be used
            directly as the window and its length will be used for nperseg.
            Defaults to 'hanning'.
        nperseg : int, optional
            Length of each segment.  Defaults to half of `x` length.
        noverlap: int, optional
            Number of points to overlap between segments. If None,
            ``noverlap = nperseg / 2``.  Defaults to None.
        nfft : int, optional
            Length of the FFT used, if a zero padded FFT is desired.  If None,
            the FFT length is `nperseg`. Defaults to None.
        detrend : str or function, optional
            Specifies how to detrend each segment. If `detrend` is a string,
            it is passed as the ``type`` argument to `detrend`. If it is a
            function, it takes a segment and returns a detrended segment.
            Defaults to 'constant'.
        show : bool, optional (default = False)
            True (1) plots data in a matplotlib figure.
            False (0) to not plot.
        ax : a matplotlib.axes.Axes instance (default = None)
        scales : str, optional
            Specifies the type of scale for the plot; default is 'linear' which
            makes a plot with linear scaling on both the x and y axis.
            Use 'semilogy' to plot with log scaling only on the y axis, 'semilogx'
            to plot with log scaling only on the x axis, and 'loglog' to plot with
            log scaling on both the x and y axis.
        xlim : float, optional
            Specifies the limit for the `x` axis; use as [xmin, xmax].
            The defaukt is `None` which sets xlim to [0, Fniquist].
        units : str, optional
            Specifies the units of `x`; default is 'V'.
        Returns
        -------
        Fpcntile : 1D array
            frequency percentiles of the power spectral density
            For example, Fpcntile[50] gives the median power frequency in Hz.
        mpf : float
            Mean power frequency in Hz.
        fmax : float
            Maximum power frequency in Hz.
        Ptotal : float
            Total power in `units` squared.
        f : 1D array
            Array of sample frequencies in Hz.
        P : 1D array
            Power spectral density or power spectrum of x.
        See Also
        --------
        scipy.signal.welch
        Notes
        -----
        An appropriate amount of overlap will depend on the choice of window
        and on your requirements.  For the default 'hanning' window an
        overlap of 50% is a reasonable trade off between accurately estimating
        the signal power, while not over counting any of the data.  Narrower
        windows may require a larger overlap.
        If `noverlap` is 0, this method is equivalent to Bartlett's method [2]_.
        References
        ----------
        .. [1] P. Welch, "The use of the fast Fourier transform for the
               estimation of power spectra: A method based on time averaging
               over short, modified periodograms", IEEE Trans. Audio
               Electroacoust. vol. 15, pp. 70-73, 1967.
        .. [2] M.S. Bartlett, "Periodogram Analysis and Continuous Spectra",
               Biometrika, vol. 37, pp. 1-16, 1950.
        Examples (also from scipy.signal.welch)
        --------
        >>> import numpy as np
        >>> from psd import psd
        #Generate a test signal, a 2 Vrms sine wave at 1234 Hz, corrupted by
        # 0.001 V**2/Hz of white noise sampled at 10 kHz and calculate the PSD:
        >>> fs = 10e3
        >>> N = 1e5
        >>> amp = 2*np.sqrt(2)
        >>> freq = 1234.0
        >>> noise_power = 0.001 * fs / 2
        >>> time = np.arange(N) / fs
        >>> x = amp*np.sin(2*np.pi*freq*time)
        >>> x += np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
        >>> psd(x, fs=freq);
        """

        # calculate power
        ################################

        from scipy import signal, integrate

        if not nperseg:
            nperseg = np.ceil(len(x) / 2)
        f, P = signal.welch(x, fs, window, nperseg, noverlap, nfft, detrend)
        Area = integrate.cumtrapz(P, f, initial=0)
        Ptotal = Area[-1]
        mpf = integrate.trapz(f * P, f) / Ptotal  # mean power frequency
        fmax = f[np.argmax(P)]
        # frequency percentiles
        inds = [0]
        Area = 100 * Area / Ptotal  # + 10 * np.finfo(np.float).eps
        for i in range(1, 101):
            inds.append(np.argmax(Area[inds[-1]:] >= i) + inds[-1])
        fpcntile = f[inds]

        # make metrics
        ################################

        # make df of powers
        df_power = pd.DataFrame()
        df_power['freq'] = f
        df_power['power'] = P

        # get all power values within range
        df_totp = df_power[(df_power.freq >= tot_pwr_lst[0]) & (df_power.freq <= tot_pwr_lst[1])] 
        totp = df_totp['power'].sum()
        try:
            totf = df_totp[(df_totp['power'] == df_totp['power'].max())]['freq'].values[0]
        except:
            totf = np.nan
        print('Total power =', totp, 'Dom freq =', totf)

        # get power values in the bradygastria range
        df_bradp = df_power[(df_power.freq >= brad_pwr_lst[0]) & (df_power.freq <= brad_pwr_lst[1])] 
        bradp = df_bradp['power'].sum()
        try:
            bradf = df_bradp[(df_bradp['power'] == df_bradp['power'].max())]['freq'].values[0]
        except:
            bradf = np.nan
        print('Bradygastric power =', bradp, 'Dom freq =', bradf)

        # get power values in the normogastric range
        df_normp = df_power[(df_power.freq >= norm_pwr_lst[0]) & (df_power.freq <= norm_pwr_lst[1])]
        normp = df_normp['power'].sum()
        try:
            normf = df_normp[(df_normp['power'] == df_normp['power'].max())]['freq'].values[0]
        except:
            normf = np.nan
        print('normygastric power =', normp, 'Dom freq =', normf)

        # get power values in the tachygastria range
        df_tachp = df_power[(df_power.freq >= tach_pwr_lst[0]) & (df_power.freq <= tach_pwr_lst[1])]
        tachp = df_tachp['power'].sum()
        try:
            tachf = df_tachp[(df_tachp['power'] == df_tachp['power'].max())]['freq'].values[0]
        except:
            tachf = np.nan
        print('tachygastric power =', tachp, 'Dom freq =', tachf)

        # make df
        final_dict = {'tot_power': totp, 
                      'brad_power': bradp, 
                      'norm_power': normp, 
                      'tach_power': tachp, 
                      'brad_domf': bradf, 
                      'norm_domf': normf, 
                      'tach_domf': tachf}

        df_final = pd.DataFrame([final_dict])
        df_final = round(df_final, 4)

        # plot
        ################################

        # make plot
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        if scales.lower() == 'semilogy' or scales.lower() == 'loglog':
            ax.set_yscale('log')
        if scales.lower() == 'semilogx' or scales.lower() == 'loglog':
            ax.set_xscale('log')
        plt.plot(f, P, linewidth=2)
        ylim = ax.get_ylim()
        plt.plot([fmax, fmax], [np.max(P), np.max(P)], 'ro',
                 label='Fpeak  = %.2f' % fmax)
        plt.plot([fpcntile[50], fpcntile[50]], ylim, 'r', lw=1.5,
                 label='F50%%   = %.2f' % fpcntile[50])
        plt.plot([mpf, mpf], ylim, 'r--', lw=1.5,
                 label='Fmean = %.2f' % mpf)
        plt.plot([fpcntile[95], fpcntile[95]], ylim, 'r-.', lw=2,
                 label='F95%%   = %.2f' % fpcntile[95])
        leg = ax.legend(loc='best', numpoints=1, framealpha=.5,
                        title='Frequencies [Hz]')
        plt.setp(leg.get_title(), fontsize=12)
        plt.xlabel('Frequency [$Hz$]', fontsize=12)
        plt.ylabel('Magnitude [%s$^2/Hz$]' % units, fontsize=12)
        plt.title(title+' - Power spectral density', fontsize=12)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        plt.tight_layout()
        plt.grid()
        
        if plot_save:
            plt.savefig(plot_name)
        
        plt.show()
        
        return df_final
