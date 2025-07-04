import os
from glob import glob
import numpy as np
from typing import List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import neo
from neo.io import Spike2IO
from scipy import signal
from neo import Block
import scipy.stats as stats
from progress.bar import Bar
import sys



def find_smr_files(directory: str) -> List[str]:
    """
    Find all files ending with '.smr' in the given directory recursively.
    """
    # Expand the '~' to the full home directory path
    expanded_directory = os.path.expanduser(directory)
    file_list = glob(f"{expanded_directory}/**/*.smr", recursive=True)
    file_list.sort()
    return file_list

def load_smr_file(file_name: str) -> neo.Block:
    """
    Load a .smr file using Neo.
    """
    reader = Spike2IO(file_name)
    return reader.read_block()

def band_pass_filter(lowcut: float, highcut: float, fs: float, order: int = 1): #-> Tuple[np.ndarray, np.ndarray]:
    """
    Create a bandpass filter using Butterworth design.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    if isinstance(low, float) and isinstance(high, float):
        b, a = signal.butter(order, [low, high], btype='bandpass')
        return b, a
    return np.array([]), np.array([])

def lfp_processing(trace: np.ndarray) -> dict:
        trace_dict = {"trace": trace}
        hilbert_trace = signal.hilbert(trace_dict["trace"], axis=1)
        trace_dict.update(amplitude = np.abs(hilbert_trace))
        trace_dict.update(phase = np.angle(hilbert_trace))
        return trace_dict

def plot_channels(data: np.ndarray|dict, time: np.ndarray) -> None:
    if isinstance(data, dict):
        trace_data = data["trace"]
    else:
        trace_data = data
    fig, axes = plt.subplots(len(trace_data), 1, figsize=(10, 6), sharex=True)
    for i, channel in enumerate(trace_data):
        axes[i].plot(time, channel)
        if isinstance(data, dict) and "amplitude" in data:
            axes[i].plot(time, data["amplitude"][i], color='orange', label='Amplitude')
            if "sws" in data:
                axes[i].scatter(data["sws"]["peak_time"][i], data["sws"]["peak_amplitude"][i], color='red', marker='.')
        axes[i].set_ylabel(f'Channel {i + 1} (mV)')
    axes[-1].set_xlabel('Time (s)')
    fig.tight_layout()
    fig.show()

def plot_power_spectrum(power_df: pd.DataFrame, xlim: Tuple[float, float]= (0.0,100.0), alpha: float=0.2) -> None:
    """
    Plot the power spectrum from the power DataFrame.
    """
    channel_count = power_df["Channel"].nunique()
    fig, axes = plt.subplots(channel_count, 1, figsize=(10, 6), sharex=True)
    # Group by channel and SegmentTime
    cmap = plt.get_cmap('viridis', power_df['Index'].nunique())
    if channel_count == 1:
        axes = [axes]
    for channel_selection, channel in enumerate(power_df["Channel"].unique()):
        grouped_results = power_df[power_df["Channel"] == channel].groupby(['SegmentTime', 'Index'])
        for i, group in enumerate(grouped_results):
            segment_time, index = group[0]
            df = group[1]
            axes[channel_selection].plot(df['Frequency'], df['Power'], color=cmap(index), alpha=alpha)
    axes[-1].set_xlabel('Frequency (Hz)')
    axes[-1].set_ylabel('Power (dB)')
    axes[-1].set_xlim(xlim)
    fig.tight_layout()
    fig.show()

def combine_power_spectra(power_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Combine multiple power spectrum DataFrames into one.
    """
    combined = pd.concat(power_dfs, ignore_index=True)
    return combined

class TraceData:
    def __init__(self, file_name: str, notch: float|None = None, downsampling_frequency: float|None = None) -> None:
        self.time: np.ndarray = np.array([])
        self.trace: np.ndarray = np.array([])
        self.sampling_rate: float | None = None
        self.time_unit: str = 's'
        self.trace_unit: str = 'mV'
        self.ripple_trace: dict | None = None
        self.gamma_trace: dict | None = None
        self.sharp_wave_trace: dict | None = None
        self.notch: float | None = None
        self.file_name: str = file_name
        self.channel_count: int = 0

        self.date = self.file_name.split('/')[-1].split("_")[0]  # Get the date from the file name
        self.recording = self.file_name.split('/')[-1].split("_")[1].removesuffix(".smr")  # Get the recording from the file name
        block = load_smr_file(file_name)
        if block.segments:
            segment = block.segments[0]
            if segment.analogsignals:
                trace = segment.analogsignals[0]
                self.time = trace.times.rescale('s').magnitude
                self.trace = trace.rescale("mV").magnitude.T
                if len(self.time) == 0 or len(self.trace) == 0:
                    raise ValueError("Time or trace data is not available in the file.")
                self.sampling_rate = float(np.round(1 / np.median(np.diff(self.time))))
                self.channel_count = self.trace.shape[0]
            if isinstance(notch, (int, float)) and notch > 0 and self.sampling_rate is not None:
                self.notch = notch
                b, a = signal.iirnotch(notch, Q=40, fs=self.sampling_rate)
                self.trace = signal.filtfilt(b, a, self.trace, axis=1)
            else:
                print("No notch filter applied or invalid notch frequency provided.")
        if downsampling_frequency is not None:
            self.downsample(downsampling_frequency)

    def downsample(self, frequency: float) -> None:
        """
        Downsample the trace data to the specified frequency.
        """
        if self.sampling_rate is None:
            raise ValueError("Sampling rate is not set. Cannot downsample.")

        if frequency >= self.sampling_rate:
            raise ValueError("Downsampling frequency must be lower than the current sampling rate.")
        
        if self.trace is None or self.time is None:
            raise ValueError("Trace data or time data is not available. Cannot downsample.")

        factor = int(self.sampling_rate / frequency)
        if factor < 1:
            raise ValueError("Downsampling factor must be at least 1.")
        
        self.trace = signal.decimate(self.trace, factor, axis=1)
        self.time = self.time[::factor]
        self.sampling_rate = frequency

    def ripple_filter(self, lowcut: float = 150.0, highcut: float = 250.0, order: int = 1):
        """
        Apply a bandpass filter to the trace data to isolate ripple frequencies.
        """
        if self.trace is None or self.time is None:
            raise ValueError("Trace data or time data is not available. Cannot filter.")
        if self.sampling_rate is None:
            raise ValueError("Sampling rate is not set. Cannot filter.")
        b, a = band_pass_filter(lowcut, highcut, self.sampling_rate, order=order)

        self.ripple_trace = lfp_processing(signal.filtfilt(b, a, self.trace, axis=1))

    
    def gamma_filter(self, lowcut: float = 30.0, highcut: float = 90.0, order: int = 1):
        """
        Apply a bandpass filter to the trace data to isolate gamma frequencies.
        """
        if len(self.trace) == 0 or len(self.time) == 0:
            raise ValueError("Trace data or time data is not available. Cannot filter.")
        if self.sampling_rate is None:
            raise ValueError("Sampling rate is not set. Cannot filter.")
        b, a = band_pass_filter(lowcut, highcut, self.sampling_rate, order=order)
        self.gamma_trace = lfp_processing(signal.filtfilt(b, a, self.trace, axis=1))

    def sharp_wave_filter(self, lowcut: float = 5.0, highcut: float = 40.0, order: int = 1):
        """
        Apply a bandpass filter to the trace data to isolate sharp wave frequencies.
        """
        if self.trace is None or self.time is None:
            raise ValueError("Trace data or time data is not available. Cannot filter.")
        if self.sampling_rate is None:
            raise ValueError("Sampling rate is not set. Cannot filter.")
        b, a = band_pass_filter(lowcut, highcut, self.sampling_rate, order=order)
        self.sharp_wave_trace = lfp_processing(signal.filtfilt(b, a, self.trace, axis=1))
        zscore_sw = stats.zscore(self.sharp_wave_trace["amplitude"], axis=1)
        # find peaks in channels with minimum of 50 ms inbetween peaks and minimum height of 3
        #      
        peaks_results = [signal.find_peaks(channel_trace, height=3, distance=self.sampling_rate // 20) for channel_trace in zscore_sw]
        peaks, _ = zip(*peaks_results)
        self.sharp_wave_trace["sws"] = {"index": peaks,
                                        "peak_time": [self.time[p] for p in peaks],
                                        "peak_amplitude": [self.sharp_wave_trace["amplitude"][i][p] for i, p in enumerate(peaks)]}

    def plot_ripple(self) -> None:
        """
        Plot the ripple filtered trace data.
        """
        if hasattr(self, 'ripple_trace') and self.ripple_trace is not None and self.time is not None:
            plot_channels(self.ripple_trace, self.time)

    def plot_gamma(self) -> None:
        """
        Plot the gamma filtered trace data.
        """
        if hasattr(self, 'gamma_trace') and self.gamma_trace is not None and self.time is not None:
            plot_channels(self.gamma_trace, self.time)

    def plot_sharp_wave(self) -> None:
        """
        Plot the sharp wave filtered trace data.
        """
        if hasattr(self, 'sharp_wave_trace') and self.sharp_wave_trace is not None and self.time is not None:
            plot_channels(self.sharp_wave_trace, self.time)

    def extract_sws(self, window_size: float = 100.0) -> List[dict]:
        """
        Extract sharp wave data from the sharp wave trace.
        """
        if not hasattr(self, 'sharp_wave_trace') or self.sharp_wave_trace is None:
            raise ValueError("Sharp wave trace is not available. Cannot extract SWS.")
        if "sws" not in self.sharp_wave_trace:
            raise ValueError("SWS data is not available in the sharp wave trace.")
        def get_cutout(trace, ripple_trace, index, width):
            if hasattr(self, 'sharp_wave_trace') and self.sharp_wave_trace is not None:
                if "amplitude" not in self.sharp_wave_trace:
                    raise ValueError("Amplitude data is not available in the sharp wave trace.")
                if len(self.sharp_wave_trace["amplitude"]) == 0:
                    raise ValueError("Amplitude data is empty. Cannot extract SWS.")
            sws_array = np.zeros((len(index), width-1))
            ripple_array = np.zeros((len(index), width-1))
            for i, idx in enumerate(index):
                sws_array[i] = trace[idx - width // 2: idx + width // 2]
                ripple_array[i] = ripple_trace[idx - width // 2: idx + width // 2]
            return {"sws": sws_array, "ripple": ripple_array}
        if self.sampling_rate is None:
            raise ValueError("Sampling rate is not set. Cannot extract SWS.")
        width = int(window_size * self.sampling_rate // 1000)  # Convert window size from ms to samples
        if self.ripple_trace is None or "trace" not in self.ripple_trace:
            raise ValueError("Ripple trace is not available. Cannot extract SWS.")
        return [get_cutout(self.trace[channel_index], self.ripple_trace["trace"][channel_index], index, width) for channel_index, index in enumerate(self.sharp_wave_trace["sws"]["index"])]

    def plot(self) -> None:
        """
        Plot the trace data.
        """
        if len(self.trace) == 0 or len(self.time) == 0:
            raise ValueError("Trace data or time data is not available. Cannot plot.")
        plot_channels(self.trace, self.time)


class TraceView:
    def __init__(self, data: TraceData, window_start: float, window_size: float):
        self.window_start = window_start
        self.window_size = window_size
        self.channel_count = data.channel_count
        self.recording = data.recording
        self.date = data.date
        if data.sampling_rate is not None:
            self.sampling_rate = data.sampling_rate
        self.file_name = data.file_name
        if data.trace is None or data.time is None:
            raise ValueError("Trace data or time data is not available. Cannot create TraceView.")
        windowlength = self.sampling_rate * window_size
        self.trace = data.trace[:, :int(data.time.size // windowlength * windowlength)].reshape(self.channel_count, -1, int(windowlength))
        self.time = data.time[:int(data.time.size // windowlength * windowlength)].reshape(-1, int(windowlength))
    
    def plot(self):
        """
        Plot the trace data within the specified time window.
        """
        if len(self.trace) == 0 or len(self.time) == 0:
            raise ValueError("Trace data or time data is not available. Cannot plot.")
        plot_channels(self.trace, self.time)
        plt.title(f"Trace View: {self.file_name} [{self.window_start}s to {self.window_start + self.window_size}s]")
        plt.show()

    def calc_power_spectrum(self, nperseg: int = 1024):
        """
        Calculate the power spectrum of the trace data within the specified time window.
        """
        if len(self.trace) == 0 or len(self.time) == 0:
            raise ValueError("Trace data or time data is not available. Cannot calculate power spectrum.")
        
        self.power_df = pd.DataFrame(columns=["Frequency", "Power", "SegmentTime", "Channel"])
        for i, channel in enumerate(self.trace):
            freqs, psd = signal.welch(channel, fs=self.sampling_rate, nperseg=nperseg, average='mean', scaling='density', axis=1)
            psd_df = pd.concat([
                pd.DataFrame({
                    "Frequency": freqs,
                    "Power": power_spec,
                    "SegmentTime": j * self.window_size,
                    "Channel": i + 1,
                    "Date": self.date,
                    "Recording": self.recording,
                    "FileName": self.file_name,
                    "SegmentTimeFile": self.time[j, 0],
                    "Index": j
                })
                for j, power_spec in enumerate(psd)
            ])
            psd_df.dropna(how='any', inplace=True)
            if i == 0:
                self.power_df = psd_df
            else:
                self.power_df = pd.concat([self.power_df, psd_df], ignore_index=True)

    def update_segment_time(self, trace_view: 'TraceView'):
        """
        Merge power data from another TraceView instance.
        """
        if not isinstance(trace_view, TraceView):
            raise ValueError("Other must be an instance of TraceView.")
        if self.power_df is None or trace_view.power_df is None:
            raise ValueError("Power data is not available in one of the TraceView instances.")
        self.power_df["SegmentTime"] += trace_view.power_df["SegmentTime"].max() + self.window_size
        self.power_df["Index"] += trace_view.power_df["Index"].max() + 1
    
    def merge_concentration_data(self, concentration_data: pd.DataFrame) -> None:
        """
        Merge concentration data with the power DataFrame.
        """

        if self.power_df is None:
            raise ValueError("Power data has not been calculated. Call calc_power_spectrum() first.")
        if not isinstance(concentration_data, pd.DataFrame):
            raise ValueError("Concentration data must be a pandas DataFrame.")

        concentration_data = concentration_data[concentration_data["Recording_file"].str.contains(self.file_name.split("/")[-1])]
        # assign kainate concentration to the power_df when time_wash-in is later or equal the SegmentTimeFile 
        concentration_df = concentration_data[concentration_data["Recording_file"].str.contains(self.file_name.split("/")[-1])].copy()
        # get column which contain string washin
        wash_in_col = concentration_data.columns[["wash" in col.lower() for col in concentration_data.columns]][0]
        concentration_df.sort_values(by=wash_in_col, inplace=True, ascending=False, ignore_index=True)  # Sort by time_wash-in in descending order
        last_time_point = self.power_df["SegmentTimeFile"].max()  # Initialize last_time_point with the start time of the first trace
        self.power_df["Kainate_concentration"] = 0
        for i, row in concentration_df.iterrows():
            time_wash_in = row[wash_in_col]
            kainate_conc = row["Kainate_concentration"]
            if i == 0:
                edge_include = "both"
            else:
                edge_include = "left"
            self.power_df.loc[self.power_df["SegmentTimeFile"].between(
                time_wash_in, last_time_point, inclusive=edge_include), "Kainate_concentration"] = kainate_conc  # Set the concentration for the time range
            last_time_point = time_wash_in


class DataSet:
    def __init__(self, directory: str):
        self.directory = directory
        self.smr_files = self._get_smr_files()
        self.power_data: pd.DataFrame | None = None
        self.gamma_data: pd.DataFrame | None = None

    def _get_smr_files(self):
        return find_smr_files(self.directory)

    def load_trace_data(self, notch: float | None = None, downsampling_frequency: float | None = None) -> None:
        if not self.smr_files:
            print("No .smr files found.")
            return None
        bar = Bar('Loading files...', max=len(self.smr_files), suffix='%(percent)d%%')
        raw_list = []
        for file in self.smr_files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"File {file} does not exist.")
            raw_list.append(TraceData(file, notch=notch, downsampling_frequency=downsampling_frequency))
            sys.stdout.write(f"Processing file {file_i + 1}/{len(self.smr_files)}: {file_name}")
            sys.stdout.flush()
            bar.next()
        self.trace_data_raw = raw_list
        bar.finish()
        

    def to_trace_view(self, window_start: float = 0, window_size: float = 60, force: bool = False) -> List | None:
        """
        Convert the loaded trace data into TraceView objects.
        """
        if hasattr(self, 'trace_data') and isinstance(self.trace_data[0], TraceView) and hasattr(self, 'trace_data_raw'):
            if force:
                print("Converting trace data to TraceView objects...")
                self.trace_data = [TraceView(data, window_start, window_size) for data in self.trace_data_raw if isinstance(data, TraceData)]
            else:
                print("Trace data has already been converted to TraceView objects.")
                return self.trace_data
        if not hasattr(self, 'trace_data_raw'):
            raise ValueError("Trace data has not been loaded. Call load_trace_data() first.")
        elif isinstance(self.trace_data_raw[0], TraceData):
            self.trace_data = [TraceView(data, window_start, window_size) for data in self.trace_data_raw if isinstance(data, TraceData)]
            return None
    
    def combine_power_spectra(self, nperseg=1250) -> pd.DataFrame:
        """
        Combine power spectra from all TraceView objects into a single DataFrame.
        """
        if not hasattr(self, 'trace_data'):
            raise ValueError("Trace data has not been loaded. Call load_trace_data() first.")
        if not all(isinstance(tv, TraceView) for tv in self.trace_data):
            raise ValueError("All trace data must be converted to TraceView objects first.")
        
        last_group = ""
        last_tv = None
        for tv in self.trace_data:
            if isinstance(tv, TraceView):
                current_group = f"{tv.date}_{tv.recording}"
                tv.calc_power_spectrum(nperseg)  # Ensure power spectrum is calculated
                if current_group == last_group and last_tv is not None:
                    print(f"Continuing with {current_group}...")
                    tv.update_segment_time(last_tv)
                last_group = current_group
                last_tv = tv
        return pd.concat([tv.power_df for tv in self.trace_data if isinstance(tv, TraceView) and hasattr(tv, 'power_df')], ignore_index=True)

    def add_concentration_table(self, file_path: str) -> None:
        """
        Add a concentration table to the dataset.
        """
        concentration_data = pd.read_csv(file_path, sep=",")
        concentration_data["Date"] = concentration_data["Recording_file"].apply(lambda x: x.split("_")[0])
        concentration_data["Recording"] = concentration_data["Recording_file"].apply(lambda x: x.split("_")[1].removesuffix(".smr"))
        self.concentration_data = concentration_data
    
    def merge_concentration_data(self):
        """
        Merge concentration data with power data.
        """
        if not hasattr(self, "trace_data"):
            raise ValueError("Trace data has not been loaded. Call load_trace_data() first.")
        if not hasattr(self, 'concentration_data'):
            raise ValueError("Concentration data has not been added. Call add_concentration_table() first.")

        for tv in self.trace_data:
            if not hasattr(tv, 'power_df'):
                raise ValueError("Power data has not been calculated. Call combine_power_spectra() first.")
            if isinstance(tv, TraceView):
                tv.merge_concentration_data(self.concentration_data)
    
    def return_power_df(self) -> pd.DataFrame:
        """
        Return the combined power DataFrame from all TraceView objects.
        """
        if not hasattr(self, 'trace_data'):
            raise ValueError("Trace data has not been loaded. Call load_trace_data() first.")

        return pd.concat([tv.power_df for tv in self.trace_data if hasattr(tv, 'power_df') and isinstance(tv, TraceView)], ignore_index=True)

    def apply_fun_to_view(self, fun):
        for tv in self.trace_data:
            fun(tv)
    
    def apply_fun_to_raw_data(self, fun):
        """
        Apply a function to the raw trace data.
        """
        if not hasattr(self, 'trace_data_raw'):
            raise ValueError("Trace data has not been loaded. Call load_trace_data() first.")

        for tv in self.trace_data_raw:
            fun(tv)

    def power_df_only(self,
                    notch: float | None = 50,
                    downsampling_frequency: float | None = 1250,
                    window_start: float | None = 0,
                    window_size: float | None = 60,
                    nperseg: int | None = None) -> pd.DataFrame:
        """
        Return only the power DataFrame from the TraceView objects.
        """
        from copy import deepcopy
        power_df_out = pd.DataFrame()
        bar = Bar('Processing files...', max=len(self.smr_files), suffix='%(percent)d%%')
        last_group = ""
        last_tv = None
        for file_i, file_name in enumerate(self.smr_files):
            # make progressbar in terminal
            sys.stdout.write(f"Processing file {file_i + 1}/{len(self.smr_files)}: {file_name}")
            sys.stdout.flush()
            tv = TraceView(TraceData(file_name, notch=notch,
                                    downsampling_frequency=downsampling_frequency),
                                    window_start, window_size)
            current_group = f"{tv.date}_{tv.recording}"
            if nperseg is None:
                nperseg = int(tv.sampling_rate)
            tv.calc_power_spectrum(nperseg)  # Ensure power spectrum is calculated
            if current_group == last_group and last_tv is not None:
                tv.update_segment_time(last_tv)
            if self.concentration_data is not None and isinstance(self.concentration_data, pd.DataFrame):
                tv.merge_concentration_data(self.concentration_data)
            
            if file_i == 0:
                last_tv = tv
                power_df_out = tv.power_df
            else:
                power_df_out = pd.concat([power_df_out, tv.power_df], ignore_index=True)
            
            last_group = current_group
            last_tv = tv
            bar.next()
        bar.finish()
        return power_df_out



# calculate the full width at half maximum (FWHM) of the peak
def calculate_fwhm(df: pd.DataFrame, peak: int) -> np.ndarray:
    """    Calculate the full width at half maximum (FWHM) of the peaks in the power spectrum.
    Args:
        frequency (np.ndarray): The frequency values of the power spectrum.
        power (np.ndarray): The power values of the power spectrum.
        peaks (np.ndarray): The indices of the peaks in the power spectrum.
    Returns:
        np.ndarray: The FWHM values for each peak.
    """

    half_max = df["Power"].values[peak] / 2
    left_idx = np.where(df["Power"].values[:peak][::-1] < half_max)[0][0]
    right_idx = np.where(df["Power"].values[peak:] < half_max)[0][0]
    return df["Frequency"].values[right_idx] - df["Frequency"].values[left_idx]
