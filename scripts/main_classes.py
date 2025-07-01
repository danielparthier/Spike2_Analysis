import os
from glob import glob
import numpy as np
from typing import List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import neo
from neo.io import Spike2IO
from scipy import signal


def find_smr_files(directory: str) -> List[str]:
    """
    Find all files ending with '.smr' in the given directory recursively.
    """
    # Expand the '~' to the full home directory path
    expanded_directory = os.path.expanduser(directory)
    return glob(f"{expanded_directory}/**/*.smr", recursive=True)

def load_smr_file(file_name: str) -> neo.core.Block:
    """
    Load a .smr file using Neo.
    """
    reader = Spike2IO(file_name)
    return reader.read_block()

def plot_channels(data: np.ndarray, time: np.ndarray) -> None:
    fig, axes = plt.subplots(len(data), 1, figsize=(10, 6), sharex=True)
    for i, channel in enumerate(data):
        axes[i].plot(time, channel)
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

class DataSet:
    def __init__(self, directory: str):
        self.directory = directory
        self.smr_files = self._get_smr_files()
        self.power_data: pd.DataFrame | None = None
        self.gamma_data: pd.DataFrame | None = None

    def _get_smr_files(self):
        return find_smr_files(self.directory)


class TraceData:
    def __init__(self, file_name: str, notch: float|None = None) -> None:
        self.time: np.ndarray = np.array([])
        self.trace: np.ndarray = np.array([])
        self.sampling_rate: float | None = None
        self.time_unit: str = 's'
        self.trace_unit: str = 'mV'
        self.ripple_trace: np.ndarray | None = None
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
                b, a = signal.iirnotch(notch, Q=70, fs=self.sampling_rate)
                self.trace = signal.filtfilt(b, a, self.trace, axis=1)
            else:
                print("No notch filter applied or invalid notch frequency provided.")

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

    def ripple_filter(self, lowcut: float = 80.0, highcut: float = 250.0, order: int = 4):
        """
        Apply a bandpass filter to the trace data to isolate ripple frequencies.
        """
        if self.trace is None or self.time is None:
            raise ValueError("Trace data or time data is not available. Cannot filter.")
        if self.sampling_rate is None:
            raise ValueError("Sampling rate is not set. Cannot filter.")
        
        nyquist = 0.5 * self.sampling_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(order, [low, high], btype='bandpass', analog=True)
        self.ripple_trace = signal.filtfilt(b, a, self.trace, axis=1)
    
    def gamma_filter(self, lowcut: float = 30.0, highcut: float = 90.0, order: int = 4):
        """
        Apply a bandpass filter to the trace data to isolate gamma frequencies.
        """
        if len(self.trace) == 0 or len(self.time) == 0:
            raise ValueError("Trace data or time data is not available. Cannot filter.")
        if self.sampling_rate is None:
            raise ValueError("Sampling rate is not set. Cannot filter.")

        nyquist = 0.5 * self.sampling_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(order, [low, high], btype='bandpass', analog=True)
        self.gamma_trace = signal.filtfilt(b, a, self.trace, axis=1)

    def plot_ripple(self) -> None:
        """20
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
            freqs, psd = signal.welch(channel, fs=self.sampling_rate, nperseg=nperseg, average='mean', scaling='spectrum', axis=1)
            psd_df = pd.concat([
                pd.DataFrame({
                    "Frequency": freqs,
                    "Power": power_spec,
                    "SegmentTime": j * self.window_size,
                    "Channel": i + 1,
                    "Date": self.date,
                    "Recording": self.recording,
                    "FileName": self.file_name,
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



file_names = find_smr_files("data/")  # Example usage, replace with your directory
data_set = DataSet("data/")
data_set.smr_files
trace_data = TraceData(file_names[0], notch=50)  # Example usage, replace with your file name and notch frequency
trace_data.downsample(2500)  # Downsample to 1000 Hz
trace_data1 = TraceData(file_names[1], notch=50)  # Load another trace data file
trace_data1.downsample(2500)  # Downsample to 1000 Hz
trace_view = TraceView(trace_data, window_start=0, window_size=60)  # Create a view for the first 60 seconds
trace_view.calc_power_spectrum(nperseg=5000)  # Calculate power spectrum with nperseg of 2048
trace_view1 = TraceView(trace_data1, window_start=0, window_size=60)  # Create a view for the first 60 seconds
trace_view1.calc_power_spectrum(nperseg=5000)  # Calculate power spectrum with nperseg of 2048


trace_view1.update_segment_time(trace_view)  # Update segment time with the first trace view
trace_view1.power_df
trace_view.power_df

plot_power_spectrum(combine_power_spectra([trace_view.power_df, trace_view1.power_df]))

trace_data.plot()
trace_data.ripple_filter()  # Apply ripple filter
trace_data.plot()  # Plot the filtered trace data
trace_data.plot_ripple()  # Plot the ripple filtered trace data
trace_data.gamma_filter()  # Apply gamma filter
trace_data.plot_gamma()  # Plot the gamma filtered trace data

first_df = trace_view.power_df[trace_view.power_df["Channel"]==1 & (trace_view.power_df["Index"] ==57)]  # Display first 10 rows of power data for channel 1 and segment time 0
first_df = trace_view.power_df[trace_view.power_df["Channel"]==2]  # Display first 10 rows of power data for channel 1 and segment time 0

# find peaks in the power spectrum of channel 1 at segment time 0 between 10 and 60 Hz
peaks, _ = signal.find_peaks(first_df[first_df["Frequency"].between(5, 90)]["Power"], height=0.00001, distance=20)


max_peak = peaks[first_df[first_df["Frequency"].between(5, 90)]["Power"].values[peaks].argmax()]



fwhm = calculate_fwhm(first_df, peaks[0])
# frequency of peaks
print("Peak Frequencies:", first_df["Frequency"][peaks].values)

plot_power_spectrum(first_df, xlim=(0,200), alpha=1)
plot_power_spectrum(trace_view1.power_df, xlim=(0,200), alpha=1)

first_df["Frequency"][peaks]

