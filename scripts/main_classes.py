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
from copy import deepcopy
import sys
import pywt
from multiprocessing import Pool
from quantities import Quantity


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


def band_pass_filter(
    lowcut: float, highcut: float, fs: float, order: int = 1
):  # -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a bandpass filter using Butterworth design.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    if isinstance(low, float) and isinstance(high, float):
        b, a = signal.butter(
            order, [low, high], btype="bandpass"
        )  # pyright: ignore[reportGeneralTypeIssues]
        return b, a
    return np.array([]), np.array([])


def lfp_processing(trace: np.ndarray) -> dict:
    trace_dict = {"trace": trace}
    if trace.ndim == 1:
        hilbert_trace = signal.hilbert(trace_dict["trace"])
    else:
        hilbert_trace = signal.hilbert(trace_dict["trace"], axis=1)
    trace_dict.update(amplitude=np.abs(hilbert_trace))
    trace_dict.update(phase=np.angle(hilbert_trace))
    return trace_dict


def plot_channels(data: np.ndarray | dict, time: np.ndarray) -> None:
    if isinstance(data, dict):
        trace_data = data["trace"]
    else:
        trace_data = data
    fig, axes = plt.subplots(len(trace_data), 1, figsize=(10, 6), sharex=True)
    for i, channel in enumerate(trace_data):
        axes[i].plot(time, channel)
        if isinstance(data, dict) and "amplitude" in data:
            axes[i].plot(time, data["amplitude"][i], color="orange", label="Amplitude")
            if "sws" in data:
                axes[i].scatter(
                    data["sws"]["peak_time"][i],
                    data["sws"]["peak_amplitude"][i],
                    color="red",
                    marker=".",
                )
        axes[i].set_ylabel(f"Channel {i + 1} (mV)")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.show()


def plot_power_spectrum(
    power_df: pd.DataFrame, xlim: Tuple[float, float] = (0.0, 100.0), alpha: float = 0.2
) -> None:
    """
    Plot the power spectrum from the power DataFrame.
    """
    channel_count = power_df["Channel"].nunique()
    fig, axes = plt.subplots(channel_count, 1, figsize=(10, 6), sharex=True)
    # Group by channel and SegmentTime
    cmap = plt.get_cmap("viridis", power_df["Index"].nunique())
    if channel_count == 1:
        axes = [axes]
    for channel_selection, channel in enumerate(power_df["Channel"].unique()):
        grouped_results = power_df[power_df["Channel"] == channel].groupby(
            ["SegmentTime", "Index"]
        )
        for i, group in enumerate(grouped_results):
            segment_time, index = group[0]
            df = group[1]
            axes[channel_selection].plot(
                df["Frequency"], df["Power"], color=cmap(index), alpha=alpha
            )
    axes[-1].set_xlabel("Frequency (Hz)")
    axes[-1].set_ylabel("Power (dB)")
    axes[-1].set_xlim(xlim)
    fig.tight_layout()
    fig.show()


def combine_power_spectra(power_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Combine multiple power spectrum DataFrames into one.
    """
    combined = pd.concat(power_dfs, ignore_index=True)
    return combined


def wavelet_transform(
    sws_waves: np.ndarray,
    low: float = 150,
    high: float = 350,
    step_size: float = 0.1,
    sampling_frequency: float = 1250.0,
):
    wavelet = "cmor1.5-1.0"  # Define the wavelet type
    frequency = np.arange(
        start=low, stop=high, step=step_size
    )  # Define scales for the wavelet transform
    scales = pywt.frequency2scale(
        wavelet=wavelet, freq=frequency / sampling_frequency
    )  # Convert scales to frequencies in Hz
    wavelet_transform = pywt.cwt(
        sws_waves,
        wavelet=wavelet,
        scales=scales,
        sampling_period=1 / sampling_frequency,
        method="fft",
    )  # Perform the continuous wavelet transform
    return wavelet_transform


def get_ripple_power_frequency(
    sws_waves: np.ndarray,
    low: float = 150,
    high: float = 350,
    step_size: float = 0.1,
    sampling_frequency: float = 1250.0,
):
    wt = wavelet_transform(
        sws_waves,
        low=low,
        high=high,
        step_size=step_size,
        sampling_frequency=sampling_frequency,
    )
    frequency = np.arange(
        start=low, stop=high, step=step_size
    )  # Define scales for the wavelet transform
    wt_abs = np.abs(wt[0])
    max_vals_freq = wt_abs.max(axis=2)
    max_frequencies = frequency[
        max_vals_freq.argmax(axis=0)
    ]  # Frequencies of the maximum wavelet coefficients for each time point
    max_vals = max_vals_freq.max(
        axis=0
    )  # Maximum values of the wavelet coefficients for each time point
    z_scored_vals = (max_vals - wt_abs.mean(axis=(0, 2))) / wt_abs.std(axis=(0, 2))
    time = (
        np.arange(sws_waves.shape[1]) / sampling_frequency * 1000
    )  # Convert time to milliseconds
    time -= time[len(time) // 2]  # Center the time around zero
    ripple_peak_time = time[wt_abs.max(axis=0).argmax(axis=1)]

    return {
        "freq_max": max_frequencies,
        "power_max": max_vals,
        "z_score": z_scored_vals,
        "frequencies": wt[1],
        "power": wt_abs,
        "ripple_peak_time": ripple_peak_time,
    }


class SWR:
    def __init__(
        self,
        trace: np.ndarray = np.array([]),
        time: np.ndarray = np.array([]),
        sampling_rate: float = 1250,
        width: float = 100,
        cut_off_time: float | None = None,
        high_freq: float = 40,
        ripple_low: float = 100,
        ripple_high: float = 250,
        ripple_order: int = 1,
    ) -> None:
        self.index: np.ndarray = np.array([])
        self.peak_time: np.ndarray = np.array([])
        self.peak_amplitude: np.ndarray = np.array([])
        self.z_scored_power: np.ndarray | None = None
        self.ripple_frequency: np.ndarray | None = None
        self.ripple_power: np.ndarray | None = None
        self.wt_frequencies: np.ndarray | None = None
        self.wavelet_power: np.ndarray | None = None
        self.ripple_peak_time: np.ndarray | None = None
        self.duration: np.ndarray = np.array([])
        self.sampling_rate: float = sampling_rate
        self.start_time: float = time[0]
        self.end_time: float = time[-1]
        self.cut_off_time: float | None = cut_off_time
        self.swrs: dict = {}
        b, a = signal.butter(
            1, high_freq, btype="low", fs=self.sampling_rate
        )  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
        self.sharp_wave_trace = lfp_processing(signal.filtfilt(b, a, trace))
        b, a = band_pass_filter(
            ripple_low, ripple_high, self.sampling_rate, order=ripple_order
        )
        self.ripple_trace = lfp_processing(signal.filtfilt(b, a, trace))
        self.find_peaks(trace)

        if cut_off_time is not None:
            valid_indices = np.where(self.peak_time <= cut_off_time)[0]
            self.index = self.index[valid_indices]
            self.peak_time = self.peak_time[valid_indices]
            self.peak_amplitude = self.peak_amplitude[valid_indices]
            self.duration = self.duration[valid_indices]
        if cut_off_time is None or cut_off_time <= self.start_time:
            cut_off_time = self.end_time

        self.get_cutout(trace, width=width, cut_off_time=cut_off_time)

    def find_peaks(self, trace: np.ndarray) -> None:
        zscore_sw = stats.zscore(trace, ddof=1)
        time = np.arange(len(zscore_sw)) / self.sampling_rate + self.start_time
        peaks, _ = signal.find_peaks(
            zscore_sw, height=3, distance=self.sampling_rate // 20
        )
        self.index = peaks
        onset = np.array(
            [
                np.min(t) if len(t) > 0 else 0
                for t in [
                    (
                        np.where(
                            np.flip(zscore_sw[(p - 100) : p] <= zscore_sw[p] * 0.05)
                        )[0]
                        if p > 100
                        else []
                    )
                    for p in peaks
                ]
            ]
        )
        end = np.array(
            [
                np.min(t) if len(t) > 0 else 0
                for t in [
                    (
                        np.where(zscore_sw[p : (p + 100)] <= zscore_sw[p] * 0.05)[0]
                        if p < len(trace) - 100
                        else []
                    )
                    for p in peaks
                ]
            ]
        )
        self.duration = (end + onset) / self.sampling_rate * 1000
        self.peak_time = np.array([time[p] for p in peaks])
        self.peak_amplitude = np.array(
            [self.sharp_wave_trace["amplitude"][p] for p in peaks]
        )

    def get_cutout(
        self, trace: np.ndarray, width: float = 100, cut_off_time: float | None = None
    ) -> None:
        """
        Get a cutout of the sharp wave ripple data around the peaks.
        """
        if self.index is None or len(self.index) == 0:
            return None
        if cut_off_time is not None and cut_off_time <= self.start_time:
            raise ValueError(
                "Cut off time must be greater than the start time of the trace."
            )

        if cut_off_time is not None and cut_off_time <= self.end_time:
            index = [
                i
                for i in self.index
                if self.start_time + i / self.sampling_rate <= cut_off_time
            ]
        else:
            index = self.index
        width_int = int(self.sampling_rate * width / 1000) // 2 * 2 + 1
        sws_array = np.full((len(index), width_int), np.nan)
        ripple_array = np.full((len(index), width_int), np.nan)
        # Initialize start_idx and end_idx to safe default values
        start_idx = 0
        end_idx = width_int - 1
        for i, idx in enumerate(index):
            start_idx = idx - width_int // 2
            end_idx = idx + width_int // 2 + 1
            array_offset_start = 0
            array_offset_end = width_int
            if start_idx < 0:
                array_offset_start = -start_idx
                start_idx = 0
            if end_idx > len(trace):
                array_offset_end = width_int - 1 - (end_idx - len(trace))
                end_idx = len(trace) - 1
            ripple_array[i, array_offset_start:array_offset_end] = self.ripple_trace[
                "trace"
            ][start_idx:end_idx]
            sws_array[i, array_offset_start:array_offset_end] = trace[start_idx:end_idx]
        self.swrs = {"sws": sws_array, "ripple": ripple_array}

    def filter_swrs(
        self,
        min_duration: float = 10.0,
        max_duration: float | None = None,
        min_amplitude: float = 0.001,
        max_amplitude: float | None = None,
        min_ripple_freq: float = 0,
        max_ripple_freq: float | None = None,
        z_power: float | None = None,
        ripple_peak_time: float | None = None,
    ) -> None:
        if self.swrs is None or len(self.swrs) == 0:
            raise ValueError("No SWR data available. Cannot filter.")

        valid_indices = np.where(
            (self.duration >= min_duration) & (self.peak_amplitude >= min_amplitude)
        )[0]

        if max_duration is not None:
            valid_indices = valid_indices[self.duration[valid_indices] <= max_duration]
        if max_amplitude is not None:
            valid_indices = valid_indices[
                self.peak_amplitude[valid_indices] <= max_amplitude
            ]
        if min_ripple_freq > 0 and self.ripple_frequency is not None:
            valid_indices = valid_indices[
                self.ripple_frequency[valid_indices] >= min_ripple_freq
            ]
        if max_ripple_freq is not None and self.ripple_frequency is not None:
            valid_indices = valid_indices[
                self.ripple_frequency[valid_indices] <= max_ripple_freq
            ]
        if z_power is not None and self.z_scored_power is not None:
            valid_indices = valid_indices[self.z_scored_power[valid_indices] >= z_power]
        if ripple_peak_time is not None and self.ripple_peak_time is not None:
            valid_indices = valid_indices[
                np.abs(self.ripple_peak_time[valid_indices]) <= ripple_peak_time
            ]

        self.swrs["sws"] = self.swrs["sws"][valid_indices]
        self.swrs["ripple"] = self.swrs["ripple"][valid_indices]
        self.index = self.index[valid_indices]
        self.peak_time = self.peak_time[valid_indices]
        self.peak_amplitude = self.peak_amplitude[valid_indices]
        self.duration = self.duration[valid_indices]

        if self.ripple_frequency is not None:
            self.ripple_frequency = self.ripple_frequency[valid_indices]
        if self.ripple_power is not None:
            self.ripple_power = self.ripple_power[valid_indices]
        if self.wavelet_power is not None:
            self.wavelet_power = self.wavelet_power[:, valid_indices, :]
        if self.z_scored_power is not None:
            self.z_scored_power = self.z_scored_power[valid_indices]
        if self.ripple_peak_time is not None:
            self.ripple_peak_time = self.ripple_peak_time[valid_indices]

    def get_ripple_properties(
        self, low: float = 100, high: float = 350, step_size: float = 0.2
    ) -> None:
        if self.swrs is None or len(self.swrs) == 0:
            return None
        (
            self.ripple_frequency,
            self.ripple_power,
            self.z_scored_power,
            self.wt_frequencies,
            self.wavelet_power,
            self.ripple_peak_time,
        ) = get_ripple_power_frequency(
            self.swrs["sws"],
            low=low,
            high=high,
            step_size=step_size,
            sampling_frequency=self.sampling_rate,
        ).values()

    def properties_dict(self) -> dict:
        """
        Convert the SWR object to a dictionary.
        """
        if self.ripple_frequency is None or self.ripple_power is None:
            ripple_frequency = np.full((len(self.index),), np.nan)
            ripple_power = np.full((len(self.index),), np.nan)
            z_scored_power = np.full((len(self.index),), np.nan)
        else:
            ripple_frequency = self.ripple_frequency
            ripple_power = self.ripple_power
            z_scored_power = self.z_scored_power

        return {
            "index": self.index,
            "peak_time": self.peak_time,
            "peak_amplitude": self.peak_amplitude,
            "duration": self.duration,
            "ripple_frequency": ripple_frequency,
            "ripple_power": ripple_power,
            "z_scored_power": z_scored_power,
        }

    def plot(
        self,
        return_fig=False,
        single=False,
        average=True,
        show=True,
        ripple_average=False,
        title="SWR Traces",
    ) -> tuple | None:
        """
        Plot the SWR traces.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        if single:
            if self.swrs is None or len(self.swrs) == 0:
                raise ValueError("No SWR data available to plot.")
            ax.plot(self.swrs["sws"].T, alpha=0.1, color="black")
        if ripple_average:
            ax.plot(
                self.swrs["ripple"].mean(axis=0),
                alpha=0.6,
                color="red",
                label="Average Ripple",
            )
        if average:
            ax.plot(
                self.swrs["sws"].mean(axis=0),
                alpha=1,
                color="blue",
                label="Average SWR",
            )
        ax.set_title(title)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (mV)")
        ax.legend(loc="best")
        if show:
            fig_show = deepcopy(fig)
            fig_show.show()

        if return_fig:
            return fig, ax

    def plot_wt(
        self, return_fig: bool = False, index: int | None = None, show: bool = True
    ) -> tuple | None:
        if self.wavelet_power is None or self.wt_frequencies is None:
            print("No wavelet power data available to plot.")
            return None
        elif index is not None and (index < 0 or index >= self.wavelet_power.shape[1]):
            print("Invalid index for wavelet power data.")
            return None
        if index is None:
            wt = np.mean(self.wavelet_power.mean(axis=1))
        elif index is not None:
            wt = self.wavelet_power[:, index, :]
        else:
            print("No wavelet power data available to plot.")
            return None

        time = np.arange(wt.shape[1]) / self.sampling_rate
        fig, ax = plt.subplots(figsize=(10, 6))
        power_wt = ax.pcolormesh(
            time, self.wt_frequencies, wt, shading="gouraud", rasterized=True
        )  # Plot the wavelet transform
        plt.colorbar(power_wt, label="Magnitude", ax=ax)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title("Continuous Wavelet Transform")
        if show:
            fig_show = deepcopy(fig)
            fig_show.show()
        if return_fig:
            return fig, ax
        else:
            return None


class TraceData:
    def __init__(
        self,
        file_name: str,
        notch: float | None = None,
        downsampling_frequency: float | None = None,
    ) -> None:
        self.time: np.ndarray = np.array([])
        self.trace: np.ndarray = np.array([])
        self.sampling_rate: float | None = None
        self.time_unit: str = "s"
        self.trace_unit: str = "mV"
        self.ripple_trace: dict | None = None
        self.gamma_trace: dict | None = None
        self.sharp_wave_trace: dict | None = None
        self.notch: float | None = None
        self.file_name: str = file_name
        self.channel_count: int = 0

        self.date = self.file_name.split("/")[-1].split("_")[
            0
        ]  # Get the date from the file name
        self.recording = (
            self.file_name.split("/")[-1].split("_")[1].removesuffix(".smr")
        )  # Get the recording from the file name
        block = load_smr_file(file_name)
        if block.segments:
            # concatenate all segments into one
            trace_magnitude = np.concatenate(
                [
                    signal.magnitude
                    for segment in block.segments
                    for signal in segment.analogsignals
                ]
            )  # Concatenate all analog signals in the segment
            trace_times = np.concatenate(
                [
                    signal.times
                    for segment in block.segments
                    for signal in segment.analogsignals
                ]
            )  # Concatenate all analog signals in the segment
            self.trace = Quantity(trace_magnitude, "mV").T
            self.time = Quantity(trace_times.magnitude, "s")

            #            segment = block.segments[0]
            #            if segment.analogsignals:
            #                trace = Quantity(trace_magnitude, "mV")
            #                self.time = Quantity(trace_times, "s")
            #                self.trace = trace
            #                if len(self.time) == 0 or len(self.trace) == 0:
            #                    raise ValueError("Time or trace data is not available in the file.")
            self.sampling_rate = float(np.round(1 / np.median(np.diff(self.time))))
            self.channel_count = self.trace.shape[0]
            if (
                isinstance(notch, (int, float))
                and notch > 0
                and self.sampling_rate is not None
            ):
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
            raise ValueError(
                "Downsampling frequency must be lower than the current sampling rate."
            )

        if self.trace is None or self.time is None:
            raise ValueError(
                "Trace data or time data is not available. Cannot downsample."
            )

        factor = int(self.sampling_rate / frequency)
        if factor < 1:
            raise ValueError("Downsampling factor must be at least 1.")

        self.trace = signal.decimate(self.trace, factor, axis=1)
        self.time = self.time[::factor]
        self.sampling_rate = frequency

    def ripple_filter(
        self, lowcut: float = 150.0, highcut: float = 250.0, order: int = 1
    ):
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

    def sharp_wave_filter(self, highcut: float = 40.0, order: int = 1):
        """
        Apply a bandpass filter to the trace data to isolate sharp wave frequencies.
        """
        if self.trace is None or self.time is None:
            raise ValueError("Trace data or time data is not available. Cannot filter.")
        if self.sampling_rate is None:
            raise ValueError("Sampling rate is not set. Cannot filter.")
        # low pass filter
        b, a = signal.butter(order, highcut, btype="low")
        # b, a = band_pass_filter(lowcut, highcut, self.sampling_rate, order=order)
        self.sharp_wave_trace = lfp_processing(
            signal.filtfilt(b, a, self.trace, axis=1)
        )
        zscore_sw = stats.zscore(self.sharp_wave_trace["amplitude"], axis=1)
        # find peaks in channels with minimum of 50 ms inbetween peaks and minimum height of 3
        #
        peaks_results = [
            signal.find_peaks(
                channel_trace, height=3, distance=self.sampling_rate // 20
            )
            for channel_trace in zscore_sw
        ]
        peaks, _ = zip(*peaks_results)
        self.sharp_wave_trace["sws"] = {
            "index": peaks,
            "peak_time": [self.time[p] for p in peaks],
            "peak_amplitude": [
                self.sharp_wave_trace["amplitude"][i][p] for i, p in enumerate(peaks)
            ],
        }

    def _detect_swr_par(
        self,
        high_freq: float = 40.0,
        width: int = 100,
        cut_off_time: float | None = None,
    ) -> None:
        """
        Detect sharp wave ripples (SWR) in the trace data.
        This method uses parallel processing to speed up the detection.
        """
        if self.sampling_rate is None:
            raise ValueError("Sampling rate is not set. Cannot detect SWR.")

        with Pool() as pool:
            results = pool.starmap(
                SWR,
                [
                    (
                        channel,
                        self.time,
                        self.sampling_rate,
                        width,
                        cut_off_time,
                        high_freq,
                    )
                    for channel in self.trace
                ],
            )

        self.swrs = results

    def _detect_swr(
        self, high_freq: float = 40, width: int = 100, cut_off_time: float | None = None
    ) -> None:
        if self.sampling_rate is None:
            raise ValueError("Sampling rate is not set. Cannot detect SWR.")
        self.swrs = [
            SWR(
                trace=channel,
                time=self.time,
                sampling_rate=self.sampling_rate,
                width=width,
                cut_off_time=cut_off_time,
                high_freq=high_freq,
            )
            for channel in self.trace
        ]

    def detect_swr(
        self,
        high_freq: float = 40.0,
        width: int = 100,
        cut_off_time: float | None = None,
        parallel: bool = True,
    ) -> None:
        """
        Detect sharp wave ripples (SWR) in the trace data.
        If parallel is True, use parallel processing to speed up the detection.
        """
        if not isinstance(high_freq, (int, float)) or high_freq <= 0:
            raise ValueError("High frequency must be a positive number.")
        if not isinstance(width, int) or width <= 0:
            raise ValueError("Width must be a positive integer.")

        if parallel:
            self._detect_swr_par(
                high_freq=high_freq, width=width, cut_off_time=cut_off_time
            )
        else:
            self._detect_swr(
                high_freq=high_freq, width=width, cut_off_time=cut_off_time
            )

    def swrs_wavelet_properties(
        self, low: float = 150, high: float = 350, step_size: float = 0.2
    ) -> None:
        if not self.swrs:
            raise ValueError("No SWRs to analyze.")
        for swr in self.swrs:
            if isinstance(swr, SWR):
                swr.get_ripple_properties(low, high, step_size)

    def filter_swrs(
        self,
        min_duration: float = 10.0,
        max_duration: float | None = None,
        min_amplitude: float = 0.001,
        max_amplitude: float | None = None,
        min_ripple_freq: float = 0,
        max_ripple_freq: float | None = None,
        z_power: float | None = None,
        ripple_peak_time: float | None = None,
    ) -> None:
        if not self.swrs:
            raise ValueError("No SWRs to filter.")
        for swr in self.swrs:
            if isinstance(swr, SWR):
                swr.filter_swrs(
                    min_duration,
                    max_duration,
                    min_amplitude,
                    max_amplitude,
                    min_ripple_freq,
                    max_ripple_freq,
                    z_power=z_power,
                    ripple_peak_time=ripple_peak_time,
                )

    def get_swr_incidence(
        self, bin_size: float | None = None, bin: bool = True
    ) -> pd.DataFrame:
        swr_df = self.swr_summary()
        if swr_df.empty:
            raise ValueError("No SWRs to summarize. Cannot calculate incidence.")
        if bin_size is None and bin:
            bin_size = swr_df["peak_time"].max() + 0.1
        if bin_size is None or not bin:
            swr_df["incidence"] = swr_df.groupby(
                ["Date", "Recording", "File", "Channel"]
            )["peak_time"].diff()
            return swr_df[swr_df["incidence"].notna()][
                ["Date", "Recording", "File", "Channel", "peak_time", "incidence"]
            ]
        if not isinstance(bin_size, (int, float)) or bin_size <= 0:
            raise ValueError("Bin size must be a positive number.")
        # also add counts per bin
        return pd.concat(
            [
                channel_df.assign(
                    time_bin=(channel_df["peak_time"] // bin_size) * bin_size,
                    incidence=channel_df["peak_time"].diff(),
                )
                .groupby(["time_bin", "Date", "Recording", "File", "Channel"])
                .agg(incidence=("incidence", "mean"), count=("incidence", "size"))
                .assign(count_per_bin=lambda df: df["count"] / bin_size)
                for _, channel_df in swr_df.groupby("Channel")
            ]
        )

    def plot_swr_incidence(self) -> None:
        """
        Plot the SWR incidence over time for each channel.
        """
        incidence_df = self.get_swr_incidence(bin=False)
        for channel in incidence_df["Channel"].unique():
            channel_df = incidence_df[incidence_df["Channel"] == channel]
            plt.plot(channel_df["peak_time"], channel_df["incidence"], label=channel)
        plt.xlabel("Time (s)")
        plt.ylabel("SWR Incidence (Hz)")
        plt.title("SWR Incidence Over Time")
        plt.legend()
        plt.show()

    def swr_summary(self) -> pd.DataFrame:
        """
        Create a summary DataFrame of SWR properties.
        """
        if not self.swrs:
            raise ValueError("No SWRs to summarize.")
        summary = []
        for i, swr in enumerate(self.swrs):
            if isinstance(swr, SWR):
                swr_dict = pd.DataFrame(swr.properties_dict())
                swr_dict["Channel"] = i + 1
                summary.append(swr_dict)
        df_out = pd.concat(summary)
        df_out["Recording"] = self.recording
        df_out["Date"] = self.date
        df_out["File"] = self.file_name
        return df_out

    def plot_ripple(self) -> None:
        """
        Plot the ripple filtered trace data.
        """
        if (
            hasattr(self, "ripple_trace")
            and self.ripple_trace is not None
            and self.time is not None
        ):
            plot_channels(self.ripple_trace, self.time)

    def plot_gamma(self) -> None:
        """
        Plot the gamma filtered trace data.
        """
        if (
            hasattr(self, "gamma_trace")
            and self.gamma_trace is not None
            and self.time is not None
        ):
            plot_channels(self.gamma_trace, self.time)

    def plot_sharp_wave(self) -> None:
        """
        Plot the sharp wave filtered trace data.
        """
        if (
            hasattr(self, "sharp_wave_trace")
            and self.sharp_wave_trace is not None
            and self.time is not None
        ):
            plot_channels(self.sharp_wave_trace, self.time)

    def extract_sws(
        self, window_size: float = 100.0, cut_off_time: float = 1800.0
    ) -> List[dict]:
        """
        Extract sharp wave data from the sharp wave trace.
        """
        if not hasattr(self, "sharp_wave_trace") or self.sharp_wave_trace is None:
            raise ValueError("Sharp wave trace is not available. Cannot extract SWS.")
        if "sws" not in self.sharp_wave_trace:
            raise ValueError("SWS data is not available in the sharp wave trace.")

        def get_cutout(trace, time, ripple_trace, index, width, cut_off_time=None):
            if hasattr(self, "sharp_wave_trace") and self.sharp_wave_trace is not None:
                if "amplitude" not in self.sharp_wave_trace:
                    raise ValueError(
                        "Amplitude data is not available in the sharp wave trace."
                    )
                if len(self.sharp_wave_trace["amplitude"]) == 0:
                    raise ValueError("Amplitude data is empty. Cannot extract SWS.")
            if (cut_off_time is not None) and (cut_off_time <= time.max()):
                index = index[time[index] >= cut_off_time]  # Limit time to cut_off_time
            elif (cut_off_time is not None) and (
                (cut_off_time > time.max()) or (cut_off_time <= 0)
            ):
                print(f"Cut off time {cut_off_time} is not valid. Using all data.")

            sws_array = np.full((len(index), width - 1), np.nan)
            ripple_array = np.full((len(index), width - 1), np.nan)
            for i, idx in enumerate(index):
                start_idx = idx - width // 2
                end_idx = idx + width // 2
                array_offset_start = 0
                array_offset_end = width - 1
                if start_idx < 0:
                    array_offset_start = -start_idx
                    start_idx = 0
                if end_idx > len(trace):
                    array_offset_end = width - 1 - (end_idx - len(trace))
                    end_idx = len(trace)

                sws_array[i, array_offset_start:array_offset_end] = trace[
                    start_idx:end_idx
                ]
                ripple_array[i, array_offset_start:array_offset_end] = ripple_trace[
                    start_idx:end_idx
                ]
            return {"sws": sws_array, "ripple": ripple_array}

        if self.sampling_rate is None:
            raise ValueError("Sampling rate is not set. Cannot extract SWS.")
        width = int(
            window_size * self.sampling_rate // 1000
        )  # Convert window size from ms to samples
        if self.ripple_trace is None or "trace" not in self.ripple_trace:
            print("Ripple trace is not available. Filtering for ripple.")
            self.ripple_filter()
            if self.ripple_trace is None or "trace" not in self.ripple_trace:
                raise ValueError("Filtering failed. Cannot extract SWS.")
        return [
            get_cutout(
                self.trace[channel_index],
                self.time,
                self.ripple_trace["trace"][channel_index],
                index,
                width,
                cut_off_time,
            )
            for channel_index, index in enumerate(self.sharp_wave_trace["sws"]["index"])
        ]

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
            raise ValueError(
                "Trace data or time data is not available. Cannot create TraceView."
            )
        windowlength = self.sampling_rate * window_size
        self.trace = data.trace[
            :, : int(data.time.size // windowlength * windowlength)
        ].reshape(self.channel_count, -1, int(windowlength))
        self.time = data.time[
            : int(data.time.size // windowlength * windowlength)
        ].reshape(-1, int(windowlength))

    def plot(self):
        """
        Plot the trace data within the specified time window.
        """
        if len(self.trace) == 0 or len(self.time) == 0:
            raise ValueError("Trace data or time data is not available. Cannot plot.")
        plot_channels(self.trace, self.time)
        plt.title(
            f"Trace View: {self.file_name} [{self.window_start}s to {self.window_start + self.window_size}s]"
        )
        plt.show()

    def calc_power_spectrum(self, nperseg: int = 1024):
        """
        Calculate the power spectrum of the trace data within the specified time window.
        """
        if len(self.trace) == 0 or len(self.time) == 0:
            raise ValueError(
                "Trace data or time data is not available. Cannot calculate power spectrum."
            )

        self.power_df = pd.DataFrame(
            columns=["Frequency", "Power", "SegmentTime", "Channel"]
        )
        for i, channel in enumerate(self.trace):
            freqs, psd = signal.welch(
                channel,
                fs=self.sampling_rate,
                nperseg=nperseg,
                average="mean",
                scaling="density",
                axis=1,
            )
            psd_df = pd.concat(
                [
                    pd.DataFrame(
                        {
                            "Frequency": freqs,
                            "Power": power_spec,
                            "SegmentTime": j * self.window_size,
                            "Channel": i + 1,
                            "Date": self.date,
                            "Recording": self.recording,
                            "FileName": self.file_name,
                            "SegmentTimeFile": self.time[j, 0],
                            "Index": j,
                        }
                    )
                    for j, power_spec in enumerate(psd)
                ]
            )
            psd_df.dropna(how="any", inplace=True)
            if i == 0:
                self.power_df = psd_df
            else:
                self.power_df = pd.concat([self.power_df, psd_df], ignore_index=True)

    def update_segment_time(self, trace_view: "TraceView"):
        """
        Merge power data from another TraceView instance.
        """
        if not isinstance(trace_view, TraceView):
            raise ValueError("Other must be an instance of TraceView.")
        if self.power_df is None or trace_view.power_df is None:
            raise ValueError(
                "Power data is not available in one of the TraceView instances."
            )
        self.power_df["SegmentTime"] += (
            trace_view.power_df["SegmentTime"].max() + self.window_size
        )
        self.power_df["Index"] += trace_view.power_df["Index"].max() + 1

    def merge_concentration_data(self, concentration_data: pd.DataFrame) -> None:
        """
        Merge concentration data with the power DataFrame.
        """

        if self.power_df is None:
            raise ValueError(
                "Power data has not been calculated. Call calc_power_spectrum() first."
            )
        if not isinstance(concentration_data, pd.DataFrame):
            raise ValueError("Concentration data must be a pandas DataFrame.")

        concentration_data = concentration_data[
            concentration_data["Recording_file"].str.contains(
                self.file_name.split("/")[-1]
            )
        ]
        # assign kainate concentration to the power_df when time_wash-in is later or equal the SegmentTimeFile
        concentration_df = concentration_data[
            concentration_data["Recording_file"].str.contains(
                self.file_name.split("/")[-1]
            )
        ].copy()
        if concentration_df.empty:
            print(
                f"No concentration data found for {self.file_name}. Skipping concentration merge."
            )
            self.power_df["Kainate_concentration"] = np.nan
            return
        # get column which contain string washin
        wash_in_col = concentration_data.columns[
            ["wash" in col.lower() for col in concentration_data.columns]
        ][0]
        concentration_df.sort_values(
            by=wash_in_col, inplace=True, ascending=False, ignore_index=True
        )  # Sort by time_wash-in in descending order
        last_time_point = self.power_df[
            "SegmentTimeFile"
        ].max()  # Initialize last_time_point with the start time of the first trace
        self.power_df["Kainate_concentration"] = 0
        for i, row in concentration_df.iterrows():
            time_wash_in = row[wash_in_col]
            kainate_conc = row["Kainate_concentration"]
            if i == 0:
                edge_include = "both"
            else:
                edge_include = "left"
            self.power_df.loc[
                self.power_df["SegmentTimeFile"].between(
                    time_wash_in, last_time_point, inclusive=edge_include
                ),
                "Kainate_concentration",
            ] = kainate_conc  # Set the concentration for the time range
            last_time_point = time_wash_in


class DataSet:
    def __init__(self, directory: str):
        self.directory = directory
        self.smr_files = self._get_smr_files()
        self.power_data: pd.DataFrame | None = None
        self.gamma_data: pd.DataFrame | None = None

    def _get_smr_files(self):
        return find_smr_files(self.directory)

    def load_trace_data(
        self, notch: float | None = None, downsampling_frequency: float | None = None
    ) -> None:
        if not self.smr_files:
            print("No .smr files found.")
            return None
        bar = Bar("Loading files...", max=len(self.smr_files), suffix="%(percent)d%%")
        raw_list = []
        for file_i, file in enumerate(self.smr_files):
            if not os.path.exists(file):
                raise FileNotFoundError(f"File {file} does not exist.")
            raw_list.append(
                TraceData(
                    file, notch=notch, downsampling_frequency=downsampling_frequency
                )
            )
            sys.stdout.write(
                f"Processing file {file_i + 1}/{len(self.smr_files)}: {file}"
            )
            sys.stdout.flush()
            bar.next()
        self.trace_data_raw = raw_list
        bar.finish()

    def to_trace_view(
        self, window_start: float = 0, window_size: float = 60, force: bool = False
    ) -> List | None:
        """
        Convert the loaded trace data into TraceView objects.
        """
        if (
            hasattr(self, "trace_data")
            and isinstance(self.trace_data[0], TraceView)
            and hasattr(self, "trace_data_raw")
        ):
            if force:
                print("Converting trace data to TraceView objects...")
                self.trace_data = [
                    TraceView(data, window_start, window_size)
                    for data in self.trace_data_raw
                    if isinstance(data, TraceData)
                ]
            else:
                print("Trace data has already been converted to TraceView objects.")
                return self.trace_data
        if not hasattr(self, "trace_data_raw"):
            raise ValueError(
                "Trace data has not been loaded. Call load_trace_data() first."
            )
        elif isinstance(self.trace_data_raw[0], TraceData):
            self.trace_data = [
                TraceView(data, window_start, window_size)
                for data in self.trace_data_raw
                if isinstance(data, TraceData)
            ]
            return None

    def combine_power_spectra(self, nperseg=1250) -> pd.DataFrame:
        """
        Combine power spectra from all TraceView objects into a single DataFrame.
        """
        if not hasattr(self, "trace_data"):
            raise ValueError(
                "Trace data has not been loaded. Call load_trace_data() first."
            )
        if not all(isinstance(tv, TraceView) for tv in self.trace_data):
            raise ValueError(
                "All trace data must be converted to TraceView objects first."
            )

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
        return pd.concat(
            [
                tv.power_df
                for tv in self.trace_data
                if isinstance(tv, TraceView) and hasattr(tv, "power_df")
            ],
            ignore_index=True,
        )

    def add_concentration_table(self, file_path: str) -> None:
        """
        Add a concentration table to the dataset.
        """
        concentration_data = pd.read_csv(file_path, sep=",")
        concentration_data["Date"] = concentration_data["Recording_file"].apply(
            lambda x: x.split("_")[0]
        )
        concentration_data["Recording"] = concentration_data["Recording_file"].apply(
            lambda x: x.split("_")[1].removesuffix(".smr")
        )
        self.concentration_data = concentration_data

    def merge_concentration_data(self):
        """
        Merge concentration data with power data.
        """
        if not hasattr(self, "trace_data"):
            raise ValueError(
                "Trace data has not been loaded. Call load_trace_data() first."
            )
        if not hasattr(self, "concentration_data"):
            raise ValueError(
                "Concentration data has not been added. Call add_concentration_table() first."
            )

        for tv in self.trace_data:
            if not hasattr(tv, "power_df"):
                raise ValueError(
                    "Power data has not been calculated. Call combine_power_spectra() first."
                )
            if isinstance(tv, TraceView):
                tv.merge_concentration_data(self.concentration_data)

    def return_power_df(self) -> pd.DataFrame:
        """
        Return the combined power DataFrame from all TraceView objects.
        """
        if not hasattr(self, "trace_data"):
            raise ValueError(
                "Trace data has not been loaded. Call load_trace_data() first."
            )

        return pd.concat(
            [
                tv.power_df
                for tv in self.trace_data
                if hasattr(tv, "power_df") and isinstance(tv, TraceView)
            ],
            ignore_index=True,
        )

    def apply_fun_to_view(self, fun):
        for tv in self.trace_data:
            fun(tv)

    def apply_fun_to_raw_data(self, fun):
        """
        Apply a function to the raw trace data.
        """
        if not hasattr(self, "trace_data_raw"):
            raise ValueError(
                "Trace data has not been loaded. Call load_trace_data() first."
            )

        for tv in self.trace_data_raw:
            fun(tv)

    def apply_fun_files_list(self, fun):
        """
        Apply a function to each file in the smr_files list.
        """
        if not hasattr(self, "smr_files"):
            raise ValueError(
                "SMR files have not been loaded. Call _get_smr_files() first."
            )
        output_list = []
        bar = Bar(
            "Processing files...", max=len(self.smr_files), suffix="%(percent)d%%"
        )
        for file_i, file_name in enumerate(self.smr_files):
            # make progressbar in terminal
            sys.stdout.write(
                f"Processing file {file_i + 1}/{len(self.smr_files)}: {file_name}"
            )
            sys.stdout.flush()
            output_list.append(fun(file_name))
            bar.next()
        bar.finish()
        return output_list

    def power_df_only(
        self,
        notch: float | None = 50,
        downsampling_frequency: float | None = 1250,
        window_start: float = 0,
        window_size: float = 60,
        nperseg: int | None = None,
    ) -> pd.DataFrame:
        """
        Return only the power DataFrame from the TraceView objects.
        """
        from copy import deepcopy

        power_df_out = pd.DataFrame()
        bar = Bar(
            "Processing files...", max=len(self.smr_files), suffix="%(percent)d%%"
        )
        last_group = ""
        last_tv = None
        for file_i, file_name in enumerate(self.smr_files):
            # make progressbar in terminal
            sys.stdout.write(
                f"Processing file {file_i + 1}/{len(self.smr_files)}: {file_name}"
            )
            sys.stdout.flush()
            tv = TraceView(
                TraceData(
                    file_name,
                    notch=notch,
                    downsampling_frequency=downsampling_frequency,
                ),
                window_start,
                window_size,
            )
            current_group = f"{tv.date}_{tv.recording}"
            if nperseg is None:
                nperseg = int(tv.sampling_rate)
            tv.calc_power_spectrum(nperseg)  # Ensure power spectrum is calculated
            if current_group == last_group and last_tv is not None:
                tv.update_segment_time(last_tv)
            if self.concentration_data is not None and isinstance(
                self.concentration_data, pd.DataFrame
            ):
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
    """Calculate the full width at half maximum (FWHM) of the peaks in the power spectrum.
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
