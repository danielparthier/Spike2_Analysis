from scripts.main_classes import *
from scipy import signal

data_set = DataSet("data/")
#data_tmp = DataSet("data/")
data_set.add_concentration_table("data/Epileptiform activity/WashinTime.csv")
#data_set.smr_files = data_set.smr_files[3:10]
#data_tmp.smr_files = data_tmp.smr_files[7:10]
#data_tmp.load_trace_data(notch=50, downsampling_frequency=1250)  # Load trace data with a notch filter at 50 Hz
dt = data_set.power_df_only()
dt

dt.groupby(["Date", "Recording"])["SegmentTime"].max()

# gamma power
dt_gamma = pd.merge(dt.loc[dt[dt["Frequency"].between(0, 5)].groupby(["Channel", "Index", "Recording", "SegmentTime", "Date", "FileName", "SegmentTimeFile", "Kainate_concentration"])["Power"].idxmax()],
                   dt.loc[dt[dt["Frequency"].between(20, 80)].groupby(["Channel", "Index", "Recording", "SegmentTime", "Date", "FileName", "SegmentTimeFile", "Kainate_concentration"])["Power"].idxmax()],
                   on=["Channel", "Index", "Recording", "SegmentTime", "Date", "FileName", "SegmentTimeFile", "Kainate_concentration"], suffixes=("_low", "_high"))
# add column with "normalised_gamma"
dt_sum = dt.groupby(["Channel", "Index", "Recording", "SegmentTime", "Date", "FileName", "SegmentTimeFile", "Kainate_concentration"])["Power"].sum().reset_index()  # Calculate the sum of power for each group
dt_gamma["normalised_gamma"] = dt_gamma["Power_high"] / dt_gamma["Power_low"]  
#dt_gamma["normalised_gamma"] = dt_gamma["Power_high"] / dt_sum["Power"]
# get the power for the last 10 minutes of each Channel, Date, Recording, Kainate_concentration
dt_kainate_power = dt_gamma.groupby(["Channel", "Date", "Recording", "Kainate_concentration"])["normalised_gamma"].apply(lambda x: x.tail(10)).groupby(["Channel", "Date", "Recording", "Kainate_concentration"]).mean().reset_index()  # Calculate the mean normalized gamma power for the last 10 minutes of each Channel, Date, Recording, Kainate_concentration
dt_kainate_power_raw = dt_gamma.groupby(["Channel", "Date", "Recording", "Kainate_concentration"])["Power_high"].apply(lambda x: x.tail(10)).groupby(["Channel", "Date", "Recording", "Kainate_concentration"]).mean().reset_index()  # Calculate the mean normalized gamma power for the last 10 minutes of each Channel, Date, Recording, Kainate_concentration
dt_kainate_frequency = dt_gamma.groupby(["Channel", "Date", "Recording", "Kainate_concentration"])["Frequency_high"].apply(lambda x: x.tail(10)).groupby(["Channel", "Date", "Recording", "Kainate_concentration"]).mean().reset_index()  # Calculate the mean frequency for the last 10 minutes of each Channel, Date, Recording, Kainate_concentration

# filter dt for one date and recording and channel
plt.figure(figsize=(10, 6))
for (channel, date, recording), group in dt_kainate_power_raw.groupby(["Channel", "Date", "Recording"]):
    plt.plot(group["Kainate_concentration"], group["Power_high"], marker='o', label=f"Channel {channel}, Date {date}, Recording {recording}")
plt.xlabel("Kainate Concentration (mM)")
plt.ylabel("Power (uV^2/Hz)")
plt.title("Power vs Kainate Concentration")
#plt.legend()
plt.grid()
plt.show()


# plot normalised gamma on y axis and kainate concentration on x axis then connect dots with lines
plt.figure(figsize=(10, 6))
for (channel, date, recording), group in dt_kainate_power.groupby(["Channel", "Date", "Recording"]):
    plt.plot(group["Kainate_concentration"], group["normalised_gamma"], marker='o', label=f"Channel {channel}, Date {date}, Recording {recording}")
plt.xlabel("Kainate Concentration (mM)")
plt.ylabel("Normalized Gamma Power")
plt.title("Normalized Gamma Power vs Kainate Concentration")
#plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
for (channel, date, recording), group in dt_kainate_frequency.groupby(["Channel", "Date", "Recording"]):
    plt.plot(group["Kainate_concentration"], group["Frequency_high"], marker='o', label=f"Channel {channel}, Date {date}, Recording {recording}")
plt.xlabel("Kainate Concentration (mM)")
plt.ylabel("Frequency (Hz)")
plt.title("Frequency vs Kainate Concentration")
#plt.legend()
plt.grid()
plt.show()


cmap = plt.get_cmap('viridis', int(dt_gamma['Kainate_concentration'].max()))  # Create a colormap with a number of colors equal to the number of kainate concentrations
plt.figure(figsize=(10, 6))
for (channel, date, recording, Kainate), group in dt_gamma.groupby(["Channel", "Date", "Recording", "Kainate_concentration"]):
    Kainate = int(Kainate)  # Ensure Kainate is an integer for indexing the colormap
    if np.isnan(Kainate):
        continue
    plt.plot(group["SegmentTime"], group["normalised_gamma"], color=cmap(Kainate), alpha=0.9)
plt.xlabel("Segment Time (s)")
plt.ylabel("Normalized Gamma Power")
plt.title("Normalized Gamma Power over Segment Time")
plt.grid()
plt.show()

cmap = plt.get_cmap('viridis', dt_gamma['Kainate_concentration'].max())  # Create a colormap with a number of colors equal to the number of kainate concentrations
plt.figure(figsize=(10, 6))
for (channel, date, recording, Kainate), group in dt_gamma.groupby(["Channel", "Date", "Recording", "Kainate_concentration"]):
    plt.scatter(group["SegmentTime"], group["Frequency_high"], color=cmap(Kainate), alpha=0.3)
plt.xlabel("Segment Time (s)")

# Add colorbar for kainate concentration for concentration between 0 and 200mM
plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), label='Kainate Concentration (mM)', ticks=np.arange(0, 201, 50))
plt.ylabel("Frequency (Hz)")
plt.title("Frequency of Gamma Peaks over Segment Time")
plt.grid()
plt.show()



cmap = plt.get_cmap('viridis', dt_gamma['Kainate_concentration'].max())  # Create a colormap with a number of colors equal to the number of kainate concentrations
plt.figure(figsize=(10, 6))
for (channel, date, recording, Kainate), group in dt_gamma.groupby(["Channel", "Date", "Recording", "Kainate_concentration"]):
    plt.plot(group["SegmentTime"], group["Frequency_high"], color=cmap(Kainate), alpha=0.9)
plt.xlabel("Segment Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("Frequency over Segment Time")
plt.grid()
plt.show()





### requires to load files --> memory heavy and only if necessary
data_set.load_trace_data(notch=50, downsampling_frequency=1250)  # Load trace data with a notch filter at 50 Hz
data_set.to_trace_view(window_start=0, window_size=60)  # Create TraceView objects for the first 60 seconds
data_set.combine_power_spectra()  # Combine power spectra from all TraceView objects
data_set.merge_concentration_data()  # Merge concentration data with power data
dt = data_set.return_power_df()  # Return the combined power DataFrame from all TraceView objects


def fun_filter(td: TraceData):
    td.ripple_filter(lowcut=150, highcut=250, order=1)  # Apply ripple filter to each TraceView object
    td.gamma_filter(lowcut=.30, highcut=90, order=1)  # Apply gamma filter to each TraceView object
    td.sharp_wave_filter(lowcut=5, highcut=40, order=1)  # Apply sharp wave filter to each TraceView object



#####
#### Sharp Wave Ripple Analysis
####
# TODO: still in progress


data_set.apply_fun_to_raw_data(fun_filter)  # Apply the filter function to all TraceView objects
data_set.trace_data_raw[0].plot_gamma()  # Plot the gamma filtered data for the first TraceView object
data_set.trace_data_raw[0].plot_ripple()  # Plot the ripple filtered data for the first TraceView object
data_set.trace_data_raw[0].plot_sharp_wave()  # Plot the sharp wave filtered data for the first TraceView object
data_set.trace_data_raw[0].plot()  # Plot the original trace data for the first TraceView object
sws_array = data_set.trace_data_raw[0].extract_sws()  # Extract sharp wave ripple events from the first TraceView object
data_set.trace_data_raw[0].ripple_trace
data_set.trace_data_raw[0].gamma_trace
data_set.trace_data_raw[0].sharp_wave_trace
sw_trace = data_set.trace_data_raw[0].sharp_wave_trace

sw_trace["sws"]["index"]  # Display the indices of the sharp wave ripple events

for channel in sw_trace["sws"]["index"][0]:
    plt.plot(data_set.trace_data_raw[0].trace[0,(channel-50):(channel+50)], color="red")
plt.show()

for channel in sws_array:
    plt.plot(channel["ripple"].mean(axis=0), color='black', alpha=0.1)  # Plot the sharp wave ripple events for each channel
plt.show()

import pywt

time = np.linspace(0, len(wavelet_transform[0])/1250, len(wavelet_transform[0]))  # Create a time vector for the wavelet transform
scales = np.geomspace(5, 30, num=40)  # Define scales for the wavelet transform
scales = np.linspace(5, 30, num=40)  # Define scales for the wavelet transform


time = np.array(range(len(sws_array[0]["sws"][0]))) / 1250  # Create a time vector for the wavelet transform
wavelet = 'cmor1.5-1.0'  # Define the wavelet type
frequency = np.geomspace(60, 350, num=100)  # Define scales for the wavelet transform
scales = pywt.frequency2scale(wavelet=wavelet, freq=frequency/1250)  # Convert scales to frequencies in Hz
wavelet_transform = pywt.cwt(sws_array[0]["sws"], wavelet=wavelet, scales=scales, sampling_period=1/1250, method="fft")  # Perform the continuous wavelet transform

#wavelet = 'cmor3.0-3.0'  # Define the wavelet type
#frequency = np.geomspace(6, 100, num=50)  # Define scales for the wavelet transform
#scales = pywt.frequency2scale(wavelet=wavelet, freq=frequency/1250)  # Candonvert scales to frequencies in Hz
#wavelet_transform = pywt.cwt(data_set.trace_data[0].trace[0][0], wavelet=wavelet, scales=scales, sampling_period=1/1250, method="fft")  # Perform the continuous wavelet transform


#filtered_freq = np.where((wavelet_transform[1] > 60) & (wavelet_transform[1] < 350))  # Find indices where the scale is greater than 60

plt.figure(figsize=(10, 6))
plt.pcolormesh(time, wavelet_transform[1], np.abs(wavelet_transform[0]).mean(axis=1), shading='gouraud')  # Plot the wavelet transform
plt.colorbar(label='Magnitude')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Continuous Wavelet Transform')
plt.show()

plt.figure(figsize=(10, 6))
plt.pcolormesh(time, wavelet_transform[1], np.abs(wavelet_transform[0][:,0]), shading='gouraud')  # Plot the wavelet transform
plt.colorbar(label='Magnitude')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Continuous Wavelet Transform')
plt.show()


sw_trace['sws'].keys()  # Display the keys in the sharp wave trace dictionary
sw_trace['sws']['index'][2]
[data_set.trace_data_raw[0].time[index] for index in data_set.trace_data_raw[3].sharp_wave_trace["sws"]["index"]]
data_set.trace_data_raw[1].sharp_wave_trace["sws"]["peak_time"]
data_set.trace_data_raw[0].time


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
                print(data["sws"]["index"][i])
                axes[i].scatter(data["sws"]["peak_time"][i], data["sws"]["peak_amplitude"][i], color='red', marker='.')
        axes[i].set_ylabel(f'Channel {i + 1} (mV)')
    axes[-1].set_xlabel('Time (s)')
    fig.tight_layout()
    fig.show()

plot_channels(data_set.trace_data_raw[0].sharp_wave_trace, data_set.trace_data_raw[0].time)  # Plot the sharp wave trace for the first TraceView object

sw_trace['sws']["peak_time"][1]

peaks_results = [signal.find_peaks(channel_trace, height=3) for channel_trace in zscore_sw]
peaks, _ = zip(*peaks_results)  # to np array
[data_set.trace_data_raw[0].time[p] for p in peaks]

for i, channel_peaks in enumerate(peaks):
    print(f"Channel {i+1} Peaks: {channel_peaks}")  # Display the peaks for each channel
    peaks[i] = np.array(channel_peaks)


data_set.trace_data_raw[0].time[peaks[0]]
np.array(peaks)  # Convert the peaks to a NumPy array

[sw_trace["amplitude"][p] for p in peaks]  # Get the amplitude of the sharp wave trace at the detected peaks

ripple_trace = data_set.trace_data_raw[0].sharp_wave_trace
test_hilbert = signal.hilbert(ripple_trace, axis=1)  # Apply Hilbert transform to the ripple trace
stats.zscore(np.abs(test_hilbert), axis=1)  # Compute the z-score of the absolute value of the Hilbert transform
phase = np.angle(test_hilbert)
freq = np.unwrap(phase)
inst_freq = np.diff(freq, axis=1, prepend=freq[:, 0:1])  # Compute the frequency from the phase
inst_freq /= (2.0 * np.pi) * 10000  # Convert from radians to Hz
plt.plot(inst_freq.T)
plt.show()

plt.plot(stats.zscore(np.abs(test_hilbert), axis=1).T)  # Plot the real part of the ripple trace
plt.show()
# z score 
import scipy.stats as stats
stats.zscore(ripple_trace)

sw_power_trace = stats.zscore(np.abs(test_hilbert))
peaks, heights = signal.find_peaks(sw_power_trace, height=3)  # Find peaks in the Hilbert transform of the ripple trace
  # Display the heights of the detected peaks
plt.plot(stats.zscore(np.abs(test_hilbert)))
plt.scatter(peaks, heights['peak_heights'], color='red', marker='o')  # Plot the ripple trace
plt.show()

trace_view1.update_segment_time(trace_view)  # Update segment time with the first trace view
trace_view1.power_df
trace_view.power_df

plot_power_spectrum(trace_view.power_df)

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


lowcut = 30.0
highcut = 90.0
fs = 2500
nyquist = 0.5 * fs
low = lowcut / nyquist
high = highcut / nyquist
filter_output = signal.butter(N=200, output="sos", Wn=[low, high], btype='band')


#b, a = band_pass_filter(lowcut=30, highcut=90, fs=2500, output="sos")
w, h = signal.freqz(b, a, worN=1024, fs=2500)
# plot frequency response with x in Hz
plt.plot(w, 20 * np.log10(abs(h)))
plt.title('Butterworth filter frequency response')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [dB]')
plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
plt.axvline(100, color='green') # cutoff frequency
plt.show()

fwhm = calculate_fwhm(first_df, peaks[0])
# frequency of peaks
print("Peak Frequencies:", first_df["Frequency"][peaks].values)

plot_power_spectrum(first_df, xlim=(0,200), alpha=1)
plot_power_spectrum(trace_view1.power_df, xlim=(0,200), alpha=1)

first_df["Frequency"][peaks]


# median filter über einzelpunkte

# power over time
# Konzentrationen einfügen
# letzte 10 min vor konzentrationsswitch
# ripples nur vor ersten washin
# power spectrum normalisieren auf slow oscillation

# 
