from scripts.main_classes import *
import re
from progress.bar import Bar
import os
import pandas as pd


data_set = DataSet("data/")

string_filter = re.compile(r".*\_[0-9].smr$")
filtered_files = [s for s in data_set.smr_files if not string_filter.match(s)]
data_set.smr_files = filtered_files

swr_incidence_out = []
swr_incidence_summary_out = []
swr_summary_out = []

if not os.path.exists("output"):
    os.makedirs("output")
bar = Bar('Loading files...', max=len(data_set.smr_files), suffix='%(percent)d%%')
for file_i, file in enumerate(data_set.smr_files):
    trace_data = TraceData(file, notch=50, downsampling_frequency=2500)
    trace_data.detect_swr(high_freq=40, cut_off_time=1800)
    trace_data.filter_swrs(min_duration=20.0, max_duration=250.0)
    trace_data.swrs_wavelet_properties(step_size=0.2)
    trace_data.filter_swrs(ripple_peak_time=10.0) 
    fig, ax = plt.subplots(figsize=(10, 7), ncols=len(trace_data.swrs), nrows=2)
    for i, channel in enumerate(trace_data.swrs):
        if channel is None:
            print(f"Channel {i+1} has no SWRs detected.")
            continue
        if channel.wavelet_power is None:
            print(f"Channel {i+1} has no wavelet power computed.")
            continue
        if channel.swrs["sws"] is None or channel.swrs["ripple"] is None:
            print(f"Channel {i+1} has no SWR data.")
            continue
        if channel.wt_frequencies is None:
            print(f"Channel {i+1} has no wavelet frequencies computed.")
            continue
        time = np.arange(channel.wavelet_power.shape[2]) / channel.sampling_rate * 1000
        ax[1, i].pcolormesh(time, channel.wt_frequencies, channel.wavelet_power.mean(axis=1), shading='gouraud', rasterized=True)
        ax[1, i].set_title(f"Channel {i + 1}")
        ax[1, i].set_xlabel('Time (ms)')
        ax[1, i].set_ylabel('Frequency (Hz)')
        ax[0, i].plot(time, channel.swrs["sws"].T, alpha=0.1, color='black')
        ax[0, i].plot(time, channel.swrs["ripple"].mean(axis=0), alpha=0.6, color='red', label='Average Ripple')
        ax[0, i].plot(time, channel.swrs["sws"].mean(axis=0), alpha=1, color='blue', label='Average SWR')
        ax[0, i].set_title(f"Channel {i + 1}")
        ax[0, i].set_xlabel("Time (ms)")
        ax[0, i].set_ylabel("Amplitude (mV)")
    fig.suptitle(f"{trace_data.date} {trace_data.recording} SWR Wavelet Properties", fontsize=16)
    fig.tight_layout()
    fig.savefig(f"output/{trace_data.date}_{trace_data.recording}_swr_wavelet.png")

    incidence_df = trace_data.get_swr_incidence(bin=False)
    # save incidence_df to a csv file
    incidence_df.to_csv(f"output/{trace_data.date}_{trace_data.recording}_swr_incidence.csv", index=True)
    # get incidence summary
    incidence_summary = trace_data.get_swr_incidence()
    incidence_summary.to_csv(f"output/{trace_data.date}_{trace_data.recording}_swr_incidence_summary.csv", index=True)
    # save swr summary
    swr_summary = trace_data.swr_summary()
    swr_summary.to_csv(f"output/{trace_data.date}_{trace_data.recording}_swr_summary.csv", index=True)
    swr_incidence_out.append(deepcopy(incidence_df))
    swr_incidence_summary_out.append(deepcopy(incidence_summary))
    swr_summary_out.append(deepcopy(swr_summary))
    sys.stdout.write(f"Processing file {file_i + 1}/{len(data_set.smr_files)}: {file}")
    sys.stdout.flush()
    bar.next()
pd.concat(swr_incidence_out).to_csv("output/swr_incidence_all.csv", index=True)
pd.concat(swr_incidence_summary_out).to_csv("output/swr_incidence_summary_all.csv", index=True)
pd.concat(swr_summary_out).to_csv("output/swr_summary_all.csv", index=True)
swr_summary_df = pd.concat(swr_summary_out)
swr_summary_df_out = swr_summary_df.groupby(["Date", "Recording", "File", "Channel"])[["ripple_frequency", "ripple_power", "duration", "peak_amplitude"]].mean()
swr_summary_df_out["incidence (count/1800s)"] = swr_summary_df.groupby(["Date", "Recording", "File", "Channel"]).size() / 1800
swr_summary_df_out.to_csv("output/swr_summary_mean.csv", index=True)
bar.finish()
