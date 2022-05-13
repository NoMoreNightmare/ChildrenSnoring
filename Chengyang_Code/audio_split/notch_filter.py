import scipy.signal as signal
import numpy as np
import csv
import sys
import matplotlib.pyplot as plt
from pyAudioAnalysis import audioBasicIO as aIO

FREQ_INC_AMT = .01  # Hz


def notch_filter(input_signal, sampling_period, freq=60, r=.99):
    N = np.cos(2 * np.pi * freq * sampling_period)
    # Equation from textbook
    b = [1, -2 * N, 1]
    a = [1., -2. * r * N, r ** 2]
    b_0 = sum(a) / sum(b)

    b = np.array(b) * b_0
    return signal.lfilter(b, a, input_signal)


"""
band_pass: Returns a band-passed version of the input signal with cutoff
  frequencies w_lo and w_hi given in Hz.
"""


def band_pass(input_signal, w_lo, w_hi):
    w_lo = float(w_lo) / 500
    w_hi = float(w_hi) / 500
    b, a = signal.butter(1, [w_lo, w_hi], "bandpass")

    return signal.lfilter(b, a, input_signal)


"""
adaptive_notch - applies the notch filter and perterbs the frequency to try to
    adapt to the input signal. 
parameters:
  input_signal: Array containing the values of the input signal.
  sampling_period: Float representing the time interval between samples (in
    seconds)
  start_freq: The frequency the notch filter will start at - a best guess of
    where the noise will be.
  r: The radius of the zero from the origin (controls the steepness of the
    notch filter).
  b_0: Normalizing constant.
  time_to_adapt: Time in seconds allowed for the filter to come to equilibrium
    each time the notch freqency is perterbed.
returns:
  An array containing the filtered input signal.
"""


def adaptive_notch(input_signal, sampling_period=.001, start_freq=50, r=.99,
                   samples_to_adapt=600, double_filter=False):
    # Calculate the number of samples in each segment.
    samples_per_segment = 3 * samples_to_adapt
    samples_btwn_segments = 2 * samples_to_adapt

    freq = start_freq
    next_segment = []
    output = []
    next_segment[:] = input_signal[0:samples_per_segment]

    if double_filter:
        double_filtered_segment = band_pass(next_segment, 50, 60)
        filtered_segment = notch_filter(double_filtered_segment, sampling_period, freq, r)
        prev_e = energy(filtered_segment[samples_to_adapt:])
    else:
        filtered_segment = notch_filter(next_segment, sampling_period, freq, r)
        prev_e = energy(filtered_segment[samples_to_adapt:])

    output = np.concatenate([output, filtered_segment])

    last_inc_dir = 1
    i = samples_btwn_segments
    energies = [prev_e]
    while i < (len(input_signal) - samples_to_adapt):
        end = i + samples_per_segment
        if end > len(input_signal):
            end = len(input_signal)
        next_segment[:] = input_signal[i:end]

        if double_filter:
            doubled_filtered_segment = band_pass(next_segment, 50, 60)
            filtered_segment = notch_filter(doubled_filtered_segment, sampling_period, freq, r)
            e = energy(filtered_segment[samples_to_adapt:])
        else:
            filtered_segment = notch_filter(next_segment, sampling_period, freq, r)
            e = energy(filtered_segment[samples_to_adapt:])
        # print ("energy", e)
        energies.append(e)
        if e < prev_e:
            # Move frequency in the same direction as last time.
            freq += last_inc_dir * FREQ_INC_AMT
        else:
            # Move frequency in the opposite direction from last time.
            last_inc_dir *= -1
            freq += last_inc_dir * FREQ_INC_AMT
        output = np.concatenate([output, filtered_segment[samples_to_adapt:]])
        prev_e = e
        i += samples_btwn_segments

    return output, energies


def energy(signal):
    return sum(np.power(signal, 2))


def mse(signal_a, signal_b):
    return ((np.array(signal_a) - np.array(signal_b)) ** 2).mean()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "please give commandline argument [path to signal with noise]" +
            " [path to signal without noise]")
        exit()
    noise_file = sys.argv[1]
    no_noise_file = sys.argv[2]
    double_filter=False
    if len(sys.argv) > 2:
        # optional commandline arg determines if energy is based on band-passed
        # filtered signal (true) or just the filtered signal (false)
        double_filter = bool(int(sys.argv[2]))
        print("double_filter:", double_filter)

    sampling_period = .01

    [fs, noise_data] = aIO.read_audio_file(noise_file)
    # noise_data = aIO.stereo_to_mono(noise_data)

    # [nfs, no_noise_signal] = aIO.read_audio_file(no_noise_file)
    # no_noise_signal = aIO.stereo_to_mono(no_noise_signal)

    # filtered = band_pass(noise_data, 40, 50)
    # print(filtered)
    # filtered = notch_filter(filtered, sampling_period, freq=30)

    filtered, energies = adaptive_notch(
        noise_data, double_filter=double_filter)
    print(filtered)
    # print ("energies:", energies)
    """
  with open("output_4-28-001.csv", "wb") as adapt_output:
    writer = csv.writer(adapt_output)
    for item in adapt_filtered:
      writer.writerow([item])
  """

    # calculate the mean squared error.
    # mse_std_notch = mse(no_noise_signal, filtered)
    # mse_adapt_notch = mse(no_noise_signal, adapt_filtered)
    # print ("standard notch error:", mse_std_notch)
    # print ("adaptive notch error:", mse_adapt_notch)

    # plot the original unfiltered noise data
    plt.figure()
    plt.plot(noise_data)
    plt.title('unfiltered')

    plt.figure()
    plt.plot(filtered)
    plt.title('standard notch filter output')

    # plt.figure()

    # plt.plot(adapt_filtered)
    # plt.title('adaptive notch filter output')
    #
    # plt.figure()
    # plt.plot(band_pass(noise_data[:5000], 59, 60))
    # plt.title("ahhhhhhhhhhhhhhhhhhhh")

    plt.show()
