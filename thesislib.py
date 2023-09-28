import matplotlib.pyplot as plt
from scipy.special import erfc
import pandas as pd
import numpy as np

# Low and high ends of signal-to-noise ratios for
# 32, 64, 128, and 256 QAM constellations in dB.
QAM32_SNR_LOW = 19.0
QAM64_SNR_LOW = 22.0
QAM128_SNR_LOW = 25.0
QAM256_SNR_LOW = 28.0

QAM32_SNR_HIGH = 24.0
QAM64_SNR_HIGH = 27.0
QAM128_SNR_HIGH = 30.0
QAM256_SNR_HIGH = 33.0

# Generate a new 32-QAM constellation with arbitrary scale
def qam32unscaled_new():
    # 6x6 square of points placed 1 unit apart
    im, re = np.mgrid[-2.5:2.5:6j, -2.5:2.5:6j]
    # actual signals have real part and imaginary part
    qam = re + im*1j
    # Remove corner points
    qam = np.delete(qam, 35)
    qam = np.delete(qam, 30)
    qam = np.delete(qam, 5)
    qam = np.delete(qam, 0)

    return qam

# Generate a new 64-QAM constellation with arbitrary scale
def qam64unscaled_new():
    # 8x8 square of points 1 unit apart
    im, re = np.mgrid[-3.5:3.5:8j, -3.5:3.5:8j]
    qam = re + im*1j

    return qam.flatten()

# Generate a new 128-QAM constellation with arbitrary scale
def qam128unscaled_new():
    # 12x12 grid of points 1 unit apart
    im, re = np.mgrid[-5.5:5.5:12j, -5.5:5.5:12j]
    qam = re + im*1j
    # Remove 4 points from each corner
    qam = np.delete(qam, 143)
    qam = np.delete(qam, 142)
    qam = np.delete(qam, 133)
    qam = np.delete(qam, 132)
    
    qam = np.delete(qam, 131)
    qam = np.delete(qam, 130)
    qam = np.delete(qam, 121)
    qam = np.delete(qam, 120)
    
    qam = np.delete(qam, 23)
    qam = np.delete(qam, 22)
    qam = np.delete(qam, 13)
    qam = np.delete(qam, 12)
    
    qam = np.delete(qam, 11)
    qam = np.delete(qam, 10)
    qam = np.delete(qam, 1)
    qam = np.delete(qam, 0)

    return qam

# Generate a new 256-QAM constellation with arbitrary scale
def qam256unscaled_new():
    # 16x16 grid of points 1 unit apart
    im, re = np.mgrid[-7.5:7.5:16j, -7.5:7.5:16j]
    qam = re + im*1j
    return qam.flatten()

# Converts a decibel value to a linear value
#  input: dB - decibel value
#  output: equivalent linear value
def linearize_dB(dB):
    return 10**(dB / 10)

# Calculate the N_0 number used to generate AWGN
#  input: snr - signal to noise ratio in dB
#  output: Value of N_0 to use in AWGN calculation
def calculate_n0(snr):
    return 1 / linearize_dB(snr)

# Calculate the average energy of a M-QAM constellation
#  input: qam - Numpy array representing QAM constellation
#  output: Average energy of the constellation as defined in the notebook
def calc_average_energy(qam):
    return np.sum(np.abs(qam)) / len(qam)

# Generate a new 32-QAM constellation scaled for an average energy of 1
def qam32_new():
    # 6x6 square of points placed 1 unit apart
    im, re = np.mgrid[-2.5:2.5:6j, -2.5:2.5:6j]
    qam = re + im*1j
    # Remove corner points
    qam = np.delete(qam, 35)
    qam = np.delete(qam, 30)
    qam = np.delete(qam, 5)
    qam = np.delete(qam, 0)
    # distance between points such that the average energy of the constellation is equal to 1
    dist = 1 / calc_average_energy(qam)

    return qam * dist # scale the distance between the points

# Generate a new 64-QAM constellation scaled for an average energy of 1
def qam64_new():
    im, re = np.mgrid[-3.5:3.5:8j, -3.5:3.5:8j]
    qam = (re + im*1j).flatten()
    dist = 1 / calc_average_energy(qam)

    return qam * dist

# Generate a new 128-QAM constellation scaled for an average energy of 1
def qam128_new():
    im, re = np.mgrid[-5.5:5.5:12j, -5.5:5.5:12j]
    qam = re + im*1j
    qam = np.delete(qam, 143)
    qam = np.delete(qam, 142)
    qam = np.delete(qam, 133)
    qam = np.delete(qam, 132)
    
    qam = np.delete(qam, 131)
    qam = np.delete(qam, 130)
    qam = np.delete(qam, 121)
    qam = np.delete(qam, 120)
    
    qam = np.delete(qam, 23)
    qam = np.delete(qam, 22)
    qam = np.delete(qam, 13)
    qam = np.delete(qam, 12)
    
    qam = np.delete(qam, 11)
    qam = np.delete(qam, 10)
    qam = np.delete(qam, 1)
    qam = np.delete(qam, 0)

    dist = 1 / calc_average_energy(qam)

    return qam * dist

# Generate a new 256-QAM constellation scaled for an average energy of 1
def qam256_new():
    im, re = np.mgrid[-7.5:7.5:16j, -7.5:7.5:16j]
    qam = (re + im*1j).flatten()
    dist = 1 / calc_average_energy(qam)
    return qam * dist

# Read a stream from "data/qam32_samples.csv" and return it as an np array
#  input: stream_name - the name of the dataframe column to return
#         stream_len - the number of data points to return from the df. Everything afterwards will be truncated.
#  output: Tuple containing phase offset, signal to noise ratio used to make the data, and a numpy array with stream of length stream_len
#          (phase_offset, snr, signal_stream)
def stream32_from_sample(stream_name, stream_len):
    data = pd.read_csv("data/qam32_samples.csv")[stream_name][0:stream_len+1] # +1 for the phase offset
    data = data.map(complex) # pd reads data as strings, so this converts them to complex numbers
    phase_offset = data[0].real # real part of first element in data is the phase offset
    snr = data[0].imag          # imaginary part of first element is the signal to noise ratio used to create the data
    data = data[1:data.size] # get rid of the phase offset in the stream data

    return (phase_offset, snr, data)

# Read a stream from "data/qam32_samples.csv" and return it as an np array
#  input: stream_name - the name of the dataframe column to return
#         stream_len - the number of data points to return from the df. Everything afterwards will be truncated.
#  output: Tuple containing phase offset, signal to noise ratio used to make the data, and a numpy array with stream of length stream_len
#          (phase_offset, snr, signal_stream)
def stream64_from_sample(stream_name, stream_len):
    data = pd.read_csv("data/qam64_samples.csv")[stream_name][0:stream_len+1] # +1 for the phase offset
    data = data.map(complex) # pd reads data as strings, so this converts them to complex numbers
    phase_offset = data[0].real # real part of first element in data is the phase offset
    snr = data[0].imag          # imaginary part of first element is the signal to noise ratio used to create the data
    data = data[1:data.size] # get rid of the phase offset in the stream data

    return (phase_offset, snr, data)

# Read a stream from "data/qam32_samples.csv" and return it as an np array
#  input: stream_name - the name of the dataframe column to return
#         stream_len - the number of data points to return from the df. Everything afterwards will be truncated.
#  output: Tuple containing phase offset, signal to noise ratio used to make the data, and a numpy array with stream of length stream_len
#          (phase_offset, snr, signal_stream)
def stream128_from_sample(stream_name, stream_len):
    data = pd.read_csv("data/qam128_samples.csv")[stream_name][0:stream_len+1] # +1 for the phase offset
    data = data.map(complex) # pd reads data as strings, so this converts them to complex numbers
    phase_offset = data[0].real # real part of first element in data is the phase offset
    snr = data[0].imag          # imaginary part of first element is the signal to noise ratio used to create the data
    data = data[1:data.size] # get rid of the phase offset in the stream data

    return (phase_offset, snr, data)

# Read a stream from "data/qam32_samples.csv" and return it as an np array
#  input: stream_name - the name of the dataframe column to return
#         stream_len - the number of data points to return from the df. Everything afterwards will be truncated.
#  output: Tuple containing phase offset, signal to noise ratio used to make the data, and a numpy array with stream of length stream_len
#          (phase_offset, snr, signal_stream)
def stream256_from_sample(stream_name, stream_len):
    data = pd.read_csv("data/qam256_samples.csv")[stream_name][0:stream_len+1] # +1 for the phase offset
    data = data.map(complex) # pd reads data as strings, so this converts them to complex numbers
    phase_offset = data[0].real # real part of first element in data is the phase offset
    snr = data[0].imag          # imaginary part of first element is the signal to noise ratio used to create the data
    data = data[1:data.size] # get rid of the phase offset in the stream data

    return (phase_offset, snr, data)


# Rotate a set of signals by some angle
#  input: signals - Numpy array of signals to rotate
#         angle_degrees - angle to rotate signals in degrees
#  output: New Numpy array of rotated signals
def rotated(signals, angle_degrees):
    return signals * np.exp(np.radians(angle_degrees) * 1j)

# Generate AWGN
#  input: snr - signal to noise ratio in dB
#         k - how many samples to generate
#  output: Numpy array of length k with complex AWGN
def awgn_noise(snr, k):
    N0 = calculate_n0(snr)
    std_dev = np.sqrt(N0 / 2.0)
    noise_re = np.random.normal(0.0, std_dev, k)
    noise_im = np.random.normal(0.0, std_dev, k) * 1j
    return noise_re + noise_im

# Calculate the symbol error probability for N-QAM
#  input: snr - signal to noise ratio in dB
#  output: Symbol error probability for given snr value(s) of snr
def symbol_err_prob(N, snr):
    sqrt_term = (3 * linearize_dB(snr)) / (2 * (N - 1))
    return 2 * erfc(np.sqrt(sqrt_term))

def get_received_stream(sent_stream, phase_offset_deg, snr):
    K = len(sent_stream)
    noise = awgn_noise(snr, K)
    return rotated(sent_stream, phase_offset_deg) + noise

# Calculate the log-likelihood function given a phase offset guess
#  input: qam - the constellation being compared against
#         received_stream - the set of signals received as an np array
#         snr - the signal to noise ratio used to calculate the received signals (in dB)
#         theta_deg - the guess of the phase offset for the received data (in degrees)
#  output: The value of the log likelihood function evaluated at theta for some set of signals and constellation
def get_log_likelihood(qam, received_stream, snr, theta_deg):
    linearized_snr = linearize_dB(snr)
    log_likelihood = 0
    
    for received_signal in received_stream:
        # Rotate the received signal clockwise by the guessed theta
        estimated_signal = rotated(received_signal, -theta_deg)
        # Log-Likelihood function as defined in chapter 1
        log_likelihood += np.log(np.sum(np.exp(-linearized_snr * np.abs(estimated_signal - qam)**2)))
        
    return log_likelihood

# Calculate the log-likelihood function given an array of phase offset guesses
#  input: qam - the constellation being compared against
#         received_stream - the set of signals received as an np array
#         snr - the signal to noise ratio used to calculate the received signals (in dB)
#         theta_arr_deg - Numpy array of theta guesses in degrees
#  output: An array of values of the log likelihood function evaluated at each respective value theta
def get_log_likelihood_arr(qam, received_stream, snr, theta_arr_deg):
    arr_len = len(theta_arr_deg)
    log_likelihoods = np.zeros(arr_len)
    
    for i in range(arr_len):
        log_likelihoods[i] = get_log_likelihood(qam, received_stream, snr, theta_arr_deg[i])
        
    return log_likelihoods

# Performs a second order Newton's method to the log-likelihood function
# Performs the differentiation numerically by plugging in points with a delta
# See: https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization
#  input: qam - the QAM constellation np array
#         received_stream - np array of received signals
#         snr - signal to noise ratio in dB
#         theta_guess - the initial estimate of theta that will be fine tuned by Newton's Method
#  output: The new estimation for the value of theta
def second_order_newtons_method(qam, received_stream, snr, theta_guess):
    delta = 1e-6
    y = lambda theta: get_log_likelihood(qam, received_stream, snr, theta)
    
    # Rise / Run
    slope1_1 = (y(theta_guess + delta) - y(theta_guess)) / delta
    # Calculate second slope so the 2nd derivative can be estimated
    slope1_2 = (y(theta_guess + 2*delta) - y(theta_guess + delta)) / delta
    second_deriv_1 = (slope1_2 - slope1_1) / delta
    t_1 = -slope1_1 / second_deriv_1
    
    theta_guess += t_1 # 1st order Newton's method
    
    # Repeat process for 2nd order
    slope2_1 = (y(theta_guess + delta) - y(theta_guess)) / delta
    slope2_2 = (y(theta_guess + 2*delta) - y(theta_guess + delta)) / delta
    second_deriv_2 = (slope1_2 - slope1_1) / delta
    t_2 = -slope2_1 / second_deriv_2
    
    return theta_guess + t_2 # 2nd order Newton's method (final estimation)