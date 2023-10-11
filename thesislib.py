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

# Convert values from radians^2 to degrees^2
#  input: rad2 - value in radians^2
#  output: Returns the value converted to degrees^2
def rad2_to_deg2(rad2):
    c = 180**2 / np.pi**2
    return rad2 * c

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

# Calculate the average power of a M-QAM constellation
#  input: qam - Numpy array representing QAM constellation
#  output: Average power of the constellation
def calc_average_power(qam):
    return np.sum(np.abs(qam)**2) / len(qam)

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
    dist = 1 / np.sqrt(calc_average_power(qam))

    return qam * dist # scale the distance between the points

# Generate a new 64-QAM constellation scaled for an average energy of 1
def qam64_new():
    im, re = np.mgrid[-3.5:3.5:8j, -3.5:3.5:8j]
    qam = (re + im*1j).flatten()
    dist = 1 / np.sqrt(calc_average_power(qam))

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

    dist = 1 / np.sqrt(calc_average_power(qam))

    return qam * dist

# Generate a new 256-QAM constellation scaled for an average energy of 1
def qam256_new():
    im, re = np.mgrid[-7.5:7.5:16j, -7.5:7.5:16j]
    qam = (re + im*1j).flatten()
    dist = 1 / np.sqrt(calc_average_power(qam))

    return qam * dist

# Generates a 32-QAM constellation split up by energy of the points for the SML algorithm
# Returns 2D array where each element of the sub arrays are equivalent in energy (distance from origin)
def sml_qam32_new():
    qam32 = qam32_new()

    energies = sorted(set(np.round(np.abs(qam32), 6)))
    partitioned_qam32 = np.ma.zeros((5,8)).astype(complex)
    
    temp = [0 for _ in range(5)] # Used to keep track of how many elems in each group
    
    # Sort each symbol into its energy level
    for symbol in qam32:
        level = get_closest_energy_level(energies, symbol)
        partitioned_qam32[level][temp[level]] = symbol
        temp[level] += 1
    
    partitioned_qam32 = np.ma.masked_equal(partitioned_qam32, 0.) # Mask all blank values

    return (partitioned_qam32, energies)

# Generates a 64-QAM constellation split up by energy of the points
def sml_qam64_new():
    qam64 = qam64_new()

    energies = sorted(set(np.round(np.abs(qam64), 6)))
    partitioned_qam64 = np.ma.zeros((9,12)).astype(complex)
    
    temp = [0 for _ in range(9)]
    
    for symbol in qam64:
        level = get_closest_energy_level(energies, symbol)
        partitioned_qam64[level][temp[level]] = symbol
        temp[level] += 1
    
    partitioned_qam64 = np.ma.masked_equal(partitioned_qam64, 0.)

    return (partitioned_qam64, energies)

# Generates a 128-QAM constellation split up by energy of the points
def sml_qam128_new():
    qam128 = qam128_new()

    energies = sorted(set(np.round(np.abs(qam128), 6)))
    partitioned_qam128 = np.ma.zeros((16,16)).astype(complex)
    
    temp = [0 for _ in range(16)]
    
    for symbol in qam128:
        level = get_closest_energy_level(energies, symbol)
        partitioned_qam128[level][temp[level]] = symbol
        temp[level] += 1
    
    partitioned_qam128 = np.ma.masked_equal(partitioned_qam128, 0.)

    return (partitioned_qam128, energies)

# Generates a 256-QAM constellation split up by energy of the points
def sml_qam256_new():
    qam256 = qam256_new()

    energies = sorted(set(np.round(np.abs(qam256), 6)))
    partitioned_qam256 = np.ma.zeros((32,16)).astype(complex)
    
    temp = [0 for _ in range(32)]
    
    for symbol in qam256:
        level = get_closest_energy_level(energies, symbol)
        partitioned_qam256[level][temp[level]] = symbol
        temp[level] += 1
    
    partitioned_qam256 = np.ma.masked_equal(partitioned_qam256, 0.)

    return (partitioned_qam256, energies)

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
    data = np.array(data[1:data.size]) # get rid of the phase offset in the stream data

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
    data = np.array(data[1:data.size]) # get rid of the phase offset in the stream data

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
    data = np.array(data[1:data.size]) # get rid of the phase offset in the stream data

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
    data = np.array(data[1:data.size]) # get rid of the phase offset in the stream data

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

# Calculate the Cramer-Rao Bound for some SNR and K
#  input: snr - Signal to noise ratio in dB
#         k - vector length (number of symbols sent)
#  output: The cramer rao bound for the specified values in radians^2
def get_crb(snr, k):
    return 1.0 / (2.0 * linearize_dB(snr) * k)

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
"""def get_log_likelihood(qam, received_stream, snr, theta_deg):
    linearized_snr = linearize_dB(snr)
    log_likelihood = 0
    
    for received_signal in received_stream:
        # Rotate the received signal clockwise by the guessed theta
        estimated_signal = rotated(received_signal, -theta_deg)
        # Log-Likelihood function as defined in chapter 1
        log_likelihood += np.log(np.sum(np.exp(-linearized_snr * np.abs(estimated_signal - qam)**2)))
        
    return log_likelihood"""

# Vectorized Log-likelihood function, functionally the same as commented out funciton above
def get_log_likelihood(qam, received_stream, snr, theta_deg):
    linearized_snr = linearize_dB(snr)
    qam_size = len(qam)
    k = len(received_stream)

    new_received = np.array(np.split(np.repeat(received_stream, qam_size), k))
    new_qam = np.tile(qam, (k,1))
    inner_sums = np.sum(np.exp(-linearized_snr * np.abs(rotated(new_received, -theta_deg) - new_qam)**2), axis=1)

    return np.sum(np.log(inner_sums))

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

# Performs a first order Newton's method to the log-likelihood function
# Performs the differentiation numerically by plugging in points with a delta
# See: https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization
#  input: qam - the QAM constellation np array
#         received_stream - np array of received signals
#         snr - signal to noise ratio in dB
#         theta_guess - the initial estimate of theta that will be fine tuned by Newton's Method
#  output: The new estimation for the value of theta
def newtons_method_ml(qam, received_stream, snr, theta_guess):
    delta = 1e-3
    y = lambda theta: get_log_likelihood(qam, received_stream, snr, theta)
    
    # # Rise / Run
    # slope1 = (y(theta_guess + delta) - y(theta_guess)) / delta
    # # Calculate second slope so the 2nd derivative can be estimated
    # slope2 = (y(theta_guess + 2*delta) - y(theta_guess + delta)) / delta
    # second_deriv = (slope2 - slope1) / delta
    # t = -slope1 / second_deriv
    
    # This performs the same code as above, but more optimized:
    y1 = y(theta_guess)
    y2 = y(theta_guess + delta)
    y3 = y(theta_guess + 2*delta)
    t = -delta * ( (y2 - y1) / (y1 - 2*y2 + y3) ) # only 1 division required

    return theta_guess + t # new guess from 1st order Newton's method

"""def newtons_methodB(qam, received_stream, snr, theta_guess):
    delta = 1e-4
    y = lambda theta: get_log_likelihood(qam, received_stream, snr, theta)
    
    # Rise / Run
    y1 = y(theta_guess)
    y2 = y(theta_guess + delta)
    t = delta * y1 / (y2 - y1)
    
    return theta_guess + t # new guess from 1st order Newton's method"""

# Performs a second order Newton's method to the ML log-likelihood function
# (first order Newton's method applied 2x)
#  input: qam - the QAM constellation np array
#         received_stream - np array of received signals
#         snr - signal to noise ratio in dB
#         theta_guess - the initial estimate of theta that will be fine tuned by Newton's Method
#  output: The new estimation for the value of theta
def second_order_newtons_method_ml(qam, received_stream, snr, theta_guess):
    theta_guess = newtons_method_ml(qam, received_stream, snr, theta_guess) # 1st order
    theta_guess = newtons_method_ml(qam, received_stream, snr, theta_guess) # 2nd order

    if (theta_guess < 0):
        theta_guess += 90
    elif (theta_guess > 90):
        theta_guess -= 90
    
    return theta_guess 

# Chooses a random phase offset and uses input to generate data and perform ML estimation to estimate the chose phase offset
#  input: qam - QAM constellation to use
#         snr - signal to noise ratio used to generate noise in dB
#         k - vector length
#  output: A tuple containing the true chosen phase offset, the estimatied phase offset, and the sent and received streams
def ml_estimation(qam, snr, k):
    theta_guesses = np.arange(0,91,2)
    phase_offset = np.random.uniform(0,90) # Choose random value for theta from [0, 90)
    sent_stream = np.random.choice(qam, k)
    received_stream = get_received_stream(sent_stream, phase_offset, snr)
    log_likelihood_vals = get_log_likelihood_arr(qam, received_stream, snr, theta_guesses)

    # Best estimate of theta based on inital 2 degree incremental evaluations of log-likelihood
    best_theta_guess = theta_guesses[np.argmax(log_likelihood_vals)]
    # Fine optimization using 2nd order Newton's method
    best_theta_guess = second_order_newtons_method_ml(qam, received_stream, snr, best_theta_guess)

    return (phase_offset, best_theta_guess, sent_stream, received_stream)


# Measures the average performence of the Maximum Likelihood estimation algorithm over a series of iterations
#  input: qam - the QAM constellation to use
#         snr - the signal to noise ratio in dB
#         k - the vector length
#         iters - how many tests to perform (1000 in paper)
#  output: The average squared error for all of the iterations tested
def ml_estimation_performance_test(qam, snr, k, iters):
    print(f"ML For: {len(qam)}-QAM: k = {k}, snr = {snr}")
    total_squared_error = 0

    # Squared error as defined on page 12 in chapter 1
    squared_err = lambda true_phase_offset, theta_guess: min(
        np.radians(true_phase_offset - theta_guess)**2,
        np.radians(true_phase_offset - theta_guess + 90)**2,
        np.radians(true_phase_offset - theta_guess - 90)**2,
    )

    for i in range(iters):
        phase_offset, best_theta_guess, sent_stream, received_stream = ml_estimation(qam, snr, k)

        total_squared_error += squared_err(phase_offset, best_theta_guess)

        # Something's gone wrong
        if (best_theta_guess < 0 or best_theta_guess > 90):
            data = pd.DataFrame({"sent": np.insert(sent_stream, 0, 0), "received": np.insert(received_stream, 0, phase_offset + snr*1j)})
            data.to_csv(f"logs/err_{len(qam)}QAM_iter{i}_infolog.csv")
            print(f"Error for {len(qam)}-QAM: SNR = {snr}, K = {k}, True Offset = {phase_offset}, Best Estimate = {best_theta_guess}, Sq. Err = {squared_err(phase_offset, best_theta_guess)}")
            print(f"Logged error to: logs/err_{len(qam)}QAM_iter{i}_infolog.csv")

        if (i % 100 == 0):
            print(f"  iteration = {i}, phase_offset = {phase_offset}, best_guess = {best_theta_guess}")
    
    print(f"Finished with mean squared error of {total_squared_error / iters}")

    return total_squared_error / iters


# The following 4 functions all perform the same suite of tests for the ML
# estimator algorithm as explained in ch. 1 of the thesis on each of their
# respective QAM constellation sizes and logs the results to a csv file.
#  input: None
#  output: returns the dataframes that get logged as CSVs

### 32-QAM ###
def calculate_ml_results_32():
    k_vals = np.arange(10,101,10)
    qam = qam32_new()
    err_data = []
    snr_data = []

    for k in k_vals:
        snr_data.append(QAM32_SNR_LOW)
        err_data.append(ml_estimation_performance_test(qam, QAM32_SNR_LOW, k, 1000))
    for k in k_vals:
        snr_data.append(QAM32_SNR_HIGH)
        err_data.append(ml_estimation_performance_test(qam, QAM32_SNR_HIGH, k, 1000))

    # Need to double length of k_vals
    data32 = pd.DataFrame({"K": np.append(k_vals, k_vals), "SNR": snr_data, "ML Results": err_data})
    data32.to_csv("data/qam32_ML_results.csv")

    return data32

### 64-QAM ###
def calculate_ml_results_64():
    k_vals = np.arange(10,101,10)
    qam = qam64_new()
    err_data = []
    snr_data = []

    for k in k_vals:
        snr_data.append(QAM64_SNR_LOW)
        err_data.append(ml_estimation_performance_test(qam, QAM64_SNR_LOW, k, 1000))
    for k in k_vals:
        snr_data.append(QAM64_SNR_HIGH)
        err_data.append(ml_estimation_performance_test(qam, QAM64_SNR_HIGH, k, 1000))

    # Need to double length of k_vals
    data64 = pd.DataFrame({"K": np.append(k_vals, k_vals), "SNR": snr_data, "ML Results": err_data})
    data64.to_csv("data/qam64_ML_results.csv")

    return data64

### 128-QAM ###
def calculate_ml_results_128():
    k_vals = np.arange(10,101,10)
    qam = qam128_new()
    err_data = []
    snr_data = []

    for k in k_vals:
        snr_data.append(QAM128_SNR_LOW)
        err_data.append(ml_estimation_performance_test(qam, QAM128_SNR_LOW, k, 1000))
    for k in k_vals:
        snr_data.append(QAM128_SNR_HIGH)
        err_data.append(ml_estimation_performance_test(qam, QAM128_SNR_HIGH, k, 1000))

    # Need to double length of k_vals
    data128 = pd.DataFrame({"K": np.append(k_vals, k_vals), "SNR": snr_data, "ML Results": err_data})
    data128.to_csv("data/qam128_ML_results.csv")

    return data128

### 256-QAM ###
def calculate_ml_results_256():
    k_vals = np.arange(10,101,10)
    qam = qam256_new()
    err_data = []
    snr_data = []

    for k in k_vals:
        snr_data.append(QAM256_SNR_LOW)
        err_data.append(ml_estimation_performance_test(qam, QAM256_SNR_LOW, k, 1000))
    for k in k_vals:
        snr_data.append(QAM256_SNR_HIGH)
        err_data.append(ml_estimation_performance_test(qam, QAM256_SNR_HIGH, k, 1000))

    # Need to double length of k_vals
    data256 = pd.DataFrame({"K": np.append(k_vals, k_vals), "SNR": snr_data, "ML Results": err_data})
    data256.to_csv("data/qam256_ML_results.csv")

    return data256

# Calculates which level of energy is closest to that of the received symbol
#  input: energies - The energy levels of the SML QAM
#         received_symbol - Complex number representing the received symbol
#  output: Returns the index of the np array in the QAM list that has the closest energy level
def get_closest_energy_level(energies, received_symbol):
    return np.argmin((np.abs(received_symbol) - energies)**2)

# Calculates which energy levels will be used in the suboptimal log-likelihood calculation
#  input: energies - The energy levels of the SML QAM
#         received_symbol - Complex number representing the received symbol
#         p - The number of surrounding energy levels to include
#  output: Returns an array of the indices of the energy levels
def get_energy_levels(energies, received_symbol, p):
    closest = get_closest_energy_level(energies, received_symbol)
    return np.arange(max(closest - p, 0), min(closest + p + 1, np.size(energies)))

# Calculates the suboptimal log-likelihood function given a phase offset guess
#  input: sml_qam - the QAM constellation being compared against (partitioned into energy levels)
#         energies - an array of the energy levels for the sml_qam
#         received_stream - the set of signals received as an np array
#         snr - the signal to noise ratio used to calculate the received signals (in dB)
#         theta_deg - the guess of the phase offset for the received data (in degrees)
#         p - the number of surrounding energy levels to include in the calculation
#  output: The value of the suboptimal log-likelihood function evaluated at theta
def get_sml_log_likelihood(sml_qam, energies, received_stream, snr, theta_deg, p):
    linearized_snr = linearize_dB(snr)
    vector_len = len(received_stream)
    sml_ll_total = 0

    for k in range(vector_len):
        received_symbol = received_stream[k]
        levels = get_energy_levels(energies, received_symbol, p)
        
        sml_ll_total += np.log(np.sum(np.exp(-linearized_snr * np.abs(rotated(received_symbol, -theta_deg) - sml_qam[levels].flatten().compressed())**2)))
    
    return sml_ll_total

# Calculates the suboptimal log-likelihood function given an array of phase offset guesses
#  input: sml_qam - the QAM constellation being compared against (partitioned into energy levels)
#         energies - an array of the energy levels for the sml_qam
#         received_stream - the set of signals received as an np array
#         snr - the signal to noise ratio used to calculate the received signals (in dB)
#         theta_deg_arr - array of guesses to perform the calculation on (in degrees)
#         p - the number of surrounding energy levels to include in the calculation
#  output: an array of sml log-likelihoods for each theta in the given theta_deg_arr
def get_sml_log_likelihood_arr(sml_qam, energies, received_stream, snr, theta_deg_arr, p):
    arr_len = len(theta_deg_arr)
    sml_ll_vals = np.zeros(arr_len)

    for i in range(arr_len):
        sml_ll_vals[i] = get_sml_log_likelihood(sml_qam, energies, received_stream, snr, theta_deg_arr[i], p)

    return sml_ll_vals

# Performs a first order Newton's method to the suboptimal log-likelihood function
# Performs the differentiation numerically by plugging in points with a delta
#  input: sml_qam - the QAM constellation being compared against (partitioned into energy levels)
#         energies - an array of the energy levels for the sml_qam
#         received_stream - the set of signals received as an np array
#         snr - the signal to noise ratio used to calculate the received signals (in dB)
#         theta_guess - the guess of the phase offset for the received data (in degrees)
#         p - the number of surrounding energy levels to include in the calculation
#  output: The new estimation for the value of theta
def newtons_method_sml(sml_qam, energies, received_stream, snr, theta_guess, p):
    delta = 1e-3
    y = lambda theta: get_sml_log_likelihood(sml_qam, energies, received_stream, snr, theta, p)
    
    y1 = y(theta_guess)
    y2 = y(theta_guess + delta)
    y3 = y(theta_guess + 2*delta)
    t = -delta * ( (y2 - y1) / (y1 - 2*y2 + y3) )

    return theta_guess + t # new guess from 1st order Newton's method


# Performs a second order Newton's method to the SML log-likelihood function
# (first order Newton's method applied 2x)
#  input: sml_qam - the QAM constellation being compared against (partitioned into energy levels)
#         energies - an array of the energy levels for the sml_qam
#         received_stream - the set of signals received as an np array
#         snr - the signal to noise ratio used to calculate the received signals (in dB)
#         theta_guess - the guess of the phase offset for the received data (in degrees)
#         p - the number of surrounding energy levels to include in the calculation
#  output: The new estimation for the value of theta
def second_order_newtons_method_sml(sml_qam, energies, received_stream, snr, theta_guess, p):
    theta_guess = newtons_method_sml(sml_qam, energies, received_stream, snr, theta_guess, p) # 1st order
    theta_guess = newtons_method_sml(sml_qam, energies, received_stream, snr, theta_guess, p) # 2nd order

    if (theta_guess < 0):
        theta_guess += 90
    elif (theta_guess > 90):
        theta_guess -= 90
    
    return theta_guess 


# Chooses a random phase offset and uses input to generate data and perform SML estimation to estimate the chose phase offset
#  input: sml_qam - the QAM constellation being compared against (partitioned into energy levels)
#         energies - an array of the energy levels for the sml_qam
#         snr - signal to noise ratio used to generate noise in dB
#         k - vector length
#         p - the number of surrounding energy levels to include in the calculation
#  output: A tuple containing the true chosen phase offset, the estimatied phase offset, and the sent and received streams
def sml_estimation(sml_qam, energies, snr, k, p):
    theta_guesses = np.arange(0,91,2)
    phase_offset = np.random.uniform(0,90) # Choose random value for theta from [0, 90)
    sent_stream = np.random.choice(sml_qam.flatten().compressed(), k)
    received_stream = get_received_stream(sent_stream, phase_offset, snr)
    log_likelihood_vals = get_sml_log_likelihood_arr(sml_qam, energies, received_stream, snr, theta_guesses, p)

    # Best estimate of theta based on inital 2 degree incremental evaluations of log-likelihood
    best_theta_guess = theta_guesses[np.argmax(log_likelihood_vals)]
    # Fine optimization using 2nd order Newton's method
    best_theta_guess = second_order_newtons_method_sml(sml_qam, energies, received_stream, snr, best_theta_guess, p)

    return (phase_offset, best_theta_guess, sent_stream, received_stream)

# Measures the average performence of the Suboptimal Maximum Likelihood estimation algorithm over a series of iterations
#  input: sml_qam - the QAM constellation being compared against (partitioned into energy levels)
#         energies - an array of the energy levels for the sml_qam
#         snr - signal to noise ratio used to generate noise in dB
#         k - vector length
#         p - the number of surrounding energy levels to include in the calculation
#         iters - how many tests to perform (1000 in paper)
#  output: The mean squared error for all of the iterations tested
def sml_estimation_performance_test(sml_qam, energies, snr, k, p, iters):
    qam_size = len(sml_qam.flatten().compressed())
    print(f"SML For: {qam_size}-QAM: k = {k}, snr = {snr}")
    total_squared_error = 0

    # Squared error as defined on page 12 in chapter 1
    squared_err = lambda true_phase_offset, theta_guess: min(
        np.radians(true_phase_offset - theta_guess)**2,
        np.radians(true_phase_offset - theta_guess + 90)**2,
        np.radians(true_phase_offset - theta_guess - 90)**2,
    )

    for i in range(iters):
        phase_offset, best_theta_guess, sent_stream, received_stream = sml_estimation(sml_qam, energies, snr, k, p)

        total_squared_error += squared_err(phase_offset, best_theta_guess)

        # Something's gone wrong
        if (best_theta_guess < 0 or best_theta_guess > 90):
            data = pd.DataFrame({"sent": np.insert(sent_stream, 0, 0), "received": np.insert(received_stream, 0, phase_offset + snr*1j)})
            data.to_csv(f"logs/err_{qam_size}QAM_iter{i}_infolog.csv")
            print(f"Error for {qam_size}-QAM: SNR = {snr}, K = {k}, True Offset = {phase_offset}, Best Estimate = {best_theta_guess}, Sq. Err = {squared_err(phase_offset, best_theta_guess)}")
            print(f"Logged error to: logs/err_{qam_size}QAM_iter{i}_infolog.csv")

        if (i % 100 == 0):
            print(f"  iteration = {i}, phase_offset = {phase_offset}, best_guess = {best_theta_guess}")
    
    print(f"Finished with mean squared error of {total_squared_error / iters}")

    return total_squared_error / iters

# The following 4 functions all perform the same suite of tests for the SML
# estimator algorithm as explained in ch. 1 of the thesis on each of their
# respective QAM constellation sizes and logs the results to a csv file.
#  input: p - the number of surrounding energy levels to include in the calculation of the suboptimal log-likelihood function
#  output: returns the dataframes that get logged as CSVs

### 32-QAM ###
def calculate_sml_results_32(p):
    k_vals = np.arange(10,101,10)
    sml_qam, energies = sml_qam32_new()
    err_data = []
    snr_data = []

    for k in k_vals:
        snr_data.append(QAM32_SNR_LOW)
        err_data.append(sml_estimation_performance_test(sml_qam, energies, QAM32_SNR_LOW, k, p, 1500))
    for k in k_vals:
        snr_data.append(QAM32_SNR_HIGH)
        err_data.append(sml_estimation_performance_test(sml_qam, energies, QAM32_SNR_HIGH, k, p, 1500))

    # Need to double length of k_vals
    data32 = pd.DataFrame({"K": np.append(k_vals, k_vals), "SNR": snr_data, "SML Results": err_data})
    data32.to_csv(f"data/qam32_SML_results_{p}p.csv")

    return data32


def calculate_sample_data():
    PHASE_VALS = [15.74324, 25.0, 45.0, 65.3546372, 75.0]
    SAMPLE_LEN = 1000

    def log_data(qam, snr_lo, snr_hi):
        sent_stream = np.insert(np.random.choice(qam, SAMPLE_LEN), 0, 0)
        
        df = pd.DataFrame({
            "Sent": sent_stream,
        })

        for index, val in enumerate(PHASE_VALS):
            received_stream_lo = np.insert(get_received_stream(sent_stream[1:], val, snr_lo), 0, val + snr_lo*1j)
            received_stream_hi = np.insert(get_received_stream(sent_stream[1:], val, snr_hi), 0, val + snr_hi*1j)

            df[f"Received {index+1} Low"] = received_stream_lo
            df[f"Received {index+1} High"] = received_stream_hi
        
        df.to_csv(f"data/qam{len(qam)}_samples.csv")

        print(f'Logged new samples to "data/qam{len(qam)}_samples.csv"')

    log_data(qam32_new(), QAM32_SNR_LOW, QAM32_SNR_HIGH)
    log_data(qam64_new(), QAM64_SNR_LOW, QAM64_SNR_HIGH)
    log_data(qam128_new(), QAM128_SNR_LOW, QAM128_SNR_HIGH)
    log_data(qam256_new(), QAM256_SNR_LOW, QAM256_SNR_HIGH)
    

def main():
    qam = qam256_new()
    df = pd.DataFrame({"Symbols": qam})
    df.to_csv("qam256.csv")

if __name__ == "__main__":
    main()