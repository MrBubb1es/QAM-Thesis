import matplotlib.pyplot as plt
from scipy.special import erfc
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