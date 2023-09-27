import matplotlib.pyplot as plt
from thesislib import *
import numpy as np

# Not a fig
# Display unscaled QAM constellations (like fig 1)
def graph_unscaled_qam_constellations():
    qam32 = qam32unscaled_new()
    qam64 = qam64unscaled_new()
    qam128 = qam128unscaled_new()
    qam256 = qam256unscaled_new()

    fig, axs = plt.subplots(2,2)
    fig.set_size_inches(10,10)
    qam32_fig, qam64_fig = axs[0]
    qam128_fig, qam256_fig = axs[1]

    qam32_fig.set_title("32-QAM (Unscaled)")
    qam32_fig.grid()
    qam32_fig.scatter(qam32.real, qam32.imag, marker='+')

    qam64_fig.set_title("64-QAM (Unscaled)")
    qam64_fig.grid()
    qam64_fig.scatter(qam64.real, qam64.imag, marker='+')

    qam128_fig.set_title("128-QAM (Unscaled)")
    qam128_fig.grid()
    qam128_fig.scatter(qam128.real, qam128.imag, marker='+')

    qam256_fig.set_title("256-QAM (Unscaled)")
    qam256_fig.grid()
    qam256_fig.scatter(qam256.real, qam256.imag, marker='+', s=16)

    fig.show()

# Not a fig
# Display constellation for different phase offsets
def phase_offset_demonstration(qam, t_vals, sent_wave, phase_offset_deg):
    received_wave = np.sin(t_vals + np.radians(phase_offset_deg))
    received_qam = rotated(qam, phase_offset_deg)
    
    fig, axs = plt.subplots(1,2)
    fig.set_size_inches(10,5)

    wave_fig, qam_fig = axs

    wave_fig.set_title(f"Waves with Phase Offset ($\\theta = {phase_offset_deg}$ degrees)")
    wave_fig.grid()
    wave_fig.plot(t_vals, sent_wave)
    wave_fig.plot(t_vals, received_wave)

    qam_fig.set_title(f"64-QAM Received Signals ($\\theta = {phase_offset_deg}$ degrees)")
    qam_fig.grid()
    qam_fig.scatter(qam.real, qam.imag, marker='+', s=16)
    qam_fig.scatter(received_qam.real, received_qam.imag, marker='+', s=16)

    plt.show()

# Display Fig 1
# Plots scaled 32, 64, 128, and 256-QAM constellations
def graph_qam_constellations():
    qam32 = qam32_new()
    qam64 = qam64_new()
    qam128 = qam128_new()
    qam256 = qam256_new()

    fig, axs = plt.subplots(2,2)
    fig.set_size_inches(10,10)
    qam32_fig, qam64_fig = axs[0]
    qam128_fig, qam256_fig = axs[1]

    qam32_fig.set_title("32-QAM")
    qam32_fig.set_xlim(-1.5, 1.5)
    qam32_fig.set_ylim(-1.5, 1.5)
    qam32_fig.grid()
    qam32_fig.scatter(qam32.real, qam32.imag, marker='+')

    qam64_fig.set_title("64-QAM")
    qam64_fig.set_xlim(-1.5, 1.5)
    qam64_fig.set_ylim(-1.5, 1.5)
    qam64_fig.grid()
    qam64_fig.scatter(qam64.real, qam64.imag, marker='+')

    qam128_fig.set_title("128-QAM")
    qam128_fig.set_xlim(-1.5, 1.5)
    qam128_fig.set_ylim(-1.5, 1.5)
    qam128_fig.grid()
    qam128_fig.scatter(qam128.real, qam128.imag, marker='+')

    qam256_fig.set_title("256-QAM")
    qam256_fig.set_xlim(-1.5, 1.5)
    qam256_fig.set_ylim(-1.5, 1.5)
    qam256_fig.grid()
    qam256_fig.scatter(qam256.real, qam256.imag, marker='+', s=16)

    fig.show()

# Display Fig 2
# Graphs P_N for N = 32, 64, 128, & 256 and for various ranges for SNR
def graph_symbol_err_prob():
    N_32_snr = np.linspace(17.0, 25.0, 40)  # Sample snr range for N = 32
    N_64_snr = np.linspace(20.0, 28.0, 40)  # Sample snr range for N = 64
    N_128_snr = np.linspace(23.0, 31.0, 40) # Sample snr range for N = 128
    N_256_snr = np.linspace(26.0, 34.0, 40) # Sample snr range for N = 256

    # Calculate symbol error probabilities with function given above
    sym_err_32 = symbol_err_prob(32, N_32_snr)
    sym_err_64 = symbol_err_prob(64, N_64_snr)
    sym_err_128 = symbol_err_prob(128, N_128_snr)
    sym_err_256 = symbol_err_prob(256, N_256_snr)

    plt.title("Symbol Error Probability ($P_N$) for Various Constellations")
    plt.ylabel("Symbol Error Probability")
    plt.xlabel("SNR ($\\gamma$) in dB per Symbol")
    plt.yscale("log")
    plt.xlim(16, 34)
    plt.ylim(1e-8, 1e-1)
    plt.grid(which="both")

    plt.plot(N_32_snr, sym_err_32)
    plt.plot(N_64_snr, sym_err_64)
    plt.plot(N_128_snr, sym_err_128)
    plt.plot(N_256_snr, sym_err_256)

    plt.legend(["32-QAM (N = 32)", "64-QAM (N = 64)", "128-QAM (N = 128)", "256-QAM (N = 256)"])

    plt.show()


# Not a Fig, used as interactive plot
#  input: qam - QAM constellation (np array)
#         snr - interactive variable, SNR in dB
#  output: None, displays plot of noisy signals
def snr_demonstration(qam, snr):
    K = 100 # Number of points per signal

    qam_noisy = qam.repeat(K)
    noise = awgn_noise(snr, len(qam_noisy))
    qam_noisy += noise

    plt.title(f"32-QAM Noisy Symbols ($\\gamma$ = {snr})")
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.grid()
    plt.scatter(qam_noisy.real, qam_noisy.imag, marker="+", s=16)
    plt.scatter(qam.real, qam.imag, marker="+", s=16)

    plt.show()

# Not a Fig, used as interactive plot
#  input: k_vals - vector lengths used as x axis values in the plot
#         snr - interactive variable, SNR in dB
#  output: None, displays plot of CRB
def crb_demonstration(k_vals, snr):
    crb_vals = 1 / (2 * k_vals * linearize_dB(snr))

    plt.title(f"Cramer-Rao Lower Bound ($\\gamma$ = {snr})")
    plt.ylabel("Mean Squared Error (MSE) in degrees$^2$")
    plt.xlabel("Vector Length (K)")
    plt.xlim(10,100)
    plt.ylim(1e-6, 1e-3)
    plt.yscale("log")
    plt.grid(which="both")
    plt.plot(k_vals, crb_vals)

    plt.show()

# Display Fig 3
