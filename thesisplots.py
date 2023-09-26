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

    wave_fig.set_title(f"Waves With Phase Offset of {phase_offset_deg} degrees")
    wave_fig.grid()
    wave_fig.plot(t_vals, sent_wave)
    wave_fig.plot(t_vals, received_wave)

    qam_fig.set_title(f"64-QAM (Unscaled) With Phase Offset of {phase_offset_deg} degrees")
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

# Fig 2 - plotted directly in notebook

# Not a Fig, used as interactive plot
def snr_demonstration(qam, snr):
    K = 100 # Number of points per signal

    qam_noisy = qam.repeat(K)
    noise = awgn_noise(snr, len(qam_noisy))
    qam_noisy += noise

    plt.title(f"32-QAM Noisy Symbols (SNR = {snr})")
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.grid()
    plt.scatter(qam_noisy.real, qam_noisy.imag, marker="+", s=16)
    plt.scatter(qam.real, qam.imag, marker="+", s=16)

    #plt.legend(["Noisy Received Data", "Data Sent"], loc="lower left")

    plt.show()

# Display Fig 3
