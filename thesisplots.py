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

# Not a Fig, graphs effects of 45 degree phase offset on a sine wave and a 64-QAM constellation
def graph_phase_offset_effects():
    phase_offset = 45.0 # degrees

    time_vals = np.linspace(0, 2*np.pi)
    phase_off_good = np.sin(time_vals)                           # No phase offset
    phase_off_err = np.sin(time_vals + np.radians(phase_offset)) # Phase offset
    qam64_good = qam64unscaled_new()                             # QAM signals as normal
    qam64_err = rotated(qam64_good, phase_offset)                # QAM signals with rotation caused by phase offset

    fig, axs = plt.subplots(2,2)
    fig.set_size_inches(10,10)
    phase_off_good_fig, phase_off_err_fig = axs[0]
    qam64_good_fig, qam64_err_fig = axs[1]

    phase_off_good_fig.set_title("No Phase Offset")
    phase_off_good_fig.grid()
    phase_off_good_fig.plot(time_vals, phase_off_good)
    phase_off_good_fig.plot(time_vals, phase_off_good)

    phase_off_err_fig.set_title(f"{phase_offset} degree Phase Offset")
    phase_off_err_fig.grid()
    phase_off_err_fig.plot(time_vals, phase_off_good)
    phase_off_err_fig.plot(time_vals, phase_off_err)

    qam64_good_fig.set_title("64-QAM Received Signals No Phase Offset")
    qam64_good_fig.grid()
    qam64_good_fig.scatter(qam64_good.real, qam64_good.imag, marker='+')

    qam64_err_fig.set_title(f"64-QAM Received Signals ($\\theta = {phase_offset}$ degrees)")
    qam64_err_fig.grid()
    qam64_err_fig.scatter(qam64_err.real, qam64_err.imag, marker='+', s=16, c="orange")

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
    crb_vals = rad2_to_deg2(get_crb(snr, k_vals))

    plt.title(f"Cramer-Rao Lower Bound ($\\gamma$ = {snr})")
    plt.ylabel("Mean Squared Error (MSE) in degrees$^2$")
    plt.xlabel("Vector Length (K)")
    plt.xlim(10,100)
    plt.ylim(1e-2, 1e2)
    plt.yscale("log")
    plt.grid(which="both")
    plt.plot(k_vals, crb_vals)

    plt.show()

# Not a Fig, used as an interactive plot
#  input: qam - the QAM constellation used
#         noise - AWGN to be added to the received signals. Normally done in get_received_stream, but precalculated here
#         sent_stream - np array of all signals from QAM that were sent. Used to calculate received signals
#         snr - SNR used to calculate noise. Fixed in this function
#         phase_offset - the actual phase offset of the data in degrees
#         theta_guess - the angle to correct the received stream by to see the affect of different correction angles in degrees
#  output: None, displays a plot of both the data received and the log-likelihood
def log_likelihood_demonstration(qam, noise, sent_stream, snr, phase_offset, theta_guess):
    received_stream = rotated(sent_stream, phase_offset) + noise
    corrected_stream = rotated(received_stream, -theta_guess)

    theta_vals = np.arange(0,91,2)
    ll_vals = get_log_likelihood_arr(qam, received_stream, snr, theta_vals)

    fig, axs = plt.subplots(1,2)
    signals_fig, ll_fig = axs

    fig.set_size_inches(10,5)

    signals_fig.set_title(f"Received Signals ($\\theta$ = {phase_offset} degrees)")
    signals_fig.set_xlim(-1.5, 1.5)
    signals_fig.set_ylim(-1.5, 1.5)
    signals_fig.scatter(received_stream.real, received_stream.imag, marker="+", s=16)
    signals_fig.scatter(corrected_stream.real, corrected_stream.imag, marker="+", s=16)
    signals_fig.scatter(qam.real, qam.imag, marker="+", s=16)
    signals_fig.legend(["Received Signals", f"Corrected by {theta_guess} deg", "32-QAM Symbols"], bbox_to_anchor=[.5,-.1])

    ll_fig.set_title(f"Log-Likelihood Function for 32-QAM Constellation")
    ll_fig.set_ylabel("Log-likelihood")
    ll_fig.set_xlabel("Theta in degrees")
    ll_fig.set_xlim(0,90)
    ll_fig.plot(theta_vals, ll_vals, zorder=1)
    ll_fig.scatter(theta_guess, get_log_likelihood(qam, received_stream, snr, theta_guess), c="orange", zorder=2)

    fig.show()



# Display Fig 3
# Graphs a typical log likelihood function for a 128-QAM constellation
def graph_log_likelihood():
    snr = QAM128_SNR_LOW # It is unknown what value was chosen for this plot in the actual figure
    K = 100
    actual_phase_offset = 45.0
    qam128 = qam128_new()

    signal_stream = np.random.choice(qam128, K)
    received_stream = get_received_stream(signal_stream, actual_phase_offset, snr)

    theta_vals = np.linspace(0,90,361)
    log_likelihood_vals = get_log_likelihood_arr(qam128, received_stream, snr, theta_vals)

    plt.title(f"Typical Log-Likelihood Function for 128-QAM Constellation ($\\theta$ = {actual_phase_offset} degrees)")
    plt.ylabel("Log-likelihood")
    plt.xlabel("Theta in degrees")
    plt.xlim(0,90)
    plt.plot(theta_vals, log_likelihood_vals)

    plt.show()

# Not a Fig, Graphs log-likelihood function for 128-QAM constellation at 2 degree increments and 2nd order newton's guess
def graph_ml_estimator():
    qam128 = qam128_new()

    true_offset, snr, received_stream = stream128_from_sample("Received 1 Low", 100) # Read 100 vals of received stream from qam128_samples

    theta_vals = np.arange(0,91,2)
    log_likelihood_vals = get_log_likelihood_arr(qam128, received_stream, snr, theta_vals)

    best_theta_guess = theta_vals[np.argmax(log_likelihood_vals)]

    plt.title("Log-Likelihood at $2^{\\circ}$ Increments")
    plt.ylabel("Log-Likelihood")
    plt.xlabel("Theta in degrees")
    plt.xlim(0,90)
    plt.scatter(theta_vals, log_likelihood_vals)
    plt.scatter(best_theta_guess, log_likelihood_vals.max())

    print(f"True Phase Offset: {true_offset}")
    print(f"Best Theta Guess: {best_theta_guess}")

# Not a Fig, graphs a closer look at how Newton's method improves the phase offset estimate
def graph_ml_estimator_with_newtons():
    qam = qam128_new()
    phase_offset, snr, received_stream = stream128_from_sample("Received 1 Low", 100)

    theta_vals = np.linspace(0,90,361) # Looking at many more values than will be considered in the ML algorithm
    ll_vals = get_log_likelihood_arr(qam, received_stream, snr, theta_vals)

    best_estimate = theta_vals[np.argmax(ll_vals[::8]) * 8] # Only look at every 2 degrees as this is how the ML algorithm works
    new_best_estimate = second_order_newtons_method_ml(qam, received_stream, snr, best_estimate)

    ll_new_best = get_log_likelihood(qam, received_stream, snr, new_best_estimate)

    fig, axs = plt.subplots(1,2)
    fig.set_size_inches(10,5)
    wide_fig, zoom_fig = axs

    wide_fig.set_title("ML with Newton's Method Applied")
    wide_fig.set_ylabel("Log-Likelihood")
    wide_fig.set_xlabel("Theta in degrees")
    wide_fig.set_xlim(0,90)
    wide_fig.plot(theta_vals, ll_vals)
    wide_fig.scatter(theta_vals[::8], ll_vals[::8])
    wide_fig.scatter(best_estimate, ll_vals[::8].max(), zorder=3)
    wide_fig.scatter(new_best_estimate, ll_new_best, marker="x", s=16, c="green", zorder=4)

    zoom_fig.set_title("ML with Newton's Method Applied (Zoomed in)")
    zoom_fig.set_ylabel("Log-Likelihood")
    zoom_fig.set_xlabel("Theta in degrees")
    zoom_fig.set_xlim(10,20)
    zoom_fig.plot(theta_vals[40:81], ll_vals[40:81])
    zoom_fig.scatter(theta_vals[40:81:8], ll_vals[40:81:8])
    zoom_fig.scatter(best_estimate, ll_vals[::8].max(), zorder=3)
    zoom_fig.scatter(new_best_estimate, ll_new_best, marker="x", c="green", zorder=4)

    fig.show()

    print(f"Actual Phase Offset: {phase_offset}")
    print(f"Best Rough Estimate: {best_estimate}")
    print(f"Best Guess with Newton's Method: {new_best_estimate}")

# Displays Figs 4-7
# Graphs the precalculated results of the ML estimator algorithm
def graph_ml_results():
    fig, axs = plt.subplots(4)
    fig.set_size_inches(5,20)
    ml32_fig, ml64_fig, ml128_fig, ml256_fig = axs

    def plot_ml_data(qam_size, ml_data, ml_fig, y_lo, y_hi):
        k_lo = ml_data["K"][0:10]
        k_hi = ml_data["K"][10:20]
        snr_lo = ml_data["SNR"][0]
        snr_hi = ml_data["SNR"][10]
        mean_sq_err_lo = np.array(ml_data["ML Results"][0:10])
        mean_sq_err_hi = np.array(ml_data["ML Results"][10:20])

        k_vals = np.arange(10,101,1)
        crb_lo = get_crb(snr_lo, k_vals)
        crb_hi = get_crb(snr_hi, k_vals)

        crb_lo = rad2_to_deg2(crb_lo)
        crb_hi = rad2_to_deg2(crb_hi)
        mean_sq_err_lo = rad2_to_deg2(mean_sq_err_lo)
        mean_sq_err_hi = rad2_to_deg2(mean_sq_err_hi)

        ml_fig.plot(k_vals, crb_lo)
        ml_fig.plot(k_vals, crb_hi)
        ml_fig.scatter(k_lo, mean_sq_err_lo, marker="x")
        ml_fig.scatter(k_hi, mean_sq_err_hi, marker="x")

        ml_fig.set_title(f"ML Estimation Performance for {qam_size}-QAM")
        ml_fig.set_ylabel("MSE in degrees squared")
        ml_fig.set_xlabel("Vector Length")
        ml_fig.set_xticks([10,20,30,40,50,60,70,80,90,100])
        ml_fig.set_xlim(10,100)
        ml_fig.set_ylim(y_lo, y_hi)
        ml_fig.set_yscale("log")
        ml_fig.grid(which="both")
        ml_fig.legend([f"CRB fo {snr_lo} dB", f"CRB for {snr_hi} dB", f"Simulation Results for {snr_lo} dB", f"Simulation Results for {snr_hi} dB"])

        return ml_fig

    ml32_data = pd.read_csv("data/qam32_ML_results.csv")
    ml64_data = pd.read_csv("data/qam64_ML_results.csv")
    ml128_data = pd.read_csv("data/qam128_ML_results.csv")
    ml256_data = pd.read_csv("data/qam256_ML_results.csv")

    ml32_fig = plot_ml_data(32, ml32_data, ml32_fig, 1e-2, 1e2)
    ml64_fig = plot_ml_data(64, ml64_data, ml64_fig, 1e-2, 1e2)
    ml128_fig = plot_ml_data(128, ml128_data, ml128_fig, 1e-2, 1e2)
    ml256_fig = plot_ml_data(256, ml256_data, ml256_fig, 1e-3, 1e2)

    fig.show()

# Not a Fig, used as an interactive plot
#  input: symbol - which symbol to use to show nearby energy levels [1,8]
#         p - how many nearby energy levels to highlight [0,5]
#  output: None, displays a scatterplot of 256-QAM constellation with nearby energy levels highlighted
def energy_levels_demonstration(symbol, p):
    qam256 = qam256_new()
    sml_qam256, energies = sml_qam256_new()
    received_symbol = qam256[136 + 17*(symbol-1)] # Chooses point along 45 degree diagonal going up and right from the center
    energy_level = get_closest_energy_level(energies, received_symbol)

    plt.scatter(qam256.real, qam256.imag, marker="+", c="pink", zorder=1) # Base 256-QAM
    plt.scatter(sml_qam256[energy_level].real, sml_qam256[energy_level].imag, marker="+", c="green", zorder=3) # Symbols on the same energy level 
    plt.scatter(received_symbol.real, received_symbol.imag, marker="+", c="blue", zorder=4) # The actual symbol chosen based on the symbol parameter

    # Highlight all nearby energy levels in red
    for i in range(1,p+1):
        below = max(energy_level - i, 0)
        above = min(energy_level + i, len(sml_qam256) - 1)
        plt.scatter(sml_qam256[below].real, sml_qam256[below].imag, marker="+", c="red", zorder=2)
        plt.scatter(sml_qam256[above].real, sml_qam256[above].imag, marker="+", c="red", zorder=2)

    plt.legend(["256-QAM Symbols", "Same Energy Level", "Received Symbol", "Similar Energy Level"], bbox_to_anchor=[.5,-.1])


    plt.title("Symbols Involved in Suboptimal Log-Likelihood Calculation")
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.grid()

    plt.show()

# Not a Fig, graphs the ML log-likelihood and SML log-likelihoods for the same received stream side by side to show the difference
def graph_both_log_likelihoods():
    qam256 = qam256_new()
    sml_qam256, energies = sml_qam256_new()
    _phase_offset, snr, received_stream = stream256_from_sample("Received 1 Low", 40)
    
    theta_vals = np.linspace(0,90,91)
    ml_ll_vals = get_log_likelihood_arr(qam256, received_stream, snr, theta_vals)
    sml_ll_vals = get_sml_log_likelihood_arr(sml_qam256, energies, received_stream, snr, theta_vals, 3)

    fig, axs = plt.subplots(1,2)
    fig.set_size_inches(10,5)
    ml_fig, sml_fig = axs

    ml_fig.set_title("Log-Likelihood for ML Estimation")
    ml_fig.set_ylabel("Log-Likelihood")
    ml_fig.set_xlabel("Theta in degrees")
    ml_fig.set_xlim(0,90)
    ml_fig.plot(theta_vals, ml_ll_vals)
    
    sml_fig.set_title("Log-Likelihood for SML Estimation ($p=3$)")
    sml_fig.set_xlabel("Theta in degrees")
    sml_fig.set_xlim(0,90)
    sml_fig.plot(theta_vals, sml_ll_vals)
   
    fig.show()

# Not a Fig, used as an interactive graph
# Plots the suboptimal log-likelihood for various values of p
def sml_p_demonstration(p):
    sml_qam128, energies = sml_qam128_new()
    _phase_offset, snr, received_stream = stream128_from_sample("Received 1 Low", 40)
    
    theta_vals = np.linspace(0,90,361)
    sml_ll_vals = get_sml_log_likelihood_arr(sml_qam128, energies, received_stream, snr, theta_vals, p)

    plt.title(f"Suboptimal Log-Likelihood ($p={p}$)")
    plt.ylabel("Log-Likelihood")
    plt.xlabel("Theta in degrees")
    plt.xlim(0,90)
    plt.plot(theta_vals, sml_ll_vals)
   
    plt.show()


# Displays Figs 8-14
# Graphs the precalculated results of the ML estimator algorithm
def graph_sml_results():
    fig, axs = plt.subplots(7,1)
    fig.set_size_inches(5,35)
    sml32_p0_fig = axs[0]
    sml32_p1_fig = axs[1]
    sml64_p1_fig = axs[2]
    sml128_p1_fig = axs[3]
    sml128_p2_fig = axs[4]
    sml256_p2_fig = axs[5]
    sml256_p3_fig = axs[6]

    def plot_sml_data(qam_size, sml_data, sml_fig, p, y_lo, y_hi):
        k_lo = sml_data["K"][0:10]
        k_hi = sml_data["K"][10:20]
        snr_lo = sml_data["SNR"][0]
        snr_hi = sml_data["SNR"][10]
        mean_sq_err_lo = np.array(sml_data["SML Results"][0:10])
        mean_sq_err_hi = np.array(sml_data["SML Results"][10:20])

        k_vals = np.arange(10,101,1)
        crb_lo = get_crb(snr_lo, k_vals)
        crb_hi = get_crb(snr_hi, k_vals)

        crb_lo = rad2_to_deg2(crb_lo)
        crb_hi = rad2_to_deg2(crb_hi)
        mean_sq_err_lo = rad2_to_deg2(mean_sq_err_lo)
        mean_sq_err_hi = rad2_to_deg2(mean_sq_err_hi)

        sml_fig.plot(k_vals, crb_lo)
        sml_fig.plot(k_vals, crb_hi)
        sml_fig.scatter(k_lo, mean_sq_err_lo, marker="x")
        sml_fig.scatter(k_hi, mean_sq_err_hi, marker="x")

        sml_fig.set_title(f"Suboptimal ML Estimation Performance for {qam_size}-QAM, p={p}")
        sml_fig.set_ylabel("MSE in degrees squared")
        sml_fig.set_xlabel("Vector Length")
        sml_fig.set_xticks([10,20,30,40,50,60,70,80,90,100])
        sml_fig.set_xlim(10,100)
        sml_fig.set_ylim(y_lo, y_hi)
        sml_fig.set_yscale("log")
        sml_fig.grid(which="both")
        sml_fig.legend([f"CRB fo {snr_lo} dB", f"CRB for {snr_hi} dB", f"Simulation Results for {snr_lo} dB", f"Simulation Results for {snr_hi} dB"])

        return sml_fig

    sml32_p0_data = pd.read_csv("data/qam32_SML_p0_results.csv")
    sml32_p1_data = pd.read_csv("data/qam32_SML_p1_results.csv")
    sml64_p1_data = pd.read_csv("data/qam64_SML_p1_results.csv")
    sml128_p1_data = pd.read_csv("data/qam128_SML_p1_results.csv")
    sml128_p2_data = pd.read_csv("data/qam128_SML_p2_results.csv")
    sml256_p2_data = pd.read_csv("data/qam256_SML_p2_results.csv")
    sml256_p3_data = pd.read_csv("data/qam256_SML_p3_results.csv")

    sml32_p0_fig = plot_sml_data(32, sml32_p0_data, sml32_p0_fig, 0, 1e-2, 1e3)
    sml32_p1_fig = plot_sml_data(32, sml32_p1_data, sml32_p1_fig, 1, 1e-2, 1e2)
    sml32_p0_fig = plot_sml_data(64, sml64_p1_data, sml64_p1_fig, 1, 1e-2, 1e3)
    sml32_p0_fig = plot_sml_data(128, sml128_p1_data, sml128_p1_fig, 1, 1e-2, 1e3)
    sml32_p0_fig = plot_sml_data(128, sml128_p2_data, sml128_p2_fig, 2, 1e-2, 1e3)
    sml32_p0_fig = plot_sml_data(256, sml256_p2_data, sml256_p2_fig, 2, 1e-3, 1e3)
    sml32_p0_fig = plot_sml_data(256, sml256_p3_data, sml256_p3_fig, 3, 1e-3, 1e3)

    fig.tight_layout()
    fig.show()