from emilie_simulation import *

### User relevant: adapt this if needed
# Setting the simulation parameters, change if needed. Everything is in SI units.
p = set_p_standard()
p["Esin"], p["nu"] = 250e9, 0.23 #TODO: are these values right?
p["sigcr"] = 1e9
p["sigau"] = 40e6
p["rhosin"] = 3440
p["rhocr"] = 7140
p["rhoau"] = 19320
p["Lside"] = 1e-3
p["hsin"] = 50e-9
p["hcr"] = 10e-9
p["hau"] = 90e-9
p["el_width"] = 5e-6
p["mpercentage_perf"] = 0.62 #how much mass density is left on the perforated area?


### User relevant: put your data here
# Put the measured frequencies here (in Hz. example below: f(1,1) = 120kHz)
freq_dict = {
    "(1,1)": 120e3,
    "(2,1)": 160e3,
    "(1,2)": 180e3
}
measured_21 = True # Set true if you have measured the (2,1) mode frequency
measured_12 = True # Set true if you have measured the (1,2) mode frequency


### Not user relevant: change nothing:
# estimating the prestress using a fit function to simulation data
p["sigsin"] = prestress_estimator(freq_dict["(1,1)"])
print("Estimated prestress: ", p["sigsin"])
sigsins = { #fitted prestresses, do not change.
    "(1,1)": 0,
    "(2,1)": 0,
    "(1,2)": 0
}
mesh = generate_mesh(p)
def fit_freq(p, mode, mesh, sigsins):
    feswave, eigenvals, multigfuwave = solve(p, freq_to_fit_to = freq_dict[mode], mode_to_fit_to = mode, ela = True)
    sigsins[mode] = p["sigsin"]
    result = result_dict(p, mesh, feswave, multigfuwave, eigenvals)
    return result
#fitting the (1,1) mode
result = fit_freq(p, "(1,1)", mesh, sigsins)
print("Prestress when fitting f(1,1): " + str(p["sigsin"]/1e6) + " MPa")
print("from that the expected second mode freqs are: f(2,1)=" + str(result["(2,1)"][0]) + " kHz and f(1,2)=" + str(result["(1,2)"][0]) + " kHz")

#fitting the (2,1) mode
if measured_21:
    result = fit_freq(p, "(2,1)", mesh, sigsins)
    print("Prestress when fitting f(2,1): " + str(p["sigsin"]/1e6) + " MPa")
    print("from that the expected other mode freqs are: f(1,1)=" + str(result["(1,1)"][0]) + " kHz and f(1,2)=" + str(result["(1,2)"][0]) + " kHz")

#fitting the (1,2) mode
if measured_12:
    result = fit_freq(p, "(1,2)", mesh, sigsins)
    print("Prestress when fitting f(1,2): " + str(p["sigsin"]/1e6) + " MPa")
    print("from that the expected other mode freqs are: f(1,1)=" + str(result["(1,1)"][0]) + " kHz and f(2,1)=" + str(result["(2,1)"][0]) + " kHz")

avg_sigsin = (sigsins["(1,1)"] + sigsins["(2,1)"] + sigsins["(1,2)"]) / (1 + measured_21 + measured_12)
print("Averaging gives an estimate of sigsin=" + str(avg_sigsin/1e6) + " MPa")