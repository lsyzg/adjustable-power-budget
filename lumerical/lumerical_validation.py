# run in server, test 'inverse design' to find alpha (db/cm) of waveguide

import sys, os
import math

sys.path.append("/opt/lumerical/v231/api/python/")
sys.path.append(os.path.dirname(__file__))

import lumapi, lumopt

WAVELENGTH_M = 1.55e-6

mode = lumapi.MODE()
mode.eval(open('lumerical_setup.lsf').read())
neff_imag = mode.get_imag()

alpha_db_per_m = (4 * math.pi * neff_imag / WAVELENGTH_M) * 10 * math.log10(math.e)
alpha_db_per_cm = alpha_db_per_m / 100

print(f"neff_imag = {neff_imag}")
print(f"alpha = {alpha_db_per_cm} dB/cm")