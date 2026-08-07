import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Adjust path to match your local Lumerical API installation
sys.path.append("/opt/lumerical/v231/api/python/")
import lumapi

# ---------------------------------------------------------------------------
# 1. Extract Pure Frequency-Domain Matrix from Lumerical Layout
# ---------------------------------------------------------------------------
print("Connecting to Lumerical session...")
mode = lumapi.MODE(hide=False)

mode.eval("""
addeme;
select("EME");
wavelength_center = get("wavelength"); # Center wavelength in meters
c_const = 299792458;

# Derive f0 and df directly:
f0 = c_const / wavelength_center;

# Define a 41.57 THz bandwidth matching your target band
df = 41.5768e12; 

f_vec = linspace(f0 - 2*df, f0 + 2*df, 1000);
sigma_f = df / (2 * sqrt(2 * log(2)));
power_spectrum = exp(-0.5 * ((f_vec - f0) / sigma_f)^2);
""")

# Pull back into Python
f_hz = np.array(mode.getv("f_vec")).flatten()
power_spectrum = np.array(mode.getv("power_spectrum")).flatten()
source_matrix_freq = np.column_stack((f_hz, power_spectrum))

print(f_hz)
print(power_spectrum)
print(source_matrix_freq)

# ---------------------------------------------------------------------------
# 2. Plotting the Spectrum
# ---------------------------------------------------------------------------
c_const = 299792458
freq_thz = f_hz / 1e12
wavelength_um = (c_const / f_hz) * 1e6

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Subplot 1: Power Spectrum vs Frequency (THz)
ax1.plot(freq_thz, power_spectrum, color='#1f77b4', linewidth=2)
ax1.set_xlabel('Frequency (THz)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Normalized Power Spectral Density $|S(f)|^2$', fontsize=11, fontweight='bold')
ax1.set_title('Spectrum vs Frequency', fontsize=12, fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.6)

# Subplot 2: Power Spectrum vs Wavelength (µm)
ax2.plot(wavelength_um, power_spectrum, color='#d62728', linewidth=2)
ax2.set_xlabel('Wavelength ($\mu$m)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Normalized Power Spectral Density $|S(\lambda)|^2$', fontsize=11, fontweight='bold')
ax2.set_title('Spectrum vs Wavelength', fontsize=12, fontweight='bold')
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

# Use this bulshit in matrix_power_budget vvvvv 


# # ---------------------------------------------------------------------------
# # 2. Transcendental Solver: tan(kh) = (gamma1 + gamma2)/(k*(1 - gamma1*gamma2/k^2))
# # ---------------------------------------------------------------------------
# def solve_transcendental_mode(k0, h, n_core, n_clad1, n_clad2):
#     n_min = max(n_clad1, n_clad2) + 1e-5
#     n_max = n_core - 1e-5
#     n_eff_grid = np.linspace(n_min, n_max, 2000)
    
#     # Transverse wavenumber & cladding decay constants
#     k = k0 * np.sqrt(n_core**2 - n_eff_grid**2)
#     gamma1 = k0 * np.sqrt(n_eff_grid**2 - n_clad1**2)
#     gamma2 = k0 * np.sqrt(n_eff_grid**2 - n_clad2**2)

#     # Residual calculation for tan(kh) = RHS
#     lhs = np.tan(k * h)
#     rhs = (gamma1 + gamma2) / (k * (1.0 - (gamma1 * gamma2) / (k**2)))
    
#     residual = np.abs(lhs - rhs)
#     best_idx = np.argmin(residual)
    
#     return n_eff_grid[best_idx], k[best_idx]


# # ---------------------------------------------------------------------------
# # 3. Calculation Pipeline Using Frequency Matrix
# # ---------------------------------------------------------------------------
# def run_frequency_domain_analysis(source_matrix, h_core_m, n_core, n_clad1, n_clad2):
#     c_const = 299792458
#     freqs = source_matrix[:, 0]
#     S_power = source_matrix[:, 1]
    
#     neff_list, k_list = [], []
#     for f in freqs:
#         k0 = 2 * np.pi * f / c_const
#         n_eff, k_val = solve_transcendental_mode(k0, h_core_m, n_core, n_clad1, n_clad2)
#         neff_list.append(n_eff)
#         k_list.append(k_val)
        
#     neff_arr = np.array(neff_list)
#     k_arr = np.array(k_list)

#     print(neff_arr)
#     print(k_arr)
    
#     # Spectrally weighted integration across the frequency spectrum
#     norm_factor = np.trapz(S_power, freqs)
#     weighted_neff = np.trapz(neff_arr * S_power, freqs) / norm_factor
#     weighted_k = np.trapz(k_arr * S_power, freqs) / norm_factor
    
#     return weighted_neff, weighted_k, freqs


# # ---------------------------------------------------------------------------
# # 4. Main Routine
# # ---------------------------------------------------------------------------
# if __name__ == "__main__":
#     h_core = 0.22e-6   # 220 nm waveguide height
#     n_core = 3.47      # Silicon
#     n_clad = 1.444     # SiO2

#     weighted_neff, weighted_k, freqs = run_frequency_domain_analysis(
#         source_matrix_freq, h_core, n_core, n_clad, n_clad
#     )

#     print("\n--- Frequency-Domain Results ---")
#     print(f"Frequency Range: {freqs[0]/1e12:.2f} THz - {freqs[-1]/1e12:.2f} THz")
#     print(f"Spectrally Weighted Effective Index (n_eff): {weighted_neff:.5f}")
#     print(f"Spectrally Weighted Transverse Wavenumber (k): {weighted_k:.4e} rad/m")