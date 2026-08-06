import sys
import os
from pathlib import Path
import math
import matplotlib.pyplot as plt
from datetime import datetime

# Adjust paths to match your local installation layout
sys.path.append("/opt/lumerical/v231/api/python/")
sys.path.append(os.path.dirname(__file__))

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import lumapi
import power_budget as pb

# ---------------------------------------------------------------------------
# Dimension Presets & Constants
# ---------------------------------------------------------------------------
WAVELENGTH_UM = 1.55
C_CONST = 299792458 
CORE_WIDTH_UM = 0.48
CORE_HEIGHT_UM = 0.22
ALPHA_CORE_DB_PER_CM = 0.01
ALPHA_CLAD_DB_PER_CM = 0.001
MMI_WIDTH_UM = 3.0
ACCESS_WIDTH_UM = CORE_WIDTH_UM  
ACCESS_LEN_UM = 5.0  
MOD_LENGTH_UM = 500.0  

script_dir = Path(__file__).resolve().parent
plots_dir = script_dir / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_SIM_FILENAME = str(script_dir / "mmi_waveguide_validation.lms")
OUTPUT_PLOT_FILENAME = str(plots_dir / f"mmi_waveguide_validation_{timestamp}.png")


def db_per_cm_to_k(alpha_db_per_cm, wavelength_um):
    """Converts a targeted dB/cm loss into an imaginary refractive index k."""
    alpha_db_per_m = alpha_db_per_cm * 100
    return alpha_db_per_m * (wavelength_um * 1e-6) / (4 * math.pi * 10 * math.log10(math.e))


# Initialize Lumerical Session with standard GUI visible
mode = lumapi.MODE(hide=False)

# ---------------------------------------------------------------------------
# Query Palik's Exact Complex Refractive Index at 1.55 um
# ---------------------------------------------------------------------------
freq_hz = C_CONST / (WAVELENGTH_UM * 1e-6)

# Pass frequency in Hz to getindex()
si_complex = mode.getindex("Si (Silicon) - Palik", freq_hz)
sio2_complex = mode.getindex("SiO2 (Glass) - Palik", freq_hz)

# Dynamic Palik Material Indices
N_CORE_PALIK = float(mode.real(si_complex))
N_CLAD_PALIK = float(mode.real(sio2_complex))

print(f"\n[Palik Database Lookup @ {WAVELENGTH_UM} um]")
print(f"  Si Core Index  (n) = {N_CORE_PALIK:.4f}")
print(f"  SiO2 Clad Index(n) = {N_CLAD_PALIK:.4f}\n")


# ---------------------------------------------------------------------------
# Waveguide FDE Simulation Function
# ---------------------------------------------------------------------------

def simulate_waveguide(width_um, height_um):
    k_core = db_per_cm_to_k(ALPHA_CORE_DB_PER_CM, WAVELENGTH_UM)
    k_clad = db_per_cm_to_k(ALPHA_CLAD_DB_PER_CM, WAVELENGTH_UM)

    mode.putv("WAVELENGTH_UM", float(WAVELENGTH_UM))
    mode.putv("N_CORE", float(N_CORE_PALIK))
    mode.putv("N_CLAD", float(N_CLAD_PALIK))
    mode.putv("width_um", float(width_um))
    mode.putv("height_um", float(height_um))
    mode.putv("k_core", float(k_core))
    mode.putv("k_clad", float(k_clad))

    script = f"""
    switchtolayout; 
    selectall; 
    delete;

    wavelength = WAVELENGTH_UM * 1e-6;
    core_w = width_um * 1e-6;
    core_h = height_um * 1e-6;

    # Core Waveguide using explicit complex dielectric index
    addrect;
    set('name', 'waveguide');
    set('x span', core_w);
    set('y span', core_h);
    set('z span', 10e-6);
    set('x', 0); set('y', 0); set('z', 0);
    set('material', '<Object defined dielectric>');
    set('index', N_CORE + 1i*k_core);

    # FDE Solver Region
    addfde;
    set('solver type', '2D Z normal');
    set('x span', 5e-6);
    set('y span', 2.5e-6);
    set('x', 0); set('y', 0); set('z', 0);
    set('background material', '<Object defined dielectric>');
    set('index', N_CLAD + 1i*k_clad);
    set('x min bc', 'PML'); set('x max bc', 'PML');
    set('y min bc', 'PML'); set('y max bc', 'PML');
    set('wavelength', wavelength);
    set('number of trial modes', 5);

    findmodes;
    
    # FIX #2: Explicit TE fundamental mode identification
    target_mode = "mode1";
    for (m=1:5) {{
        m_name = "mode" + num2str(m);
        if (getdata(m_name, "TE fraction") > 0.5) {{
            target_mode = m_name;
            break;
        }}
    }}

    neff_complex = getdata(target_mode, "neff");
    n_eff_real = real(neff_complex);
    neff_imag = imag(neff_complex);

    # FIX #1: Correct units conversion for dB/cm propagation loss
    alpha_calc = ((4*pi*neff_imag)/wavelength) * 10*log10(exp(1)) * 100;

    x = getdata(target_mode, "x");
    y = getdata(target_mode, "y");
    E2 = pinch(getelectric(target_mode));

    nx = length(x);
    ny = length(y);

    if (size(E2, 1) != nx) {{
        E2 = transpose(E2);
    }}

    p_core = 0;
    p_total = sum(E2);

    for (i = 1:nx) {{
        if (abs(x(i)) <= core_w/2) {{
            for (j = 1:ny) {{
                if (abs(y(j)) <= core_h/2) {{
                    p_core = p_core + E2(i, j);
                }}
            }}
        }}
    }}

    confinement_calc = p_core / p_total;
    """

    try:
        mode.eval(script)
    except lumapi.LumApiError as err:
        print(f"[ERROR] Failed running simulate_waveguide for width={width_um} um")
        raise err

    return (
        float(mode.getv("confinement_calc")),
        float(mode.getv("n_eff_real")),
        float(mode.getv("alpha_calc"))
    )


def analytical_waveguide(width_um, height_um):
    # FIX #3: Pass cladding refractive index explicitly for strict alignment
    confinement, n_eff = pb.strip_confinement(N_CORE_PALIK, N_CLAD_PALIK, width_um, height_um, WAVELENGTH_UM)
    alpha = pb.material_loss_db_per_cm(confinement, ALPHA_CORE_DB_PER_CM, ALPHA_CLAD_DB_PER_CM)
    return confinement, n_eff, alpha


# ---------------------------------------------------------------------------
# MMI EME Simulation Function
# ---------------------------------------------------------------------------

def simulate_mmi_eme(mmi_length_um):
    mode.putv("WAVELENGTH_UM", float(WAVELENGTH_UM))
    mode.putv("ACCESS_WIDTH_UM", float(ACCESS_WIDTH_UM))
    mode.putv("CORE_HEIGHT_UM", float(CORE_HEIGHT_UM))
    mode.putv("MMI_WIDTH_UM", float(MMI_WIDTH_UM))
    mode.putv("ACCESS_LEN_UM", float(ACCESS_LEN_UM))
    mode.putv("mmi_length_um", float(mmi_length_um))
    mode.putv("N_CORE", float(N_CORE_PALIK))
    mode.putv("N_CLAD", float(N_CLAD_PALIK))

    script = """
    switchtolayout; 
    selectall; 
    delete;

    wavelength = WAVELENGTH_UM * 1e-6;
    access_w   = ACCESS_WIDTH_UM * 1e-6;
    wg_height  = CORE_HEIGHT_UM * 1e-6;
    mmi_w      = MMI_WIDTH_UM * 1e-6;
    access_len = ACCESS_LEN_UM * 1e-6;
    mmi_len    = mmi_length_um * 1e-6;

    # Input Waveguide
    addrect; set('name','in_1');
    set('x min', -access_len - mmi_len/2); set('x max', -mmi_len/2);
    set('y', 0); set('y span', access_w);
    set('z', 0); set('z span', wg_height);
    set('material', '<Object defined dielectric>');
    set('index', N_CORE);

    # MMI Body
    addrect; set('name','MMI');
    set('x min', -mmi_len/2); set('x max', mmi_len/2);
    set('y', 0); set('y span', mmi_w);
    set('z', 0); set('z span', wg_height);
    set('material', '<Object defined dielectric>');
    set('index', N_CORE);

    # Output Waveguide
    addrect; set('name','out_1');
    set('x min', mmi_len/2); set('x max', mmi_len/2 + access_len);
    set('y', 0); set('y span', access_w);
    set('z', 0); set('z span', wg_height);
    set('material', '<Object defined dielectric>');
    set('index', N_CORE);

    # EME Region
    addeme;
    set('solver type', '3D: X Prop');
    set('wavelength', wavelength);
    set('background material', '<Object defined dielectric>');
    set('index', N_CLAD);
    set('x min', -access_len - mmi_len/2);
    set('y', 0); set('y span', mmi_w * 1.8);
    set('z', 0); set('z span', wg_height * 4);

    # FIX #4: Dense mesh cells to remove mode-mismatch and radiation artifacts
    set('number of cell groups', 3);
    set('group spans', [access_len; mmi_len; access_len]);
    set('cells', [5; 50; 5]);

    select("EME");
    set("allow custom eigensolver settings", 1);

    select("EME::Cells::cell_1");
    seteigensolver("number of trial modes", 10);

    select("EME::Cells::cell_2");
    seteigensolver("number of trial modes", 50);

    select("EME::Cells::cell_3");
    seteigensolver("number of trial modes", 10);

    # Ports
    addemeport; 
    set('port location', 'left'); 
    set('use full simulation span', 0);
    set('y', 0); set('y span', access_w * 3);
    set('z', 0); set('z span', wg_height * 3);
    set('mode selection', 'fundamental TE mode');

    addemeport; 
    set('port location', 'right'); 
    set('use full simulation span', 0);
    set('y', 0); set('y span', access_w * 3);
    set('z', 0); set('z span', wg_height * 3);
    set('mode selection', 'fundamental TE mode');

    run;
    emepropagate;

    S = getresult("EME", "user s matrix");
    if (isstruct(S)) {
        s_matrix = S.S;
    } else {
        s_matrix = S;
    }

    transmitted_power = abs(s_matrix(2, 1))^2;
    mmi_excess_loss_db = -10 * log10(transmitted_power);
    """

    try:
        mode.eval(script)
    except lumapi.LumApiError as err:
        print(f"[ERROR] Failed running simulate_mmi_eme for length={mmi_length_um} um")
        raise err

    return float(mode.getv("mmi_excess_loss_db"))


# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- 1. Waveguide FDE Sweeps ---
    widths = [0.4, 0.45, 0.48, 0.52, 0.6]
    sim_neff, ana_neff = [], []
    sim_conf, ana_conf = [], []
    sim_alpha, ana_alpha = [], []

    print("Validating Waveguide Modes using Lumerical FDE...")
    for w in widths:
        c_sim, n_sim, a_sim = simulate_waveguide(w, CORE_HEIGHT_UM)
        c_ana, n_ana, a_ana = analytical_waveguide(w, CORE_HEIGHT_UM)
        
        sim_neff.append(n_sim); ana_neff.append(n_ana)
        sim_conf.append(c_sim); ana_conf.append(c_ana)
        sim_alpha.append(a_sim); ana_alpha.append(a_ana)
        print(f"  Finished width = {w:.2f} um | n_eff = {n_sim:.5f} | Confinement = {c_sim:.4f} | Loss = {a_sim:.4e} dB/cm")

    # --- 2. MMI EME Sweeps ---
    n_eff_vertical = pb.slab_mode(N_CORE_PALIK, N_CLAD_PALIK, CORE_HEIGHT_UM, WAVELENGTH_UM)[1]
    
    optimal_length_um, _, _ = pb.mmi_derive_length_and_loss(
        n_eff_vertical, N_CLAD_PALIK, MMI_WIDTH_UM, 2, ACCESS_WIDTH_UM, WAVELENGTH_UM
    )
    if optimal_length_um <= 0:
        optimal_length_um = 4.61  # Fallback length

    lengths_um = [optimal_length_um * f for f in (0.5, 0.75, 1.0, 1.25, 1.5)]
    sim_excess, ana_excess = [], []

    print("\nValidating MMI Excess Loss using Lumerical EME...")
    for l_um in lengths_um:
        e_ana, _ = pb.mmi_excess_loss_db(
            n_eff_vertical, N_CLAD_PALIK, MMI_WIDTH_UM, 2, ACCESS_WIDTH_UM, WAVELENGTH_UM, l_um
        )
        e_sim = simulate_mmi_eme(l_um)
        sim_excess.append(e_sim)
        ana_excess.append(e_ana)
        print(f"  Finished length = {l_um:.3f} um | Excess Loss = {e_sim:.4f} dB")

    # --- 3. Plotting Grid ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Effective Index Plot
    axes[0, 0].plot(widths, sim_neff, 'o-', color='#1f77b4', linewidth=2, label='Simulated (FDE)')
    axes[0, 0].plot(widths, ana_neff, 's--', color='#ff7f0e', linewidth=2, label='Analytical (EIM)')
    axes[0, 0].set_xlabel('Core Width (µm)'); axes[0, 0].set_ylabel('Real Effective Index ($n_{eff}$)')
    axes[0, 0].set_title('Effective Index Comparison'); axes[0, 0].grid(True, linestyle='--', alpha=0.6); axes[0, 0].legend()

    # Confinement Factor Plot
    axes[0, 1].plot(widths, sim_conf, 'o-', color='#1f77b4', linewidth=2, label='Simulated (FDE)')
    axes[0, 1].plot(widths, ana_conf, 's--', color='#ff7f0e', linewidth=2, label='Analytical (EIM)')
    axes[0, 1].set_xlabel('Core Width (µm)'); axes[0, 1].set_ylabel('Confinement Factor ($\Gamma$)')
    axes[0, 1].set_title('Confinement Factor Comparison'); axes[0, 1].grid(True, linestyle='--', alpha=0.6); axes[0, 1].legend()

    # Propagation Loss Plot
    axes[1, 0].plot(widths, sim_alpha, 'o-', color='#1f77b4', linewidth=2, label='Simulated (FDE)')
    axes[1, 0].plot(widths, ana_alpha, 's--', color='#ff7f0e', linewidth=2, label='Analytical (Perturbation)')
    axes[1, 0].set_xlabel('Core Width (µm)'); axes[1, 0].set_ylabel('Loss $\\alpha$ (dB/cm)')
    axes[1, 0].set_title('Propagation Loss Comparison'); axes[1, 0].grid(True, linestyle='--', alpha=0.6); axes[1, 0].legend()

    # MMI Excess Loss Plot
    axes[1, 1].plot(lengths_um, sim_excess, 'o-', color='#1f77b4', linewidth=2, label='Simulated (EME)')
    axes[1, 1].plot(lengths_um, ana_excess, 's--', color='#ff7f0e', linewidth=2, label='Analytical (Overlap)')
    axes[1, 1].axvline(optimal_length_um, color='gray', linestyle=':', label='Derived Optimal Length')
    axes[1, 1].set_xlabel('MMI Length (µm)'); axes[1, 1].set_ylabel('Excess Loss (dB)')
    axes[1, 1].set_title('MMI Excess Loss Comparison'); axes[1, 1].grid(True, linestyle='--', alpha=0.6); axes[1, 1].legend()

    fig.suptitle('power_budget.py Analytical Model vs Lumerical Simulation', fontsize=14, fontweight='bold')
    plt.tight_layout()

    plt.savefig(OUTPUT_PLOT_FILENAME, dpi=300)
    print(f"\nSummary plot saved successfully to '{OUTPUT_PLOT_FILENAME}'.")
    plt.show()