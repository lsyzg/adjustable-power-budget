import sys, os
from pathlib import Path
import math
import matplotlib.pyplot as plt

# Adjust paths to match your local installation layout
sys.path.append("/opt/lumerical/v231/api/python/")
sys.path.append(os.path.dirname(__file__))

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import lumapi
import power_budget as pb

# ---------------------------------------------------------------------------
# Dimension Presets (Matching gui.py & power_budget.py defaults)
# ---------------------------------------------------------------------------
WAVELENGTH_UM = 1.55
N_CORE = 3.45
N_CLAD = 1.44
CORE_WIDTH_UM = 0.48
CORE_HEIGHT_UM = 0.22
ALPHA_CORE_DB_PER_CM = 0.01
ALPHA_CLAD_DB_PER_CM = 0.001
MMI_WIDTH_UM = 3.0
N_PORTS = 2
ACCESS_WIDTH_UM = CORE_WIDTH_UM  
ACCESS_LEN_UM = 5.0  
MOD_LENGTH_UM = 500.0  


def db_per_cm_to_k(alpha_db_per_cm, wavelength_um):
    """Converts a targeted dB/cm loss into an imaginary refractive index k."""
    alpha_db_per_m = alpha_db_per_cm * 100
    return alpha_db_per_m * (wavelength_um * 1e-6) / (4 * math.pi * 10 * math.log10(math.e))


# Initialize the Lumerical MODE Session
mode = lumapi.MODE()

# ---------------------------------------------------------------------------
# Waveguide FDE: Effective Index, Confinement, and Propagation Loss
# ---------------------------------------------------------------------------

def simulate_waveguide(width_um, height_um):
    k_core = db_per_cm_to_k(ALPHA_CORE_DB_PER_CM, WAVELENGTH_UM)
    k_clad = db_per_cm_to_k(ALPHA_CLAD_DB_PER_CM, WAVELENGTH_UM)

    mode.putv("WAVELENGTH_UM", float(WAVELENGTH_UM))
    mode.putv("N_CORE", float(N_CORE))
    mode.putv("N_CLAD", float(N_CLAD))
    mode.putv("width_um", float(width_um))
    mode.putv("height_um", float(height_um))
    mode.putv("k_core", float(k_core))
    mode.putv("k_clad", float(k_clad))

    script = """
    switchtolayout; 
    selectall; 
    delete;

    wavelength = WAVELENGTH_UM * 1e-6;
    core_w = width_um * 1e-6;
    core_h = height_um * 1e-6;

    # Core Waveguide (Centered at origin, cross-section in X-Y)
    addrect;
    set('name', 'waveguide');
    set('x span', core_w);
    set('y span', core_h);
    set('z span', 10e-6);
    set('x', 0); set('y', 0); set('z', 0);
    set('index', N_CORE + 1i*k_core);

    # FDE Solver Region (2D Z normal for X-Y cross section)
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

    # Solve modes
    findmodes;
    
    # Extract fundamental mode effective index
    neff_complex = getdata("mode1", "neff");
    n_eff_real = real(neff_complex);
    neff_imag = imag(neff_complex);
    alpha_calc = ((4*pi*neff_imag)/wavelength) * 10*log10(exp(1)) / 100;

    # Extract spatial grid and total electric field intensity
    x = getdata("mode1", "x");
    y = getdata("mode1", "y");
    E2 = pinch(getelectric("mode1"));

    nx = length(x);
    ny = length(y);

    # Align matrix orientation [Nx, Ny]
    if (size(E2, 1) != nx) {
        E2 = transpose(E2);
    }

    # Integrate intensity over core region
    p_core = 0;
    p_total = sum(E2);

    for (i = 1:nx) {
        if (abs(x(i)) <= core_w/2) {
            for (j = 1:ny) {
                if (abs(y(j)) <= core_h/2) {
                    p_core = p_core + E2(i, j);
                }
            }
        }
    }

    confinement_calc = p_core / p_total;
    """

    try:
        mode.eval(script)
    except lumapi.LumApiError as err:
        print(f"[ERROR] Failed running simulate_waveguide for width={width_um} um")
        raise err

    n_eff = mode.getv("n_eff_real")
    confinement = mode.getv("confinement_calc")
    alpha = mode.getv("alpha_calc")

    return confinement, n_eff, alpha


def analytical_waveguide(width_um, height_um):
    confinement, n_eff = pb.strip_confinement(N_CORE, N_CLAD, width_um, height_um, WAVELENGTH_UM)
    alpha = pb.material_loss_db_per_cm(confinement, ALPHA_CORE_DB_PER_CM, ALPHA_CLAD_DB_PER_CM)
    return confinement, n_eff, alpha


# ---------------------------------------------------------------------------
# MMI EME: Excess Loss vs Device Length Validation Loop
# ---------------------------------------------------------------------------

def simulate_mmi_eme(mmi_length_um):
    # Obtain precise tap coordinates computed from analytical logic
    tap_positions_um = [pb.mmi_tap_position(k, N_PORTS, MMI_WIDTH_UM) for k in range(1, N_PORTS + 1)]

    mode.putv("WAVELENGTH_UM", float(WAVELENGTH_UM))
    mode.putv("ACCESS_WIDTH_UM", float(ACCESS_WIDTH_UM))
    mode.putv("CORE_HEIGHT_UM", float(CORE_HEIGHT_UM))
    mode.putv("MMI_WIDTH_UM", float(MMI_WIDTH_UM))
    mode.putv("ACCESS_LEN_UM", float(ACCESS_LEN_UM))
    mode.putv("mmi_length_um", float(mmi_length_um))
    mode.putv("MOD_LENGTH_UM", float(MOD_LENGTH_UM))
    mode.putv("tap_positions_um", tap_positions_um)
    mode.putv("N_PORTS", int(N_PORTS))

    script = """
    switchtolayout; selectall; delete;

    wavelength = WAVELENGTH_UM * 1e-6;
    access_w = ACCESS_WIDTH_UM * 1e-6;
    wg_height = CORE_HEIGHT_UM * 1e-6;
    mmi_w = MMI_WIDTH_UM * 1e-6;
    access_len = ACCESS_LEN_UM * 1e-6;
    mmi_len = mmi_length_um * 1e-6;
    mod_len = MOD_LENGTH_UM * 1e-6;

    # Input waveguide (Port 1 side)
    addrect; set('name','in_1');
    set('x min', -access_len - mmi_len/2); set('x max', -mmi_len/2);
    set('y', 0); set('y span', access_w);
    set('z', 0); set('z span', wg_height);
    set('material', 'Si (Silicon) - Palik');

    # Central MMI region body
    addrect; set('name','MMI');
    set('x min', -mmi_len/2); set('x max', mmi_len/2);
    set('y', 0); set('y span', mmi_w);
    set('z', 0); set('z span', wg_height);
    set('material', 'Si (Silicon) - Palik');

    # Dynamically generate N balanced output ports & Modulator segments
    for(k=1:N_PORTS) {
        y_pos = tap_positions_um(k) * 1e-6;
        
        # Output Port Segment
        addrect; set('name','out_' + num2str(k));
        set('x min', mmi_len/2); set('x max', mmi_len/2 + access_len);
        set('y', y_pos); set('y span', access_w);
        set('z', 0); set('z span', wg_height);
        set('material', 'Si (Silicon) - Palik');
        
        # Modulator Arm Segment extension
        addrect; set('name','mod_' + num2str(k));
        set('x min', mmi_len/2 + access_len); set('x max', mmi_len/2 + access_len + mod_len);
        set('y', y_pos); set('y span', access_w);
        set('z', 0); set('z span', wg_height);
        set('material', 'Si (Silicon) - Palik');
    }

    # Add EME Solver Region setup
    addeme;
    set('allow cross-section extraction', true);
    set('wavelength', wavelength);
    set('x min', -access_len - mmi_len/2);
    
    # Cover total span up through the modulator arrays
    total_x_max = mmi_len/2 + access_len + mod_len;
    set('x max', total_x_max);
    set('y', 0); set('y span', mmi_w * 1.5);
    set('z', 0); set('z span', wg_height * 4);

    # Define EME Solver Groups (1: input access, 2: MMI Body, 3: outputs, 4: modulators)
    set('number of cell groups', 4);
    set('group spans', [access_len; mmi_len; access_len; mod_len]);
    set('cells', [1; 10; 1; 1]);
    set('modes', [5; 20; 5; 5]); # Balanced mode sizing targets

    # Setup Port Definitions 
    addemeport; set('port location', 'left'); set('y', 0); set('y span', access_w*2);
    
    for(k=1:N_PORTS) {
        y_pos = tap_positions_um(k) * 1e-6;
        addemeport; 
        set('port location', 'right'); 
        set('use custom port reference coordinates', true);
        set('port reference x', mmi_len/2 + access_len + mod_len);
        set('y', y_pos); set('y span', access_w*2);
    }

    # Run full EME S-Matrix cascades
    run;
    emegrow;
    
    # Query transmission from standard input port 1 to output targets 
    s_matrix = getresult("eme", "s");
    total_transmitted_power = 0;
    for(k=2:(N_PORTS+1)) {
        total_transmitted_power = total_transmitted_power + abs(s_matrix(k, 1))^2;
    }
    
    # Calculate global excess loss parameter
    mmi_excess_loss_db = -10 * log10(total_transmitted_power);
    """
    
    mode.eval(script)
    return mode.getv("mmi_excess_loss_db")

# ---------------------------------------------------------------------------
# Multi-point Validation Iteration Engine & Local native Plotting
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    widths = [0.4, 0.45, 0.48, 0.52, 0.6]
    sim_neff, ana_neff = [], []

    print("Validating Waveguide Modes using Lumerical FDE...")
    for w in widths:
        _, n_sim, _ = simulate_waveguide(w, CORE_HEIGHT_UM)
        _, n_ana = pb.strip_confinement(N_CORE, N_CLAD, w, CORE_HEIGHT_UM, WAVELENGTH_UM)
        sim_neff.append(float(n_sim))
        ana_neff.append(float(n_ana))

    # ---------------------------------------------------------------------------
    # Plotting directly in Python via Matplotlib (Bypasses Lumerical GUI)
    # ---------------------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    
    plt.plot(widths, sim_neff, 'o-', color='#1f77b4', linewidth=2, label='Rigorous Numerical Solve (FDE)')
    plt.plot(widths, ana_neff, 's--', color='#ff7f0e', linewidth=2, label='Analytical EIM Approximation')
    
    plt.xlabel('Core Width (µm)', fontsize=12)
    plt.ylabel('Real Effective Index ($n_{eff}$)', fontsize=12)
    plt.title('Silicon Waveguide Effective Index Validation', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    
    # Save the figure to file and display
    plt.savefig('waveguide_validation.png', dpi=300)
    print("Plot saved successfully to 'waveguide_validation.png'.")
    plt.show()