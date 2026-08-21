import sys
import os
from pathlib import Path
import math
import time
import matplotlib
from datetime import datetime

# uncomment when disowning
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["DISPLAY"] = ""

if os.environ.get("DISPLAY") == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
# Which validation(s) to run: "waveguide", "mmi", or "both"
RUN_VALIDATION = "mmi"

WAVELENGTH_UM = 1.55
C_CONST = 299792458
CORE_WIDTH_UM = 0.48
CORE_HEIGHT_UM = 0.22
MMI_WIDTH_UM = 3.0
ACCESS_WIDTH_UM = CORE_WIDTH_UM
ACCESS_LEN_UM = 5.0
MOD_LENGTH_UM = 500.0

# Waveguide width sweep (evenly spaced, inclusive of both endpoints)
WIDTH_SWEEP_START_UM = 0.4
WIDTH_SWEEP_STOP_UM = 0.6
WIDTH_SWEEP_STEPS = 50

# MMI length sweep, as a fraction of the derived optimal self-imaging
# length (absolute lengths aren't known until that's derived at runtime)
LENGTH_SWEEP_START_FACTOR = 0.5
LENGTH_SWEEP_STOP_FACTOR = 1.5
LENGTH_SWEEP_STEPS = 50


def linspace(start, stop, steps):
    if steps < 2:
        return [start]
    step_size = (stop - start) / (steps - 1)
    return [start + i * step_size for i in range(steps)]

N_CORE = pb.material_refractive_index("Si", WAVELENGTH_UM)
N_CLAD = pb.material_refractive_index("SiO2", WAVELENGTH_UM)
ALPHA_CORE_DB_PER_CM = pb.material_bulk_loss_db_per_cm("Si", WAVELENGTH_UM)
ALPHA_CLAD_DB_PER_CM = pb.material_bulk_loss_db_per_cm("SiO2", WAVELENGTH_UM)

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
mode = lumapi.MODE(hide=True)

# ---------------------------------------------------------------------------
# Cross-check: Lumerical's own live Palik lookup vs. materials_data.py
# (N_CORE/N_CLAD above, used for all the actual calculations below, come
# from power_budget.py's bundled dataset -- this just confirms that
# dataset roughly agrees with Lumerical's own Palik database at this
# wavelength; it isn't fed into anything downstream).
# ---------------------------------------------------------------------------
freq_hz = C_CONST / (WAVELENGTH_UM * 1e-6)

si_complex = mode.getindex("Si (Silicon) - Palik", freq_hz)
sio2_complex = mode.getindex("SiO2 (Glass) - Palik", freq_hz)

N_CORE_PALIK_LIVE = float(mode.real(si_complex))
N_CLAD_PALIK_LIVE = float(mode.real(sio2_complex))

print(f"\n[Palik Database Lookup @ {WAVELENGTH_UM} um]")
print(f"  Si Core Index  (n): Lumerical={N_CORE_PALIK_LIVE:.4f}  materials_data.py={N_CORE:.4f}")
print(f"  SiO2 Clad Index(n): Lumerical={N_CLAD_PALIK_LIVE:.4f}  materials_data.py={N_CLAD:.4f}\n")


# ---------------------------------------------------------------------------
# Waveguide FDE Simulation Function
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
    set('x span', 10e-6);
    set('y span', 4e-6);
    set('x', 0); set('y', 0); set('z', 0);
    set('background material', '<Object defined dielectric>');
    set('index', N_CLAD + 1i*k_clad);
    set('x min bc', 'PML'); set('x max bc', 'PML');
    set('y min bc', 'PML'); set('y max bc', 'PML');
    set('wavelength', wavelength);
    set('number of trial modes', 10);

    addmesh;
    set('name', 'mesh_wg');
    set('x', 0); set('y', 0); set('z', 0);
    set('x span', core_w + 1e-6);
    set('y span', core_h + 1e-6);
    set('z span', 10e-6);
    set('override x mesh', true);
    set('override y mesh', true);
    set('override z mesh', false);
    set('dx', 10e-9);
    set('dy', 10e-9);
    # set('dz', 5e-9);

    findmodes;

    target_mode = "mode1";
    found = 0;
    for (m = 1:5) {
        if (found == 0) {
            m_name = "mode" + num2str(m);
            te_frac = getdata(m_name, "TE polarization fraction");
            if (te_frac > 0.5) {
                target_mode = m_name;
                found = 1;
            }
        }
    }

    neff_complex = getdata(target_mode, "neff");
    n_eff_real = real(neff_complex);
    neff_imag = imag(neff_complex);

    # FIX #1: Correct units conversion for dB/cm propagation loss
    alpha_calc = ((4*pi*neff_imag)/wavelength) * 10*log10(exp(1)) / 100;

    x = getdata(target_mode, "x");
    y = getdata(target_mode, "y");
    E2 = pinch(getelectric(target_mode));

    nx = length(x);
    ny = length(y);

    if (size(E2, 1) != nx) {
        E2 = transpose(E2);
    }

    # FIX #4: Area-weighted confinement integral (non-uniform FDE mesh)
    dx = zeros(nx, 1);
    dx(1) = x(2) - x(1);
    dx(nx) = x(nx) - x(nx - 1);
    for (i = 2:nx - 1) {
        dx(i) = (x(i + 1) - x(i - 1)) / 2;
    }

    dy = zeros(ny, 1);
    dy(1) = y(2) - y(1);
    dy(ny) = y(ny) - y(ny - 1);
    for (j = 2:ny - 1) {
        dy(j) = (y(j + 1) - y(j - 1)) / 2;
    }

    p_core = 0;
    p_total = 0;

    for (i = 1:nx) {
        for (j = 1:ny) {
            weight = E2(i, j) * dx(i) * dy(j);
            p_total = p_total + weight;
            if (abs(x(i)) <= core_w/2 and abs(y(j)) <= core_h/2) {
                p_core = p_core + weight;
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

    return (
        float(mode.getv("confinement_calc")),
        float(mode.getv("n_eff_real")),
        float(mode.getv("alpha_calc"))
    )


def analytical_waveguide(width_um, height_um):
    confinement, n_eff = pb.strip_confinement(N_CORE, N_CLAD, width_um, height_um, WAVELENGTH_UM)
    alpha = pb.material_loss_db_per_cm(confinement, ALPHA_CORE_DB_PER_CM, ALPHA_CLAD_DB_PER_CM)
    return confinement, n_eff, alpha


# ---------------------------------------------------------------------------
# MMI EME Simulation Function
# ---------------------------------------------------------------------------

def simulate_mmi_eme(mmi_length_um):

    print('START - into mmi function now')

    mode.putv("WAVELENGTH_UM", float(WAVELENGTH_UM))
    mode.putv("ACCESS_WIDTH_UM", float(ACCESS_WIDTH_UM))
    mode.putv("CORE_HEIGHT_UM", float(CORE_HEIGHT_UM))
    mode.putv("MMI_WIDTH_UM", float(MMI_WIDTH_UM))
    mode.putv("ACCESS_LEN_UM", float(ACCESS_LEN_UM))
    mode.putv("mmi_length_um", float(mmi_length_um))
    mode.putv("N_CORE", float(N_CORE))
    mode.putv("N_CLAD", float(N_CLAD))
    mode.putv("tap1_um", float(pb.mmi_tap_position(1, 2, MMI_WIDTH_UM)))
    mode.putv("tap2_um", float(pb.mmi_tap_position(2, 2, MMI_WIDTH_UM)))

    script = """
    into_mmi = 1;
    switchtolayout;
    selectall;
    delete;

    wavelength = WAVELENGTH_UM * 1e-6;
    access_w   = ACCESS_WIDTH_UM * 1e-6;
    wg_height  = CORE_HEIGHT_UM * 1e-6;
    mmi_w      = MMI_WIDTH_UM * 1e-6;
    access_len = ACCESS_LEN_UM * 1e-6;
    mmi_len    = mmi_length_um * 1e-6;
    tap1       = tap1_um * 1e-6;
    tap2       = tap2_um * 1e-6;

    # Input Waveguide (centered)
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

    # Output Waveguide 1 (tap 1)
    addrect; set('name','out_1');
    set('x min', mmi_len/2); set('x max', mmi_len/2 + access_len);
    set('y', tap1); set('y span', access_w);
    set('z', 0); set('z span', wg_height);
    set('material', '<Object defined dielectric>');
    set('index', N_CORE);

    # Output Waveguide 2 (tap 2)
    addrect; set('name','out_2');
    set('x min', mmi_len/2); set('x max', mmi_len/2 + access_len);
    set('y', tap2); set('y span', access_w);
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
    set('y', 0); set('y span', mmi_w * 4.5); # Widen Y-span to reduce PML absorption boundary spikes
    set('y min bc', 'PML'); set('y max bc', 'PML');
    set('z', 0); set('z span', wg_height * 12);
    set('z min bc', 'PML'); set('z max bc', 'PML');

    # Cell Setup
    set('number of cell groups', 3);
    set('group spans', [access_len; mmi_len; access_len]);
    set('cells', [1; 1; 1]);

    # Solver & Energy Conservation
    select("EME");
    set("allow custom eigensolver settings", 1);
    set("energy conservation", "make passive");

    # Trial Modes Setup
    select("EME::Cells::cell_1"); seteigensolver("number of trial modes", 10);
    select("EME::Cells::cell_2"); seteigensolver("number of trial modes", 100);
    select("EME::Cells::cell_3"); seteigensolver("number of trial modes", 10);

    # Ports
    setactivesolver("EME");

    addemeport;
    set('port location', 'left');
    set('use full simulation span', 1); 
    set('mode selection', 'fundamental TE mode');

    addemeport;
    set('port location', 'right');
    set('use full simulation span', 0);
    set('y', tap1); set('y span', mmi_w * 2);
    set('z', 0); set('z span', wg_height * 6);
    set('mode selection', 'fundamental TE mode');

    addemeport;
    set('port location', 'right');
    set('use full simulation span', 0);
    set('y', tap2); set('y span', mmi_w * 2);
    set('z', 0); set('z span', wg_height * 6);
    set('mode selection', 'fundamental TE mode');
    
    addmesh;
    set('name', 'mesh_mmi');
    set('based on a structure', 1);
    set('structure', 'MMI');
    # set("override x mesh", true);
    set("override y mesh", true);
    set("override z mesh", true);
    # set("dx", 20e-9); 
    set("dy", 20e-9);
    set("dz", 10e-9);

    save('mmi_validation.lms');
    # Run Eigensolver & Propagation
    run;
    select("EME");
    # updatemodes;
    emepropagate;

    S = getresult("EME", "user s matrix");
    s_matrix = pinch(S);

    transmitted_power = abs(s_matrix(2, 1))^2 + abs(s_matrix(3, 1))^2;
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
    if RUN_VALIDATION not in ("waveguide", "mmi", "both"):
        raise ValueError('RUN_VALIDATION must be "waveguide", "mmi", or "both"')
    run_waveguide = RUN_VALIDATION in ("waveguide", "both")
    run_mmi = RUN_VALIDATION in ("mmi", "both")

    waveguide_sim_time = 0.0
    waveguide_ana_time = 0.0
    mmi_sim_time = 0.0
    mmi_ana_time = 0.0

    if run_waveguide:
        # --- Waveguide FDE Sweeps ---
        widths = linspace(WIDTH_SWEEP_START_UM, WIDTH_SWEEP_STOP_UM, WIDTH_SWEEP_STEPS)
        sim_neff, ana_neff = [], []
        sim_conf, ana_conf = [], []
        sim_alpha, ana_alpha = [], []

        print("Validating Waveguide Modes using Lumerical FDE...")
        for w in widths:
            t0 = time.perf_counter()
            c_sim, n_sim, a_sim = simulate_waveguide(w, CORE_HEIGHT_UM)
            t1 = time.perf_counter()
            c_ana, n_ana, a_ana = analytical_waveguide(w, CORE_HEIGHT_UM)
            t2 = time.perf_counter()
            waveguide_sim_time += t1 - t0
            waveguide_ana_time += t2 - t1

            sim_neff.append(n_sim); ana_neff.append(n_ana)
            sim_conf.append(c_sim); ana_conf.append(c_ana)
            sim_alpha.append(a_sim); ana_alpha.append(a_ana)
            print(f"  Finished width = {w:.2f} um | n_eff = {n_sim:.5f} | Confinement = {c_sim:.4f} | Loss = {a_sim:.4e} dB/cm")

    if run_mmi:
        # --- MMI EME Sweeps ---
        n_eff_vertical = pb.slab_mode(N_CORE, N_CLAD, CORE_HEIGHT_UM, WAVELENGTH_UM)[1]

        optimal_length_um, _, _ = pb.mmi_derive_length_and_loss(
            n_eff_vertical, N_CLAD, MMI_WIDTH_UM, 2, ACCESS_WIDTH_UM, WAVELENGTH_UM
        )
        if optimal_length_um <= 0:
            optimal_length_um = 4.61  # Fallback length

        length_factors = linspace(LENGTH_SWEEP_START_FACTOR, LENGTH_SWEEP_STOP_FACTOR, LENGTH_SWEEP_STEPS)
        lengths_um = [optimal_length_um * f for f in length_factors]
        sim_excess, ana_excess = [], []

        print("\nValidating MMI Excess Loss using Lumerical EME...")
        for l_um in lengths_um:
            t0 = time.perf_counter()
            e_ana, _ = pb.mmi_excess_loss_db(
                n_eff_vertical, N_CLAD, MMI_WIDTH_UM, 2, ACCESS_WIDTH_UM, WAVELENGTH_UM, l_um
            )
            t1 = time.perf_counter()
            e_sim = simulate_mmi_eme(l_um)
            t2 = time.perf_counter()
            mmi_ana_time += t1 - t0
            mmi_sim_time += t2 - t1
            sim_excess.append(e_sim)
            ana_excess.append(e_ana)
            print(f"  Finished length = {l_um:.3f} um | Excess Loss = {e_sim:.4f} dB")

    # --- Plotting: only build panels for what actually ran ---
    panel_fns = []

    if run_waveguide:
        def plot_neff(ax):
            ax.plot(widths, sim_neff, 'o-', color='#1f77b4', linewidth=2, label='Simulated (FDE)')
            ax.plot(widths, ana_neff, 's--', color='#ff7f0e', linewidth=2, label='Analytical (EIM)')
            ax.set_xlabel('Core Width (µm)'); ax.set_ylabel('Real Effective Index ($n_{eff}$)')
            ax.set_title('Effective Index Comparison'); ax.grid(True, linestyle='--', alpha=0.6); ax.legend()

        def plot_conf(ax):
            ax.plot(widths, sim_conf, 'o-', color='#1f77b4', linewidth=2, label='Simulated (FDE)')
            ax.plot(widths, ana_conf, 's--', color='#ff7f0e', linewidth=2, label='Analytical (EIM)')
            ax.set_xlabel('Core Width (µm)'); ax.set_ylabel('Confinement Factor ($\Gamma$)')
            ax.set_title('Confinement Factor Comparison'); ax.grid(True, linestyle='--', alpha=0.6); ax.legend()

        def plot_alpha(ax):
            ax.plot(widths, sim_alpha, 'o-', color='#1f77b4', linewidth=2, label='Simulated (FDE)')
            ax.plot(widths, ana_alpha, 's--', color='#ff7f0e', linewidth=2, label='Analytical (Perturbation)')
            ax.set_xlabel('Core Width (µm)'); ax.set_ylabel('Loss $\\alpha$ (dB/cm)')
            ax.set_title('Propagation Loss Comparison'); ax.grid(True, linestyle='--', alpha=0.6); ax.legend()

        panel_fns += [plot_neff, plot_conf, plot_alpha]

    if run_mmi:
        def plot_excess(ax):
            ax.plot(lengths_um, sim_excess, 'o-', color='#1f77b4', linewidth=2, label='Simulated (EME)')
            ax.plot(lengths_um, ana_excess, 's--', color='#ff7f0e', linewidth=2, label='Analytical (Overlap)')
            ax.axvline(optimal_length_um, color='gray', linestyle=':', label='Derived Optimal Length')
            ax.set_xlabel('MMI Length (µm)'); ax.set_ylabel('Excess Loss (dB)')
            ax.set_title('MMI Excess Loss Comparison'); ax.grid(True, linestyle='--', alpha=0.6); ax.legend()

        panel_fns.append(plot_excess)

    n_panels = len(panel_fns)
    grid_layout = {1: (1, 1), 3: (1, 3), 4: (2, 2)}[n_panels]
    fig, axes = plt.subplots(*grid_layout, figsize=(6 * grid_layout[1], 4.5 * grid_layout[0]))
    axes_flat = [axes] if n_panels == 1 else list(axes.flatten())

    for ax, plot_fn in zip(axes_flat, panel_fns):
        plot_fn(ax)

    fig.suptitle('Analytical Model vs Lumerical Simulation', fontsize=14, fontweight='bold')

    timing_lines = []
    if run_waveguide:
        timing_lines.append(
            f"Waveguide sweep -- Simulation (Lumerical): {waveguide_sim_time:.2f}s   |   "
            f"Analytical (power_budget.py): {waveguide_ana_time:.4f}s"
        )
    if run_mmi:
        timing_lines.append(
            f"MMI sweep -- Simulation (Lumerical): {mmi_sim_time:.2f}s   |   "
            f"Analytical (power_budget.py): {mmi_ana_time:.4f}s"
        )
    timing_label = "\n".join(timing_lines)

    plt.tight_layout(rect=[0, 0.02 + 0.03 * len(timing_lines), 1, 1])
    fig.text(0.5, 0.01, timing_label, ha='center', fontsize=10, color='dimgray')

    plt.savefig(OUTPUT_PLOT_FILENAME, dpi=300)
    print(f"\nSummary plot saved successfully to '{OUTPUT_PLOT_FILENAME}'.")
    print(timing_label)
    if matplotlib.get_backend().lower() != "agg":
        plt.tight_layout()
        plt.show()