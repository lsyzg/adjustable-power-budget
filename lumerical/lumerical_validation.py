"""Validate power_budget.py's analytical equations against Lumerical MODE
Solutions simulations, using the same default dimensions as gui.py, and plot
analytical vs simulated results for each major output: waveguide effective
index, confinement factor, propagation loss (alpha), and MMI excess loss.

Scope note: the waveguide FDE section (n_eff, confinement, alpha) uses
simple constant (non-dispersive) indices, not the Palik material database,
so that comparison isolates whether the analytical effective-index-method
approximation reproduces the same idealized physics as a rigorous numerical
solve. The MMI EME section instead uses the real Palik material database
(matching lumerical_mmi.lsf's approach), so that comparison also picks up
material dispersion, not just the mode-overlap/self-imaging math. Neither
section models sidewall roughness scattering (see the note at the top of
lumerical_alpha.lsf, where material-absorption-only alpha undershoots a
realistic 2 dB/cm target by an order of magnitude) -- so don't expect
either comparison to validate against a real fabricated device's loss.

The waveguide FDE portion follows the same pattern already validated
earlier via lumerical_alpha.lsf. The MMI EME portion (including its N
output ports and one modulator stand-in per port) is a best-effort
implementation -- exact addeme/addemeport property and result names can
differ across Lumerical versions, so check that section first if it errors.

No numpy/matplotlib dependency: plots render inside Lumerical's own script
engine (mode.putv + the native plot() command).
"""

import sys, os
import math

sys.path.append("/opt/lumerical/v231/api/python/")
sys.path.append(os.path.dirname(__file__))

import lumapi
import power_budget as pb

# ---------------------------------------------------------------------------
# Same default dimensions as gui.py's PowerBudgetGUI._build_inputs()
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
ACCESS_WIDTH_UM = CORE_WIDTH_UM  # "Direct" access coupling, matches gui.py default
ACCESS_LEN_UM = 5.0  # matches gui.py's length1/length2 defaults
MOD_LENGTH_UM = 500.0  # matches gui.py's modulator arm length default


def db_per_cm_to_k(alpha_db_per_cm, wavelength_um):
    """Invert the alpha(neff_imag) formula: what imaginary index k gives a
    bulk medium this alpha, i.e. treating k itself as neff_imag would be for
    an unconfined (infinite) medium of that material."""
    alpha_db_per_m = alpha_db_per_cm * 100
    return alpha_db_per_m * (wavelength_um * 1e-6) / (4 * math.pi * 10 * math.log10(math.e))


mode = lumapi.MODE(hide=True)

# ---------------------------------------------------------------------------
# Waveguide FDE: effective index, confinement factor, propagation loss
# ---------------------------------------------------------------------------

def build_waveguide(width_um, height_um):
    k_core = db_per_cm_to_k(ALPHA_CORE_DB_PER_CM, WAVELENGTH_UM)
    k_clad = db_per_cm_to_k(ALPHA_CLAD_DB_PER_CM, WAVELENGTH_UM)
    mode.eval(f"""
    switchtolayout; selectall; delete;

    wavelength = {WAVELENGTH_UM * 1e-6:e};
    core_width = {width_um * 1e-6:e};
    core_height = {height_um * 1e-6:e};

    addrect;
    set('name','waveguide');
    set('x span', core_width);
    set('y span', core_height);
    set('z span', 10e-6);
    set('x',0); set('y',0); set('z',0);
    set('index', {N_CORE:e} + 1i*{k_core:e});

    addfde;
    set('background material', '<Object defined dielectric>');
    set('index', {N_CLAD:e} + 1i*{k_clad:e});
    set('x span', 5e-6);
    set('y span', 2.5e-6);
    set('x',0); set('y',0); set('z',0);
    set('x min bc','PML'); set('x max bc','PML');
    set('y min bc','PML'); set('y max bc','PML');
    set('wavelength', wavelength);

    function get_neff(){{
        findmodes; selectmode(1);
        return real(getdata("mode1","neff"));
    }}

    function get_confinement(){{
        findmodes; selectmode(1);
        E2 = pinch(getdata("mode1","E2"));
        x = getdata("mode1","x");
        y = getdata("mode1","y");
        # outer-product mask: 1 inside the core rectangle, 0 outside.
        # Flag: verify x/y come back as the shapes pinch(E2) expects on
        # your Lumerical version -- this is the least-tested line here.
        core_mask = (abs(x) < core_width/2) * transpose(abs(y) < core_height/2);
        return sum(sum(E2*core_mask)) / sum(sum(E2));
    }}

    function get_alpha(){{
        findmodes; selectmode(1);
        neff_imag = imag(getdata("mode1","neff"));
        alpha_db_per_m = (4*pi*neff_imag/wavelength) * 10*log10(exp(1));
        return alpha_db_per_m/100;
    }}
    """)


def simulate_waveguide(width_um, height_um):
    build_waveguide(width_um, height_um)
    n_eff = mode.get_neff()
    confinement = mode.get_confinement()
    alpha = mode.get_alpha()
    return confinement, n_eff, alpha


def analytical_waveguide(width_um, height_um):
    confinement, n_eff = pb.strip_confinement(N_CORE, N_CLAD, width_um, height_um, WAVELENGTH_UM)
    alpha = pb.material_loss_db_per_cm(confinement, ALPHA_CORE_DB_PER_CM, ALPHA_CLAD_DB_PER_CM)
    return confinement, n_eff, alpha


# ---------------------------------------------------------------------------
# MMI EME: excess loss vs device length, with one MZM per output port
# ---------------------------------------------------------------------------
#
# Structure and rect positioning (material + x min/x max) follow
# lumerical_mmi.lsf's established style, generalized from its fixed 2-output
# example to N_PORTS outputs -- each output waveguide is followed by its own
# modulator stand-in (a plain waveguide segment MOD_LENGTH_UM long, matching
# how gui.py/power_budget.py already model insertion loss as alpha*length,
# not a literal two-arm interferometer). Output positions use
# pb.mmi_tap_position() rather than a fixed gap, so the simulated geometry
# matches what the analytical side assumes.
#
# Note: this section uses the real Palik material database (like
# lumerical_mmi.lsf), not the constant indices used in the waveguide FDE
# section above -- so unlike that section, this comparison also picks up
# material dispersion, not just the mode-overlap/self-imaging math itself.

def build_mmi_eme(mmi_length_um):
    tap_positions_um = [pb.mmi_tap_position(k, N_PORTS, MMI_WIDTH_UM) for k in range(1, N_PORTS + 1)]

    mode.eval(f"""
    switchtolayout; selectall; delete;

    wavelength = {WAVELENGTH_UM * 1e-6:e};
    access_w = {ACCESS_WIDTH_UM * 1e-6:e};
    wg_height = {CORE_HEIGHT_UM * 1e-6:e};
    mmi_w = {MMI_WIDTH_UM * 1e-6:e};
    access_len = {ACCESS_LEN_UM * 1e-6:e};
    mmi_len = {mmi_length_um * 1e-6:e};
    mod_len = {MOD_LENGTH_UM * 1e-6:e};

    addrect;
    set('name','in_1');
    set('x min', -access_len - mmi_len/2); set('x max', -mmi_len/2);
    set('y', 0); set('y span', access_w);
    set('z', 0); set('z span', wg_height);
    set('material', 'Si (Silicon) - Palik');

    addrect;
    set('name','MMI');
    set('x min', -mmi_len/2); set('x max', mmi_len/2);
    set('y', 0); set('y span', mmi_w);
    set('z', 0); set('z span', wg_height);
    set('material', 'Si (Silicon) - Palik');
    """)

    for i, y_um in enumerate(tap_positions_um, start=1):
        mode.eval(f"""
        addrect;
        set('name','out_{i}');
        set('x min', mmi_len/2); set('x max', mmi_len/2 + access_len);
        set('y', {y_um * 1e-6:e}); set('y span', access_w);
        set('z', 0); set('z span', wg_height);
        set('material', 'Si (Silicon) - Palik');

        addrect;
        set('name','mzm_{i}');
        set('x min', mmi_len/2 + access_len); set('x max', mmi_len/2 + access_len + mod_len);
        set('y', {y_um * 1e-6:e}); set('y span', access_w);
        set('z', 0); set('z span', wg_height);
        set('material', 'Si (Silicon) - Palik');
        """)

    mode.eval(f"""
    addeme;
    set('wavelength', wavelength);
    set('background material', 'SiO2 (Glass) - Palik');
    set('x min', -access_len - mmi_len/2);
    set('y span', mmi_w * 2);
    set('y', 0);
    set('number of cell groups', 4);
    set('group spans', [access_len; mmi_len; access_len; mod_len]);
    set('cells', [1; 1; 1; 1]);
    set('number of modes for all cell groups', 20);

    addemeport;
    set('name','port_in'); set('port location','left');
    """)

    for i, y_um in enumerate(tap_positions_um, start=1):
        mode.eval(f"""
        addemeport;
        set('name','port_out{i}'); set('port location','right');
        set('y', {y_um * 1e-6:e}); set('y span', access_w * 3);
        """)

    mode.eval(f"""
    function get_mmi_excess_loss(){{
        run;
        emepropagate;
        s_total = 0;
        for(k = 1:{N_PORTS}){{
            port_name = 'EME::Ports::port_out' + num2str(k);
            # Flag: verify this S-parameter dataset/field name against your
            # Lumerical version's EME results -- least-tested line here.
            s_param = getresult(port_name, 's');
            s_total = s_total + abs(s_param.s21)^2;
        }}
        return -10*log10(s_total);
    }}
    """)


def simulate_mmi_excess_loss(mmi_length_um):
    build_mmi_eme(mmi_length_um)
    return mode.get_mmi_excess_loss()


def analytical_mmi_excess_loss(mmi_length_um):
    n_eff_vertical = pb.slab_mode(N_CORE, N_CLAD, CORE_HEIGHT_UM, WAVELENGTH_UM)[1]
    excess_db, _num_modes = pb.mmi_excess_loss_db(
        n_eff_vertical, N_CLAD, MMI_WIDTH_UM, N_PORTS, ACCESS_WIDTH_UM, WAVELENGTH_UM, mmi_length_um
    )
    return excess_db


# ---------------------------------------------------------------------------
# Sweep, compare, and plot (natively inside Lumerical) each major output
# ---------------------------------------------------------------------------

def linspace(start, stop, n):
    if n <= 1:
        return [start]
    step = (stop - start) / (n - 1)
    return [start + i * step for i in range(n)]


def plot_comparison(x_vals, analytical_vals, simulated_vals, x_label, y_label, title):
    mode.putv("x_data", x_vals)
    mode.putv("analytical_data", analytical_vals)
    mode.putv("simulated_data", simulated_vals)
    mode.eval(f"""
    plot(x_data, analytical_data, simulated_data, "{x_label}", "{y_label}", "{title}");
    legend("analytical", "simulated");
    """)


def run_waveguide_validation():
    widths_um = linspace(CORE_WIDTH_UM * 0.7, CORE_WIDTH_UM * 1.3, 7)

    confinements_analytical, confinements_sim = [], []
    n_effs_analytical, n_effs_sim = [], []
    alphas_analytical, alphas_sim = [], []

    for w in widths_um:
        c_a, n_a, a_a = analytical_waveguide(w, CORE_HEIGHT_UM)
        c_s, n_s, a_s = simulate_waveguide(w, CORE_HEIGHT_UM)
        confinements_analytical.append(c_a); confinements_sim.append(c_s)
        n_effs_analytical.append(n_a); n_effs_sim.append(n_s)
        alphas_analytical.append(a_a); alphas_sim.append(a_s)
        print(f"width={w:.3f}um  Gamma: analytical={c_a:.4f} sim={c_s:.4f}  "
              f"n_eff: analytical={n_a:.4f} sim={n_s:.4f}  "
              f"alpha: analytical={a_a:.5f} sim={a_s:.5f}")

    plot_comparison(widths_um, confinements_analytical, confinements_sim,
                     "Core width (m)", "Confinement factor", "Confinement: analytical vs simulated")
    plot_comparison(widths_um, n_effs_analytical, n_effs_sim,
                     "Core width (m)", "Effective index", "n_eff: analytical vs simulated")
    plot_comparison(widths_um, alphas_analytical, alphas_sim,
                     "Core width (m)", "alpha (dB/cm)", "Propagation loss: analytical vs simulated")


def run_mmi_validation():
    n_eff_vertical = pb.slab_mode(N_CORE, N_CLAD, CORE_HEIGHT_UM, WAVELENGTH_UM)[1]
    optimal_length_um = pb.mmi_derive_length_and_loss(
        n_eff_vertical, N_CLAD, MMI_WIDTH_UM, N_PORTS, ACCESS_WIDTH_UM, WAVELENGTH_UM
    )[0]
    lengths_um = linspace(optimal_length_um * 0.5, optimal_length_um * 1.5, 9)

    excess_analytical, excess_sim = [], []
    for length_um in lengths_um:
        a = analytical_mmi_excess_loss(length_um)
        s = simulate_mmi_excess_loss(length_um)
        excess_analytical.append(a); excess_sim.append(s)
        print(f"MMI length={length_um:.2f}um  excess loss: analytical={a:.4f}dB sim={s:.4f}dB")

    plot_comparison(lengths_um, excess_analytical, excess_sim,
                     "MMI length (m)", "Excess loss (dB)", "MMI excess loss: analytical vs simulated")


if __name__ == "__main__":
    run_waveguide_validation()
    run_mmi_validation()
