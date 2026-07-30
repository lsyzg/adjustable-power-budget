import math

# conversions
def dbm_to_mw(p_dbm):
    return 10 ** (p_dbm / 10)

def mw_to_dbm(p_mw):
    return 10 * math.log10(p_mw)

def db_to_linear(l_db):
    return 10 ** (-l_db / 10)

# waveguide loss
"""
Solves the symmetric slab waveguide mode equation numerically since it has
no closed form. The confinement factor is derived by integrating the TE0
mode fields directly. The even mode has field cos(kappa*y) inside the core
and a decaying exponential outside it. Confinement is the fraction of mode
power inside the core.

Gamma = P_core / (P_core + P_clad)
P_core = t/2 + sin(kappa*t) / (2*kappa)
P_clad = cos(kappa*t/2)^2 / gamma

The 2D strip waveguide confinement factor comes from the effective index
method. The vertical slab is solved first to get an effective index, then
that index becomes the core index for a second slab solve across the
width. The two 1D confinement factors multiply together for the combined
2D confinement factor.
"""

def _slab_branch_root(kappa_max, thickness_um, branch_index):
    """Solve the symmetric-slab transcendental equation for one mode branch:
    even branch_index -> even (cos-core) mode, odd -> odd (sin-core) mode.
    branch_index=0 is the fundamental mode. No closed form -> bisection.
    Returns (kappa, gamma) in 1/um, or None if this branch isn't guided.
    """
    u_lo = branch_index * math.pi / 2
    u_hi = (branch_index + 1) * math.pi / 2
    u_max = kappa_max * thickness_um / 2
    if u_lo >= u_max:
        return None
    u_hi = min(u_hi, u_max)
    eps = 1e-9
    lo, hi = u_lo + eps, u_hi - eps
    if lo >= hi:
        return None

    even = (branch_index % 2 == 0)

    def g(u):
        kappa = 2 * u / thickness_um
        gamma = math.sqrt(max(kappa_max ** 2 - kappa ** 2, 0.0))
        if even:
            return kappa * math.tan(u) - gamma
        return kappa / math.tan(u) + gamma

    g_lo = g(lo)
    if g_lo * g(hi) > 0:
        return None
    for _ in range(80):
        mid = (lo + hi) / 2
        if (g(mid) < 0) == (g_lo < 0):
            lo = mid
        else:
            hi = mid
    u = (lo + hi) / 2
    kappa = 2 * u / thickness_um
    gamma = math.sqrt(max(kappa_max ** 2 - kappa ** 2, 0.0))
    return kappa, gamma


def _slab_mode_power_unnorm(kappa, gamma, thickness_um, even):
    d = thickness_um
    if even:
        p_core = d / 2 + math.sin(kappa * d) / (2 * kappa)
        p_clad = math.cos(kappa * d / 2) ** 2 / gamma
    else:
        p_core = d / 2 - math.sin(kappa * d) / (2 * kappa)
        p_clad = math.sin(kappa * d / 2) ** 2 / gamma
    return p_core, p_clad


def _slab_mode_field(y, kappa, gamma, thickness_um, even):
    d = thickness_um
    if even:
        if abs(y) <= d / 2:
            return math.cos(kappa * y)
        edge = math.cos(kappa * d / 2)
        return edge * math.exp(-gamma * (abs(y) - d / 2))
    if abs(y) <= d / 2:
        return math.sin(kappa * y)
    edge = math.sin(kappa * d / 2)
    sign = 1 if y > 0 else -1
    return sign * edge * math.exp(-gamma * (abs(y) - d / 2))


def slab_mode(n_core, n_clad, thickness_um, wavelength_um):
    """1D confinement factor and effective index of a symmetric slab's fundamental mode."""
    k0 = 2 * math.pi / wavelength_um
    kappa_max = k0 * math.sqrt(n_core ** 2 - n_clad ** 2)
    kappa, gamma = _slab_branch_root(kappa_max, thickness_um, 0)
    n_eff = math.sqrt(n_core ** 2 - (kappa / k0) ** 2)
    p_core, p_clad = _slab_mode_power_unnorm(kappa, gamma, thickness_um, True)
    confinement = p_core / (p_core + p_clad)
    return confinement, n_eff


def strip_confinement(n_core, n_clad, width_um, height_um, wavelength_um):
    """2D confinement factor for a strip waveguide via the effective index method:
    solve the vertical slab (height) for n_eff, then the horizontal slab (width)
    using that n_eff as the new 'core' index, and combine the two 1D factors.
    """
    confinement_v, n_eff_v = slab_mode(n_core, n_clad, height_um, wavelength_um)
    confinement_h, n_eff_2d = slab_mode(n_eff_v, n_clad, width_um, wavelength_um)
    return confinement_v * confinement_h, n_eff_2d


def material_loss_db_per_cm(confinement, alpha_core_db_per_cm, alpha_clad_db_per_cm):
    """Confinement-weighted material absorption loss (first-order perturbation theory)."""
    return confinement * alpha_core_db_per_cm + (1 - confinement) * alpha_clad_db_per_cm


# MMI excess loss
"""
The multimode section is reduced to a 1D lateral slab using the effective
index from the vertical confinement solve, matching standard MMI self
imaging theory. Every guided lateral mode, both even and odd, is found
through the slab mode solver. The input and each output access waveguide
mode are projected onto those lateral modes through numerical overlap
integrals, then propagated with each mode's own phase to the MMI length.
Excess loss is the shortfall between the total power reaching the N
outputs and the input power.
"""

def slab_lateral_modes(n_eff_core, n_clad, width_um, wavelength_um, max_modes=40):
    """All guided modes (even and odd) of a 1D lateral slab."""
    k0 = 2 * math.pi / wavelength_um
    kappa_max = k0 * math.sqrt(n_eff_core ** 2 - n_clad ** 2)
    modes = []
    for j in range(max_modes):
        root = _slab_branch_root(kappa_max, width_um, j)
        if root is None:
            break
        kappa, gamma = root
        even = (j % 2 == 0)
        p_core, p_clad = _slab_mode_power_unnorm(kappa, gamma, width_um, even)
        n_eff_m = math.sqrt(n_eff_core ** 2 - (kappa / k0) ** 2)
        modes.append({
            "order": j, "kappa": kappa, "gamma": gamma, "even": even,
            "norm": math.sqrt(p_core + p_clad), "n_eff": n_eff_m, "beta": k0 * n_eff_m,
        })
    return modes


def _simpson(f, a, b, n):
    if n % 2 == 1:
        n += 1
    h = (b - a) / n
    total = f(a) + f(b)
    for i in range(1, n):
        total += (4 if i % 2 else 2) * f(a + i * h)
    return total * h / 3


def _access_mode_overlap(mode, mmi_width_um, y0, access_width_um, n_eff_core, n_clad, wavelength_um):
    """Overlap integral of an MMI-region eigenmode against the fundamental mode
    of a single-mode access waveguide of width access_width_um centered at y0."""
    k0 = 2 * math.pi / wavelength_um
    kappa_max = k0 * math.sqrt(n_eff_core ** 2 - n_clad ** 2)
    a_kappa, a_gamma = _slab_branch_root(kappa_max, access_width_um, 0)
    a_p_core, a_p_clad = _slab_mode_power_unnorm(a_kappa, a_gamma, access_width_um, True)
    a_norm = math.sqrt(a_p_core + a_p_clad)

    span = max(mmi_width_um, access_width_um) * 1.5

    def integrand(y):
        phi = _slab_mode_field(y, mode["kappa"], mode["gamma"], mmi_width_um, mode["even"]) / mode["norm"]
        psi = _slab_mode_field(y - y0, a_kappa, a_gamma, access_width_um, True) / a_norm
        return phi * psi

    return _simpson(integrand, -span, span, 2000)


def mmi_tap_position(port_index_1based, n_ports, mmi_width_um):
    """Standard symmetric output-port positions across the MMI width."""
    k = port_index_1based
    return (2 * k - n_ports - 1) * mmi_width_um / (2 * n_ports)


def _mmi_projections(n_eff_core, n_clad, mmi_width_um, n_ports, access_width_um, wavelength_um):
    modes = slab_lateral_modes(n_eff_core, n_clad, mmi_width_um, wavelength_um)
    c = [_access_mode_overlap(m, mmi_width_um, 0.0, access_width_um, n_eff_core, n_clad, wavelength_um)
         for m in modes]
    port_overlaps = []
    for k in range(1, n_ports + 1):
        yk = mmi_tap_position(k, n_ports, mmi_width_um)
        port_overlaps.append([
            _access_mode_overlap(m, mmi_width_um, yk, access_width_um, n_eff_core, n_clad, wavelength_um)
            for m in modes
        ])
    return modes, c, port_overlaps


def _mmi_total_power(modes, c, port_overlaps, length_um):
    total = 0.0
    for overlaps in port_overlaps:
        re = im = 0.0
        for mode, cm, ov in zip(modes, c, overlaps):
            phase = -mode["beta"] * length_um
            re += cm * math.cos(phase) * ov
            im += cm * math.sin(phase) * ov
        total += re * re + im * im
    return total


def mmi_excess_loss_db(n_eff_core, n_clad, mmi_width_um, n_ports, access_width_um,
                        wavelength_um, mmi_length_um):
    """Excess loss in dB at a given device length. Computed as -10*log10 of
    the total power reaching all N outputs."""
    modes, c, port_overlaps = _mmi_projections(
        n_eff_core, n_clad, mmi_width_um, n_ports, access_width_um, wavelength_um)
    power = _mmi_total_power(modes, c, port_overlaps, mmi_length_um)
    return -10 * math.log10(power), len(modes)


def mmi_derive_length_and_loss(n_eff_core, n_clad, mmi_width_um, n_ports, access_width_um, wavelength_um):
    """Numerically finds the device length that minimizes excess loss, searching
    the general-interference self-imaging bracket from 0 to 3*L_pi. A numerical
    search is used instead of a fixed named formula because which self-imaging
    mechanism applies, general, paired, or symmetric, depends on the input
    position and N's parity. Searching directly stays robust to that.
    Returns length_um, excess_loss_db, and num_modes.
    """
    modes, c, port_overlaps = _mmi_projections(
        n_eff_core, n_clad, mmi_width_um, n_ports, access_width_um, wavelength_um)

    def total_power(length_um):
        return _mmi_total_power(modes, c, port_overlaps, length_um)

    beta0, beta1 = modes[0]["beta"], modes[1]["beta"]
    bracket = 3 * math.pi / (beta0 - beta1)

    n_scan = 400
    best_l, best_p = 0.0, -1.0
    for i in range(n_scan + 1):
        length_um = bracket * i / n_scan
        power = total_power(length_um)
        if power > best_p:
            best_p, best_l = power, length_um

    step = bracket / n_scan
    lo, hi = max(0.0, best_l - step), best_l + step
    gr = (math.sqrt(5) - 1) / 2
    c1, c2 = hi - gr * (hi - lo), lo + gr * (hi - lo)
    f1, f2 = total_power(c1), total_power(c2)
    for _ in range(60):
        if f1 < f2:
            lo, c1, f1 = c1, c2, f2
            c2 = lo + gr * (hi - lo)
            f2 = total_power(c2)
        else:
            hi, c2, f2 = c2, c1, f1
            c1 = hi - gr * (hi - lo)
            f1 = total_power(c1)
    best_length = (lo + hi) / 2
    best_power = total_power(best_length)
    return best_length, -10 * math.log10(best_power), len(modes)

# propagation loss
def propagation_loss_db(alpha_db_per_cm, length_um):
    length_cm = length_um * 1e-4
    return alpha_db_per_cm * length_cm

# mmi
def mmi_total_loss_db(n_ports, excess_loss_db):
    return 10 * math.log10(n_ports) + excess_loss_db

# modulator  power penalty
def er_db_from_splitting_ratio(gamma):
    """MZM extinction ratio limited by a non-ideal (gamma:1-gamma) splitter/combiner
    coupling ratio; gamma=0.5 (ideal 3dB) gives infinite ER."""
    cross_term = 2 * math.sqrt(gamma * (1 - gamma))
    return 10 * math.log10((1 + cross_term) / (1 - cross_term))

def er_power_penalty_db(er_db):
    er_linear = 10 ** (er_db / 10)
    return 10 * math.log10((er_linear + 1) / (er_linear - 1))

# photodetector
def pd_metrics(responsivity_a_per_w, p_pd_dbm):
    p_pd_w = dbm_to_mw(p_pd_dbm) * 1e-3
    return responsivity_a_per_w * p_pd_w    

# modulation efficiency
def capacitance_fF(c_per_um_fF, length_um):
    return c_per_um_fF * length_um

def modulation_efficiency_capacitive_pJ(C_fF, V):
    C_F = C_fF * 1e-15
    E_J = 0.25 * C_F * V ** 2
    return E_J * 1e12

def modulation_efficiency_power_pJ(p_drive_w, bit_rate_bps):
    e_bit_j = p_drive_w / bit_rate_bps
    return e_bit_j * 1e12

# power budget
def compute_results(p_laser_dbm, l_in_db, l_prop1_db, l_mmi_total_db, l_prop2_db,
                          l_mod_insertion_db, l_mod_er_penalty_db, l_out_db):
    stages = []
    p = p_laser_dbm
    stages.append(("Laser", p))

    p -= l_in_db
    stages.append(("Edge coupler (in)", p))

    p -= l_prop1_db
    stages.append(("Waveguide (pre-MMI)", p))

    p -= l_mmi_total_db
    stages.append(("MMI", p))

    p -= l_prop2_db
    stages.append(("Waveguide (post-MMI)", p))

    p -= l_mod_insertion_db
    stages.append(("Modulator (insertion loss)", p))

    p -= l_mod_er_penalty_db
    stages.append(("Modulator (ER penalty)", p))

    p -= l_out_db
    stages.append(("Edge coupler (out)", p))

    total_loss_db = p_laser_dbm - p
    stages.append(("Total loss (dB)", total_loss_db))

    return stages

def available_power_db(p_laser_dbm, p_sensitivity_dbm):
    return p_laser_dbm - p_sensitivity_dbm

def power_budget_db(p_laser_dbm, p_sensitivity_dbm, total_loss_db):
    return available_power_db(p_laser_dbm, p_sensitivity_dbm) - total_loss_db
