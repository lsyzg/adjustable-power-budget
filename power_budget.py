import math

EPSILON_0 = 8.8541878128e-12

# conversions
def dbm_to_mw(p_dbm):
    return 10 ** (p_dbm / 10)

def mw_to_dbm(p_mw):
    return 10 * math.log10(p_mw)

def db_to_linear(l_db):
    return 10 ** (-l_db / 10)

# wall-plug efficiency
def laser_electrical_power_w(p_laser_dbm, wallplug_efficiency):
    p_optical_w = dbm_to_mw(p_laser_dbm) * 1e-3
    return p_optical_w / wallplug_efficiency


# waveguide loss
def _slab_branch_root(kappa_max, thickness_um, branch_index):
    # solving the mode transcendental eq
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
    # take the power normalization of even and odd modes w closed integral approximation
    d = thickness_um
    if even:
        p_core = d / 2 + math.sin(kappa * d) / (2 * kappa)
        p_clad = math.cos(kappa * d / 2) ** 2 / gamma
    else:
        p_core = d / 2 - math.sin(kappa * d) / (2 * kappa)
        p_clad = math.sin(kappa * d / 2) ** 2 / gamma
    return p_core, p_clad


def _slab_mode_field(y, kappa, gamma, thickness_um, even):
    # Marcatili's method to solve for 2D wg
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
    # calculate confinement and neff from core and clad indices, thickness, wavelength
    k0 = 2 * math.pi / wavelength_um
    kappa_max = k0 * math.sqrt(n_core ** 2 - n_clad ** 2)
    kappa, gamma = _slab_branch_root(kappa_max, thickness_um, 0)
    n_eff = math.sqrt(n_core ** 2 - (kappa / k0) ** 2)
    p_core, p_clad = _slab_mode_power_unnorm(kappa, gamma, thickness_um, True)
    confinement = p_core / (p_core + p_clad)
    return confinement, n_eff


def strip_confinement(n_core, n_clad, width_um, height_um, wavelength_um):
    # effective index method to find confinement, solve for 1 axis, plug neff into other
    confinement_v, n_eff_v = slab_mode(n_core, n_clad, height_um, wavelength_um)
    confinement_h, n_eff_2d = slab_mode(n_eff_v, n_clad, width_um, wavelength_um)
    return confinement_v * confinement_h, n_eff_2d


def material_loss_db_per_cm(confinement, alpha_core_db_per_cm, alpha_clad_db_per_cm):
    # material loss based on confinement using first-order perturburation theory
    return confinement * alpha_core_db_per_cm + (1 - confinement) * alpha_clad_db_per_cm


# MMI excess loss
def slab_lateral_modes(n_eff_core, n_clad, width_um, wavelength_um, max_modes=40):
    # find possible modes and return their calculated qualities
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
    # Simpson's rule
    if n % 2 == 1:
        n += 1
    h = (b - a) / n
    total = f(a) + f(b)
    for i in range(1, n):
        total += (4 if i % 2 else 2) * f(a + i * h)
    return total * h / 3


def _access_mode_overlap(mode, mmi_width_um, y0, access_width_um, n_eff_core, n_clad, wavelength_um):
    # find the overlap of supported modes with the fundamental mode
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
    # evenly space ports
    return (2 * port_index_1based - n_ports - 1) * mmi_width_um / (2 * n_ports)


def _mmi_projections(n_eff_core, n_clad, mmi_width_um, n_ports, access_width_um, wavelength_um):
    # calculate mode overlap at the input, then iterate on output port locations to calculate overlap
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
    # plug in calculated values, take the imaginary and real component and mult by overlap power, then take the imaginary attenuation and add everything together
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
    # excess loss is negative total power in db
    modes, c, port_overlaps = _mmi_projections(
        n_eff_core, n_clad, mmi_width_um, n_ports, access_width_um, wavelength_um)
    power = _mmi_total_power(modes, c, port_overlaps, mmi_length_um)
    return -mw_to_dbm(power), len(modes)


def mmi_derive_length_and_loss(n_eff_core, n_clad, mmi_width_um, n_ports, access_width_um, wavelength_um):
    # calculates the optimal length of mmi for self projection, sweep from 0 to 3L_pi to find length w most power
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
    return best_length, -mw_to_dbm(best_power), len(modes)

# propagation loss
def propagation_loss_db(alpha_db_per_cm, length_um):
    length_cm = length_um * 1e-4
    return alpha_db_per_cm * length_cm

# mmi
def mmi_total_loss_db(n_ports, excess_loss_db):
    return 10 * math.log10(n_ports) + excess_loss_db

# modulator power penalty
def er_db_from_splitting_ratio(gamma):
    # power must be in core or cladding, so ratio is always gamma:1-gamma and ideal is 3db
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
def mzm_capacitance_fF(kappa, length_um, height_um, width_um):
    # calculate modulation losses due to capacitance, C = kappa*e_0*A/d, A = l*h of waveguide, d is waveguide width
    length_m = length_um * 1e-6
    height_m = height_um * 1e-6
    width_m = width_um * 1e-6
    area_m2 = length_m * height_m
    d_m = width_m
    C_F = kappa * EPSILON_0 * area_m2 / d_m
    return C_F * 1e15

def modulation_efficiency_capacitive_pJ(C_fF, V):
    # on average 1/4 of bits require change in modulation electrode
    C_F = C_fF * 1e-15
    E_J = 0.25 * C_F * V ** 2
    return E_J * 1e12

def traveling_wave_drive_power_w(vpp, z0_ohm):
    return vpp ** 2 / (4 * z0_ohm)

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
