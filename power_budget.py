import math

# constants
Q_ELECTRON = 1.602176634e-19
H_PLANCK = 6.62607015e-34
C_LIGHT = 2.99792458e8

# conversions
def dbm_to_mw(p_dbm):
    return 10 ** (p_dbm / 10)

def mw_to_dbm(p_mw):
    return 10 * math.log10(p_mw)

def db_to_linear(l_db):
    return 10 ** (-l_db / 10)

# propagation loss
def propagation_loss_db(alpha_db_per_cm, length_um):
    length_cm = length_um * 1e-4
    return alpha_db_per_cm * length_cm

def total_propagation_loss_db(segments):
    return sum(propagation_loss_db(alpha, length_um) for alpha, length_um in segments)

# mmi
def mmi_split_loss_db(n_ports):
    return 10 * math.log10(n_ports)

def mmi_total_loss_db(n_ports, excess_loss_db):
    return mmi_split_loss_db(n_ports) + excess_loss_db

# modulator  power penalty
def er_power_penalty_db(er_db):
    er_linear = 10 ** (er_db / 10)
    return -10 * math.log10((er_linear + 1) / (er_linear - 1))

# photodetector
def qe_from_responsivity(responsivity_a_per_w, wavelength_nm):
    wavelength_m = wavelength_nm * 1e-9
    return (responsivity_a_per_w * H_PLANCK * C_LIGHT) / (Q_ELECTRON * wavelength_m)

def qe_penalty_db(eta_qe):
    return -10 * math.log10(eta_qe)

def pd_metrics(responsivity_a_per_w, p_pd_dbm):
    p_pd_w = dbm_to_mw(p_pd_dbm) * 1e-3
    photocurrent_a = responsivity_a_per_w * p_pd_w
    return photocurrent_a

# modulation efficiency
def modulation_efficiency_capacitive_pJ(C_fF, V):
    C_F = C_fF * 1e-15
    E_J = 0.5 * C_F * V ** 2
    return E_J * 1e12

def modulation_efficiency_power_pJ(p_drive_w, bit_rate_bps):
    e_bit_j = p_drive_w / bit_rate_bps
    return e_bit_j * 1e12

# power budget
def compute_power_budget(p_laser_dbm, l_in_db, l_prop1_db, l_mmi_total_db, l_prop2_db,
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

    p += l_mod_er_penalty_db
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
