"""Parse JPL Horizons vector output and generate trajectory JSON for the simulator.

Usage:
  python3 gen_trajectory.py                    # default: 2026-04-01 22:24 UTC
  python3 gen_trajectory.py 2026-04-02T18:24   # custom launch time (UTC)
"""
import json, re, math, sys
from datetime import datetime, timezone
from scipy.optimize import least_squares, minimize
from numba import njit

def parse_horizons(filepath):
    """Parse Horizons vector output into list of {jd, x, y, z, vx, vy, vz}."""
    records = []
    with open(filepath) as f:
        lines = f.readlines()

    in_data = False
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == '$$SOE':
            in_data = True
            i += 1
            continue
        if line == '$$EOE':
            break
        if not in_data:
            i += 1
            continue

        # Line 1: JD = date string
        m = re.match(r'(\d+\.\d+)\s*=\s*A\.D\.\s*(.+)', line)
        if not m:
            i += 1
            continue
        jd = float(m.group(1))
        date_str = m.group(2).strip()

        # Line 2: " X = val Y = val Z = val"
        i += 1
        vals = re.findall(r'[-+]?\d+\.?\d*E[+-]\d+', lines[i])
        if len(vals) < 3:
            i += 1
            continue
        x, y, z = float(vals[0]), float(vals[1]), float(vals[2])

        # Line 3: " VX= val VY= val VZ= val"
        i += 1
        vals = re.findall(r'[-+]?\d+\.?\d*E[+-]\d+', lines[i])
        vx, vy, vz = float(vals[0]), float(vals[1]), float(vals[2])

        # Line 4: LT, RG, RR (skip)
        i += 1

        records.append({
            'jd': jd, 'date': date_str,
            'x': x, 'y': y, 'z': z,
            'vx': vx, 'vy': vy, 'vz': vz,
        })
        i += 1

    return records


def compute_met_hours(jd, launch_jd):
    return (jd - launch_jd) * 24.0


def utc_to_jd(dt):
    """Convert datetime (UTC) to Julian Date."""
    # JD = 367*Y - int(7*(Y+int((M+9)/12))/4) + int(275*M/9) + D + 1721013.5 + UT/24
    y, m, d = dt.year, dt.month, dt.day
    ut = dt.hour + dt.minute / 60 + dt.second / 3600
    if m <= 2:
        y -= 1
        m += 12
    A = int(y / 100)
    B = 2 - A + int(A / 4)
    return int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + ut / 24 + B - 1524.5


# Default: 2026-04-01 22:35 UTC (Artemis II actual launch)
DEFAULT_LAUNCH_UTC = "2026-04-01T22:35"

def parse_launch_time():
    """Parse launch time from CLI arg or use default. Returns JD."""
    if len(sys.argv) > 1:
        launch_str = sys.argv[1]
    else:
        launch_str = DEFAULT_LAUNCH_UTC

    dt = datetime.fromisoformat(launch_str).replace(tzinfo=timezone.utc)
    jd = utc_to_jd(dt)
    print(f"Launch time: {dt.isoformat()} (JD {jd:.6f})", file=sys.stderr)
    return jd


def merge_fine_and_coarse(fine_sc, fine_moon, coarse_sc, coarse_moon):
    """Merge 1-min near-Earth data with 10-min cruise data.
    Use fine data for the overlap period, then switch to coarse."""
    if not fine_sc:
        return coarse_sc, coarse_moon

    fine_end_jd = fine_sc[-1]['jd']
    print(f"  Fine data: {len(fine_sc)} pts, up to JD {fine_end_jd:.3f}", file=sys.stderr)

    # Keep all fine data, then append coarse data after the fine period
    merged_sc = list(fine_sc)
    merged_moon = list(fine_moon)
    for i, sc in enumerate(coarse_sc):
        if sc['jd'] > fine_end_jd:
            merged_sc.append(sc)
            merged_moon.append(coarse_moon[min(i, len(coarse_moon)-1)])

    return merged_sc, merged_moon


def cross(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]

def norm(v):
    r = math.sqrt(sum(x*x for x in v))
    return [x/r for x in v] if r > 0 else v

def mag(v):
    return math.sqrt(sum(x*x for x in v))

def scale_vec(v, s):
    return [x*s for x in v]

def add_vec(a, b):
    return [a[i]+b[i] for i in range(3)]


def dot(a, b):
    return sum(a[i]*b[i] for i in range(3))


# ============================================================
# Lambert solver (universal variable / Stumpff functions)
# ============================================================
def stumpff_c(z):
    if z > 1e-6:
        return (1 - math.cos(math.sqrt(z))) / z
    elif z < -1e-6:
        return (math.cosh(math.sqrt(-z)) - 1) / (-z)
    return 0.5

def stumpff_s(z):
    if z > 1e-6:
        sq = math.sqrt(z)
        return (sq - math.sin(sq)) / (sq ** 3)
    elif z < -1e-6:
        sq = math.sqrt(-z)
        return (math.sinh(sq) - sq) / (sq ** 3)
    return 1.0 / 6.0

def lambert(r1_vec, r2_vec, tof_sec, mu=398600.4418):
    """Solve Lambert's problem: find the orbit connecting r1 and r2 in tof seconds.
    Returns (v1, v2) velocity vectors, or None if no solution."""
    r1 = mag(r1_vec)
    r2 = mag(r2_vec)
    cos_dnu = max(-1, min(1, dot(r1_vec, r2_vec) / (r1 * r2)))

    # Use cross product to determine prograde/retrograde
    c = cross(r1_vec, r2_vec)
    # Prograde: angular momentum in +Z direction (typical for LEO launches from KSC)
    sin_dnu = math.sqrt(max(0, 1 - cos_dnu ** 2))
    if c[2] < 0:
        sin_dnu = -sin_dnu

    A = sin_dnu * math.sqrt(r1 * r2 / (1 - cos_dnu))
    if abs(A) < 1e-10:
        return None

    def tof_from_z(z):
        cz = stumpff_c(z)
        sz = stumpff_s(z)
        y = r1 + r2 + A * (z * sz - 1) / math.sqrt(max(cz, 1e-12))
        if y < 0:
            return float('inf')
        chi = math.sqrt(y / max(cz, 1e-12))
        return (chi ** 3 * sz + A * math.sqrt(max(y, 0))) / math.sqrt(mu)

    # Bisection to find z
    # z > 0: elliptic, z = 0: parabolic, z < 0: hyperbolic
    # Start with safe range and widen carefully
    z_lo, z_hi = -2.0, 4 * math.pi ** 2
    # Widen lower bound carefully (avoid overflow in cosh for large |z|)
    for _ in range(30):
        try:
            t_lo = tof_from_z(z_lo)
            if t_lo > tof_sec:
                break
            z_lo *= 1.5
            if z_lo < -100:
                break
        except (OverflowError, ValueError):
            z_lo /= 1.5
            break

    for _ in range(200):
        z_mid = (z_lo + z_hi) / 2
        t_mid = tof_from_z(z_mid)
        if abs(t_mid - tof_sec) < 1e-4:
            break
        if t_mid < tof_sec:
            z_hi = z_mid
        else:
            z_lo = z_mid

    z = (z_lo + z_hi) / 2
    cz = stumpff_c(z)
    sz = stumpff_s(z)
    y = r1 + r2 + A * (z * sz - 1) / math.sqrt(max(cz, 1e-12))

    f = 1 - y / r1
    g = A * math.sqrt(max(y, 0) / mu)
    g_dot = 1 - y / r2

    if abs(g) < 1e-12:
        return None

    v1 = [(r2_vec[i] - f * r1_vec[i]) / g for i in range(3)]
    v2 = [(g_dot * r2_vec[i] - r1_vec[i]) / g for i in range(3)]
    return v1, v2


import numpy as np

_J2 = 1.08263e-3
_J3 = -2.5327e-6
_R_E_J2 = 6378.137
_MU = 398600.4418
_MU_MOON = 4902.8
_NO_MOON = np.zeros(3)

@njit(cache=True)
def _rk4_step(s, dt, moon_pos):
    """Numba-optimized RK4 step with Earth J2 + Moon gravity."""
    def deriv(s):
        r2 = s[0]*s[0] + s[1]*s[1] + s[2]*s[2]
        r = math.sqrt(r2)
        r3 = r2 * r
        ax = -_MU*s[0]/r3
        ay = -_MU*s[1]/r3
        az = -_MU*s[2]/r3
        # J2 + J3
        z = s[2]; z2 = z*z; z2_r2 = z2 / r2
        Re2 = _R_E_J2 * _R_E_J2
        fJ2 = 1.5 * _J2 * _MU * Re2 / (r2 * r3)
        ax += fJ2 * s[0] * (5*z2_r2 - 1)
        ay += fJ2 * s[1] * (5*z2_r2 - 1)
        az += fJ2 * s[2] * (5*z2_r2 - 3)
        # J3 (zonal harmonic, derived from the degree-3 Legendre term)
        Re3 = Re2 * _R_E_J2
        r5 = r2 * r3
        r7 = r5 * r2
        r9 = r7 * r2
        fJ3 = 0.5 * _J3 * _MU * Re3
        common_xy = 5.0 * fJ3 * z * (3.0 * r2 - 7.0 * z2) / r9
        ax += common_xy * s[0]
        ay += common_xy * s[1]
        az += fJ3 * (-3.0 * r2 * r2 + 30.0 * z2 * r2 - 35.0 * z2 * z2) / r9
        # Moon
        if moon_pos[0] != 0.0 or moon_pos[1] != 0.0 or moon_pos[2] != 0.0:
            dx = s[0]-moon_pos[0]; dy = s[1]-moon_pos[1]; dz = s[2]-moon_pos[2]
            rm = math.sqrt(dx*dx+dy*dy+dz*dz); rm3 = rm*rm*rm
            rm0 = math.sqrt(moon_pos[0]**2+moon_pos[1]**2+moon_pos[2]**2); rm03 = rm0**3
            ax += -_MU_MOON*dx/rm3 - _MU_MOON*moon_pos[0]/rm03
            ay += -_MU_MOON*dy/rm3 - _MU_MOON*moon_pos[1]/rm03
            az += -_MU_MOON*dz/rm3 - _MU_MOON*moon_pos[2]/rm03
        out = np.empty(6)
        out[0]=s[3]; out[1]=s[4]; out[2]=s[5]; out[3]=ax; out[4]=ay; out[5]=az
        return out

    k1 = deriv(s)
    k2 = deriv(s + dt*0.5*k1)
    k3 = deriv(s + dt*0.5*k2)
    k4 = deriv(s + dt*k3)
    return s + dt/6.0 * (k1 + 2*k2 + 2*k3 + k4)

@njit(cache=True)
def _propagate_segment(state, dt, total_sec, moon_pos):
    """Propagate a segment using RK4. Returns final state."""
    s = state.copy()
    elapsed = 0.0
    while elapsed < total_sec:
        this_dt = min(dt, total_sec - elapsed)
        s = _rk4_step(s, this_dt, moon_pos)
        elapsed += this_dt
    return s


@njit(cache=True)
def _propagate_segment_linear_moon(state, dt, total_sec, moon_pos_ref, moon_vel_ref, seg_start_sec, moon_ref_sec):
    """Propagate a segment using RK4 with linearly moving moon ephemeris."""
    s = state.copy()
    elapsed = 0.0
    while elapsed < total_sec:
        this_dt = min(dt, total_sec - elapsed)
        mid_t = seg_start_sec + elapsed + 0.5 * this_dt
        moon_pos = moon_pos_ref + moon_vel_ref * (mid_t - moon_ref_sec)
        s = _rk4_step(s, this_dt, moon_pos)
        elapsed += this_dt
    return s


@njit(cache=True)
def _propagate_burn_linear_moon(state, dt, burn_sec, dv_total, dv_radial_total,
                                moon_pos_ref, moon_vel_ref, burn_start_sec, moon_ref_sec):
    """Propagate while applying a finite burn with linearly moving moon ephemeris."""
    s = state.copy()
    elapsed = 0.0
    while elapsed < burn_sec:
        this_dt = min(dt, burn_sec - elapsed)
        mid_t = burn_start_sec + elapsed + 0.5 * this_dt
        moon_pos = moon_pos_ref + moon_vel_ref * (mid_t - moon_ref_sec)
        s = _rk4_step(s, this_dt, moon_pos)
        dv_step = dv_total * this_dt / burn_sec
        dv_radial_step = dv_radial_total * this_dt / burn_sec
        vx = s[3]; vy = s[4]; vz = s[5]
        rx = s[0]; ry = s[1]; rz = s[2]
        vmag = math.sqrt(vx*vx + vy*vy + vz*vz)
        rmag = math.sqrt(rx*rx + ry*ry + rz*rz)
        if vmag > 0.0:
            s[3] += vx / vmag * dv_step
            s[4] += vy / vmag * dv_step
            s[5] += vz / vmag * dv_step
        if rmag > 0.0:
            s[3] += rx / rmag * dv_radial_step
            s[4] += ry / rmag * dv_radial_step
            s[5] += rz / rmag * dv_radial_step
        elapsed += this_dt
    return s

# Wrapper to accept lists (for compatibility)
def rk4_step(state, dt, mu=398600.4418, moon_pos=None, mu_moon=4902.8):
    s = np.array(state, dtype=np.float64)
    mp = np.array(moon_pos, dtype=np.float64) if moon_pos else _NO_MOON
    result = _rk4_step(s, dt, mp)
    return result.tolist()


def get_era(jd):
    """Earth Rotation Angle (IAU 2000)."""
    return 2 * math.pi * (0.7790572732640 + 1.00273781191135448 * (jd - 2451545.0))


def ksc_icrf(jd):
    """Kennedy Space Center position in ICRF at given JD.
    KSC: lat 28.5724°N, lon -80.649°W (= 279.351°E), alt ~0 km."""
    lat = math.radians(28.5724)
    lon = math.radians(279.351)
    R = 6378.0  # km

    # ECEF position
    x_ecef = R * math.cos(lat) * math.cos(lon)
    y_ecef = R * math.cos(lat) * math.sin(lon)
    z_ecef = R * math.sin(lat)

    # ECEF → ICRF: rotate by ERA around Z
    era = get_era(jd)
    x_icrf = x_ecef * math.cos(era) - y_ecef * math.sin(era)
    y_icrf = x_ecef * math.sin(era) + y_ecef * math.cos(era)
    z_icrf = z_ecef
    return [x_icrf, y_icrf, z_icrf]


def synthesize_early_trajectory(first_sc, first_moon, launch_jd, SCALE):
    """Synthesize MET 0 to first Horizons point using 3-phase orbital mechanics.

    Phase 0: Launch ascent (T+0 to T+0.13h)
    Phase 1: Insertion orbit (T+0.13h to burn1) — perigee 27km, apogee 2222km
    Phase 2: After burn1 (perigee raise, ~44 m/s at apogee) — coast to burn2
    Phase 3: After burn2 (apogee raise, ~2276 m/s at perigee, 18min) — coast to Horizons
    Burns optimized to match Horizons endpoint (position + velocity direction).
    """
    first_met_h = compute_met_hours(first_sc['jd'], launch_jd)
    print(f"  Synthesizing MET 0 to {first_met_h:.1f}h", file=sys.stderr)

    r_target = [first_sc['x'], first_sc['y'], first_sc['z']]
    v_target = [first_sc['vx'], first_sc['vy'], first_sc['vz']]
    moon_pos_scene = [first_moon['x'] * SCALE, first_moon['y'] * SCALE, first_moon['z'] * SCALE]
    moon_dist_km = mag([first_moon['x'], first_moon['y'], first_moon['z']])
    moon_km = [first_moon['x'], first_moon['y'], first_moon['z']]
    moon_vel_km = [first_moon['vx'], first_moon['vy'], first_moon['vz']]

    MU = 398600.4418
    R_E = 6378

    # --- Orbital plane + basis vectors from Horizons ---
    ksc_pos = ksc_icrf(launch_jd)
    print(f"  KSC ICRF: ({ksc_pos[0]:.0f}, {ksc_pos[1]:.0f}, {ksc_pos[2]:.0f}) km", file=sys.stderr)

    h_vec_t34 = norm(cross(r_target, v_target))

    # Correct orbital plane from T+3.4h back to T+0
    # Total J2 precession over 3 orbit phases: ΔRAAN=-0.39°, Δω=+0.64°
    # Rotate h_vec around Z by -ΔRAAN to get T+0 plane
    # Also rotate basis vectors by -Δω for perigee argument
    _dRaan_total = math.radians(-0.39)  # RAAN shift T+0→T+3.4h
    _domega_total = math.radians(0.64)  # ω shift T+0→T+3.4h
    cos_r = math.cos(-_dRaan_total); sin_r = math.sin(-_dRaan_total)
    h_vec = norm([
        h_vec_t34[0]*cos_r - h_vec_t34[1]*sin_r,
        h_vec_t34[0]*sin_r + h_vec_t34[1]*cos_r,
        h_vec_t34[2]
    ])

    ksc_in_plane = [ksc_pos[i] - dot(ksc_pos, h_vec) * h_vec[i] for i in range(3)]
    e1_base = norm(ksc_in_plane)
    e2_base = norm(cross(h_vec, e1_base))

    # --- Orbital parameters ---
    INS_PERI = R_E + 27;   INS_APO = R_E + 2222
    RAISED_PERI = R_E + 185
    CHECKOUT_APO = R_E + 70000

    a_ins = (INS_PERI + INS_APO) / 2
    e_ins = (INS_APO - INS_PERI) / (INS_APO + INS_PERI)
    p_ins = a_ins * (1 - e_ins**2)
    omega_ins = math.sqrt(MU / a_ins**3)

    def v_at(r, a):
        return math.sqrt(MU * (2/r - 1/a))

    ASCENT_H = 0.13
    BURN1_H = 0.82   # perigee raise (ICPS) ~T+49min
    BURN2_H = 1.8    # apogee raise (ICPS) ~T+1:48, 18min burn
    BURN2_DURATION_SEC = 18.0 * 60.0
    dt = 10.0  # 10-second steps (numba JIT for speed)

    def kepler_solve_burn(M, ecc, a, tol=1e-10):
        E = M
        for _ in range(50):
            dE = (M - E + ecc * math.sin(E)) / (1 - ecc * math.cos(E))
            E += dE
            if abs(dE) < tol: break
        nu = 2 * math.atan2(math.sqrt(1+ecc)*math.sin(E/2), math.sqrt(1-ecc)*math.cos(E/2))
        r = a * (1 - ecc * math.cos(E))
        return nu, r

    # Nominal Δv
    a_after1 = (RAISED_PERI + INS_APO) / 2
    a_after2 = (RAISED_PERI + CHECKOUT_APO) / 2
    M_b1 = omega_ins * BURN1_H * 3600
    nu_b1_nom, r_b1_mag_nom = kepler_solve_burn(M_b1, e_ins, a_ins)
    dv1_nom = v_at(INS_APO, a_after1) - v_at(r_b1_mag_nom, a_ins)
    dv2_nom = v_at(RAISED_PERI, a_after2) - v_at(RAISED_PERI, a_after1)
    print(f"  Nominal: dv1={dv1_nom*1000:.0f} m/s, dv2={dv2_nom*1000:.0f} m/s", file=sys.stderr)

    # --- Propagation with burns + launch azimuth parameter ---
    def rotate_basis(angle):
        """Rotate e1/e2 in the orbital plane by angle (radians)."""
        c, s = math.cos(angle), math.sin(angle)
        e1r = [c*e1_base[i] + s*e2_base[i] for i in range(3)]
        e2r = [-s*e1_base[i] + c*e2_base[i] for i in range(3)]
        return e1r, e2r

    # J2 secular precession of insertion orbit (T+0 → burn1)
    # Computed from 3-orbit analysis: investment + intermediate + checkout
    # RAAN shift: -0.177° for insertion orbit over 0.82h
    # ω shift: +0.289° for insertion orbit over 0.82h
    # These values correct for the fact that h_vec is from T+3.4h, not T+0
    _inc_rad = math.acos(max(-1, min(1, h_vec[2])))
    _n_ins = math.sqrt(MU / a_ins**3)
    _p_factor = (R_E / a_ins)**2 / (1 - e_ins**2)**2
    _dRaan_dt = -1.5 * _n_ins * _J2 * _p_factor * math.cos(_inc_rad)
    _domega_dt = 1.5 * _n_ins * _J2 * _p_factor * (2 - 2.5 * math.sin(_inc_rad)**2)

    def get_burn1_state(azimuth, b1h=BURN1_H):
        """Compute burn1 state on insertion orbit (Kepler equation, fast)."""
        e1r, e2r = rotate_basis(azimuth)
        omega = math.sqrt(MU / a_ins**3)
        M = omega * b1h * 3600
        nu, r_mag = kepler_solve_burn(M, e_ins, a_ins)
        r = add_vec(scale_vec(e1r, r_mag * math.cos(nu)),
                    scale_vec(e2r, r_mag * math.sin(nu)))
        v_dir = norm([-math.sin(nu) * e1r[i] + math.cos(nu) * e2r[i] for i in range(3)])
        v_mag = v_at(r_mag, a_ins)
        return r, scale_vec(v_dir, v_mag), v_dir, norm(r)

    moon_np = np.array(moon_km, dtype=np.float64)
    moon_vel_np = np.array(moon_vel_km, dtype=np.float64)
    moon_ref_sec = first_met_h * 3600.0

    def moon_pos_at_met(met_h):
        dt_sec = met_h * 3600.0 - moon_ref_sec
        return [moon_km[i] + moon_vel_km[i] * dt_sec for i in range(3)]

    def orbital_altitudes(state):
        r_vec = state[:3].tolist() if hasattr(state[:3], 'tolist') else list(state[:3])
        v_vec = state[3:].tolist() if hasattr(state[3:], 'tolist') else list(state[3:])
        rmag = mag(r_vec)
        vmag = mag(v_vec)
        energy = vmag * vmag / 2 - MU / rmag
        if energy >= 0:
            return float('inf'), float('inf')
        hmag = mag(cross(r_vec, v_vec))
        ecc_term = 1 + 2 * energy * hmag * hmag / (MU * MU)
        ecc = math.sqrt(max(0.0, ecc_term))
        a = -MU / (2 * energy)
        rp = a * (1 - ecc) - R_E
        ra = a * (1 + ecc) - R_E
        return rp, ra

    def evaluate_trajectory(dv1, dv2, dv1r, dv2r, azimuth, b1h=BURN1_H, b2h=BURN2_H):
        r_b1, v_b1, v_b1_dir, r_b1_hat = get_burn1_state(azimuth, b1h)
        st_ins = list(r_b1) + list(v_b1)
        ins_seg = (b1h - ASCENT_H) * 3600
        elapsed_back = 0.0
        min_alt_phase1 = mag(r_b1) - R_E
        while elapsed_back < ins_seg:
            this_dt = min(dt, ins_seg - elapsed_back)
            moon_pos = moon_pos_at_met(b1h - (elapsed_back + 0.5 * this_dt) / 3600)
            st_ins = rk4_step(st_ins, -this_dt, moon_pos=moon_pos)
            min_alt_phase1 = min(min_alt_phase1, mag(st_ins[:3]) - R_E)
            elapsed_back += this_dt
        meco_state = np.array(st_ins, dtype=np.float64)
        v_after = add_vec(v_b1, add_vec(scale_vec(v_b1_dir, dv1), scale_vec(r_b1_hat, dv1r)))
        st = np.array(list(r_b1) + v_after, dtype=np.float64)
        burn1_after_state = st.copy()
        burn2_start_h = b2h - BURN2_DURATION_SEC / 7200.0
        burn2_end_h = b2h + BURN2_DURATION_SEC / 7200.0
        # Segment 1: burn1 → burn2 start
        st = _propagate_segment_linear_moon(
            st, dt, (burn2_start_h - b1h) * 3600, moon_np, moon_vel_np, b1h * 3600.0, moon_ref_sec
        )
        mid_state = st.copy()
        # Finite burn2 centered at b2h
        st = _propagate_burn_linear_moon(
            st, dt, BURN2_DURATION_SEC, dv2, dv2r, moon_np, moon_vel_np, burn2_start_h * 3600.0, moon_ref_sec
        )
        post_burn2_state = st.copy()
        # Segment 2: burn2 end → Horizons
        st = _propagate_segment_linear_moon(
            st, dt, (first_met_h - burn2_end_h) * 3600, moon_np, moon_vel_np, burn2_end_h * 3600.0, moon_ref_sec
        )
        return {
            'pos': st[:3].tolist(),
            'vel': st[3:].tolist(),
            'mid_state': mid_state,
            'post_burn2_state': post_burn2_state,
            'burn1_after_state': burn1_after_state,
            'meco_state': meco_state,
            'min_alt_phase1': min_alt_phase1,
        }

    def propagate(dv1, dv2, dv1r, dv2r, azimuth, b1h=BURN1_H, b2h=BURN2_H):
        result = evaluate_trajectory(dv1, dv2, dv1r, dv2r, azimuth, b1h, b2h)
        return result['pos'], result['vel']

    def propagate_segment_state(state, start_h, end_h):
        state_np = np.array(state, dtype=np.float64)
        if end_h <= start_h:
            return state_np.copy()
        return _propagate_segment_linear_moon(
            state_np, dt, (end_h - start_h) * 3600, moon_np, moon_vel_np, start_h * 3600.0, moon_ref_sec
        )

    def propagate_burn_state(state, burn_center_h, dv_prograde, dv_radial):
        state_np = np.array(state, dtype=np.float64)
        burn_start_h = burn_center_h - BURN2_DURATION_SEC / 7200.0
        return _propagate_burn_linear_moon(
            state_np, dt, BURN2_DURATION_SEC, dv_prograde, dv_radial,
            moon_np, moon_vel_np, burn_start_h * 3600.0, moon_ref_sec
        )

    # 9 params: Δv (4) + azimuth + burn times (2) + insertion orbit adjustment (2)
    # Insertion orbit starts at 27×2222km, allowed ±10% adjustment
    x0 = [dv1_nom, dv2_nom, 0, 0, 0, BURN1_H, BURN2_H, 1.0, 1.0]
    bounds = [
        (-0.08, 0.12),          # dv1
        (1.5, 3.5),             # dv2
        (-0.08, 0.08),          # dv1r
        (-0.08, 0.08),          # dv2r
        (-0.2, 0.2),            # azimuth
        (0.6, 1.1),             # burn1 effective time
        (1.5, 2.2),             # burn2 effective time
        (0.85, 1.15),           # ins_peri scale (±15%)
        (0.9995, 1.0005),       # ins_apo effectively fixed at public 2222 km
    ]

    def cost_bounded(x):
        for i, (lo, hi) in enumerate(bounds):
            if x[i] < lo or x[i] > hi:
                return 1e12
        try:
            # Temporarily adjust insertion orbit
            nonlocal a_ins, e_ins, omega_ins
            ip = INS_PERI * x[7] + R_E * (1 - x[7])  # scale altitude, not radius
            ia = INS_APO * x[8] + R_E * (1 - x[8])
            a_ins = (ip + ia) / 2
            e_ins = (ia - ip) / (ia + ip)
            omega_ins = math.sqrt(MU / a_ins**3)
            traj = evaluate_trajectory(x[0], x[1], x[2], x[3], x[4], x[5], x[6])
            pos, vel = traj['pos'], traj['vel']
            pe = mag([pos[i] - r_target[i] for i in range(3)])
            ve = mag([vel[i] - v_target[i] for i in range(3)]) * 1000
            mid_peri, mid_apo = orbital_altitudes(traj['mid_state'])
            fin_peri, fin_apo = orbital_altitudes(traj['post_burn2_state'])
            meco_alt = mag(traj['meco_state'][:3].tolist()) - R_E
            ins_pen = ((ip - R_E - 27.0) / 27.0) ** 2 + ((ia - R_E - 2222.0) / 2222.0) ** 2
            mid_pen = ((mid_peri - 185.0) / 185.0) ** 2 + ((mid_apo - 2222.0) / 2222.0) ** 2
            fin_pen = ((fin_peri - 185.0) / 185.0) ** 2 + ((fin_apo - 70000.0) / 70000.0) ** 2
            time_pen = ((x[5] - 49.0 / 60.0) / 0.15) ** 2 + ((x[6] - 108.0 / 60.0) / 0.15) ** 2
            radial_pen = (x[2] / 0.02) ** 2 + (x[3] / 0.02) ** 2
            altitude_pen = 0.0
            if traj['min_alt_phase1'] < 0:
                altitude_pen += ((traj['min_alt_phase1']) / 20.0) ** 2
            if meco_alt < 0:
                altitude_pen += (meco_alt / 20.0) ** 2
            return pe + ve + 4000 * ins_pen + 4000 * mid_pen + 3000 * fin_pen + 300 * time_pen + 1000 * radial_pen + 8000 * altitude_pen
        except:
            return 1e12
        finally:
            # Restore
            a_ins = (INS_PERI + INS_APO) / 2
            e_ins = (INS_APO - INS_PERI) / (INS_APO + INS_PERI)
            omega_ins = math.sqrt(MU / a_ins**3)

    # Stage 1: match position (fast convergence)
    res1 = minimize(cost_bounded, x0, method='Nelder-Mead',
                    options={'maxiter': 10000, 'xatol': 1e-5, 'fatol': 0.1, 'adaptive': True})

    # Stage 2: refine velocity from Stage 1 result (higher velocity weight)
    def cost_vel_focus(x):
        for i, (lo, hi) in enumerate(bounds):
            if x[i] < lo or x[i] > hi:
                return 1e12
        try:
            nonlocal a_ins, e_ins, omega_ins
            ip = INS_PERI * x[7] + R_E * (1 - x[7])
            ia = INS_APO * x[8] + R_E * (1 - x[8])
            a_ins = (ip + ia) / 2
            e_ins = (ia - ip) / (ia + ip)
            omega_ins = math.sqrt(MU / a_ins**3)
            traj = evaluate_trajectory(x[0], x[1], x[2], x[3], x[4], x[5], x[6])
            pos, vel = traj['pos'], traj['vel']
            pe = mag([pos[i] - r_target[i] for i in range(3)])
            ve = mag([vel[i] - v_target[i] for i in range(3)]) * 2000  # 2x higher velocity weight
            mid_peri, mid_apo = orbital_altitudes(traj['mid_state'])
            fin_peri, fin_apo = orbital_altitudes(traj['post_burn2_state'])
            meco_alt = mag(traj['meco_state'][:3].tolist()) - R_E
            ins_pen = ((ip - R_E - 27.0) / 27.0) ** 2 + ((ia - R_E - 2222.0) / 2222.0) ** 2
            mid_pen = ((mid_peri - 185.0) / 185.0) ** 2 + ((mid_apo - 2222.0) / 2222.0) ** 2
            fin_pen = ((fin_peri - 185.0) / 185.0) ** 2 + ((fin_apo - 70000.0) / 70000.0) ** 2
            time_pen = ((x[5] - 49.0 / 60.0) / 0.15) ** 2 + ((x[6] - 108.0 / 60.0) / 0.15) ** 2
            radial_pen = (x[2] / 0.02) ** 2 + (x[3] / 0.02) ** 2
            altitude_pen = 0.0
            if traj['min_alt_phase1'] < 0:
                altitude_pen += ((traj['min_alt_phase1']) / 20.0) ** 2
            if meco_alt < 0:
                altitude_pen += (meco_alt / 20.0) ** 2
            return pe + ve + 4000 * ins_pen + 4000 * mid_pen + 3000 * fin_pen + 300 * time_pen + 1000 * radial_pen + 8000 * altitude_pen
        except:
            return 1e12
        finally:
            a_ins = (INS_PERI + INS_APO) / 2
            e_ins = (INS_APO - INS_PERI) / (INS_APO + INS_PERI)
            omega_ins = math.sqrt(MU / a_ins**3)

    result = minimize(cost_vel_focus, list(res1.x), method='Nelder-Mead',
                      options={'maxiter': 20000, 'xatol': 1e-7, 'fatol': 0.001, 'adaptive': True})

    def candidate_metrics(x):
        for i, (lo, hi) in enumerate(bounds):
            if x[i] < lo or x[i] > hi:
                return None
        try:
            nonlocal a_ins, e_ins, omega_ins
            ip = INS_PERI * x[7] + R_E * (1 - x[7])
            ia = INS_APO * x[8] + R_E * (1 - x[8])
            a_ins = (ip + ia) / 2
            e_ins = (ia - ip) / (ia + ip)
            omega_ins = math.sqrt(MU / a_ins**3)
            traj = evaluate_trajectory(x[0], x[1], x[2], x[3], x[4], x[5], x[6])
            pos, vel = traj['pos'], traj['vel']
            pe = mag([pos[i] - r_target[i] for i in range(3)])
            ve = mag([vel[i] - v_target[i] for i in range(3)])
            mid_peri, mid_apo = orbital_altitudes(traj['mid_state'])
            fin_peri, fin_apo = orbital_altitudes(traj['post_burn2_state'])
            meco_alt = mag(traj['meco_state'][:3].tolist()) - R_E
            return {
                'pe': pe,
                've': ve,
                'ins_peri': ip - R_E,
                'ins_apo': ia - R_E,
                'mid_peri': mid_peri,
                'mid_apo': mid_apo,
                'fin_peri': fin_peri,
                'fin_apo': fin_apo,
                'meco_alt': meco_alt,
                'min_alt_phase1': traj['min_alt_phase1'],
            }
        except:
            return None
        finally:
            a_ins = (INS_PERI + INS_APO) / 2
            e_ins = (INS_APO - INS_PERI) / (INS_APO + INS_PERI)
            omega_ins = math.sqrt(MU / a_ins**3)

    def constrained_objective(x):
        m = candidate_metrics(x)
        if m is None:
            return 1e12
        radial_pen = (x[2] / 0.01) ** 2 + (x[3] / 0.01) ** 2
        time_pen = ((x[5] - 49.0 / 60.0) / 0.08) ** 2 + ((x[6] - 108.0 / 60.0) / 0.08) ** 2
        orbit_soft = (
            ((m['ins_peri'] - 27.0) / 5.0) ** 2 +
            ((m['ins_apo'] - 2222.0) / 80.0) ** 2 +
            ((m['mid_peri'] - 185.0) / 20.0) ** 2 +
            ((m['mid_apo'] - 2222.0) / 120.0) ** 2 +
            ((m['fin_peri'] - 185.0) / 40.0) ** 2 +
            ((m['fin_apo'] - 70000.0) / 4000.0) ** 2
        )
        return m['pe'] + m['ve'] * 1500 + 400 * radial_pen + 80 * time_pen + 50 * orbit_soft

    def make_ineq(metric_name, lo=None, hi=None):
        def fn(x):
            m = candidate_metrics(x)
            if m is None:
                return -1e9
            v = m[metric_name]
            if lo is not None and hi is None:
                return v - lo
            if hi is not None and lo is None:
                return hi - v
            return min(v - lo, hi - v)
        return fn

    constraints = [
        {'type': 'ineq', 'fun': make_ineq('min_alt_phase1', lo=0.0)},
        {'type': 'ineq', 'fun': make_ineq('meco_alt', lo=0.0)},
        {'type': 'ineq', 'fun': make_ineq('ins_peri', lo=22.0, hi=32.0)},
        {'type': 'ineq', 'fun': make_ineq('ins_apo', lo=2140.0, hi=2305.0)},
        {'type': 'ineq', 'fun': make_ineq('mid_peri', lo=150.0, hi=220.0)},
        {'type': 'ineq', 'fun': make_ineq('mid_apo', lo=2100.0, hi=2350.0)},
        {'type': 'ineq', 'fun': make_ineq('fin_peri', lo=150.0, hi=260.0)},
        {'type': 'ineq', 'fun': make_ineq('fin_apo', lo=65000.0, hi=75000.0)},
    ]

    res_constrained = minimize(
        constrained_objective,
        list(result.x),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 500, 'ftol': 1e-6, 'disp': False}
    )
    if res_constrained.success:
        result = res_constrained
        print(f"  Constrained refine succeeded ({result.nfev} evals)", file=sys.stderr)
    else:
        print(f"  Constrained refine failed: {res_constrained.message}", file=sys.stderr)
        cobyla_constraints = list(constraints)
        for i, (lo, hi) in enumerate(bounds):
            cobyla_constraints.append({'type': 'ineq', 'fun': lambda x, i=i, lo=lo: x[i] - lo})
            cobyla_constraints.append({'type': 'ineq', 'fun': lambda x, i=i, hi=hi: hi - x[i]})
        res_cobyla = minimize(
            constrained_objective,
            list(result.x),
            method='COBYLA',
            constraints=cobyla_constraints,
            options={'maxiter': 2000, 'rhobeg': 0.05, 'tol': 1e-5, 'disp': False}
        )
        if res_cobyla.success:
            result = res_cobyla
            print(f"  COBYLA refine succeeded ({result.nfev} evals)", file=sys.stderr)
        else:
            print(f"  COBYLA refine failed: {res_cobyla.message}", file=sys.stderr)
    final_x = np.array(result.x, dtype=np.float64)
    final_nfev = getattr(result, 'nfev', 0)

    traj0 = evaluate_trajectory(*final_x[:7])
    ms_x0 = np.concatenate([
        final_x,
        traj0['burn1_after_state'],
        traj0['mid_state'],
        traj0['post_burn2_state'],
    ])
    node_window = np.array([3000, 3000, 3000, 2.0, 2.0, 2.0] * 3, dtype=np.float64)
    lower_ms = np.concatenate([np.array([b[0] for b in bounds], dtype=np.float64), ms_x0[9:] - node_window])
    upper_ms = np.concatenate([np.array([b[1] for b in bounds], dtype=np.float64), ms_x0[9:] + node_window])

    def multiple_shooting_residuals(z):
        p = z[:9]
        y1 = np.array(z[9:15], dtype=np.float64)
        y2 = np.array(z[15:21], dtype=np.float64)
        y3 = np.array(z[21:27], dtype=np.float64)
        ip = INS_PERI * p[7] + R_E * (1 - p[7])
        ia = INS_APO * p[8] + R_E * (1 - p[8])
        nonlocal a_ins, e_ins, omega_ins
        a_ins = (ip + ia) / 2
        e_ins = (ia - ip) / (ia + ip)
        omega_ins = math.sqrt(MU / a_ins**3)
        r_b1, v_b1, v_b1_dir, r_b1_hat = get_burn1_state(p[4], p[5])
        expected_y1 = np.array(list(r_b1) + add_vec(v_b1, add_vec(scale_vec(v_b1_dir, p[0]), scale_vec(r_b1_hat, p[2]))), dtype=np.float64)
        burn2_start_h = p[6] - BURN2_DURATION_SEC / 7200.0
        burn2_end_h = p[6] + BURN2_DURATION_SEC / 7200.0
        prop_y2 = propagate_segment_state(y1, p[5], burn2_start_h)
        prop_y3 = propagate_burn_state(y2, p[6], p[1], p[3])
        endpoint = propagate_segment_state(y3, burn2_end_h, first_met_h)
        mid_peri, mid_apo = orbital_altitudes(y2)
        fin_peri, fin_apo = orbital_altitudes(y3)
        st_ins = list(r_b1) + list(v_b1)
        ins_seg = (p[5] - ASCENT_H) * 3600
        elapsed_back = 0.0
        min_alt_phase1 = mag(r_b1) - R_E
        while elapsed_back < ins_seg:
            this_dt = min(dt, ins_seg - elapsed_back)
            moon_pos = moon_pos_at_met(p[5] - (elapsed_back + 0.5 * this_dt) / 3600)
            st_ins = rk4_step(st_ins, -this_dt, moon_pos=moon_pos)
            min_alt_phase1 = min(min_alt_phase1, mag(st_ins[:3]) - R_E)
            elapsed_back += this_dt
        meco_alt = mag(st_ins[:3]) - R_E
        a_ins = (INS_PERI + INS_APO) / 2
        e_ins = (INS_APO - INS_PERI) / (INS_APO + INS_PERI)
        omega_ins = math.sqrt(MU / a_ins**3)
        res = []
        # Burn 1 is modeled as an impulse: position should connect, velocity may jump.
        res.extend(((y1[:3] - expected_y1[:3]) / 6.0).tolist())
        res.extend(((prop_y2[:3] - y2[:3]) / 5.0).tolist())
        res.extend(((prop_y2[3:] - y2[3:]) / 0.005).tolist())
        res.extend(((prop_y3[:3] - y3[:3]) / 5.0).tolist())
        res.extend(((prop_y3[3:] - y3[3:]) / 0.005).tolist())
        res.extend(((endpoint[:3] - np.array(r_target)) / 12.0).tolist())
        res.extend(((endpoint[3:] - np.array(v_target)) / 0.006).tolist())
        res.extend(((y1 - ms_x0[9:15]) / np.array([80, 80, 80, 0.08, 0.08, 0.08])).tolist())
        res.extend(((y2 - ms_x0[15:21]) / np.array([80, 80, 80, 0.08, 0.08, 0.08])).tolist())
        res.extend(((y3 - ms_x0[21:27]) / np.array([80, 80, 80, 0.08, 0.08, 0.08])).tolist())
        res.extend([
            (ip - R_E - 27.0) / 4.0,
            (ia - R_E - 2222.0) / 12.0,
            (mid_peri - 185.0) / 15.0,
            (mid_apo - 2222.0) / 14.0,
            (fin_peri - 185.0) / 25.0,
            (fin_apo - 70000.0) / 1800.0,
            (p[5] - 49.0 / 60.0) / 0.05,
            (p[6] - 108.0 / 60.0) / 0.05,
            (p[0] - dv1_nom) / 0.02,
            p[2] / 0.006,
            p[3] / 0.006,
            max(0.0, -min_alt_phase1) / 5.0,
            max(0.0, -meco_alt) / 5.0,
        ])
        return np.array(res, dtype=np.float64)

    res_ms = least_squares(
        multiple_shooting_residuals,
        ms_x0,
        bounds=(lower_ms, upper_ms),
        method='trf',
        loss='soft_l1',
        f_scale=1.0,
        max_nfev=1200,
        verbose=0,
    )
    def evaluate_ms_solution(z):
        p = np.array(z[:9], dtype=np.float64)
        y1 = np.array(z[9:15], dtype=np.float64)
        y2 = np.array(z[15:21], dtype=np.float64)
        y3 = np.array(z[21:27], dtype=np.float64)
        ip = INS_PERI * p[7] + R_E * (1 - p[7])
        ia = INS_APO * p[8] + R_E * (1 - p[8])
        nonlocal a_ins, e_ins, omega_ins
        a_ins = (ip + ia) / 2
        e_ins = (ia - ip) / (ia + ip)
        omega_ins = math.sqrt(MU / a_ins**3)
        r_b1, v_b1, v_b1_dir, r_b1_hat = get_burn1_state(p[4], p[5])
        burn1_expected = np.array(list(r_b1) + add_vec(v_b1, add_vec(scale_vec(v_b1_dir, p[0]), scale_vec(r_b1_hat, p[2]))), dtype=np.float64)
        burn2_start_h = p[6] - BURN2_DURATION_SEC / 7200.0
        burn2_end_h = p[6] + BURN2_DURATION_SEC / 7200.0
        prop_y2 = propagate_segment_state(y1, p[5], burn2_start_h)
        prop_y3 = propagate_burn_state(y2, p[6], p[1], p[3])
        endpoint = propagate_segment_state(y3, burn2_end_h, first_met_h)
        mid_peri, mid_apo = orbital_altitudes(y2)
        fin_peri, fin_apo = orbital_altitudes(y3)
        st_ins = list(r_b1) + list(v_b1)
        ins_seg = (p[5] - ASCENT_H) * 3600
        elapsed_back = 0.0
        min_alt_phase1 = mag(r_b1) - R_E
        while elapsed_back < ins_seg:
            this_dt = min(dt, ins_seg - elapsed_back)
            moon_pos = moon_pos_at_met(p[5] - (elapsed_back + 0.5 * this_dt) / 3600)
            st_ins = rk4_step(st_ins, -this_dt, moon_pos=moon_pos)
            min_alt_phase1 = min(min_alt_phase1, mag(st_ins[:3]) - R_E)
            elapsed_back += this_dt
        meco_state = np.array(st_ins, dtype=np.float64)
        pe = mag((endpoint[:3] - np.array(r_target)).tolist())
        va = math.degrees(math.acos(max(-1, min(1, dot(norm(endpoint[3:].tolist()), norm(v_target))))))
        continuity_pos = max(
            mag((y1[:3] - burn1_expected[:3]).tolist()),
            mag((prop_y2[:3] - y2[:3]).tolist()),
            mag((prop_y3[:3] - y3[:3]).tolist()),
        )
        continuity_vel = max(
            mag((prop_y2[3:] - y2[3:]).tolist()),
            mag((prop_y3[3:] - y3[3:]).tolist()),
        )
        a_ins = (INS_PERI + INS_APO) / 2
        e_ins = (INS_APO - INS_PERI) / (INS_APO + INS_PERI)
        omega_ins = math.sqrt(MU / a_ins**3)
        return {
            'params': p,
            'y1': y1,
            'y2': y2,
            'y3': y3,
            'meco_state': meco_state,
            'pe': pe,
            'va': va,
            'ins_peri': ip - R_E,
            'ins_apo': ia - R_E,
            'mid_peri': mid_peri,
            'mid_apo': mid_apo,
            'fin_peri': fin_peri,
            'fin_apo': fin_apo,
            'min_alt_phase1': min_alt_phase1,
            'continuity_pos': continuity_pos,
            'continuity_vel': continuity_vel,
        }
    ms_solution = None
    if res_ms.success:
        ms_eval = evaluate_ms_solution(res_ms.x)
        print(f"  Multiple shooting refine succeeded ({res_ms.nfev} evals): "
              f"pos={ms_eval['pe']:.0f}km vel={ms_eval['va']:.1f}deg "
              f"cont={ms_eval['continuity_pos']:.0f}km/{ms_eval['continuity_vel']*1000:.0f}m/s "
              f"min_alt={ms_eval['min_alt_phase1']:.0f}km", file=sys.stderr)
        if ms_eval['pe'] < 200 and ms_eval['continuity_pos'] < 50 and ms_eval['continuity_vel'] < 0.05:
            ms_solution = ms_eval
            final_x = np.array(ms_eval['params'], dtype=np.float64)
            final_nfev = res_ms.nfev
            print("  Multiple shooting solution accepted", file=sys.stderr)
        else:
            print("  Multiple shooting solution rejected (too much endpoint/continuity error)", file=sys.stderr)
    else:
        print(f"  Multiple shooting refine failed: {res_ms.message}", file=sys.stderr)

    dv1o, dv2o, dv1ro, dv2ro, az_opt, b1h_opt, b2h_opt, peri_scale, apo_scale = final_x
    # Apply final orbit adjustment
    INS_PERI_FINAL = INS_PERI * peri_scale + R_E * (1 - peri_scale)
    INS_APO_FINAL = INS_APO * apo_scale + R_E * (1 - apo_scale)
    a_ins = (INS_PERI_FINAL + INS_APO_FINAL) / 2
    e_ins = (INS_APO_FINAL - INS_PERI_FINAL) / (INS_APO_FINAL + INS_PERI_FINAL)
    omega_ins = math.sqrt(MU / a_ins**3)
    if ms_solution is None:
        fp, fv = propagate(dv1o, dv2o, dv1ro, dv2ro, az_opt, b1h_opt, b2h_opt)
        pe = mag([fp[i]-r_target[i] for i in range(3)])
        va = math.degrees(math.acos(max(-1, min(1, dot(norm(fv), norm(v_target))))))
    else:
        fp = propagate_segment_state(
            ms_solution['y3'],
            b2h_opt + BURN2_DURATION_SEC / 7200.0,
            first_met_h
        )[:3].tolist()
        fv = propagate_segment_state(
            ms_solution['y3'],
            b2h_opt + BURN2_DURATION_SEC / 7200.0,
            first_met_h
        )[3:].tolist()
        pe = ms_solution['pe']
        va = ms_solution['va']
    print(f"  Optimized ({final_nfev} evals): dv1={dv1o*1000:.0f}+{dv1ro*1000:.0f}r, "
          f"dv2={dv2o*1000:.0f}+{dv2ro*1000:.0f}r, az={math.degrees(az_opt):.1f}deg, "
          f"b1={b1h_opt:.3f}h, b2={b2h_opt:.3f}h, "
          f"peri={INS_PERI_FINAL-R_E:.0f}km({peri_scale:.3f}), apo={INS_APO_FINAL-R_E:.0f}km({apo_scale:.3f}), "
          f"pos={pe:.0f}km, vel={va:.1f}deg", file=sys.stderr)

    # Use optimized values for trajectory generation
    e1, e2 = rotate_basis(az_opt)
    # a_ins, e_ins, omega_ins already set from fixed INS_PERI/INS_APO
    BURN1_H = b1h_opt
    BURN2_H = b2h_opt

    # --- Generate points ---
    points = []
    def add_pt(met_h, pos, spd):
        de = mag(pos)
        points.append({
            'met': round(met_h, 4), 'x': round(pos[0]*SCALE, 4),
            'y': round(pos[1]*SCALE, 4), 'z': round(pos[2]*SCALE, 4),
            'mx': round(moon_pos_scene[0], 4), 'my': round(moon_pos_scene[1], 4),
            'mz': round(moon_pos_scene[2], 4),
            'spd': round(spd, 3), 'dE': round(de, 0), 'dM': round(moon_dist_km, 0),
        })

    # Kepler equation solver: M → E → nu → r
    def kepler_solve(M, ecc, tol=1e-10):
        """Solve Kepler's equation M = E - e*sin(E) for E."""
        E = M
        for _ in range(50):
            dE = (M - E + ecc * math.sin(E)) / (1 - ecc * math.cos(E))
            E += dE
            if abs(dE) < tol: break
        nu = 2 * math.atan2(math.sqrt(1+ecc)*math.sin(E/2), math.sqrt(1-ecc)*math.cos(E/2))
        r = a_ins * (1 - ecc * math.cos(E))
        return nu, r

    # --- Phase 1: Backward RK4 from burn1 to MECO (with J2, consistent with Phase 2+3) ---
    r_b1, v_b1, v_b1_dir, r_b1_hat = get_burn1_state(az_opt, b1h_opt)
    # burn1_pre = state on insertion orbit at burn1 time
    st_ins = list(r_b1) + list(v_b1)
    ins_seg = (BURN1_H - ASCENT_H) * 3600
    ins_states = []
    el = 0.0
    while el < ins_seg:
        ins_states.append((BURN1_H - el / 3600, list(st_ins)))
        this_dt = min(dt, ins_seg - el)
        moon_pos = moon_pos_at_met(BURN1_H - (el + 0.5 * this_dt) / 3600)
        st_ins = rk4_step(st_ins, -this_dt, moon_pos=moon_pos)
        el += this_dt
    ins_states.append((ASCENT_H, list(st_ins)))
    ins_states.reverse()  # chronological

    # MECO state = Phase 1 start (physically correct, RK4+J2 backward from burn1)
    meco_state = ins_states[0][1] if ms_solution is None else ms_solution['meco_state'].tolist()
    meco_pos = meco_state[:3]
    meco_spd = mag(meco_state[3:])

    # --- Phase 0: Ascent (KSC ground → MECO) ---
    n_ascent = max(2, int(ASCENT_H * 3600 / 10))
    ksc_dir = norm(ksc_pos)
    meco_dir = norm(meco_pos)
    for i in range(n_ascent + 1):
        t = i / n_ascent
        met_h = t * ASCENT_H
        dir_blend = norm([ksc_dir[j] * (1 - t) + meco_dir[j] * t for j in range(3)])
        radius = mag(ksc_pos) * (1 - t) + mag(meco_pos) * t
        pos = [dir_blend[j] * radius for j in range(3)]
        spd = 0.4 + t * (meco_spd - 0.4)
        add_pt(met_h, pos, spd)

    # --- Phase 1 points ---
    for met_h, st in ins_states:
        if met_h <= ASCENT_H:
            continue
        add_pt(met_h, st[:3], mag(st[3:]))

    # --- Phase 2+3: Two burns + propagation (mirroring propagate() exactly) ---
    if ms_solution is None:
        v_after = add_vec(v_b1, add_vec(scale_vec(v_b1_dir, dv1o), scale_vec(r_b1_hat, dv1ro)))
        st = list(r_b1) + v_after
    else:
        st = ms_solution['y1'].tolist()

    burn2_start_h = BURN2_H - BURN2_DURATION_SEC / 7200.0
    burn2_end_h = BURN2_H + BURN2_DURATION_SEC / 7200.0

    # Segment 1: burn1 → burn2 start
    seg1 = (burn2_start_h - BURN1_H) * 3600
    elapsed = 0.0; step = 0
    while elapsed < seg1:
        mh = BURN1_H + elapsed / 3600
        if step % 6 == 0:
            add_pt(mh, st[:3], mag(st[3:]))
        this_dt = min(dt, seg1 - elapsed)
        moon_pos = moon_pos_at_met(mh + 0.5 * this_dt / 3600)
        st = rk4_step(st, this_dt, moon_pos=moon_pos)
        elapsed += this_dt; step += 1
    if ms_solution is not None:
        st = ms_solution['y2'].tolist()

    # Finite burn2
    burn_elapsed = 0.0
    while burn_elapsed < BURN2_DURATION_SEC:
        mh = burn2_start_h + burn_elapsed / 3600
        if step % 6 == 0:
            add_pt(mh, st[:3], mag(st[3:]))
        this_dt = min(dt, BURN2_DURATION_SEC - burn_elapsed)
        moon_pos = moon_pos_at_met(mh + 0.5 * this_dt / 3600)
        st = rk4_step(st, this_dt, moon_pos=moon_pos)
        vd = norm(st[3:]); rd = norm(st[:3])
        dv_scale = this_dt / BURN2_DURATION_SEC
        st[3] += vd[0] * dv2o * dv_scale + rd[0] * dv2ro * dv_scale
        st[4] += vd[1] * dv2o * dv_scale + rd[1] * dv2ro * dv_scale
        st[5] += vd[2] * dv2o * dv_scale + rd[2] * dv2ro * dv_scale
        burn_elapsed += this_dt; step += 1
    if ms_solution is not None:
        st = ms_solution['y3'].tolist()

    # Segment 2: burn2 end → Horizons
    seg2 = (first_met_h - burn2_end_h) * 3600
    elapsed = 0.0; step = 0
    while elapsed < seg2:
        mh = burn2_end_h + elapsed / 3600
        if step % 6 == 0:
            add_pt(mh, st[:3], mag(st[3:]))
        this_dt = min(dt, seg2 - elapsed)
        moon_pos = moon_pos_at_met(mh + 0.5 * this_dt / 3600)
        st = rk4_step(st, this_dt, moon_pos=moon_pos)
        elapsed += this_dt; step += 1

    # Final point: propagated endpoint (should match r_target exactly)
    add_pt(first_met_h, st[:3], mag(st[3:]))

    print(f"  {len(points)} pts, MET {points[0]['met']:.2f}-{points[-1]['met']:.2f}h, "
          f"min alt {min(p['dE'] for p in points)-6378:.0f}km", file=sys.stderr)
    return points

# Mission events (MET in hours)
EVENTS = [
    {'met': 0,       'name': '打ち上げ',               'name_en': 'Launch'},
    {'met': 0.036,   'name': 'SRB分離',               'name_en': 'SRB Separation'},           # T+2:08
    {'met': 0.135,   'name': 'MECO + コアステージ分離', 'name_en': 'MECO + Core Stage Sep'},   # T+8:06
    {'met': 0.3,     'name': 'ソーラーパネル展開',      'name_en': 'Solar Array Deploy'},       # T+~18min
    {'met': 0.817,   'name': '近地点上昇（ICPS RL10）', 'name_en': 'Perigee Raise (ICPS)'},    # T+49min
    {'met': 1.8,     'name': '遠地点上昇（ICPS RL10）', 'name_en': 'Apogee Raise (ICPS)'},     # T+1:48
    {'met': 3.4,     'name': 'ICPS分離',              'name_en': 'ICPS Separation'},           # T+3:24
    {'met': 3.4,     'name': '近接運用デモ開始',       'name_en': 'Prox Ops Demo Start'},      # T+3:24
    {'met': 4.6,     'name': '近接運用デモ終了',       'name_en': 'Prox Ops Demo End'},        # T+~4:35
    {'met': 12.4,    'name': '近地点上昇（ESM AJ10）', 'name_en': 'Perigee Raise (ESM)'},     # T+~12h (Apr 2 morning)
    {'met': 25.4,    'name': 'TLI（Orion ESM AJ10）',  'name_en': 'TLI (Orion ESM)'},         # T+1d01:27
    {'met': 103.0,   'name': '月重力圏突入',           'name_en': 'Lunar SOI Entry'},
    {'met': 120.6,   'name': '月最接近',               'name_en': 'Closest Lunar Approach'},
    {'met': 139.8,   'name': '月重力圏離脱',           'name_en': 'Lunar SOI Exit'},
    {'met': 215.0,   'name': '再突入',                 'name_en': 'Entry Interface'},
    {'met': 217.8,   'name': '着水',                   'name_en': 'Splashdown'},
]


def main():
    base = 'data/horizons'

    # 10-min data (full mission)
    sc_data = parse_horizons(f'{base}/spacecraft.txt')
    moon_data = parse_horizons(f'{base}/moon.txt')

    # 1-min data (near-Earth phase, first ~28h)
    sc_fine = parse_horizons(f'{base}/spacecraft_fine.txt')
    moon_fine = parse_horizons(f'{base}/moon_fine.txt')

    if not sc_data:
        print("ERROR: no spacecraft data parsed", file=sys.stderr)
        sys.exit(1)

    print(f"Parsed coarse: {len(sc_data)} sc, {len(moon_data)} moon", file=sys.stderr)
    print(f"Parsed fine:   {len(sc_fine)} sc, {len(moon_fine)} moon", file=sys.stderr)

    # Merge: 1-min for near-Earth, 10-min for the rest
    sc_merged, moon_merged = merge_fine_and_coarse(sc_fine, moon_fine, sc_data, moon_data)
    print(f"Merged: {len(sc_merged)} points", file=sys.stderr)

    # Launch time (configurable via CLI argument)
    launch_jd = parse_launch_time()

    # Scale factor: convert km to scene units
    # Earth-Moon distance (~384,400 km) = ~100 scene units
    SCALE = 100.0 / 384400.0  # scene units per km

    # Synthesize early trajectory (MET 0 to first Horizons point)
    early_points = synthesize_early_trajectory(sc_merged[0], moon_merged[0], launch_jd, SCALE)

    trajectory = list(early_points)
    for i, sc in enumerate(sc_merged):
        moon = moon_merged[min(i, len(moon_merged)-1)]

        met_h = compute_met_hours(sc['jd'], launch_jd)
        if met_h < 0:
            continue

        # Position in scene units
        sx = sc['x'] * SCALE
        sy = sc['y'] * SCALE
        sz = sc['z'] * SCALE

        # Moon position in scene units
        mx = moon['x'] * SCALE
        my = moon['y'] * SCALE
        mz = moon['z'] * SCALE

        # Speed in km/s
        speed = math.sqrt(sc['vx']**2 + sc['vy']**2 + sc['vz']**2)

        # Distance from Earth (km)
        d_earth = math.sqrt(sc['x']**2 + sc['y']**2 + sc['z']**2)

        # Distance from Moon (km)
        dx = sc['x'] - moon['x']
        dy = sc['y'] - moon['y']
        dz = sc['z'] - moon['z']
        d_moon = math.sqrt(dx**2 + dy**2 + dz**2)

        trajectory.append({
            'met': round(met_h, 3),
            'x': round(sx, 4),
            'y': round(sy, 4),
            'z': round(sz, 4),
            'vx': round(sc['vx'], 4),
            'vy': round(sc['vy'], 4),
            'vz': round(sc['vz'], 4),
            'mx': round(mx, 4),
            'my': round(my, 4),
            'mz': round(mz, 4),
            'spd': round(speed, 3),
            'dE': round(d_earth, 0),
            'dM': round(d_moon, 0),
        })

    # Find key events
    min_moon_dist = min(trajectory, key=lambda t: t['dM'])
    print(f"Closest approach to Moon: {min_moon_dist['dM']:.0f} km at MET {min_moon_dist['met']:.1f}h", file=sys.stderr)
    print(f"Total points: {len(trajectory)}", file=sys.stderr)
    print(f"MET range: {trajectory[0]['met']:.1f}h to {trajectory[-1]['met']:.1f}h", file=sys.stderr)

    # Output JSON
    output = {
        'launch_jd': launch_jd,
        'first_met_h': trajectory[0]['met'],
        'last_met_h': trajectory[-1]['met'],
        'closest_approach_km': round(min_moon_dist['dM']),
        'closest_approach_met_h': round(min_moon_dist['met'], 1),
        'scale_km_per_unit': round(1.0 / SCALE),
        'events': EVENTS,
        'points': trajectory,
    }

    with open('trajectory.json', 'w') as f:
        json.dump(output, f, separators=(',', ':'))

    # Also output a compact version for inline embedding
    print(f"Output: trajectory.json ({len(json.dumps(output, separators=(',',':')))} bytes)", file=sys.stderr)


if __name__ == '__main__':
    main()
