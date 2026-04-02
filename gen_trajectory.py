"""Parse JPL Horizons vector output and generate trajectory JSON for the simulator.

Usage:
  python3 gen_trajectory.py                    # default: 2026-04-01 22:24 UTC
  python3 gen_trajectory.py 2026-04-02T18:24   # custom launch time (UTC)
"""
import json, re, math, sys
from datetime import datetime, timezone
from scipy.optimize import minimize

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


J2 = 1.08263e-3   # Earth J2
R_E_J2 = 6378.137  # Earth equatorial radius for J2 (km)

def rk4_step(state, dt, mu=398600.4418, moon_pos=None, mu_moon=4902.8):
    """RK4 integration step. state = [x,y,z,vx,vy,vz].
    Includes Earth J2 and optional lunar gravity."""
    def deriv(s):
        r = math.sqrt(s[0]**2 + s[1]**2 + s[2]**2)
        r2 = r * r
        r3 = r2 * r
        ax, ay, az = -mu*s[0]/r3, -mu*s[1]/r3, -mu*s[2]/r3

        # J2 perturbation disabled: introduces 41km error vs Horizons data.
        # Likely due to ICRF Z ≠ Earth spin axis (0.004° offset) or
        # interaction with our simplified orbital model. Needs investigation.
        # z2_r2 = s[2]**2 / r2
        # fJ2 = 1.5 * J2 * mu * R_E_J2**2 / (r2 * r3)
        # ax += fJ2 * s[0] * (5 * z2_r2 - 1)
        # ay += fJ2 * s[1] * (5 * z2_r2 - 1)
        # az += fJ2 * s[2] * (5 * z2_r2 - 3)

        if moon_pos:
            dx = s[0] - moon_pos[0]
            dy = s[1] - moon_pos[1]
            dz = s[2] - moon_pos[2]
            rm = math.sqrt(dx**2 + dy**2 + dz**2)
            rm3 = rm**3
            rm0 = math.sqrt(moon_pos[0]**2 + moon_pos[1]**2 + moon_pos[2]**2)
            rm03 = rm0**3
            ax += -mu_moon * dx / rm3 - mu_moon * moon_pos[0] / rm03
            ay += -mu_moon * dy / rm3 - mu_moon * moon_pos[1] / rm03
            az += -mu_moon * dz / rm3 - mu_moon * moon_pos[2] / rm03
        return [s[3], s[4], s[5], ax, ay, az]

    k1 = deriv(state)
    s2 = [state[i] + dt/2 * k1[i] for i in range(6)]
    k2 = deriv(s2)
    s3 = [state[i] + dt/2 * k2[i] for i in range(6)]
    k3 = deriv(s3)
    s4 = [state[i] + dt * k3[i] for i in range(6)]
    k4 = deriv(s4)
    return [state[i] + dt/6 * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) for i in range(6)]


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

    MU = 398600.4418
    R_E = 6378

    # --- Orbital plane + basis vectors from Horizons ---
    ksc_pos = ksc_icrf(launch_jd)
    print(f"  KSC ICRF: ({ksc_pos[0]:.0f}, {ksc_pos[1]:.0f}, {ksc_pos[2]:.0f}) km", file=sys.stderr)

    h_vec = norm(cross(r_target, v_target))
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
    dt = 10.0

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

    def get_burn1_state(azimuth, ins_peri=INS_PERI, ins_apo=INS_APO, b1h=BURN1_H, m_off=0):
        """Compute burn1 position/velocity for given launch azimuth and insertion orbit."""
        e1r, e2r = rotate_basis(azimuth)
        a = (ins_peri + ins_apo) / 2
        e = (ins_apo - ins_peri) / (ins_apo + ins_peri)
        omega = math.sqrt(MU / a**3)
        M = omega * b1h * 3600 + m_off
        nu, r_mag = kepler_solve_burn(M, e, a)
        r = add_vec(scale_vec(e1r, r_mag * math.cos(nu)),
                    scale_vec(e2r, r_mag * math.sin(nu)))
        v_dir = norm([-math.sin(nu) * e1r[i] + math.cos(nu) * e2r[i] for i in range(3)])
        v_mag = v_at(r_mag, a)
        return r, scale_vec(v_dir, v_mag), v_dir, norm(r)

    def propagate(dv1, dv2, dv1r, dv2r, azimuth, ins_peri=INS_PERI, ins_apo=INS_APO, b1h=BURN1_H, b2h=BURN2_H, m_off=0):
        r_b1, v_b1, v_b1_dir, r_b1_hat = get_burn1_state(azimuth, ins_peri, ins_apo, b1h, m_off)
        v_after = add_vec(v_b1, add_vec(scale_vec(v_b1_dir, dv1), scale_vec(r_b1_hat, dv1r)))
        st = list(r_b1) + v_after
        elapsed = 0.0; seg1 = (b2h - b1h) * 3600
        while elapsed < seg1:
            st = rk4_step(st, min(dt, seg1 - elapsed), moon_pos=moon_km)
            elapsed += min(dt, seg1 - elapsed)
        vd = norm(st[3:]); rd = norm(st[:3])
        st[3] += vd[0]*dv2 + rd[0]*dv2r
        st[4] += vd[1]*dv2 + rd[1]*dv2r
        st[5] += vd[2]*dv2 + rd[2]*dv2r
        elapsed = 0.0; seg2 = (first_met_h - b2h) * 3600
        while elapsed < seg2:
            st = rk4_step(st, min(dt, seg2 - elapsed), moon_pos=moon_km)
            elapsed += min(dt, seg2 - elapsed)
        return st[:3], st[3:]

    def cost_vec(x):
        # x = [dv1, dv2, dv1r, dv2r, azimuth, ins_peri, ins_apo]
        try:
            pos, vel = propagate(x[0], x[1], x[2], x[3], x[4], x[5], x[6])
            pe = mag([pos[i] - r_target[i] for i in range(3)])
            ve = mag([vel[i] - v_target[i] for i in range(3)]) * 1000
            return pe + ve
        except:
            return 1e12

    # 9 params: Δv (4) + azimuth + insertion orbit (2) + burn effective times (2)
    # Burn times are optimizable: impulsive approximation of finite-duration burns
    # means the "effective time" differs from the start/end time
    x0 = [dv1_nom, dv2_nom, 0, 0, 0, INS_PERI, INS_APO, BURN1_H, BURN2_H, 0]
    bounds = [
        (-0.5, 1.0),            # dv1
        (1.5, 3.5),             # dv2
        (-0.3, 0.3),            # dv1r
        (-0.3, 0.3),            # dv2r
        (-0.2, 0.2),            # azimuth
        (R_E - 200, R_E + 200), # ins_peri
        (R_E + 500, R_E + 5000),# ins_apo
        (0.6, 1.1),             # burn1 effective time
        (1.5, 2.2),             # burn2 effective time
        (-3.14, 3.14),          # M_offset (mean anomaly offset, radians)
    ]

    def cost_bounded(x):
        for i, (lo, hi) in enumerate(bounds):
            if x[i] < lo or x[i] > hi:
                return 1e12
        try:
            pos, vel = propagate(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9])
            pe = mag([pos[i] - r_target[i] for i in range(3)])
            ve = mag([vel[i] - v_target[i] for i in range(3)]) * 1000
            return pe + ve
        except:
            return 1e12

    result = minimize(cost_bounded, x0, method='Nelder-Mead',
                      options={'maxiter': 20000, 'xatol': 1e-6, 'fatol': 0.01, 'adaptive': True})

    dv1o, dv2o, dv1ro, dv2ro, az_opt, ins_peri_opt, ins_apo_opt, b1h_opt, b2h_opt, m_offset_opt = result.x
    fp, fv = propagate(dv1o, dv2o, dv1ro, dv2ro, az_opt, ins_peri_opt, ins_apo_opt, b1h_opt, b2h_opt, m_offset_opt)
    pe = mag([fp[i]-r_target[i] for i in range(3)])
    va = math.degrees(math.acos(max(-1, min(1, dot(norm(fv), norm(v_target))))))
    print(f"  Optimized ({result.nfev} evals): dv1={dv1o*1000:.0f}+{dv1ro*1000:.0f}r, "
          f"dv2={dv2o*1000:.0f}+{dv2ro*1000:.0f}r, az={math.degrees(az_opt):.1f}deg, "
          f"peri={ins_peri_opt-R_E:.0f}km, apo={ins_apo_opt-R_E:.0f}km, "
          f"b1={b1h_opt:.3f}h, b2={b2h_opt:.3f}h, M0={math.degrees(m_offset_opt):.1f}deg, "
          f"pos={pe:.0f}km, vel={va:.1f}deg", file=sys.stderr)

    # Use optimized values for trajectory generation
    e1, e2 = rotate_basis(az_opt)
    a_ins = (ins_peri_opt + ins_apo_opt) / 2
    e_ins = (ins_apo_opt - ins_peri_opt) / (ins_apo_opt + ins_peri_opt)
    p_ins = a_ins * (1 - e_ins**2)
    omega_ins = math.sqrt(MU / a_ins**3)
    INS_PERI = ins_peri_opt
    INS_APO = ins_apo_opt
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

    # Phase 0+1 combined: ascent then insertion orbit
    # Compute Phase 1's state at ASCENT_H for smooth connection
    M_ascent = omega_ins * ASCENT_H * 3600
    nu_ascent, _ = kepler_solve(M_ascent, e_ins)

    # Phase 0: Ascent (T+0 to T+0.13h)
    # At MECO (T+8min), spacecraft is already on the 27x2222km insertion orbit.
    # Ascent smoothly transitions from ground to Kepler orbit position.
    M_meco = omega_ins * ASCENT_H * 3600 + m_offset_opt
    nu_meco, r_meco = kepler_solve(M_meco, e_ins)
    r_meco = max(r_meco, R_E + 10)

    n_ascent = max(2, int(ASCENT_H * 3600 / 10))
    for i in range(n_ascent + 1):
        t = i / n_ascent
        met_h = t * ASCENT_H
        # Radius: ground → Kepler radius at MECO
        r = R_E + t * (r_meco - R_E)
        # Angle: 0 → nu_meco
        ang = t * nu_meco
        pos = add_vec(scale_vec(e1, r*math.cos(ang)), scale_vec(e2, r*math.sin(ang)))
        spd = 0.4 + t * (v_at(r_meco, a_ins) - 0.4)
        add_pt(met_h, pos, spd)

    # Phase 1: Insertion orbit (T+0.13h to burn1), every 10 seconds
    # Already on Kepler orbit, no blending needed
    for i in range(1, int((BURN1_H - ASCENT_H) * 3600 / 10) + 1):
        ts = ASCENT_H * 3600 + i * 10
        met_h = ts / 3600
        if met_h > BURN1_H: break
        M = omega_ins * ts + m_offset_opt
        nu, r = kepler_solve(M, e_ins)
        r = max(r, R_E + 10)
        # Position in orbital plane using true anomaly
        # nu is measured from perigee; in our basis, perigee is at e1 direction
        pos = add_vec(scale_vec(e1, r*math.cos(nu)), scale_vec(e2, r*math.sin(nu)))
        add_pt(met_h, pos, v_at(r, a_ins))

    # Phase 2+3: Two burns + propagation (mirroring propagate() exactly)
    r_b1, v_b1, v_b1_dir, r_b1_hat = get_burn1_state(az_opt, ins_peri_opt, ins_apo_opt, b1h_opt, m_offset_opt)
    v_after = add_vec(v_b1, add_vec(scale_vec(v_b1_dir, dv1o), scale_vec(r_b1_hat, dv1ro)))
    st = list(r_b1) + v_after

    # Segment 1: burn1 → burn2
    seg1 = (BURN2_H - BURN1_H) * 3600
    elapsed = 0.0; step = 0
    while elapsed < seg1:
        mh = BURN1_H + elapsed / 3600
        if step % 6 == 0:
            add_pt(mh, st[:3], mag(st[3:]))
        this_dt = min(dt, seg1 - elapsed)
        st = rk4_step(st, this_dt, moon_pos=moon_km)
        elapsed += this_dt; step += 1

    # Apply burn2
    vd = norm(st[3:]); rd = norm(st[:3])
    st[3] += vd[0]*dv2o + rd[0]*dv2ro
    st[4] += vd[1]*dv2o + rd[1]*dv2ro
    st[5] += vd[2]*dv2o + rd[2]*dv2ro

    # Segment 2: burn2 → Horizons
    seg2 = (first_met_h - BURN2_H) * 3600
    elapsed = 0.0; step = 0
    while elapsed < seg2:
        mh = BURN2_H + elapsed / 3600
        if step % 6 == 0:
            add_pt(mh, st[:3], mag(st[3:]))
        this_dt = min(dt, seg2 - elapsed)
        st = rk4_step(st, this_dt, moon_pos=moon_km)
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
