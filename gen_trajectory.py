"""Parse JPL Horizons vector output and generate trajectory JSON for the simulator.

Usage:
  python3 gen_trajectory.py                    # default: 2026-04-01 22:24 UTC
  python3 gen_trajectory.py 2026-04-02T18:24   # custom launch time (UTC)
"""
import json, re, math, sys
from datetime import datetime, timezone

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


# Default: 2026-04-01 22:24 UTC (Artemis II nominal launch)
DEFAULT_LAUNCH_UTC = "2026-04-01T22:24"

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


def rk4_step(state, dt, mu=398600.4418, moon_pos=None, mu_moon=4902.8):
    """RK4 integration step. state = [x,y,z,vx,vy,vz].
    If moon_pos is provided, includes lunar gravity (3-body)."""
    def deriv(s):
        r = math.sqrt(s[0]**2 + s[1]**2 + s[2]**2)
        r3 = r**3
        ax, ay, az = -mu*s[0]/r3, -mu*s[1]/r3, -mu*s[2]/r3
        if moon_pos:
            dx = s[0] - moon_pos[0]
            dy = s[1] - moon_pos[1]
            dz = s[2] - moon_pos[2]
            rm = math.sqrt(dx**2 + dy**2 + dz**2)
            rm3 = rm**3
            # Moon direct + indirect term
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
    """Synthesize MET 0 to first Horizons point.

    Strategy:
    1. KSC position at T+0
    2. Short ascent to LEO altitude (T+0 to T+0.13h)
    3. Lambert problem: find the Keplerian arc from LEO position (T+0.13h)
       to the first Horizons point (T+3.6h)
    4. Forward propagate along the Lambert orbit
    This represents the combined effect of all burns as a single transfer orbit.
    """
    first_met_h = compute_met_hours(first_sc['jd'], launch_jd)
    print(f"  Synthesizing MET 0 to {first_met_h:.1f}h", file=sys.stderr)

    r_target = [first_sc['x'], first_sc['y'], first_sc['z']]
    v_target = [first_sc['vx'], first_sc['vy'], first_sc['vz']]

    moon_pos = [first_moon['x'] * SCALE, first_moon['y'] * SCALE, first_moon['z'] * SCALE]
    moon_dist_km = mag([first_moon['x'], first_moon['y'], first_moon['z']])

    # --- KSC position ---
    ksc_pos = ksc_icrf(launch_jd)
    print(f"  KSC ICRF: ({ksc_pos[0]:.0f}, {ksc_pos[1]:.0f}, {ksc_pos[2]:.0f}) km", file=sys.stderr)

    # Orbital plane from Horizons data
    h_vec = norm(cross(r_target, v_target))
    # Project KSC onto orbital plane and raise to LEO altitude
    ksc_dot_h = dot(ksc_pos, h_vec)
    ksc_in_plane = [ksc_pos[i] - ksc_dot_h * h_vec[i] for i in range(3)]
    ksc_dir = norm(ksc_in_plane)
    v_dir = norm(cross(h_vec, ksc_dir))

    LEO_R = 6578.0  # km
    LEO_OMEGA = math.sqrt(398600 / LEO_R ** 3)

    # LEO position at T+0.13h (after ascent): ~1 minute into orbit
    ascent_end_h = 0.13
    ascent_end_sec = ascent_end_h * 3600
    angle_at_ascent = LEO_OMEGA * ascent_end_sec
    r_leo = add_vec(
        scale_vec(ksc_dir, LEO_R * math.cos(angle_at_ascent)),
        scale_vec(v_dir, LEO_R * math.sin(angle_at_ascent))
    )

    # --- Two-burn transfer: LEO → intermediate → high-elliptical ---
    BURN1_H = 0.82   # perigee raise (~T+49min): prograde, raises apogee
    BURN2_H = 1.5    # apogee raise (~T+1.5h): prograde, raises perigee

    # LEO state at burn1
    burn1_sec = BURN1_H * 3600
    angle_at_burn1 = LEO_OMEGA * burn1_sec
    r_burn1 = add_vec(
        scale_vec(ksc_dir, LEO_R * math.cos(angle_at_burn1)),
        scale_vec(v_dir, LEO_R * math.sin(angle_at_burn1))
    )
    v_leo_dir = norm([
        -math.sin(angle_at_burn1) * ksc_dir[i] + math.cos(angle_at_burn1) * v_dir[i]
        for i in range(3)
    ])
    v_leo = scale_vec(v_leo_dir, LEO_R * LEO_OMEGA)

    # Use shooting method: find burn1 Δv (prograde) that, after propagation
    # through burn2 point and to T+3.6h, best matches the Horizons position.
    # Burn2 is also prograde, we optimize both magnitudes.
    moon_km = [first_moon['x'], first_moon['y'], first_moon['z']]
    dt_rk4 = 10.0  # 10-second integration steps

    # Radial direction at burn1 (perpendicular to velocity, in orbital plane)
    r_burn1_dir = norm(r_burn1)

    def propagate_two_burns(dv1, dv2, dv1_radial=0, dv2_radial=0):
        """Apply burns and propagate. Returns final (position, velocity)."""
        # Burn 1: prograde + radial components
        v_after1 = add_vec(v_leo,
            add_vec(scale_vec(v_leo_dir, dv1), scale_vec(r_burn1_dir, dv1_radial)))
        state = list(r_burn1) + v_after1

        steps_to_burn2 = int((BURN2_H - BURN1_H) * 3600 / dt_rk4)
        for _ in range(steps_to_burn2):
            state = rk4_step(state, dt_rk4, moon_pos=moon_km)

        # Burn 2: prograde + radial
        v_dir2 = norm(state[3:])
        r_dir2 = norm(state[:3])
        state[3] += v_dir2[0] * dv2 + r_dir2[0] * dv2_radial
        state[4] += v_dir2[1] * dv2 + r_dir2[1] * dv2_radial
        state[5] += v_dir2[2] * dv2 + r_dir2[2] * dv2_radial

        steps_to_end = int((first_met_h - BURN2_H) * 3600 / dt_rk4)
        for _ in range(steps_to_end):
            state = rk4_step(state, dt_rk4, moon_pos=moon_km)

        return state[:3], state[3:]  # position, velocity

    def cost(dv1, dv2, dv1r=0, dv2r=0):
        """Cost: position error + velocity direction error."""
        pos, vel = propagate_two_burns(dv1, dv2, dv1r, dv2r)
        pos_err = mag([pos[i] - r_target[i] for i in range(3)])
        # Velocity direction error (angle between vectors, scaled to km)
        vel_dir = norm(vel)
        tgt_dir = norm(v_target)
        cos_angle = max(-1, min(1, dot(vel_dir, tgt_dir)))
        angle_err = math.acos(cos_angle)  # radians
        # Weight: 1 degree of direction error ≈ 100 km position equivalent
        return pos_err + angle_err * 180 / math.pi * 100

    # Multi-level optimization: 4 params (dv1, dv2 prograde + radial)
    best_cost = float('inf')
    best_params = [1.0, 2.0, 0.0, 0.0]  # dv1, dv2, dv1_radial, dv2_radial

    # Level 1: coarse prograde-only search
    for dv1 in [x * 0.5 for x in range(0, 14)]:
        for dv2 in [x * 0.5 for x in range(0, 14)]:
            try:
                c = cost(dv1, dv2)
                if c < best_cost:
                    best_cost = c
                    best_params = [dv1, dv2, 0, 0]
            except:
                continue

    # Level 2-5: refine all 4 params with decreasing step
    for step in [0.1, 0.02, 0.005, 0.001]:
        prev = list(best_params)
        rng = range(-5, 6)
        for d1 in [prev[0] + x * step for x in rng]:
            for d2 in [prev[1] + x * step for x in rng]:
                for d1r in [prev[2] + x * step * 0.5 for x in rng]:
                    for d2r in [prev[3] + x * step * 0.5 for x in rng]:
                        try:
                            c = cost(d1, d2, d1r, d2r)
                            if c < best_cost:
                                best_cost = c
                                best_params = [d1, d2, d1r, d2r]
                        except:
                            continue

    best_dv1, best_dv2, best_dv1r, best_dv2r = best_params
    final_pos, final_vel = propagate_two_burns(best_dv1, best_dv2, best_dv1r, best_dv2r)
    best_err = mag([final_pos[i] - r_target[i] for i in range(3)])
    vel_angle = math.degrees(math.acos(max(-1, min(1, dot(norm(final_vel), norm(v_target))))))


    print(f"  Two-burn solution: Δv1={best_dv1:.3f}+{best_dv1r:.3f}r km/s, "
          f"Δv2={best_dv2:.3f}+{best_dv2r:.3f}r km/s, "
          f"pos error={best_err:.0f} km, vel angle={vel_angle:.1f}°", file=sys.stderr)

    # --- Generate trajectory with the best burns ---
    v_after1 = add_vec(v_leo,
        add_vec(scale_vec(v_leo_dir, best_dv1), scale_vec(r_burn1_dir, best_dv1r)))
    r_burn1_final = list(r_burn1)

    # --- Generate trajectory points ---
    points = []

    # Phase 1: Launch ascent (T+0 to T+0.13h)
    for i in range(20):
        t = i / 19
        met_h = t * ascent_end_h
        t_sec = met_h * 3600
        r_asc = 6378 + t * 200
        angle = LEO_OMEGA * t_sec * t
        pos = add_vec(
            scale_vec(ksc_dir, r_asc * math.cos(angle)),
            scale_vec(v_dir, r_asc * math.sin(angle))
        )
        spd = 0.4 + t * 7.4
        d_earth = mag(pos)
        points.append({
            'met': round(met_h, 4),
            'x': round(pos[0] * SCALE, 4), 'y': round(pos[1] * SCALE, 4), 'z': round(pos[2] * SCALE, 4),
            'mx': round(moon_pos[0], 4), 'my': round(moon_pos[1], 4), 'mz': round(moon_pos[2], 4),
            'spd': round(spd, 3), 'dE': round(d_earth, 0), 'dM': round(moon_dist_km, 0),
        })

    # Phase 2: LEO circular orbit (T+0.13h to T+0.82h, ~half orbit)
    n_leo_steps = int((BURN1_H - ascent_end_h) * 3600 / 30)  # every 30 sec
    for i in range(1, n_leo_steps + 1):
        t_sec = (ascent_end_h * 3600) + i * 30
        met_h = t_sec / 3600
        angle = LEO_OMEGA * t_sec
        pos = add_vec(
            scale_vec(ksc_dir, LEO_R * math.cos(angle)),
            scale_vec(v_dir, LEO_R * math.sin(angle))
        )
        d_earth = mag(pos)
        points.append({
            'met': round(met_h, 4),
            'x': round(pos[0] * SCALE, 4), 'y': round(pos[1] * SCALE, 4), 'z': round(pos[2] * SCALE, 4),
            'mx': round(moon_pos[0], 4), 'my': round(moon_pos[1], 4), 'mz': round(moon_pos[2], 4),
            'spd': round(7.8, 3), 'dE': round(d_earth, 0), 'dM': round(moon_dist_km, 0),
        })

    # Phase 3: Two-burn transfer (T+0.82h to T+3.6h)
    state = list(r_burn1_final) + list(v_after1)
    dt_prop = 10.0
    sample_every = 6  # every 60 seconds
    burn2_applied = False
    total_steps = int((first_met_h - BURN1_H) * 3600 / dt_prop)

    for step in range(total_steps + 1):
        elapsed_sec = step * dt_prop
        met_h = BURN1_H + elapsed_sec / 3600

        # Apply burn2 at the right time
        if not burn2_applied and met_h >= BURN2_H:
            v_dir2 = norm(state[3:])
            r_dir2 = norm(state[:3])
            state[3] += v_dir2[0] * best_dv2 + r_dir2[0] * best_dv2r
            state[4] += v_dir2[1] * best_dv2 + r_dir2[1] * best_dv2r
            state[5] += v_dir2[2] * best_dv2 + r_dir2[2] * best_dv2r
            burn2_applied = True

        if step % sample_every == 0:
            pos = state[:3]
            spd = mag(state[3:])
            d_earth = mag(pos)
            points.append({
                'met': round(met_h, 4),
                'x': round(pos[0] * SCALE, 4), 'y': round(pos[1] * SCALE, 4), 'z': round(pos[2] * SCALE, 4),
                'mx': round(moon_pos[0], 4), 'my': round(moon_pos[1], 4), 'mz': round(moon_pos[2], 4),
                'spd': round(spd, 3), 'dE': round(d_earth, 0), 'dM': round(moon_dist_km, 0),
            })

        state = rk4_step(state, dt_prop, moon_pos=moon_km)

    # Blend last portion only if error is still significant
    if best_err > 100:  # > 100 km
        BLEND_H = 0.3
        blend_start_met = first_met_h - BLEND_H
        for p in points:
            if p['met'] > blend_start_met:
                t = (p['met'] - blend_start_met) / BLEND_H
                t = t * t * (3 - 2 * t)
                p['x'] = round(p['x'] * (1 - t) + r_target[0] * SCALE * t, 4)
                p['y'] = round(p['y'] * (1 - t) + r_target[1] * SCALE * t, 4)
                p['z'] = round(p['z'] * (1 - t) + r_target[2] * SCALE * t, 4)
                target_spd = mag(v_target)
                p['spd'] = round(p['spd'] * (1 - t) + target_spd * t, 3)
                pos_km = [p['x'] / SCALE, p['y'] / SCALE, p['z'] / SCALE]
                p['dE'] = round(mag(pos_km), 0)
        print(f"  Blending applied (error {best_err:.0f} km > 100 km threshold)", file=sys.stderr)

    print(f"  Synthesized {len(points)} points, "
          f"MET {points[0]['met']:.2f}h to {points[-1]['met']:.2f}h, "
          f"min altitude {min(p['dE'] for p in points) - 6378:.0f} km",
          file=sys.stderr)

    return points


# Mission events (MET in hours)
EVENTS = [
    {'met': 0,       'name': '打ち上げ',               'name_en': 'Launch'},
    {'met': 0.033,   'name': 'SRB分離',               'name_en': 'SRB Separation'},
    {'met': 0.133,   'name': 'コアステージ分離 (MECO)', 'name_en': 'Core Stage Sep (MECO)'},
    {'met': 0.817,   'name': '近地点上昇噴射',         'name_en': 'Perigee Raise Burn'},
    {'met': 1.5,     'name': '遠地点上昇噴射',         'name_en': 'Apogee Raise Burn'},
    {'met': 3.404,   'name': 'ICPS分離',              'name_en': 'ICPS Separation'},
    {'met': 25.617,  'name': 'TLI（月遷移噴射）',      'name_en': 'Trans-Lunar Injection'},
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
