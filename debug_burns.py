"""Debug: compute and visualize the early trajectory burns."""
import json, math, sys

# Load trajectory data
with open('trajectory.json') as f:
    d = json.load(f)

pts = d['points']
first = None
for p in pts:
    if p.get('vx') is not None and p['met'] > 3:
        first = p
        break

if not first:
    print("ERROR: no Horizons data point with velocity found")
    sys.exit(1)

print(f"=== Horizons start point ===")
print(f"MET: {first['met']:.2f}h")
print(f"Pos: ({first['x']:.4f}, {first['y']:.4f}, {first['z']:.4f}) scene units")
print(f"Vel: ({first['vx']:.4f}, {first['vy']:.4f}, {first['vz']:.4f}) km/s")
print(f"dE: {first['dE']:.0f} km, spd: {first['spd']:.3f} km/s")
print()

# Check first 10 points for altitude
print("=== First 20 trajectory points ===")
for i, p in enumerate(pts[:20]):
    alt = p['dE'] - 6378
    print(f"[{i:3d}] MET {p['met']:7.4f}h  dE={p['dE']:8.0f}km  alt={alt:8.0f}km  spd={p['spd']:.1f}km/s  pos=({p['x']:.3f},{p['y']:.3f},{p['z']:.3f})")

print()

# Check around burn times
print("=== Around burn1 (T+0.82h) ===")
for p in pts:
    if 0.7 <= p['met'] <= 1.0:
        alt = p['dE'] - 6378
        print(f"MET {p['met']:7.4f}h  dE={p['dE']:8.0f}km  alt={alt:8.0f}km  spd={p['spd']:.1f}")

print()
print("=== Around burn2 (T+1.8h) ===")
for p in pts:
    if 1.6 <= p['met'] <= 2.0:
        alt = p['dE'] - 6378
        print(f"MET {p['met']:7.4f}h  dE={p['dE']:8.0f}km  alt={alt:8.0f}km  spd={p['spd']:.1f}")

print()

# Check for Earth penetration
print("=== Points inside Earth (alt < 0) ===")
for i, p in enumerate(pts):
    if p['dE'] < 6378:
        print(f"[{i:3d}] MET {p['met']:7.4f}h  dE={p['dE']:8.0f}km  alt={p['dE']-6378:.0f}km")

# Check position jumps
print()
print("=== Position jumps > 0.5 scene units ===")
for i in range(1, min(300, len(pts))):
    dx = ((pts[i]['x']-pts[i-1]['x'])**2 + (pts[i]['y']-pts[i-1]['y'])**2 + (pts[i]['z']-pts[i-1]['z'])**2)**0.5
    if dx > 0.5:
        print(f"[{i:3d}] MET {pts[i-1]['met']:.4f}->{pts[i]['met']:.4f}h  jump={dx:.2f} units ({dx*3844:.0f}km)")

print()

# Orbital parameters at key points
MU = 398600
R_E = 6378
print("=== Orbital analysis ===")

# Insertion orbit
print(f"Insertion orbit: peri={R_E+27}km ({27}km alt), apo={R_E+2222}km ({2222}km alt)")
a_ins = (R_E+27 + R_E+2222) / 2
e_ins = (R_E+2222 - R_E-27) / (R_E+2222 + R_E+27)
print(f"  a={a_ins:.0f}km, e={e_ins:.4f}")
print(f"  v_peri={math.sqrt(MU*(2/(R_E+27) - 1/a_ins)):.3f} km/s")
print(f"  v_apo={math.sqrt(MU*(2/(R_E+2222) - 1/a_ins)):.3f} km/s")
print(f"  period={2*math.pi*math.sqrt(a_ins**3/MU)/3600:.2f}h")
omega = math.sqrt(MU / a_ins**3)
print(f"  mean motion omega={omega:.6f} rad/s")
print(f"  at T+0.82h ({0.82*3600:.0f}s): angle={omega*0.82*3600:.2f} rad = {math.degrees(omega*0.82*3600):.1f} deg")

# How far is the spacecraft at T+0.82h?
angle_b1 = omega * 0.82 * 3600
# True anomaly mapping for ellipse
M = angle_b1  # mean anomaly
# Solve Kepler's equation: M = E - e*sin(E)
E = M
for _ in range(20):
    E = M + e_ins * math.sin(E)
nu = 2 * math.atan2(math.sqrt(1+e_ins) * math.sin(E/2), math.sqrt(1-e_ins) * math.cos(E/2))
r_b1 = a_ins * (1 - e_ins * math.cos(E))
print(f"  at burn1: E={math.degrees(E):.1f}deg, nu={math.degrees(nu):.1f}deg, r={r_b1:.0f}km, alt={r_b1-R_E:.0f}km")
print(f"  (should be near apogee={R_E+2222}={R_E+2222}km for burn1)")
