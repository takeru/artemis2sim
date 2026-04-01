#!/bin/bash
# Fetch Artemis II and Moon ephemeris from JPL Horizons API
# Earth-centered ICRF vectors
#
# Usage:
#   ./fetch.sh                          # default window (Apr 2-10)
#   ./fetch.sh 2026-04-03 2026-04-12    # custom start/end dates
#
# Spacecraft ID: -1024 (Artemis II / Integrity / Orion EM-2)
# Moon ID: 301

set -euo pipefail
cd "$(dirname "$0")"

START="${1:-2026-04-02}"
END="${2:-2026-04-10}"

# Add time component
START_TIME="${START}+02:00"
END_TIME="${END}+23:00"
# Fine data: first 28h from start
FINE_END=$(date -j -f "%Y-%m-%d" "$START" "+%Y-%m-%d" 2>/dev/null || echo "$START")
FINE_END_TIME="${FINE_END}+06:00"

echo "Data window: $START to $END"
echo ""

echo "Fetching Artemis II spacecraft vectors (10-min)..."
curl -s "https://ssd.jpl.nasa.gov/api/horizons.api?\
format=text&\
COMMAND='-1024'&\
OBJ_DATA='NO'&\
MAKE_EPHEM='YES'&\
EPHEM_TYPE='VECTORS'&\
CENTER='500@399'&\
START_TIME='${START_TIME}'&\
STOP_TIME='${END_TIME}'&\
STEP_SIZE='10+min'&\
VEC_CORR='NONE'&\
REF_PLANE='FRAME'&\
REF_SYSTEM='ICRF'" > spacecraft.txt
echo "  -> spacecraft.txt ($(wc -l < spacecraft.txt) lines)"

echo "Fetching Moon vectors (10-min)..."
curl -s "https://ssd.jpl.nasa.gov/api/horizons.api?\
format=text&\
COMMAND='301'&\
OBJ_DATA='NO'&\
MAKE_EPHEM='YES'&\
EPHEM_TYPE='VECTORS'&\
CENTER='500@399'&\
START_TIME='${START_TIME}'&\
STOP_TIME='${END_TIME}'&\
STEP_SIZE='10+min'&\
VEC_CORR='NONE'&\
REF_PLANE='FRAME'&\
REF_SYSTEM='ICRF'" > moon.txt
echo "  -> moon.txt ($(wc -l < moon.txt) lines)"

echo "Fetching fine spacecraft vectors (1-min, first 28h)..."
curl -s "https://ssd.jpl.nasa.gov/api/horizons.api?\
format=text&\
COMMAND='-1024'&\
OBJ_DATA='NO'&\
MAKE_EPHEM='YES'&\
EPHEM_TYPE='VECTORS'&\
CENTER='500@399'&\
START_TIME='${START_TIME}'&\
STOP_TIME='${FINE_END_TIME}'&\
STEP_SIZE='1+min'&\
VEC_CORR='NONE'&\
REF_PLANE='FRAME'&\
REF_SYSTEM='ICRF'" > spacecraft_fine.txt
echo "  -> spacecraft_fine.txt ($(wc -l < spacecraft_fine.txt) lines)"

echo "Fetching fine Moon vectors (1-min, first 28h)..."
curl -s "https://ssd.jpl.nasa.gov/api/horizons.api?\
format=text&\
COMMAND='301'&\
OBJ_DATA='NO'&\
MAKE_EPHEM='YES'&\
EPHEM_TYPE='VECTORS'&\
CENTER='500@399'&\
START_TIME='${START_TIME}'&\
STOP_TIME='${FINE_END_TIME}'&\
STEP_SIZE='1+min'&\
VEC_CORR='NONE'&\
REF_PLANE='FRAME'&\
REF_SYSTEM='ICRF'" > moon_fine.txt
echo "  -> moon_fine.txt ($(wc -l < moon_fine.txt) lines)"

echo ""
echo "Done. Run 'python3 gen_trajectory.py' to generate trajectory.json"
