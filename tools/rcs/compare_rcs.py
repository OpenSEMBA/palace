#!/usr/bin/env python3
"""Compare Palace bistatic RCS output to the PEC sphere Mie series."""
import csv
import sys
import numpy as np

sys.path.insert(0, ".")
from mie_pec import mie_pec

rcs_file = sys.argv[1] if len(sys.argv) > 1 else "postpro/sphere/farfield-rcs.csv"
radius = float(sys.argv[2]) if len(sys.argv) > 2 else 0.159154943
freq = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3e9

rows = []
with open(rcs_file) as f:
    r = csv.reader(f)
    header = next(r)
    for line in r:
        vals = [x.strip() for x in line]
        rows.append((float(vals[1]), float(vals[2]), float(vals[3]), float(vals[4])))

print(f"{'theta':>6} {'phi':>5} {'palace':>10} {'mie':>10} {'err dB':>8}")
errs = []
for exc, theta_deg, phi_deg, rcs in rows:
    th = np.radians(theta_deg)
    ph = np.radians(phi_deg)
    et, ep = mie_pec(radius, freq, [th], [ph])
    sig = abs(et[0]) ** 2 + abs(ep[0]) ** 2
    err = 10 * np.log10(max(rcs, 1e-30) / max(sig, 1e-30))
    errs.append(err)
    if abs(theta_deg % 30) < 1e-9:
        print(f"{theta_deg:6.0f} {phi_deg:5.0f} {rcs:10.5f} {sig:10.5f} {err:8.2f}")

errs = np.array(errs)
print(f"\nn = {len(errs)}  mean err = {errs.mean():.2f} dB  "
      f"rms err = {np.sqrt((errs**2).mean()):.2f} dB  "
      f"max |err| = {np.abs(errs).max():.2f} dB")
