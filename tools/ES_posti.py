# example usage:
# python3 ./tools/ES_posti.py ./output/initial_field_spectrum.dat

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# user params (can be overridden from command line)
import argparse

parser = argparse.ArgumentParser(description='Plot energy spectrum from a file')
parser.add_argument('fname', nargs='?', default='Energy-spectrum.dat',
                    help='Energy spectrum filename (default: Energy-spectrum.dat)')
parser.add_argument('--kmin', type=int, default=2, help='minimum k to consider for fit')
parser.add_argument('--window', type=int, default=16, help='sliding window size in k')
args = parser.parse_args()

fname = args.fname
kmin_fit = args.kmin
window = args.window
target_slope = -5.0/3.0

# load spectrum (skip comments)
data = np.loadtxt(fname, comments='#')
k = data[:,0].astype(int)
Ek = data[:,1]

# remove zeros and small entries
mask = (k >= kmin_fit) & (Ek > 0)
k = k[mask]
Ek = Ek[mask]

logk = np.log10(k)
logE = np.log10(Ek)

best = None
best_idx = -1

for i in range(0, len(k) - window + 1):
    xi = logk[i:i+window]
    yi = logE[i:i+window]
    slope, intercept, r_value, p_value, std_err = stats.linregress(xi, yi)
    # slope is d(logE)/d(logk)
    # compare to target
    err = abs(slope - target_slope)
    if best is None or err < best[0]:
        best = (err, slope, intercept, r_value, i, i+window-1)

err, slope, intercept, r_value, i0, i1 = best
k0 = int(k[i0])
k1 = int(k[i1])
print("Best fit window k=[%d..%d], slope = %.4f, target = %.4f, r^2=%.4e" % (k0,k1,slope,target_slope,r_value**2))

# plot
plt.figure(figsize=(8,6))
plt.loglog(k, Ek, label='E(k)')
# plot fitted line over best window
kx = np.linspace(k0, k1, 100)
# transform back: logE = slope*logk + intercept
logE_fit = slope * np.log10(kx) + intercept
plt.loglog(kx, 10**logE_fit, '--', label='fit slope %.3f'%slope)
# overlay -5/3 line scaled to passing roughly through middle of fit window
midk = np.sqrt(k0*k1)
midE = 10 ** (slope * np.log10(midk) + intercept)
C = midE * (midk ** (5.0/3.0))  # scale such that E(k) ~ C * k^{-5/3}
plt.loglog(kx, C * kx**(-5.0/3.0), ':', label='-5/3 scaled')

plt.xlabel('k')
plt.ylabel('E(k)')
plt.title('Energy spectrum and detected −5/3 window: k=[%d..%d], slope=%.3f' % (k0,k1,slope))
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.4)
plt.savefig('spectrum_kolmo.png', dpi=200)
plt.show()
