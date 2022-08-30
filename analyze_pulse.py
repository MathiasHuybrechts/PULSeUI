import numpy as np
import matplotlib.pyplot as plt
import re
import pandas
import glob
import math
from collections import namedtuple
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-l", "--left", help="left index, start polyfit here")
parser.add_argument("-r", "--right", help="right index, stop polyfit here")
args = parser.parse_args()
left_index = int(args.left)
right_index = int(args.right)


# Do you want to Print Plots of the slopes?
do_plots = True

# Get all files in the specified folder
files = glob.glob('Measurements/Test/measurement_*.npy')
filepath = "Measurements/Test/"

measurement = namedtuple('measurement', ['title', 'spectra', 'times'])
measurements = []
names = []

for file in files:

    if 'times' in file:
        continue
    val = file.split('measurement_')[1].split('.npy')[0]
    if val == '':
        continue

    names.append(val)
    measurements.append(
        measurement(
            title=val,
            spectra=np.load(filepath + 'measurement_%s.npy' % val, allow_pickle=True),
            times=np.load(filepath + 'measurement_times_%s.npy' % val, allow_pickle=True)))


def dose(filename):
    """
    Args:
        filename: Name of the measurement
    Returns:
        If dose is stored in filename, return dose else return None
    """
    match = re.search(r'_(\d+)ug_', filename)
    if match:
        return match.group(1)
    return None


def fix_measurement(meas):
    """
    Args:
        meas: Namedtuple which stores the title, spectra and times
    Return:
        data: Augmented meas namedtuple with all the spectra and times appended in the same list
    """
    times = meas.times          # Array with all times of all measurements
    spectra = meas.spectra      # Array with all spectra of all measurements
    ns = []
    nt = []

    for spectrum_index in range(len(spectra)):
        if np.shape(spectra[spectrum_index]) == (256, 2):
            ns.append(spectra[spectrum_index])
            nt.append(times[spectrum_index])
    data = measurement(meas.title, np.array(ns), np.array(nt))
    return data


initial_slope_length = 120


def find_dip(frequencies, spectrum, prev_min):
    """
    Find the resonance dip of the intensity curve with respect to wavelength.
    Args:
        frequencies:
        spectrum: the record.py measurement values
        prev_min:
    Returns:
        fitted_freq_min:
        pos_min:
    """
    pos_min = np.argmin(spectrum)
    if prev_min is not None and abs(prev_min - pos_min) > 3:
        pos_min = prev_min
    if prev_min is not None and frequencies[pos_min] > 645:
        pos_min = prev_min

    rs = []
    frq = []

    for left_margin in [9, 10, 11, 12]:
        for right_margin in [8, 9, 10, 11]:
            left = pos_min - left_margin
            right = pos_min + right_margin

            fit, residuals, rank, singular_values, rcond = np.polyfit(frequencies[left:right], spectrum[left:right], 2, full=True)

            ls = np.linspace(frequencies[left:right].min(), frequencies[left:right].max(), 20000)

            fitted_freq_min = ls[np.argmin(np.poly1d(fit)(ls))]

            rs.append(residuals[0] / (left_margin + right_margin))
            frq.append(fitted_freq_min)

    fitted_freq_min_idx = np.argmin(rs)
    fitted_freq_min = frq[fitted_freq_min_idx]
    return fitted_freq_min, pos_min


def process_measurement(meas):
    """
    Args:
        meas: Namedtuple which stores the title, spectra and times
    Return:
        Array of results:
    """
    meas = fix_measurement(meas)

    spectra = meas.spectra
    times = meas.times

    spectra_moving_avg_width = 10
    light_measurement = np.sum(spectra[0:10, :, 1], axis=0) / spectra_moving_avg_width

    relative_spectra = spectra[:, :, 1] / light_measurement
    freqs = spectra[:, :, 0]

    results = []

    for i in range(len(spectra)):
        relative_spectrum = (np.sum(spectra[i:i + spectra_moving_avg_width, :, 1],
                                    axis=0) / spectra_moving_avg_width) + 2000
        relative_spectrum /= light_measurement + 2000

        pos_min = None
        if np.min(relative_spectrum) < 0.97:        # was 0.97
            time = times[i] - times[0]
            s = relative_spectrum
            f = freqs[i]

            fitted_freq_min, pos_min = find_dip(f, s, pos_min)

            if 500 < fitted_freq_min < 680:
                results.append(np.array([time, fitted_freq_min]))
    print(np.array(results))
    return np.array(results)


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def find_slope(m, left_index=None, right_index=None):
    r = process_measurement(m)
    print("length of results:", len(r))
    if len(r) < 100:
        print("Not enough values from process_meas found for %s to complete slope calculation" %(m.title))
        return float("nan"), float("nan")

    if left_index is not None:
        a = left_index
    else:
        a = r[0][0] + 30

    aidx = np.argwhere(r[:, 0] > a)[0][0]
    maxsearch = a + 50
    maxsearchidx = np.argwhere(r[:, 0] > maxsearch)[0][0]

    # find minimum beyond a this
    a2idx = np.argmin(r[aidx:maxsearchidx, 1])
    a2 = r[aidx + a2idx][0]

    if a2 - a < 50:
        a = a2

    if right_index is not None:
        b = right_index
    else:
        b = a + initial_slope_length

    aidx = np.argwhere(r[:, 0] > a)[0][0]
    bidx = np.argwhere(r[:, 0] > b)[0][0]
    print("aidx: %d, bidx: %d" %(aidx, bidx))

    fits = []
    for aoffset in range(-50, 30, 2):
        for boffset in range(-30, 50, 2):
            if aidx + aoffset < 0:
                continue
            if (bidx + boffset) >= len(r):       # jaz - was causing index errors
                continue                        # jaz
            tofit_times = np.log(r[aidx + aoffset:bidx + boffset, 0])
            tofit_shifts = np.log(r[aidx + aoffset:bidx + boffset, 1])

            # print(np.shape(r), aidx+aoffset, bidx+boffset, np.shape(tofit_times), np.shape(tofit_shifts))
            (fit, residuals, rank, singular_values, rcond) = np.polyfit(tofit_times, tofit_shifts, 1, full=True)
            res = residuals[0]
            resper = res / (bidx + boffset - aidx + aoffset)
            #             print("aoffset=%3d boffset=%3d slope=%5.5f residual=%4.2f residual per datapoint promille=%4.2f" % (aoffset, boffset, fit[0], res, 100*resper))

            fit_a = r[aidx + aoffset, 0]
            fit_b = r[bidx + boffset, 0]
            fits.append((resper, aidx + aoffset, fit_a, bidx + boffset, fit_b, fit))

    bestfit = sorted(fits, key=lambda x: x[0])[0]
    (best_resper, best_a_offset, best_a, best_b_offset, best_b, fit) = bestfit

    #     print("bestfit: res=%d a=%2.2f b=%2.2f")
    fit_times = np.linspace(np.log(a), np.log(b), 1000)
    fit_vals = np.poly1d(fit)(fit_times)
    slope = fit[0]

    print("measurement %s first dip %2.2f using a=%d b=%d slope length=%d slope=%e" % (m.title, r[0][0], best_a, best_b, best_b - best_a, slope))
    if do_plots:
        plt.figure(figsize=(8, 4))
        plt.title("i=%d %s, shift slope %e" % (i, m.title, slope))
        plt.xlabel("Time")
        plt.ylabel("Wavelength")
        plt.plot(r[:, 0], r[:, 1])
        plt.plot(np.exp(fit_times), np.exp(fit_vals))
        plt.axvline(a, c='c')
        plt.axvline(b, c='c')
        plt.show()
    return slope, best_resper


index = []
slopes = []
residuals = []
titles = []
results = {}

for i in range(0, len(measurements)):
    m = measurements[i]
    print(m.title)
    slope, best_resper = find_slope(m, left_index, right_index)
    results[m.title] = (slope, best_resper)


for n, (slope, residual) in results.items():
    if math.isnan(slope):
        continue
    index.append(dose(n))
    slopes.append(slope)
    residuals.append(residual)
    titles.append(n)

df = pandas.DataFrame({"title": titles, "slope": slopes, "residual": residuals, "real": index})
plt.scatter(df['real'], df['slope'])


print("index: ", df.index, "slope: ", df['slope'])
(fit, residuals, rank, singular_values, rcond) = np.polyfit(df['real'], df['slope'], 1, full=True)


ls = np.linspace(-5, 20, 10000)
print("fit: ", fit)
vals = np.poly1d(fit)(ls)
plt.plot(ls, vals, '-')
plt.xlabel("Concentration")
plt.ylabel("Slope")
#plt.show()

# widen a bit
conc2SlopeX = np.linspace(-30, 30, 2000) # used to be LS
conc2SlopeY = np.poly1d(fit)(conc2SlopeX) # used to be vals
df['slope']


def slope2conc(slope):
    global conc2SlopeX, conc2SlopeY
    idx = np.argwhere(conc2SlopeY > slope)
    if not idx.any():
        return float("NaN")
    idx = idx[0]
    estimate = conc2SlopeX[idx][0]
    return estimate

estimates = []
for x in df['slope']:
    estimate = slope2conc(x)
    # Values are always between 0 and 10
    estimates.append(min(10, max(0, estimate)))
df['estimate'] = estimates
df['r2'] = (df['estimate'] - df['real']).pow(2)
avg_error = (df['real'] - df['estimate']).abs().mean()
print("residuals %e, rank %s, singular_values %s, rcond %s" % (residuals, rank, singular_values, rcond))
print("total r2: %2.2f on %d measurements (avg absolute error=%2.2f)" % (df['r2'].sum(), len(df), avg_error))
print("y = %8e x + %8e" % (fit[0], fit[1]))
calibration_r2 = df['r2'].sum()
plt.figure()


for i in set(df.index):
    entries = df[df['real'] == i]
    if len(entries) > 2:
        mean = entries['slope'].mean()
        std = entries['slope'].std()
        print("i = %d mean = %e std = %e" % (i, mean, std))
        plt.errorbar([i], [mean], [std], c='r', linestyle='None', marker='^')

plt.show()

# Save calibration to disk.
np.save('images/calibration_conc2SlopeX.npy', conc2SlopeX)
np.save('images/calibration_conc2SlopeY.npy', conc2SlopeY)

print("yay the code is done!")
