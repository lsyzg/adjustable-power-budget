# run in server, test 'inverse design' to find alpha (db/cm) of waveguide

import sys, os

sys.path.append("/opt/lumerical/v231/api/python/")
sys.path.append(os.path.dirname(__file__))

import lumapi

TARGET_ALPHA_DB_PER_CM = 2.0  # <- set this to the alpha you want the waveguide to hit

mode = lumapi.MODE()
mode.eval(open('lumerical_setup.lsf').read())


def linspace(start, stop, n):
    if n <= 1:
        return [start]
    step = (stop - start) / (n - 1)
    return [start + i * step for i in range(n)]


def alpha_for_dimensions(x_span_m, y_span_m):
    mode.switchtolayout()
    mode.setnamed('waveguide', 'x span', x_span_m)
    mode.setnamed('waveguide', 'y span', y_span_m)
    return mode.get_alpha()


x_spans = linspace(3e-7, 7e-7, 9)    # waveguide width sweep, 300-700 nm
y_spans = linspace(1.5e-7, 3e-7, 7)  # waveguide height sweep, 150-300 nm

best = None
for x_span in x_spans:
    for y_span in y_spans:
        alpha = alpha_for_dimensions(x_span, y_span)
        error = abs(alpha - TARGET_ALPHA_DB_PER_CM)
        if best is None or error < best["error"]:
            best = {"x_span": x_span, "y_span": y_span, "alpha": alpha, "error": error}

print(f"target alpha = {TARGET_ALPHA_DB_PER_CM} dB/cm")
print(f"closest match: x span = {best['x_span'] * 1e9:.1f} nm, "
      f"y span = {best['y_span'] * 1e9:.1f} nm, alpha = {best['alpha']:.4f} dB/cm")
