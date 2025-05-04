import math
import colorsys


def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip('#')
    r = int(hex_str[0:2], 16) / 255.0
    g = int(hex_str[2:4], 16) / 255.0
    b = int(hex_str[4:6], 16) / 255.0
    return (r, g, b)


def rgb_to_lin(c):
    if c <= 0.04045:
        return c / 12.92
    else:
        return ((c + 0.055) / 1.055) ** 2.4


def lin_to_xyz(lin_rgb):
    r, g, b = lin_rgb
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    return (x, y, z)


def xyz_to_lab(xyz):
    x, y, z = xyz
    xn, yn, zn = 0.95047, 1.0, 1.08883

    def f(t):
        delta = 6/29
        if t > delta**3:
            return t ** (1/3)
        else:
            return t / (3 * delta**2) + 4/29

    fx = f(x / xn)
    fy = f(y / yn)
    fz = f(z / zn)

    l_val = 116 * fy - 16
    a_val = 500 * (fx - fy)
    b_val = 200 * (fy - fz)
    return (l_val, a_val, b_val)


def rgb_to_lab(rgb):
    r, g, b = rgb
    lin_r = rgb_to_lin(r)
    lin_g = rgb_to_lin(g)
    lin_b = rgb_to_lin(b)
    xyz = lin_to_xyz((lin_r, lin_g, lin_b))
    return xyz_to_lab(xyz)


def deltaE(lab1, lab2):
    return math.sqrt((lab1[0]-lab2[0])**2 + (lab1[1]-lab2[1])**2 + (lab1[2]-lab2[2])**2)


def hsv_to_rgb(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h/360.0, s, v)
    return (r, g, b)


def rgb_to_hex(rgb):
    r, g, b = rgb
    r_int = max(0, min(255, int(r * 255 + 0.5)))
    g_int = max(0, min(255, int(g * 255 + 0.5)))
    b_int = max(0, min(255, int(b * 255 + 0.5)))
    return "#{:02x}{:02x}{:02x}".format(r_int, g_int, b_int)


COLOR_CYCLE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    "#393b79", "#5254a3", "#6b6ecf", "#9c9ede", "#637939",
    "#8ca252", "#b5cf6b", "#cedb9c", "#8c6d31", "#bd9e39"
]

lab_cache = {}
for color in COLOR_CYCLE:
    rgb = hex_to_rgb(color)
    lab = rgb_to_lab(rgb)
    lab_cache[color] = lab

existing_hex = set(COLOR_CYCLE)
new_colors = []

for i in range(30):
    colorA = COLOR_CYCLE[i]
    colorB = COLOR_CYCLE[(i+1) % 30]
    labA = lab_cache[colorA]
    labB = lab_cache[colorB]

    candidates = []
    for h in range(0, 360, 30):
        candidate_rgb = hsv_to_rgb(h, 1.0, 1.0)
        candidate_hex = rgb_to_hex(candidate_rgb)
        candidate_lab = rgb_to_lab(candidate_rgb)
        d1 = deltaE(candidate_lab, labA)
        d2 = deltaE(candidate_lab, labB)
        score = min(d1, d2)
        candidates.append((candidate_hex, candidate_rgb, score))

    candidates.sort(key=lambda x: x[2], reverse=True)
    chosen_hex = None
    for cand in candidates:
        hex_val = cand[0]
        if hex_val not in existing_hex:
            chosen_hex = hex_val
            break
    if chosen_hex is None:
        chosen_hex = candidates[0][0]

    existing_hex.add(chosen_hex)
    new_colors.append(chosen_hex)

new_array = []
for i in range(30):
    new_array.append(COLOR_CYCLE[i])
    new_array.append(new_colors[i])

print(new_array)
