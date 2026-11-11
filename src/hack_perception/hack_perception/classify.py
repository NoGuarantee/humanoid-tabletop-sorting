"""Tray colour and object category for one table cluster. Pure numpy, no ROS.

Trays are the three plastics from the scene: red, blue, green. The id comes from the task
(`/sim/task` field `trays`); the known layout is only the fallback before that message arrives.

An object is the catalogue entry, among the names in `/sim/objects/names`, whose size is closest
to the cluster. The category is the one from that message, so the same size list sorts whatever
the task assigned. A colour that cannot belong to the chosen object adds a penalty: on the closest
cross-category pair (the clock and the sweet boxes, 2.6 cm) it turns a contradiction into a low
confidence instead of a wrong tray. `brain` skips anything under its confidence threshold.

CLIP is not loaded. These fifteen objects already separate by size and colour, and another model
would sit on the same GPU as the simulator and cuRobo. It becomes worth it if real frames put two
categories inside the noise with a high confidence.
"""
import math

import numpy as np

# long, short, height in metres, category, colours a camera view of the object can show.
# Sizes are the axis-aligned box of the mesh standing upright: max(x, y), min(x, y), z.
# Colours are loose where size already decides, and tight where two categories are close.
CATALOG = {
    "Schleich_African_Black_Rhino": (0.138, 0.041, 0.060, "figure", "grey orange"),
    "3D_Dollhouse_Sink": (0.085, 0.073, 0.131, "figure", "white grey yellow orange"),
    "LADYBUG_BEAD": (0.084, 0.057, 0.030, "figure", "red orange"),
    "Threshold_Bamboo_Ceramic_Soap_Dish": (0.125, 0.098, 0.051, "dish", "orange yellow white"),
    "Cole_Hardware_Bowl_Scirocco_YellowBlue": (0.115, 0.113, 0.062, "dish", "yellow cyan blue white"),
    "Ecoforms_Cup_B4_SAN": (0.100, 0.100, 0.072, "dish", "orange green grey yellow white"),
    "Nescafe_Memento_Latte_Caramel_8_08_oz_23_g_packets_64_oz_184_g": (0.092, 0.064, 0.161, "box", "white red orange grey"),
    "Brother_Ink_Cartridge_Magenta_LC75M": (0.088, 0.031, 0.147, "box", "magenta blue white"),
    "Philips_EcoVantage_43_W_Light_Bulbs_Natural_Light_2_pack": (0.107, 0.067, 0.108, "box", "blue white cyan grey"),
    "Razer_Taipan_White_Ambidextrous_Gaming_Mouse": (0.182, 0.100, 0.040, "electronics", "white grey dark"),
    "JBL_Charge_Speaker_portable_wireless_wired_Green": (0.176, 0.073, 0.075, "electronics", "green"),
    "Crosley_Alarm_Clock_Vintage_Metal": (0.109, 0.054, 0.162, "electronics", "grey white dark"),
    "Nestle_Nips_Hard_Candy_Peanut_Butter": (0.088, 0.042, 0.153, "box", "orange red yellow"),
    "Nips_Hard_Candy_Rich_Creamy_Butter_Rum_4_oz_1133_g": (0.089, 0.043, 0.153, "box", "orange red yellow"),
    "009_gelatin_box": (0.101, 0.089, 0.030, "box", "red yellow white"),
}

# A wrong colour costs this much, in metres of size distance. Below the clock/nips gap (2.6 cm),
# above a size error of about a centimetre: it breaks a tie and refuses a contradiction.
COLOR_PENALTY = 0.018

# Plastics of the three trays, as set on the material. Chromaticity, so a darker light does not move them.
TRAY_RGB = {"red": (0.85, 0.25, 0.20), "blue": (0.20, 0.45, 0.85), "green": (0.25, 0.70, 0.30)}
TRAY_CHROMA = {name: np.array(rgb, float) / sum(rgb) for name, rgb in TRAY_RGB.items()}
DEFAULT_TRAY = {"red": "tray_left", "blue": "tray_right", "green": "tray_center"}
TRAY_COLOR_MAX = 0.22  # further than this from every plastic is not a tray colour

# A whole tray is 0.32 x 0.22. Below this the frame cut it, or it is only a wall; above it an object is merged in.
TRAY_LONG = (0.28, 0.42)
TRAY_SHORT = (0.18, 0.30)


def _colours(spec):
    return spec[4].split()


POOLS = {
    "4": (
        "Schleich_African_Black_Rhino", "3D_Dollhouse_Sink", "LADYBUG_BEAD",
        "Threshold_Bamboo_Ceramic_Soap_Dish", "Cole_Hardware_Bowl_Scirocco_YellowBlue", "Ecoforms_Cup_B4_SAN",
        "Nescafe_Memento_Latte_Caramel_8_08_oz_23_g_packets_64_oz_184_g", "Brother_Ink_Cartridge_Magenta_LC75M",
        "Philips_EcoVantage_43_W_Light_Bulbs_Natural_Light_2_pack",
    ),
    "6": (
        "Razer_Taipan_White_Ambidextrous_Gaming_Mouse", "JBL_Charge_Speaker_portable_wireless_wired_Green",
        "Crosley_Alarm_Clock_Vintage_Metal", "Nestle_Nips_Hard_Candy_Peanut_Butter",
        "Nips_Hard_Candy_Rich_Creamy_Butter_Rum_4_oz_1133_g", "009_gelatin_box",
    ),
}


def names_for(pool):
    """{name: category} of one sorting pool. For tests and offline replay; the node uses /sim/objects/names."""
    return {name: CATALOG[name][3] for name in POOLS[str(pool)]}


def _pixels(rgb, pixels):
    """Nx3 float colours in 0..1, or an empty array."""
    if rgb is None or pixels is None or len(pixels) == 0:
        return np.zeros((0, 3))
    cols = np.asarray(rgb, np.float32).reshape(-1, 3)[np.asarray(pixels)]
    if cols.size and cols.max() > 1.5:
        cols = cols / 255.0
    return cols


def color_tag(rgb, pixels):
    """Coarse colour of a cluster: red, orange, yellow, green, cyan, blue, magenta, white, grey, dark, or ""."""
    cols = _pixels(rgb, pixels)
    if len(cols) < 40:
        return ""
    spread = cols.max(axis=1) - cols.min(axis=1)
    vivid = spread > 0.12
    if float(vivid.mean()) < 0.30:
        value = float(np.median(cols.mean(axis=1)))
        if value > 0.62:
            return "white"
        if value > 0.25:
            return "grey"
        return "dark"
    r, g, b = (float(x) for x in np.median(cols[vivid], axis=0))
    hi, lo = max(r, g, b), min(r, g, b)
    d = hi - lo
    if d < 1e-6:
        return "grey"
    if hi == r:
        hue = ((g - b) / d) % 6.0
    elif hi == g:
        hue = (b - r) / d + 2.0
    else:
        hue = (r - g) / d + 4.0
    hue *= 60.0
    if hue < 15.0 or hue >= 345.0:
        return "red"
    if hue < 45.0:
        return "orange"
    if hue < 75.0:
        return "yellow"
    if hue < 170.0:
        return "green"
    if hue < 200.0:
        return "cyan"
    if hue < 255.0:
        return "blue"
    return "magenta"


def _confidence(cost, margin):
    """1 when the size fits and the next category is far, 0 when it does not. `brain` keeps values from 0.6."""
    fit = max(0.0, 1.0 - cost / 0.045)
    sep = max(0.0, min(1.0, (margin - 0.008) / 0.018))
    return fit * sep


def classify_object(size, rgb, pixels, names):
    """(name, category, confidence) of one cluster. Empty when `names` has nothing from the catalogue.

    `names` is {object name: category} from /sim/objects/names. `size` is (along, across, height) in metres.
    """
    if not names:
        return "", "", 0.0
    long, short = sorted((float(size[0]), float(size[1])), reverse=True)
    height = float(size[2])
    tag = color_tag(rgb, pixels)
    found = []
    for name, category in names.items():
        spec = CATALOG.get(name)
        if spec is None:
            continue
        dist = math.hypot(long - spec[0], short - spec[1], height - spec[2])
        penalty = COLOR_PENALTY if tag and tag not in _colours(spec) else 0.0
        found.append((dist + penalty, name, str(category)))
    if not found:
        return "", "", 0.0
    found.sort()
    cost, name, category = found[0]
    margin = 0.05  # no other category in this task: the size fit alone decides
    for other, _, other_cat in found[1:]:
        if other_cat != category:
            margin = other - cost
            break
    return name, category, float(_confidence(cost, margin))


def _chroma(rgb, pixels):
    cols = _pixels(rgb, pixels)
    if len(cols) < 40:
        return None
    spread = cols.max(axis=1) - cols.min(axis=1)
    vivid = cols[spread > 0.08]
    if len(vivid) < 40:
        return None
    med = np.median(vivid, axis=0)
    total = float(med.sum())
    if total < 1e-6:
        return None
    return med / total


def tray_color(rgb, pixels):
    """red / blue / green, or "" when the cluster is not one of the three plastics."""
    chroma = _chroma(rgb, pixels)
    if chroma is None:
        return ""
    name, dist = "", 1e9
    for label, proto in TRAY_CHROMA.items():
        gap = float(np.linalg.norm(chroma - proto))
        if gap < dist:
            name, dist = label, gap
    return name if dist <= TRAY_COLOR_MAX else ""


def tray_body(size):
    """True for a whole tray footprint. A cut tray and a lone wall are rejected: a bad centre drops the object."""
    long, short = sorted((float(size[0]), float(size[1])), reverse=True)
    return TRAY_LONG[0] <= long <= TRAY_LONG[1] and TRAY_SHORT[0] <= short <= TRAY_SHORT[1]


def find_trays(clusters, rgb, tray_of_color):
    """Trays among clusters already rejected as trays by the geometry.

    Each cluster needs size, pixels, center, yaw, top_z, n_points, rejected.
    `tray_of_color` maps red/blue/green to a tray id. One id: the cluster with the most points.
    Returns dicts: id, color, center (x, y, z of the opening), yaw, size, n_points.
    """
    found = {}
    for cluster in clusters:
        if cluster.rejected != "tray" or not tray_body(cluster.size):
            continue
        color = tray_color(rgb, cluster.pixels)
        tray_id = tray_of_color.get(color, "")
        if not tray_id:
            continue
        previous = found.get(tray_id)
        if previous is not None and cluster.n_points <= previous["n_points"]:
            continue
        found[tray_id] = {
            "id": tray_id, "color": color, "yaw": float(cluster.yaw),
            "center": (float(cluster.center[0]), float(cluster.center[1]), float(cluster.top_z)),
            "size": (float(cluster.size[0]), float(cluster.size[1]), float(cluster.size[2])),
            "n_points": int(cluster.n_points),
        }
    return list(found.values())


def _check():
    """Exact catalogue sizes, noisy sizes, and the three tray plastics. Exits non-zero on a wrong confident label."""
    pools = {pool: names_for(pool) for pool in POOLS}
    failed = 0
    for pool, names in pools.items():
        for name, spec in ((n, CATALOG[n]) for n in names):
            tag = _colours(spec)[0]
            got, category, conf = classify_object((spec[0], spec[1], spec[2]), _swatch(tag), np.arange(80), names)
            if got != name or category != spec[3] or conf < 0.6:
                print(f"FAIL exact pool {pool} {name}: got {got} {category} {conf:.2f}")
                failed += 1
    rng = np.random.default_rng(0)
    accepted = wrong = 0
    for pool, names in pools.items():
        items = list(names)
        for _ in range(400):
            name = items[int(rng.integers(0, len(items)))]
            spec = CATALOG[name]
            noise = rng.normal(0.0, 0.008, 3)
            size = (spec[0] + noise[0], spec[1] + noise[1], spec[2] + noise[2])
            tag = _colours(spec)[int(rng.integers(0, len(_colours(spec))))]
            got, category, conf = classify_object(size, _swatch(tag), np.arange(80), names)
            if conf < 0.6:
                continue
            accepted += 1
            if category != spec[3]:
                wrong += 1
                print(f"FAIL noise pool {pool} {name} -> {got} {category} {conf:.2f} size {size}")
    print(f"noise: {accepted} accepted, {wrong} wrong category")
    paints = {"red": (217, 64, 51), "blue": (51, 115, 217), "green": (64, 179, 77)}
    for color, paint in paints.items():
        if tray_color(_swatch_rgb(paint), np.arange(100)) != color:
            print(f"FAIL tray {color}")
            failed += 1
        dark = tuple(int(0.4 * c) for c in paint)  # dimmer light, same chromaticity
        pale = tuple(int(0.75 * c + 0.25 * 255) for c in paint)  # a white wash over the plastic
        if tray_color(_swatch_rgb(dark), np.arange(100)) != color or tray_color(_swatch_rgb(pale), np.arange(100)) != color:
            print(f"FAIL lit tray {color} dark {dark} pale {pale}")
            failed += 1
    if tray_color(_swatch("grey"), np.arange(100)) != "":
        print("FAIL grey read as a tray")
        failed += 1
    if not tray_body((0.32, 0.22, 0.07)) or tray_body((0.20, 0.22, 0.07)) or tray_body((0.50, 0.22, 0.07)):
        print("FAIL tray footprint gate")
        failed += 1
    red = _swatch_rgb(paints["red"])

    def tray_cluster(points, x):
        return type("C", (), {"rejected": "tray", "size": (0.32, 0.22, 0.07), "pixels": np.arange(100),
                              "center": np.array([x, 0.69, -0.11]), "yaw": 1.2, "top_z": -0.08, "n_points": points})()

    wall = tray_cluster(500, 0.34)
    wall.size = (0.12, 0.02, 0.07)
    found = find_trays([tray_cluster(10, 0.30), tray_cluster(80, 0.34), wall], red, DEFAULT_TRAY)
    if len(found) != 1 or found[0]["id"] != "tray_left" or found[0]["n_points"] != 80 or found[0]["center"][2] != -0.08:
        print(f"FAIL find_trays {found}")
        failed += 1
    print("ok" if failed == 0 and wrong == 0 else "FAILED")
    return failed + wrong


def _swatch(tag):
    rgb = {
        "red": (210, 40, 35), "orange": (210, 110, 30), "yellow": (220, 200, 40), "green": (40, 170, 50),
        "cyan": (40, 180, 190), "blue": (40, 70, 200), "magenta": (190, 50, 180), "white": (230, 228, 224),
        "grey": (130, 125, 120), "dark": (25, 24, 23),
    }[tag]
    return _swatch_rgb(rgb)


def _swatch_rgb(rgb):
    image = np.empty((10, 10, 3), np.uint8)
    image[:] = rgb
    return image


if __name__ == "__main__":
    raise SystemExit(_check())


def merge_trays(model_trays, rejected, rgb, tray_of_color, measured=()):
    """Prefer fitted openings, then measured plastic rims, then colour clusters."""
    trays = {t.id: (t.id, t.color, t.center, t.yaw, t.size) for t in model_trays}
    for t in measured:
        trays.setdefault(t.id, (t.id, t.color, t.center, t.yaw, t.size))
    if set(tray_of_color.values()) - trays.keys():
        for t in find_trays(rejected, rgb, tray_of_color):
            trays.setdefault(t["id"], (t["id"], t["color"], t["center"], t["yaw"], t["size"]))
    return list(trays.values())
