import argparse
import os
from array import array

import yaml

try:
    import ROOT
    ROOT.gROOT.SetBatch(True)
except Exception as exc:
    raise RuntimeError("ROOT is required to merge signal ROOT files") from exc


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build per-era combined histograms (concatenated categories) from existing "
            "plotIt configs, normalizing by bin width."
        )
    )
    parser.add_argument(
        "PLOT_PREFIX",
        help="Prefix for plot/histogram names (e.g. SR, DY, TT).",
    )
    parser.add_argument(
        "--config-2022",
        dest="config_2022",
        default="/afs/cern.ch/work/a/aguzel/private/wwbb-run3-datacards/output/"
                "SR_v1.4.8_heuristic_v4-5_DYfix/plotit/plots_2022_combined.yml",
        help="Absolute path to the 2022 plotIt YAML (combined).",
    )
    parser.add_argument(
        "--config-2023",
        dest="config_2023",
        default="/afs/cern.ch/work/a/aguzel/private/wwbb-run3-datacards/output/"
                "SR_v1.4.8_heuristic_v4-5_DYfix/plotit/plots_2023_combined.yml",
        help="Absolute path to the 2023 plotIt YAML (combined).",
    )
    parser.add_argument(
        "--output-suffix",
        default="_with_combined",
        help="Suffix to append to output YAML file names before the extension.",
    )
    parser.add_argument(
        "--blind",
        action="store_true",
        help="Enable blinding (set blinded-range in output plots)",
    )
    return parser.parse_args()


_args = _parse_args()
PLOT_PREFIX = _args.PLOT_PREFIX.rstrip("_")
PLOT_NAME_PREFIX = f"{PLOT_PREFIX}_"
ENABLE_BLINDING = _args.blind

GGF_PREFIX = "ggHH_kl_1_kt_1_hbbhww_"
VBF_PREFIX = "qqHH_CV_1_C2V_1_kl_1_hbbhww_"
ALT_GGF_PREFIX = "HH_ggf_"
ALT_VBF_PREFIX = "HH_vbf_"
ALLOWED_SIGNAL_PREFIXES = (GGF_PREFIX, VBF_PREFIX, ALT_GGF_PREFIX, ALT_VBF_PREFIX)
COMBINED_HIST = f"{PLOT_NAME_PREFIX}combined"
# Plot styling constants
Y_MAX = 1_000_000_000.0
HLINE_Y = 1_000_000.0
# Default guesses; actual category names are detected dynamically from the YAML.
COMBINED_CATEGORIES = [
    ("Resolved 1b", f"{PLOT_NAME_PREFIX}resolved1b"),
    ("Resolved 2b", f"{PLOT_NAME_PREFIX}resolved2b"),
    ("Boosted", f"{PLOT_NAME_PREFIX}boosted"),
    ("VBF resolved", f"{PLOT_NAME_PREFIX}vbf_resolved"),
    ("VBF boosted", f"{PLOT_NAME_PREFIX}vbf_boosted"),
]
COMBINED_BIN_UNIT = 1.0
COMBINED_X_TITLE_OFFSET = 12.4

def prune_non_sm_signals(cfg):
    """Drop non-SM HH signals from files and plot sample lists."""

    files = cfg.get("files", {})
    drop = [
        name
        for name, info in files.items()
        if info.get("type") == "signal"
        and not any(name.startswith(prefix) for prefix in ALLOWED_SIGNAL_PREFIXES)
    ]
    for name in drop:
        files.pop(name, None)

    if drop:
        drop_set = set(drop)
        for plot in cfg.get("plots", {}).values():
            samples = plot.get("samples")
            if samples:
                plot["samples"] = [s for s in samples if s not in drop_set]

prune_non_sm_signals_dataclass = prune_non_sm_signals  # keep name for reuse in functions


def strip_blinding(cfg):
    """Remove per-plot blinding range while keeping global blinding style settings."""

    cfg.pop("blinded-range", None)


def remap_groups(cfg):
    """Regroup backgrounds: Higgs (ZH, tHq, ttH, tHW), keep TTbar/DY/single-top, others -> other_bkg."""

    files = cfg.get("files", {})
    for info in files.values():
        g = info.get("group")
        if info.get("type") == "signal":
            continue
        if g in HIGGS_GROUPS:
            info["group"] = "SingleHiggs"
        elif g in SINGLE_TOP_GROUPS:
            info["group"] = "SingleTop"
        elif g in KEEP_GROUPS:
            info["group"] = g
        else:
            info["group"] = "other_bkg"

    groups = cfg.setdefault("groups", {})
    groups.setdefault("SingleHiggs", {"legend": "Single Higgs", "fill-color": "#6c9bd2"})
    groups.setdefault("SingleTop", {"legend": "Single top", "fill-color": "#f2b86c"})
    groups.setdefault("other_bkg", {"legend": "Other bkg", "fill-color": "#a0a0a0"})
    # Preserve existing keep groups; ensure they exist if missing.
    for g in KEEP_GROUPS:
        groups.setdefault(g, {"legend": g})


COMBINED_TOTAL_WIDTH = None
COMBINED_BOUNDARIES = None
LINE_STYLE = {
    "line-width": 2,
    "line-color": "#000000",
    "line-type": 1,
}
KEEP_GROUPS = {"DY", "TTbar", "data_obs", "HH_ggf", "HH_vbf"}
SINGLE_TOP_GROUPS = {"ST_tW", "ST_tchan"}
HIGGS_GROUPS = {"ZH", "tHq", "ttH", "tHW"}


def detect_categories(plot_names):
    """Infer the category plot names from the YAML keys.

    Preference order: resolved1b, resolved2b, boosted. Falls back to the
    default COMBINED_CATEGORIES if detection fails.
    """

    keywords = [
        ("resolved1b", "Resolved 1b"),
        ("resolved2b", "Resolved 2b"),
        ("boosted", "Boosted"),
        ("vbf_resolved", "VBF resolved"),
        ("vbf_boosted", "VBF boosted"),
    ]
    lowered = {name.lower(): name for name in plot_names}
    categories = []
    for key, label in keywords:
        match = next((orig for low, orig in lowered.items() if key in low), None)
        if match:
            categories.append((label, match))

    return categories if categories else list(COMBINED_CATEGORIES)


def build_combined_hist(root_path, categories, combined_hist_name, normalize_bin_width=True):
    """Concatenate resolved1b/2b/boosted histograms (nominal and syst) into one histogram.

    For every variation suffix ("" for nominal, "__systUp", ...), this builds
    a TH1 named COMBINED_HIST{suffix} with bin edges shifted so categories are
    consecutive on one x-axis. Bin labels are set only for the nominal.
    """

    f_in = ROOT.TFile.Open(root_path, "UPDATE")
    if not f_in or f_in.IsZombie():
        raise RuntimeError(f"Failed to open ROOT file for update: {root_path}")

    # Drop any previously written combined (nominal or syst) histograms.
    for key in list(f_in.GetListOfKeys()):
        name = key.GetName()
        if name.startswith(combined_hist_name):
            f_in.Delete(f"{name};*")

    cat_names = [name for _, name in categories]
    nominals = {}
    for cat_name in cat_names:
        h = f_in.Get(cat_name)
        if not h:
            f_in.Close()
            return False
        nominals[cat_name] = h

    # Build bin edges once from the nominal shapes.
    edges = [0.0]
    gap = 0.0  # zero gap to avoid an empty bin at the right edge
    offset = 0.0
    cat_bins = {}
    boundaries = []
    for idx, cat_name in enumerate(cat_names):
        hist = nominals[cat_name]
        nbins = hist.GetXaxis().GetNbins()
        cat_bins[cat_name] = nbins
        if idx > 0:
            boundaries.append(offset)
        for ibin in range(1, nbins + 1):
            edges.append(offset + ibin * COMBINED_BIN_UNIT)
        offset = edges[-1] + gap

    for i in range(len(edges) - 1):
        if not edges[i] < edges[i + 1]:
            f_in.Close()
            raise RuntimeError(f"Non-increasing edges in {root_path}: {edges[i]} >= {edges[i+1]}")

    arr = array("d", edges)

    # Collect all variations ("" for nominal) across categories.
    variations = {"": nominals}
    for key in f_in.GetListOfKeys():
        name = key.GetName()
        obj = key.ReadObj()
        if not (obj and obj.InheritsFrom("TH1")):
            continue
        for cat_name in cat_names:
            prefix = f"{cat_name}__"
            if name.startswith(prefix):
                suffix = name[len(cat_name):]
                variations.setdefault(suffix, {})[cat_name] = obj

    for suffix, cat_map in variations.items():
        if any(cat_name not in cat_map for cat_name in cat_names):
            continue
        hname = f"{combined_hist_name}{suffix}"
        combined = ROOT.TH1D(hname, nominals[cat_names[0]].GetTitle(), len(edges) - 1, arr)
        if not combined:
            f_in.Close()
            raise RuntimeError(f"Failed to create {hname} in {root_path}")
        combined.Sumw2()

        start_bin = 1
        for (label, _), cat_name in zip(categories, cat_names):
            hist = cat_map[cat_name]
            nbins = cat_bins[cat_name]
            if hist.GetXaxis().GetNbins() != nbins:
                f_in.Close()
                raise RuntimeError(f"Bin mismatch for {cat_name}{suffix} in {root_path}")
            for ibin in range(1, nbins + 1):
                width = hist.GetXaxis().GetBinWidth(ibin) if normalize_bin_width else 1.0
                target_bin = start_bin + ibin - 1
                content = hist.GetBinContent(ibin)
                error = hist.GetBinError(ibin)
                combined.SetBinContent(target_bin, content / width)
                combined.SetBinError(target_bin, error / width)
            if suffix == "":
                mid_bin = start_bin + nbins // 2
                combined.GetXaxis().SetBinLabel(mid_bin, label)
            start_bin += nbins

        xaxis = combined.GetXaxis()
        xaxis.SetTitle("ML score")
        xaxis.SetTitleOffset(COMBINED_X_TITLE_OFFSET)
        combined.SetDirectory(f_in)
        combined.Write()

    f_in.Close()

    global COMBINED_TOTAL_WIDTH, COMBINED_BOUNDARIES
    if COMBINED_TOTAL_WIDTH is None:
        COMBINED_TOTAL_WIDTH = edges[-1] - edges[0]
    if COMBINED_BOUNDARIES is None:
        COMBINED_BOUNDARIES = boundaries
    return True


def process_single_config(config_path):
    """Build a combined plot for a single era config, writing a new YAML alongside it."""

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    prune_non_sm_signals(cfg)
    remap_groups(cfg)

    plotdefaults = cfg.get("plotdefaults", {})
    strip_blinding(plotdefaults)
    strip_blinding(cfg.get("configuration", {}))

    # Force a consistent global y-max used by plotIt when defaults propagate.
    if "y-axis-range" in plotdefaults:
        ymin, _ = plotdefaults.get("y-axis-range", [1.0e-5, Y_MAX])
        plotdefaults["y-axis-range"] = [ymin, Y_MAX]

    # Derive ROOT directory from the config.
    root_dir = cfg.get("configuration", {}).get("root", "root")
    if not os.path.isabs(root_dir):
        root_dir = os.path.join(os.path.dirname(config_path), root_dir)

    categories = detect_categories(cfg.get("plots", {}).keys())

    global COMBINED_TOTAL_WIDTH, COMBINED_BOUNDARIES
    COMBINED_TOTAL_WIDTH = None
    COMBINED_BOUNDARIES = None
    for root_fname in list(cfg.get("files", {}).keys()):
        build_combined_hist(os.path.join(root_dir, root_fname), categories, COMBINED_HIST)

    plots = cfg.setdefault("plots", {})
    base_plot_names = tuple(cat_name for _, cat_name in categories)
    base_plots = [plots[name] for name in base_plot_names if name in plots]
    if base_plots:
        max_lin = max(
            cfg_plot.get("y-axis-range", [0.0, 0.0])[1]
            for cfg_plot in base_plots
            if "y-axis-range" in cfg_plot
        )
        max_log = max(
            cfg_plot.get("log-y-axis-range", [0.0, 0.0])[1]
            for cfg_plot in base_plots
            if "log-y-axis-range" in cfg_plot
        )
        combined_cfg = dict(base_plots[0])
        combined_cfg.pop("blinded-range", None)
        strip_blinding(combined_cfg)
        labels_str = " | ".join(label for label, _ in categories)
        combined_cfg["x-axis"] = ""
        x_max = COMBINED_TOTAL_WIDTH if COMBINED_TOTAL_WIDTH is not None else 3.0
        combined_cfg["x-axis-range"] = [0.0, x_max]
        combined_cfg["show-overflow"] = False
        combined_cfg["x-axis-label-size"] = 0.07
        combined_cfg["x-axis-hide-ticks"] = False
        combined_cfg["y-axis-range"] = [0.0, Y_MAX]
        combined_cfg["ratio-y-axis-range"] = list(
            plotdefaults.get("ratio-y-axis-range", [0.5, 1.5])
        )
        combined_cfg["log-y-axis-range"] = [0.01, Y_MAX]
        if COMBINED_BOUNDARIES:
            combined_cfg["vertical-lines"] = [
                {**LINE_STYLE, "value": b} for b in COMBINED_BOUNDARIES
            ] + [
                {**LINE_STYLE, "pad-location": "bottom", "value": b}
                for b in COMBINED_BOUNDARIES
            ]

            # Explicit lines with y extents to cover full pad height (top and ratio pads).
            y_min, y_max = combined_cfg.get("y-axis-range", [0.0, Y_MAX])
            line_height_top = max(y_max, Y_MAX)
            ratio_min, ratio_max = combined_cfg.get("ratio-y-axis-range", [0.5, 1.5])
            line_height_ratio = ratio_max

            top_lines = [
                {"value": [[b, y_min], [b, line_height_top]], **LINE_STYLE}
                for b in COMBINED_BOUNDARIES
            ]
            bottom_lines = [
                {
                    "pad-location": "bottom",
                    "value": [[b, ratio_min], [b, line_height_ratio]],
                    **LINE_STYLE,
                }
                for b in COMBINED_BOUNDARIES
            ]

            horizontal_line = {
                "value": [[0.0, HLINE_Y], [x_max, HLINE_Y]],
                **LINE_STYLE,
            }

            combined_cfg["lines"] = top_lines + bottom_lines + [horizontal_line]
        plots[COMBINED_HIST] = combined_cfg

    BLIND_RANGE = [0.25, 0.999]
    if ENABLE_BLINDING:
        for plot_cfg in cfg.get("plots", {}).values():
            plot_cfg["blinded-range"] = BLIND_RANGE

    base, ext = os.path.splitext(config_path)
    out_path = f"{base}{_args.output_suffix}{ext}"
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f)

    print(
        f"Built combined plot (bin-width normalized) for {config_path} -> {out_path} "
        f"with {len(cfg.get('plots', {}))} plots."
    )


if __name__ == "__main__":
    for cfg_path in (_args.config_2022, _args.config_2023):
        process_single_config(cfg_path)

