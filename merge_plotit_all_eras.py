import argparse
import os
from array import array

import yaml
from produceDataCards import Datacard

try:
    import ROOT
    ROOT.gROOT.SetBatch(True)
except Exception as exc:
    raise RuntimeError("ROOT is required to merge signal ROOT files") from exc

RUN3_ERA = "Run3"
def _parse_args():
    parser = argparse.ArgumentParser(
        description="Merge plotIt YAMLs and ROOT files across eras into a Run3 configuration."
    )
    parser.add_argument(
        "PLOTIT_DIR",
        help="Path to the plotIt directory containing plots_<ERA>.yml and root/",
    )
    parser.add_argument(
        "PLOT_PREFIX",
        help="Prefix for plot/histogram names (e.g. SR, DY, TT).",
    )
    parser.add_argument(
        "eras",
        nargs="?",
        default=RUN3_ERA,
        help="Eras to merge (comma-separated, e.g. 2022,2023 or Run3 for all eras).",
    )
    return parser.parse_args()


_args = _parse_args()
ERAS = _args.eras.split(",") if _args.eras != RUN3_ERA else ["2022", "2022EE", "2023", "2023BPix"]
PLOTIT_DIR = os.path.abspath(_args.PLOTIT_DIR)
PLOT_PREFIX = _args.PLOT_PREFIX.rstrip("_")
PLOT_NAME_PREFIX = f"{PLOT_PREFIX}_"
ROOT_DIR = os.path.join(PLOTIT_DIR, "root")

GGF_PREFIX = "ggHH_kl_1_kt_1_hbbhww_"
VBF_PREFIX = "qqHH_CV_1_C2V_1_kl_1_hbbhww_"
ALLOWED_SIGNAL_PREFIXES = (GGF_PREFIX, VBF_PREFIX)
COMBINED_HIST = f"{PLOT_NAME_PREFIX}combined"
COMBINED_CATEGORIES = [
    ("Resolved 1b", f"{PLOT_NAME_PREFIX}resolved1b"),
    ("Resolved 2b", f"{PLOT_NAME_PREFIX}resolved2b"),
    ("Boosted", f"{PLOT_NAME_PREFIX}boosted"),
]
COMBINED_BIN_UNIT = 1.0
COMBINED_X_TITLE_OFFSET = 12.4

# Load per-era plotIt configs.
configs = []
configs_by_era = {}
for era in ERAS:
    with open(f"{PLOTIT_DIR}/plots_{era}.yml", "r") as f:
        cfg = yaml.safe_load(f)
        configs.append(cfg)
        configs_by_era[era] = cfg

# Era subsets to merge separately.
ERA_GROUPS = {
    "Run3": ERAS,
    ERAS[0]: [ERAS[0], ERAS[1]],
    # "2022": ["2022", "2022EE"],
    # "2023": ["2023", "2023BPix"],
}
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

# Use plotIt configuration from the combined config file.
COMBINED_CONFIG_PATH = "/afs/cern.ch/work/a/aguzel/private/wwbb-run3-datacards/config/config_combined_sr.yml"


def strip_blinding(cfg):
    """Remove per-plot blinding range while keeping global blinding style settings."""

    cfg.pop("blinded-range", None)


class YamlIncludeSafeLoader(yaml.SafeLoader):
    pass


def _construct_include(loader, node):
    return None


YamlIncludeSafeLoader.add_constructor("!include", _construct_include)

with open(COMBINED_CONFIG_PATH, "r") as f:
    combined_config = yaml.load(f, Loader=YamlIncludeSafeLoader)
combined_plotit = combined_config.get("plotIt", {})

COMBINED_TOTAL_WIDTH = None


def build_combined_hist(root_path):
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
        if name.startswith(COMBINED_HIST):
            f_in.Delete(f"{name};*")

    cat_names = [name for _, name in COMBINED_CATEGORIES]
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
    for cat_name in cat_names:
        hist = nominals[cat_name]
        nbins = hist.GetXaxis().GetNbins()
        cat_bins[cat_name] = nbins
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
        hname = f"{COMBINED_HIST}{suffix}"
        combined = ROOT.TH1D(hname, nominals[cat_names[0]].GetTitle(), len(edges) - 1, arr)
        if not combined:
            f_in.Close()
            raise RuntimeError(f"Failed to create {hname} in {root_path}")
        combined.Sumw2()

        start_bin = 1
        for (label, _), cat_name in zip(COMBINED_CATEGORIES, cat_names):
            hist = cat_map[cat_name]
            nbins = cat_bins[cat_name]
            if hist.GetXaxis().GetNbins() != nbins:
                f_in.Close()
                raise RuntimeError(f"Bin mismatch for {cat_name}{suffix} in {root_path}")
            for ibin in range(1, nbins + 1):
                target_bin = start_bin + ibin - 1
                combined.SetBinContent(target_bin, hist.GetBinContent(ibin))
                combined.SetBinError(target_bin, hist.GetBinError(ibin))
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

    global COMBINED_TOTAL_WIDTH
    if COMBINED_TOTAL_WIDTH is None:
        COMBINED_TOTAL_WIDTH = edges[-1] - edges[0]
    return True


def build_merged_output(target_era, era_subset):
    """Build and write a merged plotIt config for a given era subset."""

    merged = Datacard.merge_plotIt([configs_by_era[e] for e in era_subset])
    prune_non_sm_signals(merged)

    merged["configuration"] = dict(combined_plotit.get("configuration", {}))
    merged.setdefault("configuration", {})
    merged["configuration"].setdefault("root", "root")
    merged["configuration"]["margin-bottom"] = 0.16
    merged["configuration"]["eras"] = [target_era]
    merged["configuration"]["blinded-range-fill-color"] = '#29556270'
    merged["configuration"]["blinded-range-fill-style"] = 1001
    strip_blinding(merged["configuration"])

    if "legend" in combined_plotit:
        merged["legend"] = combined_plotit["legend"]
    if "plotdefaults" in combined_plotit:
        merged["plotdefaults"] = combined_plotit["plotdefaults"]
    strip_blinding(merged.get("plotdefaults", {}))

    era_lumis = {
        era: float(configs_by_era[era]["configuration"]["luminosity"][era])
        for era in era_subset
    }
    total_lumi = sum(era_lumis.values())
    merged["configuration"]["luminosity"] = {target_era: total_lumi}

    for filename, info in merged.get("files", {}).items():
        orig_era = info.get("era")
        info["era"] = target_era
        if info.get("type") != "data" and orig_era in era_lumis:
            info["scale"] = info.get("scale", 1.0) * (era_lumis[orig_era] / total_lumi)
        if info.get("type") == "signal":
            if filename.startswith(GGF_PREFIX):
                info["group"] = "HH_ggf"
            elif filename.startswith(VBF_PREFIX):
                info["group"] = "HH_vbf"

    def merge_signal_files(prefix, out_name, legend, line_color, group_name):
        signal_files = [
            fname
            for fname, info in merged.get("files", {}).items()
            if info.get("type") == "signal" and fname.startswith(prefix)
        ]
        if not signal_files:
            return

        def add_hist(name, obj, sums_map):
            if name not in sums_map:
                sums_map[name] = obj.Clone(name)
                sums_map[name].SetName(name)
                sums_map[name].SetDirectory(0)
            else:
                sums_map[name].Add(obj)

        sums = {}
        for fname in signal_files:
            path = os.path.join(ROOT_DIR, fname)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing ROOT file: {path}")
            in_file = ROOT.TFile.Open(path, "READ")
            if not in_file or in_file.IsZombie():
                raise RuntimeError(f"Failed to open ROOT file: {path}")
            for key in in_file.GetListOfKeys():
                key_name = key.GetName()
                obj = key.ReadObj()
                if not obj:
                    continue
                if obj.InheritsFrom("TH1"):
                    add_hist(key_name, obj, sums)
                elif obj.InheritsFrom("TDirectory"):
                    for subkey in obj.GetListOfKeys():
                        sub_name = subkey.GetName()
                        sub_obj = subkey.ReadObj()
                        if sub_obj and sub_obj.InheritsFrom("TH1"):
                            add_hist(sub_name, sub_obj, sums)
            in_file.Close()

        out_path = os.path.join(ROOT_DIR, out_name)
        if os.path.exists(out_path):
            os.remove(out_path)
        out_file = ROOT.TFile.Open(out_path, "RECREATE")
        if not out_file or out_file.IsZombie():
            raise RuntimeError(f"Failed to create ROOT file: {out_path}")
        out_file.cd()
        for hist in sums.values():
            hist.Write()
        out_file.Write()
        out_file.Close()

        check_file = ROOT.TFile.Open(out_path, "READ")
        if not check_file or check_file.IsZombie():
            raise RuntimeError(f"Failed to re-open ROOT file: {out_path}")
        written = check_file.GetListOfKeys().GetSize()
        check_file.Close()
        if written != len(sums):
            raise RuntimeError(
                f"Merged ROOT file {out_path} has {written} histograms, expected {len(sums)}"
            )

        for fname in signal_files:
            merged["files"].pop(fname, None)

        merged["files"][out_name] = {
            "cross-section": 1.0 / total_lumi,
            "era": target_era,
            "generated-events": 1.0,
            "group": group_name,
            "legend": legend,
            "line-color": line_color,
            "type": "signal",
        }

    merge_signal_files(
        GGF_PREFIX,
        f"HH_ggf_{target_era}.root",
        "HH_{ggf}^{kappa_lambda=1}->bbWW",
        3,
        "HH_ggf",
    )
    merge_signal_files(
        VBF_PREFIX,
        f"HH_vbf_{target_era}.root",
        "HH_{vbf}->bbWW",
        6,
        "HH_vbf",
    )

    merged.setdefault("groups", {})
    merged["groups"].update({
        "HH_ggf": {
            "legend": "HH_{ggf}^{kappa_lambda=1}->bbWW",
            "line-color": 1,
        },
        "HH_vbf": {
            "legend": "HH_{vbf}->bbWW",
            "line-color": 2,
        },
    })

    DEFAULT_RATIO_RANGE = list(merged.get("plotdefaults", {}).get("ratio-y-axis-range", [0.5, 1.5]))
    for plot_cfg in merged.get("plots", {}).values():
        plot_cfg["era"] = target_era
        plot_cfg["ratio-y-axis-range"] = DEFAULT_RATIO_RANGE
        strip_blinding(plot_cfg)

    global COMBINED_TOTAL_WIDTH
    COMBINED_TOTAL_WIDTH = None
    for root_fname in list(merged.get("files", {}).keys()):
        build_combined_hist(os.path.join(ROOT_DIR, root_fname))

    plots = merged.setdefault("plots", {})
    base_plot_names = tuple(cat_name for _, cat_name in COMBINED_CATEGORIES)
    base_plots = [plots[name] for name in base_plot_names if name in plots]
    if base_plots:
        max_lin = max(cfg.get("y-axis-range", [0.0, 0.0])[1] for cfg in base_plots if "y-axis-range" in cfg)
        max_log = max(cfg.get("log-y-axis-range", [0.0, 0.0])[1] for cfg in base_plots if "log-y-axis-range" in cfg)
        combined_cfg = dict(base_plots[0])
        combined_cfg.pop("blinded-range", None)
        strip_blinding(combined_cfg)
        combined_cfg["x-axis"] = "DL score (resolved1b | resolved2b | boosted)"
        x_max = COMBINED_TOTAL_WIDTH if COMBINED_TOTAL_WIDTH is not None else 3.0
        combined_cfg["x-axis-range"] = [0.0, x_max]
        combined_cfg["show-overflow"] = False
        combined_cfg["x-axis-label-size"] = 0
        combined_cfg["x-axis-hide-ticks"] = True
        combined_cfg["y-axis-range"] = [0.0, max_lin]
        combined_cfg["ratio-y-axis-range"] = [0.5, 1.5]
        combined_cfg["log-y-axis-range"] = [0.01, max_log]
        plots[COMBINED_HIST] = combined_cfg

    BLIND_RANGE = [0.25, 0.999]
    for plot_cfg in merged.get("plots", {}).values():
        plot_cfg["blinded-range"] = BLIND_RANGE

    out_path = f"{PLOTIT_DIR}/plots_{target_era}_combined.yml"
    with open(out_path, "w") as f:
        yaml.safe_dump(merged, f)

    print(
        f"Merged plotIt config written to {out_path} with {len(merged.get('files', {}))} files and {len(merged.get('plots', {}))} plots."
    )


if __name__ == "__main__":
    for target, eras in ERA_GROUPS.items():
        build_merged_output(target, eras)

