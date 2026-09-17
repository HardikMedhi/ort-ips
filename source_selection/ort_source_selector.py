import argparse
import re
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
from pathlib import Path
from astropy.table import Table

# Managing Functions

def main(args:tuple):
    """Run the full source-selection workflow for a catalog and pointing location."""
    cat_filepath, n, pointing_ra, pointing_dec, dontsaveplot = args

    tel_info = read_yaml_file()

    cat_data = Table.read(cat_filepath).to_pandas()
    name_col = get_colnames(cat_data)[-1]
    cat_data = cat_data.drop_duplicates(subset=name_col)

    df_infov = get_infov_sources(cat_data, pointing_ra, pointing_dec, tel_info)
    if df_infov.empty:
        print("No sources fall within the module's FOV at the given pointing coordinates.")
        return

    df_selected = select_sources(df_infov, n, tel_info)
    
    print(f"Total sources in module FOV: {len(df_infov)}")
    print("\nSelected Independent Beam Sources:")
    print(df_selected)

    fig, ax = plot(cat_data, df_infov, df_selected, pointing_ra, pointing_dec, tel_info)
    if not dontsaveplot:
        folder_path = Path(__file__).parent / "output"
        folder_path.mkdir(exist_ok=True)

        cat_filename = cat_filepath.stem
        filepath = folder_path / f"{cat_filename}_{pointing_ra:.1f}_{pointing_dec:.1f}_{n}beams.jpeg"

        fig.savefig(filepath, bbox_inches="tight")
        print(f"Plot saved to {filepath}.")
    plt.show()

def get_args() -> tuple[Path, int, float, float, bool]:
    """Parse command-line arguments for the catalog path, pointing, and beam count."""
    parser = argparse.ArgumentParser(
        description="This program filters a catalog of sources to obtain a list of sources that" \
        "fall within a module's FoV and can be observed by N independent beams simultaneously."
    )

    parser.add_argument("-f", "--cat-filepath", required=True, type=Path,
                        help="Filepath to the catalog of sources")
    parser.add_argument("-n", type=int, required=True,
                        help="Integer number of independent synthesized beams")
    parser.add_argument("-ra", "--pointing-ra", dest="ra", type=float, required=True,
                        help="Pointing right ascension in degrees")
    parser.add_argument("-dec", "--pointing-dec", dest="dec", type=float, required=True,
                        help="Pointing declination in degrees")
    parser.add_argument("--dontsaveplot", action="store_true",
                        help="(Optional) If invoked, the plot will not be saved, just displayed.")

    args = parser.parse_args()
    cat_filepath = Path(args.cat_filepath)
    n = int(args.n)
    ra = args.ra
    dec = args.dec
    dont_save_plot = args.dontsaveplot

    return cat_filepath, n, ra, dec, dont_save_plot

# Core Functions

def calculate_beam_fwhm(dec_deg:float, tel_info:dict, is_module:bool=False) -> tuple[float, float] | tuple[np.ndarray, np.ndarray]:
    """Return the RA and Dec beam widths for a given declination and telescope setup."""
    dec_rad = np.radians(np.atleast_1d(dec_deg))
    L_ns = tel_info['len_ns_mod'] if is_module else tel_info['len_ns_total']

    width_dec_deg = np.degrees(tel_info['lam'] / (L_ns * np.cos(dec_rad)))
    width_ra_deg = np.ones_like(dec_rad) * np.degrees(tel_info['lam'] / tel_info['len_ew_total'])

    if np.isscalar(dec_deg):
        return float(width_ra_deg[0]), float(width_dec_deg[0])
    return width_ra_deg, width_dec_deg

def get_infov_sources(df:pd.DataFrame, pointing_ra:float, pointing_dec:float, tel_info:dict) -> pd.DataFrame:
    """Filter catalog sources that lie within the telescope module field of view."""
    ra_col, dec_col, _, _ = get_colnames(df)

    mod_fwhm_ra, mod_fwhm_dec = calculate_beam_fwhm(
        pointing_dec, tel_info, is_module=True
    )

    dra = df[ra_col].values - pointing_ra
    dra = (dra + 180) % 360 - 180
    dra = np.abs(dra * np.cos(np.deg2rad(pointing_dec)))
    ddec = np.abs(df[dec_col].values - pointing_dec)

    mask_fov = (dra <= (mod_fwhm_ra / 2)) & (ddec <= (mod_fwhm_dec / 2))
    df_infov = df[mask_fov].copy().reset_index(drop=True)

    return df_infov

def select_sources(df:pd.DataFrame, num_beams:int, tel_info:dict) -> pd.DataFrame:
    """Choose a non-overlapping subset of bright sources for independent beams."""
    ra_col, dec_col, flux_col, _ = get_colnames(df)

    num_srcs = len(df)
    ras_local = df[ra_col].values
    decs_local = df[dec_col].values

    ras_beam, decs_beam = calculate_beam_fwhm(decs_local, tel_info, is_module=False)

    G = nx.Graph()
    G.add_nodes_from(range(num_srcs))

    for i in range(num_srcs):
        for j in range(i + 1, num_srcs):
            avg_dec = (decs_local[i] + decs_local[j]) / 2.0

            delta_ra = ((ras_local[i] - ras_local[j] + 180.0) % 360.0 - 180.0) * np.cos(np.deg2rad(avg_dec))
            delta_dec = decs_local[i] - decs_local[j]

            collision_ra = (ras_beam[i] + ras_beam[j]) / 2.0
            collision_dec = (decs_beam[i] + decs_beam[j]) / 2.0

            distance_metric = (delta_ra / collision_ra) ** 2 + (delta_dec / collision_dec) ** 2

            if distance_metric < 1.0:
                G.add_edge(i, j)

    # Maximum Independent Set
    mis_indices = list(nx.algorithms.approximation.maximum_independent_set(G))
    mis_df = df.iloc[mis_indices].copy()

    selected_df = (
        mis_df.sort_values(by=flux_col, ascending=False).head(num_beams).copy()
    )

    return selected_df

# Plotting Function

def plot(
        catalogue_df:pd.DataFrame, in_fov_df:pd.DataFrame, selected_df:pd.DataFrame, 
        pointing_ra:float, pointing_dec:float, tel_info:dict
    ) -> tuple:
    """Create a diagnostic sky plot showing the FOV, catalog sources, and selected beams."""
    fig, ax = plt.subplots(figsize=(10, 8))

    ra_col, dec_col, flux_col, name_col = get_colnames(catalogue_df)

    # 1. Background catalog sources
    cos_dec_ptr = np.cos(np.radians(pointing_dec))
    nearby_mask = (
        np.abs((catalogue_df[ra_col] - pointing_ra) * cos_dec_ptr) <= 2.0
    ) & (np.abs(catalogue_df[dec_col] - pointing_dec) <= 2.0)
    nearby_df = catalogue_df[nearby_mask]

    ax.scatter(
        nearby_df[ra_col],
        nearby_df[dec_col],
        c="gray",
        s=20,
        alpha=0.4,
        label="Catalog Sources",
    )

    # 2. In-FOV catalog sources
    if not in_fov_df.empty:
        ax.scatter(
            in_fov_df[ra_col],
            in_fov_df[dec_col],
            c="tab:blue",
            s=30,
            label="Sources in Module FOV",
        )

    # 3. Single Module FOV Bounding Box
    mod_fwhm_ra, mod_fwhm_dec = calculate_beam_fwhm(pointing_dec, tel_info, is_module=True)
    mod_fwhm_ra_sky = mod_fwhm_ra / cos_dec_ptr

    fov_rect = patches.Rectangle(
        xy=(
            pointing_ra - mod_fwhm_ra_sky / 2.0,
            pointing_dec - mod_fwhm_dec / 2.0,
        ),
        width=mod_fwhm_ra_sky,
        height=mod_fwhm_dec,
        linewidth=2.0,
        edgecolor="black",
        facecolor="none",
        linestyle="--",
        label="Module FOV Boundary",
    )
    ax.add_patch(fov_rect)

    # 4. Synthesized Beam Ellipses around selected targets
    if not selected_df.empty:
        ax.scatter(
            selected_df[ra_col],
            selected_df[dec_col],
            c="red",
            s=90,
            marker="*",
            zorder=5,
            label="Selected Independent Beams",
        )

        for _, row in selected_df.iterrows():
            src_ra, src_dec = row[ra_col], row[dec_col]
            b_ra, b_dec = calculate_beam_fwhm(src_dec, tel_info, is_module=False)
            b_ra_sky = b_ra / np.cos(np.radians(src_dec))

            # Draw full synthesized beam ellipse
            ellipse = patches.Ellipse(
                xy=(src_ra, src_dec),
                width=b_ra_sky,
                height=b_dec,
                angle=0.0,
                edgecolor="crimson",
                facecolor="red",
                alpha=0.3,
                linewidth=1.2,
                zorder=4,
            )
            ax.add_patch(ellipse)

            ax.annotate(
                row[name_col],
                (src_ra, src_dec),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
                color="darkred",
                weight="bold",
            )

    # Formatting
    ax.set_xlabel("Right Ascension (deg)")
    ax.set_ylabel("Declination (deg)")
    ax.set_title(
        f"ORT Beam Layout | Pointing: ({pointing_ra:.2f}°, {pointing_dec:.2f}°)"
    )
    ax.invert_xaxis()
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")

    return fig, ax

# Utility Functions

def read_yaml_file() -> dict:
    """Load the telescope configuration dictionary from the YAML metadata file."""
    filepath_tel_info = Path(__file__).parent / "tel_info.yaml"
    with open(filepath_tel_info, "r") as f:
        info = yaml.safe_load(f)
    return info['ort']

def get_colnames(df:pd.DataFrame) -> tuple[str, str, str, str]:
    """Identify the RA, Dec, flux, and source-name columns from a catalog DataFrame."""
    ra_pattern = re.compile(r'^ra(?:[_ -]?j2000)?$', re.IGNORECASE)
    dec_pattern = re.compile(r'^dec(?:[_ -]?j2000)?$', re.IGNORECASE)
    flux_pattern = re.compile(r'^(?:flux|flux[_ -]?jy|s[_ -]?327|int[_ -]?flux)$', re.IGNORECASE)
    name_pattern = re.compile(r'^(?:name|source[_ -]?name|id)$', re.IGNORECASE)

    ra_col = next((col for col in df.columns if ra_pattern.match(str(col))), None)
    dec_col = next((col for col in df.columns if dec_pattern.match(str(col))), None)

    flux_col = next((col for col in df.columns if flux_pattern.match(str(col))), None)
    name_col = next((col for col in df.columns if name_pattern.match(str(col))), None)

    if any(column is None for column in (ra_col, dec_col, flux_col, name_col)):
        raise ValueError(
            f"Could not find all required columns in the catalog.\n"
            f"Available columns: {list(df.columns)}\n"
            f"Expected RA pattern: {ra_pattern.pattern}\n"
            f"Expected Dec pattern: {dec_pattern.pattern}\n"
            f"Expected flux pattern: {flux_pattern.pattern}\n"
            f"Expected name pattern: {name_pattern.pattern}"
        )

    return ra_col, dec_col, flux_col, name_col


if __name__ == "__main__":
    args = get_args()
    main(args)