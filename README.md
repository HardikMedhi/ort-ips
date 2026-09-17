# ORT IPS Observation Support Code

This repository hosts code to support Interplanetary Scintillation (IPS) observations with the Ooty Radio Telescope (ORT). It is intended to assist with practical planning and preparation for ORT IPS observing work.

The project will continue to grow over time and may include multiple tasks related to ORT IPS operations, planning, and supporting workflows. At present, one of the included components is source-selection support for candidate target preparation.

## Repository layout

- `source_selection/` — tools for selecting suitable ORT targets based on telescope geometry, pointing, and beam constraints.
- `tel_info.yaml` — telescope configuration and beam metadata used by the source-selection routines.
- `requirements.txt` — Python dependencies for the project.
- generated output plots — saved under the relevant script directory when plotting is enabled.

## Source selection module

The `source_selection` component prepares candidate source lists for ORT IPS observations. It reads a source catalog and identifies sources that:

- fall within the module field of view at a given pointing,
- are compatible with the requested number of independent beams,
- and are selected to minimize beam overlap while prioritizing the strongest sources.

The source-selection step constructs a graph in which sources that overlap in beam footprint are treated as adjacent. A maximum independent set from graph theory is then computed to find a set of mutually non-overlapping sources, and the strongest among them are ranked by flux for the final selection.

The current script also removes duplicate source entries using the source name column before selection, and optionally saves both a plot and a CSV file of the chosen targets.

## Operating `ort_source_selector.py`

The main entry point is `source_selection/ort_source_selector.py`.

### Required arguments

- `-f` or `--cat-filepath`: path to the source catalog file
- `-n`: number of independent beams to select
- `-ra` or `--pointing-ra`: right ascension of the telescope pointing in degrees
- `-dec` or `--pointing-dec`: declination of the telescope pointing in degrees

### Example

```bash
python source_selection/ort_source_selector.py -f ../ort_cat.fits -n 10 -ra 0.8 -dec -14
```

This command will:

1. read the catalog from `../ort_cat.fits`,
2. filter sources within the relevant ORT module field of view,
3. select up to 10 non-overlapping independent beam targets,
4. print the selected sources, and
5. generate a figure showing the pointing geometry and chosen sources unless plotting is disabled.

---

Author: Hardik Medhi
