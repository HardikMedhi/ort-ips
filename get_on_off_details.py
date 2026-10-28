#!/usr/bin/env python3
"""
Module to extract target source and its corresponding single cold-sky region 
information from an observation schedule FITS file using Astropy via exact RA string matching.
"""

import sys
import argparse
from astropy.table import Table

# Default FITS file path (update as needed)
DEFAULT_FITS_PATH = "ort_mwa_cold_comb_cat.fits"

def load_observation_table(fits_path=DEFAULT_FITS_PATH):
    """
    Loads the observation FITS file into an Astropy Table.
    
    Parameters:
    -----------
    fits_path : str
        Path to the FITS file.
        
    Returns:
    --------
    astropy.table.Table
        The loaded table containing observation data.
    """
    try:
        return Table.read(fits_path, format='fits')
    except Exception as e:
        raise IOError(f"Failed to read FITS file at '{fits_path}': {e}")

def _clean_val(val):
    """Helper to safely clean Astropy column values (bytes to str, quantities to scalar)."""
    if hasattr(val, 'value'):
        val = val.value
    if isinstance(val, bytes):
        val = val.decode('utf-8')
    return str(val).strip()

def get_source_info(table, source_name):
    """
    Extracts target source information (including flux) and its single corresponding 
    cold-sky region by matching the exact RA string.
    
    Parameters:
    -----------
    table : astropy.table.Table
        The observation table.
    source_name : str
        The name of the target source to search for.
        
    Returns:
    --------
    dict
        A structured dictionary containing 'target' details and 'cold_sky' details.
    """
    # Clean table names for comparison
    table_names = [_clean_val(name) for name in table['source_name']]
    target_name_clean = source_name.strip()
    
    # Find target source row index
    mask_target = [name.lower() == target_name_clean.lower() for name in table_names]
    
    if not any(mask_target):
        raise ValueError(f"Source '{source_name}' not found in the FITS table.")
    
    target_idx = mask_target.index(True)
    target_row = table[target_idx]
    
    target_ra = _clean_val(target_row['raj2000'])
    target_dec = _clean_val(target_row['decj2000'])
    target_flux = float(target_row['flux_jy'].value) if hasattr(target_row['flux_jy'], 'value') else float(target_row['flux_jy'])
    target_time = float(target_row['time_min'].value) if hasattr(target_row['time_min'], 'value') else float(target_row['time_min'])
    
    target_info = {
        'source_name': _clean_val(target_row['source_name']),
        'raj2000': target_ra,
        'decj2000': target_dec,
        'flux_jy': target_flux,
        'time_min': target_time
    }
    
    # Find the corresponding cold-sky region: exact same RA string, different row index
    cold_sky_info = None
    for i, row in enumerate(table):
        if i == target_idx:
            continue
        row_ra = _clean_val(row['raj2000'])
        if row_ra == target_ra:
            cs_time = float(row['time_min'].value) if hasattr(row['time_min'], 'value') else float(row['time_min'])
            cold_sky_info = {
                'source_name': _clean_val(row['source_name']),
                'raj2000': row_ra,
                'decj2000': _clean_val(row['decj2000']),
                'time_min': cs_time
            }
            break
            
    return {
        'target': target_info,
        'cold_sky': cold_sky_info
    }

def format_output(info):
    """
    Formats the extracted dictionary into a human-readable string for terminal printing.
    """
    target = info['target']
    cs = info['cold_sky']
    
    lines = [
        "=== Target Source Information ===",
        f"Source Name : {target['source_name']}",
        f"RA (J2000)  : {target['raj2000']}",
        f"Dec (J2000) : {target['decj2000']}",
        f"Flux        : {target['flux_jy']} Jy",
        f"Obs Time    : {target['time_min']} minutes",
        "",
        "=== Cold-Sky Region Information ==="
    ]
    
    if cs:
        lines.extend([
            f"Source Name : {cs['source_name']}",
            f"RA (J2000)  : {cs['raj2000']}",
            f"Dec (J2000) : {cs['decj2000']}",
            f"Obs Time    : {cs['time_min']} minutes"
        ])
    else:
        lines.append("No corresponding cold-sky region found with matching RA string.")
        
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Extract target source and associated cold-sky region details from observation FITS table.")
    parser.add_argument("source_name", type=str, help="Name of the target source to query.")
    parser.add_argument("--fits", type=str, default=DEFAULT_FITS_PATH, help="Path to the observation FITS file.")
    
    args = parser.parse_args()
    
    try:
        table = load_observation_table(args.fits)
        info = get_source_info(table, args.source_name)
        print(format_output(info))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()