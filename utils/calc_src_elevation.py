import argparse
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astroplan import Observer, FixedTarget

import get_tel_info

def main(args:tuple):
    ra, dec, datetime, tel = args

    tel_info = get_tel_info.main(tel)

    elevation, azimuth, is_up = get_elevation(ra, dec, datetime, tel_info)

    print(f"\nSource Coordinates (RA/Dec J2000): {ra} {dec}")
    print(f"Observing datetime: {datetime} | Telescope: {tel}")
    print(f"\nElevation of source = {elevation:.2f}°")
    print(f"Azimuth of source = {azimuth:.2f}°")
    print(f"Is the source up? {is_up}")

def get_args() -> tuple[str, str, str, str]:
    parser = argparse.ArgumentParser(
        description="Calculate the elevation of a source," \
        "given its coordinates, observing date-time and telescope name.",
        prefix_chars="@"
    )

    parser.add_argument("ra", type=str,
                        help="RAJ2000 Coordinate of the source in HMS format.")
    parser.add_argument('dec', type=str,
                        help="DecJ2000 Coordinate of the source in DMS format.")
    parser.add_argument('datetime', type=str,
                        help="Datetime of observation in IST (YYYY-MM-DDTHH:MM:SS format).")
    parser.add_argument('@tel', type=str, default='ort',
                        help="(Optional) Telescope Name. Default is ort. Current options are ort/gmrt only.")

    args = parser.parse_args()

    ra = args.ra
    dec = args.dec
    datetime = args.datetime
    tel = args.tel

    return ra, dec, datetime, tel

def get_elevation(ra:str, dec:str, datetime:str, tel_info:dict) -> tuple[float, float, bool]:
    observer = Observer(
        latitude=tel_info['lat']*u.degree,
        longitude=tel_info['lon']*u.degree,
        elevation=tel_info['height']*u.m
    )

    dt = Time(datetime, format='isot', scale='utc') - 5.5 * u.hour

    coord = SkyCoord(
        ra=ra, dec=dec,
        unit=(u.hour, u.degree),
        frame='icrs'
    )
    target = FixedTarget(coord)

    altaz = observer.altaz(dt, target)
    elevation = altaz.alt.deg
    azimuth = altaz.az.deg

    is_up = observer.target_is_up(dt, target)

    return elevation, azimuth, is_up


if __name__ == "__main__":
    args = get_args()
    main(args)