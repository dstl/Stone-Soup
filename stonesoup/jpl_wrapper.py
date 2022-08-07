# -*- coding: utf-8 -*-
"""Uses the available API's courtesy of the Jet Propulsion Laboratory on small celestial bodies. Creates easier access to these API's and smoother transitions
into Python and the existing Stone Soup framework.

"""

from .types.orbitalstate import *
import requests
from platform.base import FixedPlatform
from .astronomical_conversions import geodetic_to_cartesian


def get_sb(query: dict):
    r"""
    Sends request to JPL's small body database with query specified in parameters. SBDB wrapper function.
    If covariance is in query, will return a GaussianOrbitalState instead of OrbitalState.

    See https://ssd-api.jpl.nasa.gov/doc/sbdb.html for further details on query parameters.

    Parameters
        ----------
        query: dict
            Query payload. input as python dictionary, {"KEY1" : "VALUE1 , ... , KEYN : VALUEN"}

        Returns
        -------
        OrbitalState
            State based on the orbital elements of the object with additional metadata from the query as a dictionary object.
    """
    if query is None:
        raise TypeError("Query empty.")
    if 'sstr' not in query.keys() or 'spk' not in query.keys() or 'des' not in query.keys():
        raise RuntimeError("Missing one of the following mandatory search fields: \'sstr\', \'spk\', or \'des\'")
    r = requests.get('https://ssd-api.jpl.nasa.gov/sbdb_query.api', params=query).json()
    # initialize empty array for state vector
    idx = [None] * 6
    # parse through orbital elements into keplerian state vector
    for x in r['orbit']['elements']:
        if x['name'] == 'i':
            idx[0] = x['value']
        if x['name'] == 'om':
            idx[1] = x['value']
        if x['name'] == 'e':
            idx[2] = x['value']
        if x['name'] == 'w':
            idx[3] = x['value']
        if x['name'] == 'ma':
            idx[4] = x['value']
        if x['name'] == 'n':
            idx[5] = x['value']
    # check if covariance is specified in query
    if query['covar'] == 'mat':
        return GaussianOrbitalState(state_vector = [x['value'] for x in [r['orbit']['elements'][i] for i in idx]], coordinates = 'TLE')
    else:
        return OrbitalState(state_vector = [x['value'] for x in [r['orbit']['elements'][i] for i in idx]],coordinates = 'TLE',
                        metadata=[r.get(m) for m in ["object","signature"]]
                        covar=r['orbit']['covariance']['data'])



def get_astro_radar(query: dict):
    r"""
    Similar to get_sb but uses the sbdb-radar API, which is a subset of the original with only items that contain radar astronometry. Also contains information
    on the measuring radars. 

    See https://ssd-api.jpl.nasa.gov/doc/sb_radar.html for further details on query parameters.

    Parameters
        ----------
        query: dict
            Query payload. input as python dictionary, {"KEY1" : "VALUE1 , ... , KEYN : VALUEN"}

        Returns
        -------
        OrbitalState
            State based on the orbital elements of the object with additional metadata from the query as a dictionary object.
    """
    if query is None:
        raise TypeError("Query empty.")
    r = requests.get('https://ssd-api.jpl.nasa.gov/sb_radar.api', params=query).json()
    #TODO Add cylindrical coordinates
    if query['coords'] == 'True' :
        coords = [[x["latitude"],x["longitude"]] for x in r["coords"]]
        rdr_stns = [FixedPlatform(state=StateVector(geodetic_to_caresian(x)), position_mapping=(0,1,2) for x in coords]
        return rdr_stns
    else:
        raise RuntimeWarning("Please use the get_sb method. No coordinates were requested in query.")
 
    
