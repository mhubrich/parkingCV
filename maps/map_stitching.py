'''
Source: GooMPy: Google Maps for Python
https://github.com/simondlevy/GooMPy

This is a modified version of above source.
Changes:
* The Google logo is cut out of the tiles
* skimage instead of PIL for image manipulation
* No `GRABRATE` (Google Maps doesn't seem to have a limitation for that anymore
* Map tiles are fetched without labels (e.g. city or street names)
'''
import os
import sys
import math
import time
import numpy as np
import skimage.io

from skimage.color import rgba2rgb


try:
    from key import _KEY
except:
    _KEY = None

_EARTHPIX = 268435456  # Number of pixels in half the earth's circumference at zoom = 21
_DEGREE_PRECISION = 4  # Number of decimal places for rounding coordinates
_HEIGHT_LOGO = 22      # Height of the Google logo in pixels
_MAX_TILESIZE = 640    # Largest tile we can grab without paying

_TILESIZE_h = _MAX_TILESIZE - 2 * _HEIGHT_LOGO # Largest tile height without logo
_TILESIZE_w = _MAX_TILESIZE                    # Largest tile width

_pixrad = _EARTHPIX / math.pi


def _roundto(value, digits):
    return int(value * 10**digits) / 10.**digits


def _pixels_to_degrees(pixels, zoom):
    return pixels * 2 ** (21 - zoom)


def _grab_tile(lat, lon, zoom, scale, maptype, _TILESIZE_h, _TILESIZE_w):
    urlbase = ('https://maps.googleapis.com/maps/api/staticmap?'
               'style=feature:all|element:labels|visibility:off'
               '&center=%f,%f&zoom=%d&scale=%d&maptype=%s&size=%dx%d')
    if _KEY:
        urlbase += '&key=' + _KEY

        specs = lat, lon, zoom, scale, maptype, _TILESIZE_w

    filename = 'mapscache/' + ('%f_%f_%d_%d_%s_%d_%d' % (specs + (_TILESIZE_h,)))

    if maptype == 'roadmap':
        filename += '.png'
    else:
        filename += '.jpg'

    if os.path.isfile(filename):
        tile = skimage.io.imread(filename)
    else:
        url = urlbase % (specs + (_TILESIZE_w,))
        tile = skimage.io.imread(url)
        # Some tiles are in mode `RGBA` and need to be converted
        if tile.shape[-1] == 4:
            tile = rgba2rgb(tile)
        tile = tile[_HEIGHT_LOGO:-_HEIGHT_LOGO,:]
        if not os.path.exists('mapscache'):
            os.mkdir('mapscache')
        skimage.io.imsave(filename, tile)

    return tile


def _pix_to_lon(j, lonpix, ntiles, _TILESIZE_w, zoom):
    return math.degrees((lonpix + _pixels_to_degrees(((j)-ntiles/2)*_TILESIZE_w, zoom) - _EARTHPIX) / _pixrad)

def _pix_to_lat(k, latpix, ntiles, _TILESIZE_h, zoom):
    return math.degrees(math.pi/2 - 2 * math.atan(math.exp(((latpix + _pixels_to_degrees((k-ntiles/2)*_TILESIZE_h, zoom)) - _EARTHPIX) / _pixrad)))

def fetchTiles(latitude, longitude, zoom, scale, maptype, radius_meters=None, default_ntiles=4):
    '''
    Fetches tiles from GoogleMaps at the specified coordinates, zoom level (0-22), and map type ('roadmap',
    'terrain', 'satellite', or 'hybrid').  The value of radius_meters deteremines the number of tiles that will be
    fetched; if it is unspecified, the number defaults to default_ntiles.  Tiles are stored as JPEG images
    in the mapscache folder.
    '''
    if scale != 1:
        raise ValueError('Currently, only `scale=1` supported.')

    latitude = _roundto(latitude, _DEGREE_PRECISION)
    longitude = _roundto(longitude, _DEGREE_PRECISION)

    # https://groups.google.com/forum/#!topic/google-maps-js-api-v3/hDRO4oHVSeM
    pixels_per_meter = 2**zoom / (156543.03392 * math.cos(math.radians(latitude)))

    # number of tiles required to go from center latitude to desired radius in meters
    ntiles = default_ntiles if radius_meters is None else int(round(2 * pixels_per_meter / (_TILESIZE_w /2./ radius_meters)))

    lonpix = _EARTHPIX + longitude * math.radians(_pixrad)

    sinlat = math.sin(math.radians(latitude))
    latpix = _EARTHPIX - _pixrad * math.log((1 + sinlat)/(1 - sinlat)) / 2

    bigsize_h = ntiles * _TILESIZE_h
    bigsize_w = ntiles * _TILESIZE_w
    bigimage = np.zeros((bigsize_h, bigsize_w, 3), dtype=np.uint8)

    for j in range(ntiles):
        lon = _pix_to_lon(j, lonpix, ntiles, _TILESIZE_w, zoom)
        for k in range(ntiles):
            lat = _pix_to_lat(k, latpix, ntiles, _TILESIZE_h, zoom)
            tile = _grab_tile(lat, lon, zoom, scale, maptype, _TILESIZE_h, _TILESIZE_w)
            bigimage[k*_TILESIZE_h:(k+1)*_TILESIZE_h, j*_TILESIZE_w:(j+1)*_TILESIZE_w] = tile

    return bigimage
