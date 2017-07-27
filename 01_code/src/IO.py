"""Some Import and Export or In and Out helpers."""

import numpy as np
import pandas as pd
from skimage.io import imread

from os import listdir
from os.path import isfile, isdir, join
import glob


def load_grid(dataset):
    """Return ILD/grid data

    Parameters
    ----------
    dataset : str
        'TUB_2015'/ 'ITA_2017' / 'ISF_2013'

    Returns
    -------
    grid : pandas.dataframe

    Example
    -------
    grid = src.IO.load_ild()
    """
    pathRoot = './../02_data/' + dataset + '/'
    pathCsv = pathRoot + 'Csv/'
    folders = [f for f in listdir(pathRoot) if isdir(join(pathRoot, f))]

    if ('Csv' in folders):
        filesCsv = [f for f in listdir(pathCsv) if isfile(join(pathCsv, f))]

        if ('.DS_Store' in filesCsv):
            del filesCsv[0]
        if ('grid.csv' in filesCsv):
            grid = pd.read_csv(pathCsv + 'grid.csv',
                               header=None, names=['azimuth', 'elevation'])

    return grid


def load_HRTF_img(dataset, plane, ear):
    """Load plotted spherical harmonics images.

    Parameters
    ----------
    dataset : str
        'TUB_2015'/ 'ITA_2017' / 'ISF_2013' / 'SADIE' / 'CIPIC'
    plane : str
        'horizontal' / 'median'
    ear : str
        'left' / 'right'

    Returns
    -------
    samples : (40, P) numpy.ndarray
        Pixel values
    imgDim1 : int
    imgDim2 : int
    """

    pathRoot = './../02_data/' + dataset + '/'
    pathImg = pathRoot + 'Img/'

    if ear == 'right':
        pathSide = pathImg + 'Right/'

        if plane == 'horizontal':
            pathPlane = pathSide + 'Horizontal/'
            samples, imgDim1, imgDim2 = read_img(pathPlane)

        if plane == 'median':
            pathPlane = pathSide + 'Median/'
            samples, imgDim1, imgDim2 = read_img(pathPlane)

    if ear == 'left':
        pathSide = pathImg + 'Left/'

        if plane == 'horizontal':
            pathPlane = pathSide + 'Horizontal/'
            samples, imgDim1, imgDim2 = read_img(pathPlane)

        if plane == 'median':
            pathPlane = pathSide + 'Median/'
            samples, imgDim1, imgDim2 = read_img(pathPlane)

    return samples, imgDim1, imgDim2


def read_img(path):
    """Load images to array from directory

    Parameters
    ----------
    path : str

    Returns
    -------
    samples : (40, P) numpy.ndarray
        Pixel values
    imgDim1 : int
    imgDim2 : int
    """

    pathsFull = glob.glob(path + '*.png')
    filesFull = [f for f in listdir(path) if isfile(join(path, f))]

    if ('reference.png' in filesFull) is True:
        del filesFull[0]
        del pathsFull[0]

    img = imread(pathsFull[0], as_grey=True)
    imgDim1, imgDim2 = np.shape(img)
    samples = np.zeros([len(pathsFull), (imgDim1*imgDim2)])

    for i in range(len(pathsFull)):
        # read images
        img = imread(pathsFull[i], as_grey=True)
        img = img.reshape(-1, 1)
        img = np.squeeze(img)
        samples[i, :] = img

    return samples, imgDim1, imgDim2
