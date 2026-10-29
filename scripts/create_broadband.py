import numpy as np
import scipy.ndimage as ndimage
from scipy.optimize import curve_fit

import matplotlib
from matplotlib import pyplot as plt

from PyAstronomy import pyasl
import pyphot
from pyphot import (unit, Filter)
from reproject import reproject_exact, reproject_adaptive, reproject_interp, reproject_from_healpix

from astropy.io import (fits, ascii)
from astropy import wcs
import astropy.units as u
from astropy.visualization import SqrtStretch, PercentileInterval

from photutils import aperture_photometry, RectangularAperture
from photutils import ApertureStats

import os
import csv
import glob


def produce_broadband(data, head, filename, filter_name):
    
    cube_spec = data
    size      = [head['NAXIS2'], head['NAXIS1']]
    wave_spec = head['CRVAL3'] + np.arange(head['NAXIS3']) *head['CD3_3']

    #####broad-band r-band
    img_bband = np.zeros([size[0], size[1]])
    
    for id_x in np.arange(size[1]-1):   
        for id_y in np.arange(size[0]-1):
            flux_spec  = cube_spec[:, id_y, id_x]

            wv = wave_spec*unit['AA']
            fl = flux_spec*unit['erg/s/cm**2/AA']
            flux_syn  = filter_name.get_flux(wv, fl) #erg/s/cm**2/AA
            
            img_bband[id_y, id_x] = flux_syn   #erg/s/cm**2/AA
    
    hdu = fits.PrimaryHDU(header=head, data=img_bband)
    hdu.writeto(filename, overwrite=True)      
    return img_bband




def filter_(survey):
    img_dir = '/media/yong-group/DGIS1/MUSE/MUSE_reduced_LX/3_MUSE-fluxcal/broad_band_image/'
    if survey   ==  'DESI':
        filt_dir = img_dir+'CTIO_DECam_filtercurve/'
        
        filcont  = ascii.read(filt_dir+'CTIO_DECam.g.dat')
        wave     = np.array(filcont['col1'])*unit['AA']
        transmit = np.array(filcont['col2'])
        filter_1 = Filter(wave, transmit, name='CT_g', dtype='energy', unit='Angstrom')

        filcont  = ascii.read(filt_dir+'CTIO_DECam.r.dat') 
        wave     = np.array(filcont['col1'])*unit['AA']
        transmit = np.array(filcont['col2'])
        filter_2 = Filter(wave, transmit, name='CT_r', dtype='energy', unit='Angstrom')

        filcont  = ascii.read(filt_dir+'CTIO_DECam.i.dat') 
        wave     = np.array(filcont['col1'])*unit['AA']
        transmit = np.array(filcont['col2'])
        filter_3 = Filter(wave, transmit, name='CT_i', dtype='energy', unit='Angstrom')

        
    elif survey == 'PanStarrs':
        filt_dir = img_dir+'PanStarrs_filtercurve/'
        
        filcont  = ascii.read(filt_dir+'PAN-STARRS_PS1.g.dat') 
        wave     = np.array(filcont['col1'])*unit['AA']
        transmit = np.array(filcont['col2'])
        filter_1 = Filter(wave, transmit, name='PS_g', dtype='energy', unit='Angstrom')


        filcont  = ascii.read(filt_dir+'PAN-STARRS_PS1.r.dat')
        wave     = np.array(filcont['col1'])*unit['AA']
        transmit = np.array(filcont['col2'])
        filter_2 = Filter(wave, transmit, name='PS_r', dtype='energy', unit='Angstrom')

        filcont  = ascii.read(filt_dir+'PAN-STARRS_PS1.i.dat') 
        wave     = np.array(filcont['col1'])*unit['AA']
        transmit = np.array(filcont['col2'])
        filter_3 = Filter(wave, transmit, name='PS_i', dtype='energy', unit='Angstrom')

    else:
        filt_dir = img_dir+'SkyMapper_filtercurve/'
        
        filcont  = ascii.read(filt_dir+'SkyMapper_SkyMapper.g.dat') 
        wave     = np.array(filcont['col1'])*unit['AA']
        transmit = np.array(filcont['col2'])
        filter_1 = Filter(wave, transmit, name='SM_g', dtype='energy', unit='Angstrom')


        filcont  = ascii.read(filt_dir+'SkyMapper_SkyMapper.r.dat') 
        wave     = np.array(filcont['col1'])*unit['AA']
        transmit = np.array(filcont['col2'])
        filter_2 = Filter(wave, transmit, name='SM_r', dtype='energy', unit='Angstrom')

        filcont  = ascii.read(filt_dir+'SkyMapper_SkyMapper.i.dat') 
        wave     = np.array(filcont['col1'])*unit['AA']
        transmit = np.array(filcont['col2'])
        filter_3 = Filter(wave, transmit, name='SM_i', dtype='energy', unit='Angstrom')
        
    return filter_1,filter_2,filter_3
    
        

def create_broadband(file,survey,save_dir,band='all'):

        filter_all = filter_(survey)
        
        hdu     = fits.open(file)
        head    = hdu[1].header
        data    = hdu[1].data*10**(-20)

        num_xspa   = head['NAXIS1']
        num_yspa   = head['NAXIS2']
        num_spec   = head['NAXIS3']

        wave_spec  = head['CRVAL3'] + np.arange(head['NAXIS3']) *head['CD3_3']

        date =  file.split('FINAL_')[1].split('.')[0][2:-3]
         
        if band == 'all':
            file_name = save_dir+filter_all[0].name+'_'+date+'.fits'
            img_syn_1 = produce_broadband(data, head, file_name, filter_all[0])
            file_name = save_dir+filter_all[1].name+'_'+date+'.fits'
            img_syn_2 = produce_broadband(data, head, file_name, filter_all[1])
            file_name = save_dir+filter_all[2].name+'_'+date+'.fits'
            img_syn_3 = produce_broadband(data, head, file_name, filter_all[2])
            
        else:
            i = int(band)
            file_name = save_dir+filter_all[i].name+'_'+date+'.fits'
            img_syn   = produce_broadband(data, head, file_name, filter_all[i])

        return
    
    
    
    
    
    
    
    
    
    
    