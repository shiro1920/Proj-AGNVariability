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

img_dir = '/media/yong-group/DGIS1/MUSE/MUSE_reduced_LX/3_MUSE-fluxcal/broad_band_image/'
color_t  = np.array(['r','b','g','darkorange','cyan','m','gold','brown','purple','darkgoldenrod',\
                    'hotpink','grey','mediumpurple','lime','y','darkred','darkcyan','olive','sandybrown',\
                    'deepskyblue','palevioletred','slategrey','mediumvioletred','steelblue'])

def read_ds9reg(filename, head):
    import warnings
    warnings.filterwarnings("ignore")

    csv_file   = open(filename) 
    rows       = list( csv.reader(csv_file) )
    rows       = rows[3]
    ra_cen     = np.float(rows[0][4:]) #deg
    dec_cen    = np.float(rows[1])     #deg
    width      = np.abs(np.float(rows[2][:-1])) #arcsec
    height     = np.abs(np.float(rows[3][:-1])) #arcsec
    angle      = np.float(rows[4][:-1])
    ###convert to pixels
    w       = wcs.WCS(head, naxis=2)
    if 'CDELT1' in head:
        arc_pix = np.abs(head['CDELT1'])*3600 #arcsec/pixel
    if 'CD1_1' in head:
        arc_pix = np.abs(head['CD1_1'])*3600 #arcsec/pixel
        
    
    pos     = np.array([ra_cen, dec_cen])
    pos     = w.wcs_world2pix([pos], 0)
    xc      = pos[0,0]
    yc      = pos[0,1]
    width   = width/arc_pix
    height  = height/arc_pix
    PA      = (angle+180.0)/180.0*np.pi 
    
    return xc, yc, width, height, PA

def DESI_broadband(head_bb,img_bb,aper_dir,filter):
        
    xc, yc, width, height, PA = read_ds9reg(aper_dir+'ds9_source.reg', head_bb)
    xc_sky, yc_sky, width_sky, height_sky, PA_sky = read_ds9reg(aper_dir+'ds9_sky.reg', head_bb)
    
    aperture_sky = RectangularAperture([xc_sky, yc_sky], width_sky, height_sky, theta=PA_sky)
    sky  = np.float(ApertureStats(img_bb, aperture_sky).median)
    
    aperture   = RectangularAperture([xc, yc], width, height, theta=PA)    
    phot_table = aperture_photometry(img_bb-sky, aperture)
    phot_bb    = np.float(phot_table['aperture_sum'])     
    flux       = phot_bb*3.6307805477010325e-29*3e18/(filter.leff.value**2)
    
    return flux 
    
def PanStarrs_broadband(file, aper_dir, filter):
    hdu        = fits.open(file)
    head_bb    = hdu[0].header
    img_bb     = hdu[0].data
    
    xc, yc, width, height, PA = read_ds9reg(aper_dir+'ds9_source.reg', head_bb)
    xc_sky, yc_sky, width_sky, height_sky, PA_sky = read_ds9reg(aper_dir+'ds9_sky.reg', head_bb)
    
    aperture_sky = RectangularAperture([xc_sky, yc_sky], width_sky, height_sky, theta=PA_sky)
    sky  = np.float(ApertureStats(img_bb, aperture_sky).median)
    
    aperture   = RectangularAperture([xc, yc], width, height, theta=PA)
    
    phot_table = aperture_photometry(img_bb-sky, aperture)
    phot_bb    = np.float(phot_table['aperture_sum'])
    EXPTIME    = head_bb['EXPTIME']
    ABmag      = -2.5*np.log10(phot_bb)+25+2.5*np.log10(EXPTIME)
    flux       = 10**(-ABmag/2.5)*3630.0*1.0e-23*3.0e18/filter.leff.value**2    

    return flux
    
def SkyMapper_broadband(file, aper_dir, filter):
    hdu        = fits.open(file)
    head_bb    = hdu[0].header
    img_bb     = hdu[0].data
    
    xc, yc, width, height, PA = read_ds9reg(aper_dir+'ds9_source.reg', head_bb)
    xc_sky, yc_sky, width_sky, height_sky, PA_sky = read_ds9reg(aper_dir+'ds9_sky.reg', head_bb)
    
    aperture_sky = RectangularAperture([xc_sky, yc_sky], width_sky, height_sky, theta=PA_sky)
    sky  = np.float(ApertureStats(img_bb, aperture_sky).median)
    
    aperture   = RectangularAperture([xc, yc], width, height, theta=PA)
    
    phot_table = aperture_photometry(img_bb-sky, aperture)
    phot_bb    = np.float(phot_table['aperture_sum'])
    ZPAPPROX   = head_bb['ZPAPPROX']
    ABmag      = ZPAPPROX - 2.5*np.log10(phot_bb)
    flux       = 10**(-ABmag/2.5)*3630.0*1.0e-23*3.0e18/filter.leff.value**2    

    return flux 

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

def survey_broadband(objname,survey,aper_dir):
    filt_0,filt_1,filt_2 = filter_(survey)
    
    if survey  == 'DESI':
        img_file= glob.glob(img_dir+objname+'/DESI/cutout*')[0]        
        hdu    = fits.open(img_file)
        header = hdu[0].header
        img    = hdu[0].data
        img_0  = img[0]
        img_1  = img[1]
        img_2  = img[2] 
        phot_0 = DESI_broadband(header,img_0,aper_dir,filt_0)
        phot_1 = DESI_broadband(header,img_1,aper_dir,filt_1)
        phot_2 = DESI_broadband(header,img_2,aper_dir,filt_2)
        
    elif survey== 'PanStarrs':        
        PS_g_file = glob.glob(img_dir+objname+'/PanStarrs/*stk.g*fits')[0]
        PS_r_file = glob.glob(img_dir+objname+'/PanStarrs/*stk.r*fits')[0]
        PS_i_file = glob.glob(img_dir+objname+'/PanStarrs/*stk.i*fits')[0]
        
        phot_0 = PanStarrs_broadband(PS_g_file, aper_dir, filt_0)
        phot_1 = PanStarrs_broadband(PS_r_file, aper_dir, filt_1)
        phot_2 = PanStarrs_broadband(PS_i_file, aper_dir, filt_2)
        
        
    elif survey== 'SkyMapper':
        SM_g_file = glob.glob(img_dir+objname+'/SkyMapper/SkyMapper_g_*fits')[0]
        SM_r_file = glob.glob(img_dir+objname+'/SkyMapper/SkyMapper_r_*fits')[0]
        SM_i_file = glob.glob(img_dir+objname+'/SkyMapper/SkyMapper_i_*fits')[0]
        
        phot_0 = SkyMapper_broadband(SM_g_file,aper_dir,filt_0)
        phot_1 = SkyMapper_broadband(SM_r_file,aper_dir,filt_1)
        phot_2 = SkyMapper_broadband(SM_i_file,aper_dir,filt_2)
        
    return phot_0,phot_1,phot_2 

def MUSE_broadband(syn_0,syn_1,syn_2,aper_dir):
 
    img_syn_0  = syn_0[0].data
    img_syn_1  = syn_1[0].data            
    img_syn_2  = syn_2[0].data

    head_syn   = syn_0[0].header

    xc, yc, width, height, PA = read_ds9reg(aper_dir+'ds9_source.reg', head_syn)
    xc_sky, yc_sky, width_sky, height_sky, PA_sky = read_ds9reg(aper_dir+'ds9_sky.reg',head_syn)
    aperture     = RectangularAperture([xc,yc],width, height, theta=PA)
    aperture_sky = RectangularAperture([xc_sky,yc_sky],width_sky, height_sky, theta=PA_sky)

    sky  = np.float(ApertureStats(img_syn_0, aperture_sky).median)
    phot_syn_0 = np.float(ApertureStats(img_syn_0-sky, aperture).sum )

    sky  = np.float(ApertureStats(img_syn_1, aperture_sky).median)
    phot_syn_1 = np.float(ApertureStats(img_syn_1-sky, aperture).sum )

    sky  = np.float(ApertureStats(img_syn_2, aperture_sky).median)
    phot_syn_2 = np.float(ApertureStats(img_syn_2-sky, aperture).sum )            

    return phot_syn_0, phot_syn_1, phot_syn_2
        
    
    
def check_phot_aperture(objname,survey,aper_dir,band):
    img_dir = '/media/yong-group/DGIS1/MUSE/MUSE_reduced_LX/3_MUSE-fluxcal/broad_band_image/'
    if band == 'g':
        i = 0 
    elif band == 'r':
        i = 1
    else:
        i = 2
        
    if survey  == 'DESI':
        img_file= glob.glob(img_dir+objname+'/DESI/cutout*')[0]        
        hdu     = fits.open(img_file)
        header  = hdu[0].header
        img     = hdu[0].data[i]
        
    elif survey== 'PanStarrs': 
        files  = glob.glob(img_dir+objname+'/PanStarrs/*stk.'+band+'*fits')[0]
        hdu    = fits.open(files)
        img    = hdu[0].data 
        header = hdu[0].header
        
    else:
        files  = glob.glob(img_dir+objname+'/SkyMapper/SkyMapper_'+band+'_*fits')[0]
        hdu    = fits.open(files)
        img    = hdu[0].data 
        header = hdu[0].header
     
    
    transform = SqrtStretch() + PercentileInterval(90.)
    fig ,axs  = plt.subplots(3,3,figsize=(30, 30),dpi=100)
    matplotlib.rc('xtick', labelsize=12)
    matplotlib.rc('ytick', labelsize=12)

    xc, yc, width, height, PA = read_ds9reg(aper_dir+'ds9_source.reg', header)
    xc_sky, yc_sky, width_sky, height_sky, PA_sky = read_ds9reg(aper_dir+'ds9_sky.reg', header)

    aperture     = RectangularAperture([xc, yc], width, height, theta=PA)
    aperture_sky = RectangularAperture([xc_sky, yc_sky], width_sky, height_sky, theta=PA_sky)

    axs[0][0].imshow(transform(img),origin='lower')
    aperture.plot(axs[0][0],color='red',lw=2)
    aperture_sky.plot(axs[0][0],color='yellow',lw=2)
           
    single_files  = glob.glob(img_dir+objname+'/'+survey+'/single_observation/*_'+band+'*')
    
    sin_aper_dir  = aper_dir
    for i,file in enumerate(single_files):
        if i < 8:
            sin_hdu   = fits.open(file)
            sin_header= sin_hdu[0].header
            sin_img   = sin_hdu[0].data 

            xc, yc, width, height, PA = read_ds9reg(sin_aper_dir+'ds9_source.reg', sin_header)
            xc_sky,yc_sky,width_sky,height_sky, PA_sky = read_ds9reg(sin_aper_dir+'ds9_sky.reg',sin_header)

            sin_aperture     = RectangularAperture([xc, yc], width, height, theta=PA)
            sin_aperture_sky = RectangularAperture([xc_sky, yc_sky], width_sky, height_sky, theta=PA_sky)

            j  = i+1
            jj = int(j//3)
            ii = int(j%3)
            axs[jj][ii].imshow(transform(sin_img),origin='lower')
            sin_aperture.plot(axs[jj][ii],color='red',lw=2)
            sin_aperture_sky.plot(axs[jj][ii],color='yellow',lw=2)
    plt.show()

def compare_photometry(survey,objname,phot_aper_dir,spec_dir,file_dir,refdate=None,det_date=[None]):
    filter_0,filter_1,filter_2 = filter_(survey)
    filter_wave = [filter_0.leff.value,filter_1.leff.value,filter_2.leff.value]
    phot_survey = np.array(survey_broadband(objname,survey,phot_aper_dir))
    
    fig  = plt.figure(figsize=(8, 4),dpi=100)
    matplotlib.rc('xtick', labelsize=10)
    matplotlib.rc('ytick', labelsize=10)           
    ax   = fig.add_subplot(111)
    
    files= glob.glob(spec_dir+'*DATACUBE*')
    files= np.sort(files)
    
    for f,file in enumerate(files):
        date = file.split('FINAL_')[1].split('.')[0][2:-3]
        if date not in det_date:
            syn_0_file = glob.glob(file_dir+'/*_g_'+date+'.fits')[0]
            syn_1_file = glob.glob(file_dir+'/*_r_'+date+'.fits')[0]
            syn_2_file = glob.glob(file_dir+'/*_i_'+date+'.fits')[0]
            syn_0 = fits.open(syn_0_file)
            syn_1 = fits.open(syn_1_file)
            syn_2 = fits.open(syn_2_file)
            phot_MUSE = np.array(MUSE_broadband(syn_0,syn_1,syn_2,phot_aper_dir))

            if date==refdate:
                color_f = 'black'
            else:
                color_f = color_t[f]
            ax.plot(filter_wave,phot_MUSE/phot_survey ,'o:',ms=7,alpha=0.8,color=color_f,label=date)
   
    ax.legend( loc='best',ncol=3,fontsize = 10 )    
    wave = np.linspace(filter_wave[0],filter_wave[-1],1000)       
    ax.plot(wave, np.ones(len(wave)), '--', color='black')
    ax.set_ylim(0,2)
    
    plt.show()
        
        
def compare_spec(spec_dir,spec_aper_dir,refdate=None,det_date=[None],ylim=None,save_dir=None,plot_ratio=False,w_bin=8800,\
                f_bin = None,i_bin=None): 
    
    files= glob.glob(spec_dir+'*DATACUBE*')
    files= np.sort(files)
    
    fig  = plt.figure(figsize=(14, 24),dpi=100)
    matplotlib.rc('xtick', labelsize=12)
    matplotlib.rc('ytick', labelsize=12)
    axs1 = fig.add_subplot(611)
    axs2 = fig.add_subplot(612)
    axs3 = fig.add_subplot(613)

    
    files= glob.glob(spec_dir+'*DATACUBE*')
    files= np.sort(files)
    
    if refdate:        
        ref_file = glob.glob(spec_dir+'*DATACUBE*'+refdate+'*')[0]
        ref_hdu  = fits.open(ref_file)
        ref_spec = ref_hdu[1].data
        ref_head = ref_hdu[1].header
        ref_wave = ref_head['CRVAL3'] + np.arange(ref_head['NAXIS3']) *ref_head['CD3_3']
        ref_wave = ref_wave
        
        xc,yc,height,width,PA=read_ds9reg(spec_aper_dir+'ds9_source.reg',ref_head)
        xc_sky,yc_sky,height_sky,width_sky,PA_sky = read_ds9reg(spec_aper_dir+'ds9_sky.reg',ref_head)
        
        ref_syn_spec_sky = np.nanmedian(np.nanmedian(ref_spec[:,int(yc_sky-height_sky/2):int(yc_sky+height_sky/2),int(xc_sky-width_sky/2):int(xc_sky+width_sky/2)],axis=1),axis=1)
        ref_syn_spec = np.nanmedian(np.nanmedian(ref_spec[:,int(yc-height/2):int(yc+height/2),int(xc-width/2):int(xc+width/2)],axis=1),axis=1)
        ref_syn_spec = ref_syn_spec-ref_syn_spec_sky        

        num,wave_bin = np.histogram(ref_wave,bins=10)
        wave_bin[-2] = w_bin
        
        ref_bin_spec = np.full(len(wave_bin)-1,np.nan)
        ref_bin_wave = np.full(len(wave_bin)-1,np.nan)
        for w in np.arange(len(wave_bin)-1):
            ref_bin_spec[w] = np.nanmedian(ref_syn_spec[(~(ref_wave<wave_bin[w]))&(ref_wave<wave_bin[w+1])])
            ref_bin_wave[w] = np.nanmedian(ref_wave[(~(ref_wave<wave_bin[w]))&(ref_wave<wave_bin[w+1])])
            
        axs1.plot(ref_wave,ref_syn_spec,color='black',label = refdate)
        axs2.plot(ref_wave,ref_syn_spec,color='black',label = refdate)
        axs3.plot(ref_bin_wave,ref_bin_spec,color='black',label = refdate)        
        if plot_ratio:    
            axs4 = fig.add_subplot(614)
            axs5 = fig.add_subplot(615)
            axs6 = fig.add_subplot(616)
            axs4.plot(ref_wave,ref_syn_spec,color='black',label = refdate)
            axs5.plot(ref_wave,ref_syn_spec,color='black',label = refdate)
        
    
    if save_dir:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    
    bin_all  = []
    for f,file in enumerate(files):
        hdu  = fits.open(file)
        spec = hdu[1].data
        head = hdu[1].header
        var  = hdu[2].data
        err  = np.sqrt(var)

        wave = head['CRVAL3'] + np.arange(head['NAXIS3']) *head['CD3_3']
        
        date = file.split('FINAL_')[1].split('.')[0][2:-3]
        if date not in det_date:
            xc,yc,height,width,PA=read_ds9reg(spec_aper_dir+'ds9_source.reg',head)
            xc_sky,yc_sky,height_sky,width_sky,PA_sky = read_ds9reg(spec_aper_dir+'ds9_sky.reg',head)

            syn_spec_sky = np.nanmedian(np.nanmedian(spec[:,int(yc_sky-height_sky/2):int(yc_sky+height_sky/2),int(xc_sky-width_sky/2):int(xc_sky+width_sky/2)],axis=1),axis=1)
            syn_spec = np.nanmedian(np.nanmedian(spec[:,int(yc-height/2):int(yc+height/2),int(xc-width/2):int(xc+width/2)],axis=1),axis=1)
            syn_spec = syn_spec-syn_spec_sky

            num,wave_bin = np.histogram(wave,bins=10)
            
            wave_bin[-2] = w_bin

            bin_spec = np.full(len(wave_bin)-1,np.nan)
            bin_wave = np.full(len(wave_bin)-1,np.nan)   
            for w in np.arange(len(wave_bin)-1):
                selec = (~(wave<wave_bin[w]))&(wave<wave_bin[w+1])
                bin_spec[w] = np.nanmedian(syn_spec[selec])
                bin_wave[w] = np.nanmedian(wave[selec])
                
            if date!=refdate:
                axs1.plot(wave, syn_spec,color=color_t[f],alpha=0.6,label=date)
                axs2.plot(wave, syn_spec+6*(f%4+1),color=color_t[f],alpha=0.6,label=date)
                axs3.plot(bin_wave,bin_spec,color=color_t[f],label=date)  
            bin_all.append(bin_spec)
            
            if refdate and plot_ratio:
                ratio = ref_bin_spec/bin_spec

                if np.all(~np.isnan(ratio)):
                    p30 = np.array([1,1,1.2,1])
                    p3_fit,p3_cov = curve_fit(tripoly,bin_wave[i_bin:f_bin],ratio[i_bin:f_bin],p30)

                    new_syn_spec  = np.copy(syn_spec)
                    factor= tripoly(wave,*p3_fit)
                    new_syn_spec  = new_syn_spec*factor
                    
                    if save_dir:
                        new_spec = np.copy(spec)
                        new_var  = np.copy(var)
                        nam_y,nam_x = spec.shape[1],spec.shape[2]
                        for jj in np.arange(nam_y):
                            for ii in np.arange(nam_x):
                                new_spec[:,jj,ii] = spec[:,jj,ii]*factor
                                new_var[:,jj,ii]  = (err[:,jj,ii]*factor)**2

                        hdu_prim  = fits.PrimaryHDU(data=hdu[0].data,header=hdu[0].header)
                        hdu_data  = fits.ImageHDU(data=new_spec, header=hdu[1].header)
                        hdu_var   = fits.ImageHDU(data=new_var , header=hdu[2].header)
                        hdulist   = fits.HDUList([hdu_prim, hdu_data,hdu_var])
                        
                        
                        name = file.split(spec_dir)[1]
                        hdulist.writeto(save_dir+name, overwrite=True)

                    
                    if date!=refdate:
                        axs4.plot(wave, new_syn_spec,color=color_t[f],alpha=0.6,label = date)
                        axs5.plot(wave, new_syn_spec+6*(f%4+1), alpha=0.8 , color=color_t[f],label=date)
        
                        axs6.plot(bin_wave,ratio,'o', color=color_t[f],label = date)
                        axs6.plot(wave,tripoly(wave,*p3_fit),'-.',color=color_t[f])
                    
                    
            if (f>1)&(f%4==0):
                axs1.legend( loc='best',ncol=3,fontsize = 10 )
                axs2.legend( loc='best',ncol=3,fontsize = 10 )
                axs3.legend( loc='best',ncol=3,fontsize = 10 )
                    
                if refdate:
                    axs1.set_ylim(np.nanmin(ref_bin_spec)/2,np.nanmax(ref_bin_spec)*3)
                    axs2.set_ylim(np.nanmin(ref_bin_spec)/2,np.nanmax(ref_bin_spec)*3+20)                    
                    if plot_ratio:
                        axs4.legend( loc='best',ncol=3,fontsize = 10 )
                        axs5.legend( loc='best',ncol=3,fontsize = 10 )
                        axs6.legend( loc='best',ncol=3,fontsize = 10 )
                        axs4.set_ylim(np.nanmin(ref_bin_spec)/2,np.nanmax(ref_bin_spec)*3)
                        axs5.set_ylim(np.nanmin(ref_bin_spec)/2,np.nanmax(ref_bin_spec)*3+20)
                    
                else:
                    axs1.set_ylim(np.nanmin(bin_all)/3,np.nanmax(bin_all)*3)
                    axs2.set_ylim(np.nanmin(bin_all)/3,np.nanmax(bin_all)*3+20)
                   
                if ylim:
                    axs1.set_ylim(ylim[0],ylim[1])
                    axs2.set_ylim(ylim[0],ylim[1]+20)                    
                    
                plt.show()        
                print('**********************************************************************')
                print('**********************************************************************')            

                fig  = plt.figure(figsize=(14, 24),dpi=100)
                matplotlib.rc('xtick', labelsize=12)
                matplotlib.rc('ytick', labelsize=12)
                axs1 = fig.add_subplot(611)
                axs2 = fig.add_subplot(612)
                axs3 = fig.add_subplot(613)
                
                bin_all = []
                
                if refdate:
                    axs1.plot(ref_wave,ref_syn_spec,color='black',label = refdate)
                    axs2.plot(ref_wave,ref_syn_spec,color='black',label = refdate)
                    axs3.plot(ref_bin_wave,ref_bin_spec,color='black',label = refdate)                    
                    if plot_ratio:
                        axs4 = fig.add_subplot(614)
                        axs5 = fig.add_subplot(615)
                        axs6 = fig.add_subplot(616)
                        axs4.plot(ref_wave,ref_syn_spec,color='black',label = refdate)
                        axs5.plot(ref_wave,ref_syn_spec,color='black',label = refdate)
            
    axs1.legend( loc='best',ncol=3,fontsize = 10 )
    axs2.legend( loc='best',ncol=3,fontsize = 10 )
    axs3.legend( loc='best',ncol=3,fontsize = 10 )
        
        
    if refdate:
        if plot_ratio:
            axs4.legend( loc='best',ncol=3,fontsize = 10 )
            axs5.legend( loc='best',ncol=3,fontsize = 10 )
            axs6.legend( loc='best',ncol=3,fontsize = 10 )
            axs4.set_ylim(np.nanmin(ref_bin_spec)/2,np.nanmax(ref_bin_spec)*3)
            axs5.set_ylim(np.nanmin(ref_bin_spec)/2,np.nanmax(ref_bin_spec)*3+20)    

        axs1.set_ylim(np.nanmin(ref_bin_spec)/2,np.nanmax(ref_bin_spec)*3)
        axs2.set_ylim(np.nanmin(ref_bin_spec)/2,np.nanmax(ref_bin_spec)*3+20)
    else:
        axs1.set_ylim(np.nanmin(bin_all)/3,np.nanmax(bin_all)*3)
        axs2.set_ylim(np.nanmin(bin_all)/3,np.nanmax(bin_all)*3+20)
        
    if ylim:
        axs1.set_ylim(ylim[0],ylim[1])
        axs2.set_ylim(ylim[0],ylim[1]+20)
        
    plt.show()        
            

      
    
def tripoly(x,*p):
    return p[3]*x**3+p[2]*x**2+p[1]*x+p[0]



def check_phot_aperture_single(objname,file,survey,aper_dir,band):
    img_dir = '/media/yong-group/DGIS1/MUSE/MUSE_reduced_LX/3_MUSE-fluxcal/broad_band_image/'
    if band == 'g':
        i = 0 
    elif band == 'r':
        i = 1
    else:
        i = 2
        
    if survey  == 'DESI':
        img_file= glob.glob(img_dir+objname+'/DESI/cutout*')[0]        
        hdu     = fits.open(img_file)
        header  = hdu[0].header
        img     = hdu[0].data[i]
        
    elif survey== 'PanStarrs': 
        files  = glob.glob(img_dir+objname+'/PanStarrs/*stk.'+band+'*fits')[0]
        hdu    = fits.open(files)
        img    = hdu[0].data 
        header = hdu[0].header
        
    else:
        files  = glob.glob(img_dir+objname+'/SkyMapper/SkyMapper_'+band+'_*fits')[0]
        hdu    = fits.open(files)
        img    = hdu[0].data 
        header = hdu[0].header
     
    
    transform = SqrtStretch()+ PercentileInterval(90.)
    fig ,axs  = plt.subplots(1,2,figsize=(20, 10),dpi=100)
    matplotlib.rc('xtick', labelsize=12)
    matplotlib.rc('ytick', labelsize=12)

    xc, yc, width, height, PA = read_ds9reg(aper_dir+'ds9_source.reg', header)
    xc_sky, yc_sky, width_sky, height_sky, PA_sky = read_ds9reg(aper_dir+'ds9_sky.reg', header)

    aperture     = RectangularAperture([xc, yc], width, height, theta=PA)
    aperture_sky = RectangularAperture([xc_sky, yc_sky], width_sky, height_sky, theta=PA_sky)

    axs[0].imshow(transform(img),origin='lower')
    aperture.plot(axs[0],color='red',lw=2)
    aperture_sky.plot(axs[0],color='yellow',lw=2)
           

    sin_hdu   = fits.open(file)
    sin_header= sin_hdu[0].header
    sin_img   = sin_hdu[0].data 

    xc, yc, width, height, PA = read_ds9reg(aper_dir+'ds9_source.reg', sin_header)
    xc_sky,yc_sky,width_sky,height_sky, PA_sky = read_ds9reg(aper_dir+'ds9_sky.reg',sin_header)

    sin_aperture     = RectangularAperture([xc, yc], width, height, theta=PA)
    sin_aperture_sky = RectangularAperture([xc_sky, yc_sky], width_sky, height_sky, theta=PA_sky)

    axs[1].imshow(transform(sin_img),origin='lower')
    sin_aperture.plot(axs[1],color='red',lw=2)
    sin_aperture_sky.plot(axs[1],color='yellow',lw=2)
    plt.show()


def compare_photometry_synth(survey,objname,phot_aper_dir,file_dir):
    
    filter_0,filter_1,filter_2 = filter_(survey)
    filter_wave = [filter_0.leff.value,filter_1.leff.value,filter_2.leff.value]
    phot_survey = np.array(survey_broadband(objname,survey,phot_aper_dir))
    
    fig  = plt.figure(figsize=(8, 4),dpi=100)
    matplotlib.rc('xtick', labelsize=10)
    matplotlib.rc('ytick', labelsize=10)           
    ax   = fig.add_subplot(111)
    
    syn_0_file = glob.glob(file_dir+'/*_g_*'+'.fits')[0]
    syn_1_file = glob.glob(file_dir+'/*_r_*'+'.fits')[0]
    syn_2_file = glob.glob(file_dir+'/*_i_*'+'.fits')[0]
    syn_0 = fits.open(syn_0_file)
    syn_1 = fits.open(syn_1_file)
    syn_2 = fits.open(syn_2_file)
    phot_MUSE = np.array(MUSE_broadband(syn_0,syn_1,syn_2,phot_aper_dir))

    ratio = phot_MUSE/phot_survey
    ax.plot(filter_wave,ratio ,'o:',ms=7,color='r',label='synth')
   
    ax.legend( loc='best',ncol=3,fontsize = 10 )    
    wave = np.linspace(filter_wave[0],filter_wave[-1],1000)       
    ax.plot(wave, np.ones(len(wave)), '--', color='black')
    ax.set_ylim(0,2)
    
    plt.show()
    
    return ratio
    

