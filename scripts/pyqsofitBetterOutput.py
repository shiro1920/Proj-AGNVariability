def betterOutput(qsofit_class_q, save_path, target_name, ins_name, redshift, 
                 save_data = False, data_output=True, n_iter=100):
    """
    qsofit_class_q: class created by pyQSOFiT, with a default name 'q' in their example;
    save_path: str, final output data save path;
    target_name: str, object name;
    ins_name: str, to deal with spectra from different survey, like SDSS, GTC, or other instruments;
    redshift: float or array, to convert flux to luminosity;
    save_data: bool, default: False;
    data_output: bool, default: True;
    n_iter: int, repeat times of MCMC, the errors of line properties is simulated by MCMC method,
    
    """
    
    from tqdm import tnrange, tqdm_notebook
    from tqdm.notebook import trange
    
    ##### 分类线类型和线心波长
    def get_peak_type(line_name):
    line_center = []
    line_type   = []
    for name in line_name:
        if 'Hb_br' in name:
            line_center.append(4862.68)
            line_type.append('broad')
        if 'Hb_na' in name:
            line_center.append(4862.68)
            line_type.append('narrow')
        if 'Ha_br' in name:
            line_center.append(6564.61)
            line_type.append('broad')
        if 'Ha_na' in name:
            line_center.append(6564.61)
            line_type.append('narrow')   
        if 'OIII4959c' in name:
            line_center.append(4960.30)
            line_type.append('narrow')
        if 'OIII5007c' in name:
            line_center.append(5008.24)
            line_type.append('narrow')
        if 'OIII4959w' in name:
            line_center.append(4960.30)
            line_type.append('broad')
        if 'OIII5007w' in name:
            line_center.append(5008.24)
            line_type.append('broad')
        if "HeII4687_na" in name:
            line_center.append(4687.02)
            line_type.append('narrow')
        if "HeII4687_br" in name:
            line_center.append(4687.02)
            line_type.append('broad')
        if 'NII6549' in name:
            line_center.append(6549.85)
            line_type.append('narrow')
        if 'NII6585' in name:
            line_center.append(6585.28)
            line_type.append('narrow')
        if 'SII6718' in name:
            line_center.append(6718.29)
            line_type.append('narrow')
        if 'SII6732' in name:
            line_center.append(6732.67)
            line_type.append('narrow')
        if 'MgII_na' in name:
            line_center.append(2798.75)
            line_type.append('narrow')
        if 'MgII_br' in name:
            line_center.append(2798.75)
            line_type.append('broad')
        if 'CIII_na' in name:
            line_center.append(1908.73)
            line_type.append('narrow')
        if 'CIII_br' in name:
            line_center.append(1908.73)
            line_type.append('broad')
        if 'CIV_na' in name:
            line_center.append(1549.06)
            line_type.append('narrow')
        if 'CIV_br' in name:
            line_center.append(1549.06)
            line_type.append('broad')
        if 'Lya_na' in name:
            line_center.append(1215.67)
            line_type.append('narrow')
        if 'Lya_br' in name:
            line_center.append(1215.67)
            line_type.append('broad')
            
    return line_center, line_type
    
    ########  计算光度
    def fluxToLuminosity(flux, flux_err, z):
        """flat Universe as h=0.693, OmO=0.286, Tcmb0 = 2.725 K """
        from astropy.cosmology import WMAP9 as cosmo
        DL = cosmo.luminosity_distance(z).cgs.value  ## unit cm
        flux, flux_err = np.array(flux), np.array(flux_err)
        F_a = 4 * np.pi * DL**2                      
        L, L_err = F_a * flux * 1e-17, F_a * flux_err* 1e-17   ## unit erg/s
        return L, L_err
    
    ####  组织qsofit输出参数并分类
    
    q = qsofit_class_q
    idx = []
    idy = []
    for ii in np.arange(q.line_result_name.shape[0]):
        if "ndof" in q.line_result_name[ii]:
            idx.append(ii)
        if "whole" in q.line_result_name[ii]:
            idy.append(ii)
    line_result = q.line_result[idx[-1]+1:idy[0]]
    line_r_name  = q.line_result_name[idx[-1]+1:idy[0]]
    line_count  = line_rname.shape[0]/6
    line_name   = np.array([line_rname[6*k][:-6] for k in np.arange(int(line_count))])
    line_center, line_type = get_peak_type(line_name)
    
    p_line_name = np.array([])
    p_err_line_name = np.array([])
    p_line = np.array([])
    p_err_line = np.array([])

    for i in range(len(line_r_name)):
        if 'err' in line_r_name[i]:
            p_err_line_name = np.append(p_err_line_name, line_r_name[i])
            p_err_line = np.append(p_err_line, np.float(line_result[i]))
        else:
            p_line_name = np.append(p_line_name, line_r_name[i])
            p_line = np.append(p_line, np.float(line_result[i]))
            
    #######  计算测量量及其误差
    
    line_fwhm  = np.full_like(line_name, np.nan)
    line_sigma = np.full_like(line_name, np.nan)
    line_ew    = np.full_like(line_name, np.nan)
    line_peak  = np.full_like(line_name, np.nan)
    line_flux  = np.full_like(line_name, np.nan)
    line_logL  = np.full_like(line_name, np.nan)

    line_fwhm_err  = np.full_like(line_name, np.nan)
    line_sigma_err = np.full_like(line_name, np.nan)
    line_ew_err    = np.full_like(line_name, np.nan)
    line_peak_err  = np.full_like(line_name, np.nan)
    line_flux_err  = np.full_like(line_name, np.nan)
    line_logL_err  = np.full_like(line_name, np.nan)

    for j in trange(len(line_name), desc='lines'):
        pt_line = p_line[3*j:3*j+3]
        pt_err_line = p_err_line[3*j:3*j+3]
        
        z = redshift
        n_iter = n_iter
        norm_array = np.zeros(n_iter)
        all_fwhm   = np.full_like(norm_array, np.nan)
        all_sigma  = np.full_like(norm_array, np.nan)
        all_ew     = np.full_like(norm_array, np.nan)
        all_peak   = np.full_like(norm_array, np.nan)
        all_area   = np.full_like(norm_array, np.nan)
        
        for tra in trange(n_iter,leave=False, desc='MCMC'):
            pp_line = random.gauss(pt_line,pt_err_line)
            try:
                all_fwhm[tra], all_sigma[tra], all_ew[tra], all_peak[tra], all_area[tra] = q.line_prop(line_center[j],pp_line,line_type[j])
            except:
                all_fwhm[tra], all_sigma[tra], all_ew[tra], all_peak[tra], all_area[tra] = np.nan, np.nan, np.nan, np.nan, np.nan

        line_fwhm[j], line_sigma[j], line_ew[j], line_peak[j], line_flux[j] = np.nanmedian(all_fwhm), np.nanmedian(all_sigma), \
                                                              np.nanmedian(all_ew), np.nanmedian(all_peak), np.nanmedian(all_area)
        line_fwhm_err[j], line_sigma_err[j], line_ew_err[j], line_peak_err[j], line_flux_err[j] = np.nanstd(all_fwhm), np.nanstd(all_sigma), \
                                                                                    np.nanstd(all_ew), np.nanstd(all_peak), np.nanstd(all_area)
        flux2L, flux2L_err = fluxToLuminosity(np.float(line_flux[j]), np.float(line_flux_err[j]), z=z)
        line_logL[j], line_logL_err[j] = np.log10(flux2L), np.log10(flux2L_err)
        
    ########  写进table 
     
    data = Table()
    data['name']      = line_name
    data['peak']      = line_peak
    data['peak_err']  = line_peak_err
    data['flux']      = line_flux
    data['flux_err']  = line_flux_err
    data['L']         = line_logL
    data['L_err']     = line_logL_err
    data['FWHM']      = line_fwhm
    data['FWHM_err']  = line_fwhm_err
    data['sigma']     = line_sigma
    data['sigma_err'] = line_sigma_err
    data['ew']        = line_ew
    data['ew_err']    = line_ew_err
    
    if save_data:
        path = save_path 
        tar_name = target_name
        ins = ins_name 
        ascii.write(data,path + tar_name + ins  + "_line_properties.dat",overwrite=True)
        print("Save data completed")
        
    if data_output:  
        return data
    else:
        return