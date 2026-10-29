import numpy as np
def resample_spec_fluxconv(newwave, wave, flux, err, extrap_treatment = None):
    #newwave in AA
    #wave in AA
    #flux in erg/s/cm^2/AA
    #err  in erg/s/cm^2/AA
    #extrap_treatment = None: fill outside grids with nan
    #                 = zero: fill outside grids with zero
    def define_spec_edge(x):
        inside_edges = (x[1:] + x[:-1]) / 2
        edges = np.insert(inside_edges, 0, 2 * x[0] - inside_edges[0])
        edges = np.append(edges, 2 * x[-1] - inside_edges[-1])
        return edges

    ######define edges
    orig_edges = define_spec_edge(wave)
    fin_edges  = define_spec_edge(newwave)

    orig_low = orig_edges[:-1]
    fin_low  = fin_edges[:-1]
    orig_upp = orig_edges[1:]
    fin_upp  = fin_edges[1:]


    ########got the matrix of each original pixel's contributin to the final pixel
    l_inf = np.where(orig_low > fin_low[:, np.newaxis],
                         orig_low, fin_low[:, np.newaxis])
    l_sup = np.where(orig_upp < fin_upp[:, np.newaxis],
                         orig_upp, fin_upp[:, np.newaxis])

    ##   f=sum(f*resamp_detlam)/sum(resamp_detlam)
    ##   err = ( sum(err**2 * resamp_detlam * orig_detlam) )**2/sum(resamp_detlam)
    resamp_detlam   = (l_sup - l_inf).clip(0)  #the contribution in wavelength of each wave-grid to newwave-grid
    orig_detlam     = (orig_upp - orig_low) 
    resamp_mat      = resamp_detlam*orig_detlam # 

    left_clip = np.where(fin_edges[:-1] - orig_edges[0] < 0, 0, 1)
    right_clip = np.where(orig_edges[-1] - fin_edges[1:] < 0, 0, 1)
    keep_overlapping_matrix = left_clip * right_clip

    resamp_detlam   *= keep_overlapping_matrix[:, np.newaxis]
    resamp_mat      *= keep_overlapping_matrix[:, np.newaxis]
    
    tot_flux    = np.sum(flux * resamp_detlam, axis=-1)
    tot_wv      = np.sum(resamp_detlam, axis=-1)
    id          = tot_wv > 0
    newflux     = np.full(len(tot_wv), np.nan)
    newflux[id] = tot_flux[id]/tot_wv[id]
    
    var        = err**2
    tot_err    = np.sum( var * resamp_mat, axis=-1)**0.5
    newerr     =  np.full(len(tot_wv), np.nan)
    newerr[id] = tot_err[id]/tot_wv[id]

    
    
    if extrap_treatment == 'zero':
        idout = (fin_low < orig_low[0]) | (fin_upp > orig_upp[-1])
        newflux[idout] = 0.0
        newerr[idout]  = 0.0
    return newflux, newerr
    
wave = np.arange(10)
flux = np.full(10, 1)
err  = flux/10

newwave = np.array([9, 10, 11])
resample_spec_fluxconv(newwave, wave, flux, err, extrap_treatment = 'zero') 
