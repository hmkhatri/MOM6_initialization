"""

Graeme MacGilchrist
15/01/2020
gmacgilchrist@gmail.com

Set of functions to calculate input files for MOM6.


"""

import xarray as xr
import numpy as np

def calc_XYmeters(grid):
    '''Calculate the locations of each T point in [grid] in units of meters'''
    Xval = np.append(np.zeros(shape=(grid['lath'].size,1)),
                     grid['dxT'].cumsum('lonh').values,
                     axis=1)
    Yval = np.append(np.zeros(shape=(1,grid['lonh'].size)),
                     grid['dyT'].cumsum('lath').values,
                     axis=0)
    Xval = 0.5*(Xval[:,:-1]+Xval[:,1:])
    Yval = 0.5*(Yval[:-1,:]+Yval[1:,:])
    X = xr.DataArray(Xval,dims=['lath','lonh'],coords={'lath':grid['lath'],'lonh':grid['lonh']})
    Y = xr.DataArray(Yval,dims=['lath','lonh'],coords={'lath':grid['lath'],'lonh':grid['lonh']})
    
    return X,Y

def calc_vgrid(nk,max_depth,min_depth=0,thkcello_topcell=1,method='powerlaw'):
    '''Calculate the locations and thickness of grid cells for the vertical ocean grid'''
    z0 = min_depth + thkcello_topcell
    H = max_depth
    k = np.linspace(1,nk,num=nk)

    # Defining INTERFACE locations (i.e. grid cell interfaces)
    if method=='powerlaw':
        # Power law
        B = np.log(H/z0)/np.log(nk)
        zw = z0*k**B
    elif method=='uniform':
        zw = np.linspace(z0,H,nk)
    elif method=='exponential':
        zw = z0*np.exp(np.log(H/z0)*(k/(nk)))
        
    # Add the free surface, z*=0, as an interface (saved until this point as z0=0 messes with power law scaling)
    zw = np.append(0,zw)
    
    # Central point is THICKNESS location
    zt = (zw[1:] + zw[:-1]) / 2

    # Place in data arrays
    zw = xr.DataArray(zw,coords=[zw],dims=['NKp1'])
    zt = xr.DataArray(zt,coords=[zt],dims=['NK'])
    # Calculate thickness
    dz = zw.diff(dim='NKp1')
    dz = xr.DataArray(dz,coords=[zt],dims=['NK'])

    # Combine arrays to one dataset
    zw.name='zw'
    zt.name='zt'
    dz.name='dz'
    vgrid = xr.merge([zw,zt,dz])
    
    return vgrid

def def_sponge_dampingtimescale_north(hgrid,sponge_width,idampval):
    '''Define a sponge grid at the north of the domain based on horizontal grid shape.
    hgrid is the horizontal grid dataset
    sponge_width is the degrees of lat to damp over [must be a list, progressively decreasing in width]
    idampval is the inverse damping rate (in s-1) [must be a list] '''
    idamp = xr.zeros_like(hgrid['geolat'])
    for i in range(len(sponge_width)):
        sponge_region = hgrid['geolat']>hgrid['geolat'].max(xr.ALL_DIMS)-sponge_width[i]
        idamp=idamp+xr.zeros_like(hgrid['geolat']).where(~sponge_region,idampval[i])
    return idamp

def def_sponge_interfaceheight(vgrid,hgrid):
    '''Define a 3D array of layer interface heights (eta), to which sponge will relax.'''
    eta = xr.DataArray(-vgrid['zw'],coords=[vgrid['zw']],dims='depthe')
    eta = eta*xr.ones_like(hgrid['D'])
    return eta

def calc_distribution(coordinate,function,**kwargs):
    '''Calculate the distribution of a variable, based on a given coordinate
    e.g. linear surface distribution of temperature, where
             coordinate = Y
             val_at_mincoord = SST at south
             val_at_maxcoord = SST at north
             function = 'linear'
        Independent variable required for functions can be passed at the end of the function
    '''
    if function=='linear':
        
        A = (val_at_maxcoord-val_at_mincoord)/(coordinate.max(xr.ALL_DIMS)-coordinate.min(xr.ALL_DIMS))
        B = val_at_mincoord - coordinate.min(xr.ALL_DIMS)*A
        distribution = A*coordinate+B
    
#     if function=='exponential':
        
#     return distribution