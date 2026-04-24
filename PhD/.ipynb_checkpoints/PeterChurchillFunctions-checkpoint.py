'''
This file is a list of functions that I have used in my Master Thesis that would come in handy for my PhD. 
The functions work for NorESM and EC-Earth
'''

from scipy.special import erf
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from dask.diagnostics import ProgressBar
import dask.array as da
import statsmodels.api as sm
from scipy.odr import ODR, Model, RealData


# ==========================================================
# NorESMExtract (optimized for Dask)
# ==========================================================
def NorESMExtract_Dask(NorPath, station, VarList, Xspace, PNSD=True, chunks="auto"):
    """
    Load NorESM data efficiently with Dask and compute PNSD lazily.

    Parameters
    ----------
    NorPath : str
        Path to the NetCDF file.
    station : str
        Station name or coordinate.
    VarList : list of str
        Variables to extract from file.
    Xspace : array-like
        Size distribution bins.
    PNSD : bool, optional
        If True, compute dNdlogD for each mode.
    chunks : str or dict, optional
        Dask chunking specification. Default 'auto'.
    """

    # Open lazily using Dask
    Data = xr.open_dataset(NorPath, chunks=chunks)
    Data = Data.sel(station=station)
    ds = xr.Dataset()
    PNSD_ds = xr.Dataset()

    # Extract requested variables
    for var in VarList:
        if var in Data:
            ds[var] = Data[var]
        else:
            print(f"⚠️ Variable {var} not found in dataset.")
            
    #Create a CDNC variable
    ds['CDNC'] = Data['AWNC']/Data['FREQL']
     
    # Construct PNSD lazily
    if PNSD:
        for i in range(0, 16):
            sig, nmr, nconc = f"SIGMA{i:02d}", f"NMR{i:02d}", f"NCONC{i:02d}"
            if sig not in Data:
                continue
            PNSD_ds[sig] = Data[sig]
            PNSD_ds[nmr] = Data[nmr]
            PNSD_ds[nconc] = Data[nconc]

            PNSD_ds[f"dNdlogD{i:02d}"] = xr.apply_ufunc(
                dNdlogD,
                Data[nconc],
                Xspace,
                Data[nmr],
                Data[sig],
                dask="parallelized",
                output_dtypes=[float],
            )

        # Combine all dNdlogD modes lazily
        dNdlogD_vars = [v for v in PNSD_ds.data_vars if v.startswith("dNdlogD")]
        ds["dNdlogD"] = sum(PNSD_ds[v] for v in dNdlogD_vars)

        return ds, PNSD_ds

    else:
        for i in range(0, 16):
            sig, nmr, nconc = f"SIGMA{i:02d}", f"NMR{i:02d}", f"NCONC{i:02d}"
            if sig in Data:
                ds[sig] = Data[sig]
                ds[nmr] = Data[nmr]
                ds[nconc] = Data[nconc]
        return ds


def ECearthExtract_Dask(ECPath, station, ifsVarList, ifsVarNames, Xspace, PNSD=True, chunks="auto"):
    """
    Lazily load EC-Earth data using Dask and optionally compute PNSD.

    Parameters
    ----------
    ECPath : str
        Path to EC-Earth NetCDF file.
    station : str
        Station name or coordinate.
    ifsVarList : list of str
        Raw variable names from IFS data.
    ifsVarNames : list of str
        Desired standardized variable names.
    Xspace : array-like
        Size distribution bins.
    PNSD : bool, optional
        If True, compute dNdlogD distributions.
    chunks : str or dict, optional
        Dask chunking specification. Default 'auto'.
    """

    import xarray as xr

    #Open lazily with Dask
    Data = xr.open_dataset(ECPath, chunks=chunks)
    Data = Data.sel(station=station)

    ds = xr.Dataset()
    PNSD_ds = xr.Dataset()

    #Define variables (should be defined BEFORE usage)
    radius_variables = ['RDRY_NUS', 'RDRY_AIS', 'RDRY_ACS', 'RWET_AII', 'RDRY_COS', 'RWET_ACI', 'RWET_COI']
    Numb_variables = ['N_NUS', 'N_AIS', 'N_ACS', 'N_AII', 'N_COS', 'N_ACI', 'N_COI']
    ModesSigma = [1.59, 1.59, 1.59, 2.0, 1.59, 1.59, 2.0]

    #Clean up radius variables (convert to nm if needed)
    for r in radius_variables:
        if r in Data:
            Data[r] = Data[r].where(Data[r] > 0)
            if "units" in Data[r].attrs and Data[r].attrs["units"] == "m":
                Data[r] = Data[r] * 1e9
                Data[r].attrs["units"] = "nm"

    #Handle IFS variables (lazy reindexing)
    if len(ifsVarList) > 0:
        # Compute mean level variable
        ds["lev_ifs"] = Data["var54"].mean("time")

        for ifs, name in zip(ifsVarList, ifsVarNames):
            if ifs not in Data:
                print(f"IFS variable {ifs} not found in dataset.")
                continue
            ds[name] = Data[ifs].isel(lev=0).drop_vars("lev", errors="ignore")
            # Note: align 'lev_ifs' and 'lev' lazily
            if "lev" in Data:
                ds[name] = ds[name].interp(lev_ifs=Data["lev"], method="nearest")

        ds = ds.drop_vars("lev_ifs", errors="ignore")

    #Pressure to hPa
    if "pressure" in Data:
        ds["lev"] = Data["pressure"].mean("time") / 100
        ds["lev"].attrs["units"] = "hPa"
        
    #  CDNC 
    if "var20" in Data and "var22" in Data:
        cdnc = (Data["var20"] / Data["var22"]).where(Data["var22"] > 0)
    
        # If lev_ifs is present but lev is available, interpolate CDNC to lev
        if "lev_ifs" in cdnc.dims and "lev" in Data:
            cdnc = cdnc.interp(lev_ifs=Data["lev"])
            cdnc = cdnc.rename({"lev_ifs": "lev"})
    
        #Assign and preserve Dask-lazy behavior
        ds["CDNC"] = cdnc
        
    #Particle Number Size Distribution
    if PNSD:
        # Collect PNSD variables
        for radius, conc in zip(radius_variables, Numb_variables):
            if radius in Data and conc in Data:
                PNSD_ds[radius] = Data[radius]
                PNSD_ds[conc] = Data[conc]

        # Compute distributions lazily with Dask
        dis_variable = ["NUS_dis", "AIS_dis", "ACS_dis", "COS_dis", "AII_dis", "ACI_dis", "COI_dis"]

        for radius, conc, sigma, dist in zip(radius_variables, Numb_variables, ModesSigma, dis_variable):
            if radius not in PNSD_ds or conc not in PNSD_ds:
                continue

            PNSD_ds[dist] = xr.apply_ufunc(
                dNdlogD,
                PNSD_ds[conc],
                Xspace,
                PNSD_ds[radius] * 2,
                sigma,
                dask="parallelized",
                output_dtypes=[float],
            )

        # Combine all dNdlogD modes lazily
        dNdlogD_vars = [v for v in PNSD_ds.data_vars if v.endswith("_dis")]
        if len(dNdlogD_vars) > 0:
            PNSD_ds["dNdlogD"] = sum(PNSD_ds[v] for v in dNdlogD_vars)

        return ds, PNSD_ds

    else:
        # Return only basic variables
        for radius, conc in zip(radius_variables, Numb_variables):
            if radius in Data:
                ds[radius] = Data[radius]
            if conc in Data:
                ds[conc] = Data[conc]



        return ds


def NorComposition(NorPath, station):
    Data = xr.open_dataset(NorPath)
    Data = Data.sel(station = station)
    ds = xr.Dataset()
    ## Name of all the variables
    OA_ls = ['SOA_NA', 'SOA_A1', 'OM_AC', 'OM_AI', 
                'OM_NI', 'SOA_NA_OCW', 'SOA_A1_OCW', 'OM_AC_OCW', 
                'OM_AI_OCW', 'OM_NI_OCW']
    SO4_ls = ['SO4_NA', 'SO4_A1', 'SO4_A2', 'SO4_AC',
                 'SO4_PR', 'SO4_NA_OCW', 'SO4_A1_OCW', 'SO4_A2_OCW',
                 'SO4_AC_OCW', 'SO4_PR_OCW',]
    Seasalt_ls = ['SS_A1', 'SS_A2', 'SS_A1_OCW', 'SS_A2_OCW',]
    Dust_ls = ['DST_A2', 'DST_A2_OCW',]
    BC_ls = ['BC_N', 'BC_AX', 'BC_NI', 'BC_A',
                'BC_AI', 'BC_AC', 'BC_N_OCW', 'BC_NI_OCW',
                'BC_A_OCW', 'BC_AI_OCW', 'BC_AC_OCW',]
    
    CompositionList = [OA_ls, SO4_ls, Seasalt_ls, Dust_ls, BC_ls]
    VarMassName = ['OA_Mass','SO4_Mass','Seasalt_Mass','Dust_Mass', 'BC_Mass']
    VarFracName = ['OA_Frac','SO4_Frac','Seasalt_Frac','Dust_Frac', 'BC_Frac']    
    ## Calculate teh mass of the 5 compsition categories
    for var, varname in zip(CompositionList, VarMassName):
        ds[varname] = sum(Data[i] for i in var) 
        
    ## Calculate the total mass         
    ds['Total_Mass'] = sum(ds[i] for i in VarMassName)

    ## Find the mass fraction 
    for varFrac, varMass in zip(VarFracName, VarMassName):
        ds[varFrac] = ds[varMass]/ds['Total_Mass']
        
    return ds


def ECComposition(ECPath, station):
    Data = xr.open_dataset(ECPath)
    Data = Data.sel(station = station)
    ds = xr.Dataset()
    ## Name of all the variables

    OA_ls = ['M_SOANUS','M_POMAIS','M_SOAAIS','M_POMACS','M_SOAACS','M_POMAII', 'M_SOAAII',]
    SO4_ls = ['M_SO4NUS','M_SO4ACS', 'M_SO4AIS_es']
    Seasalt_ls= ['M_SSACS'] 
    Dust_ls = ['M_DUACI','M_DUACS'] 
    BC_ls = ['M_BCACS','M_BCAII','M_BCAIS',] 
    ## Calculate the estimate of SO4 in the Aitken mode using ratio of masses from accumulation and aitken
    Data['M_SO4AIS_es'] =(Data['M_SO4ACS'] / 
                  (Data['M_BCACS'] + Data['M_POMACS'] + Data['M_SOAACS'])
                 ) * (Data['M_BCAIS'] + Data['M_POMAIS'] + Data['M_SOAAIS'])
    
    CompositionList = [OA_ls, SO4_ls, Seasalt_ls, Dust_ls, BC_ls]
    VarMassName = ['OA_Mass','SO4_Mass','Seasalt_Mass','Dust_Mass', 'BC_Mass']
    VarFracName = ['OA_Frac','SO4_Frac','Seasalt_Frac','Dust_Frac', 'BC_Frac']    
    ## Calculate teh mass of the 5 compsition categories
    for var, varname in zip(CompositionList, VarMassName):
        ds[varname] = sum(Data[i] for i in var) 
        
    ## Calculate the total mass         
    ds['Total_Mass'] = sum(ds[i] for i in VarMassName)

    ## Find the mass fraction 
    for varFrac, varMass in zip(VarFracName, VarMassName):
        ds[varFrac] = ds[varMass]/ds['Total_Mass']
        
    return ds


def dNdlogD(N,x,mu,sigma):
    return N*np.exp(-(np.log10(x) - np.log10(mu*2))**2 / (2 * np.log10(sigma)**2))/ (np.log10(sigma) * np.sqrt(2 * np.pi))  


def dNdlogD_dask(N, x, mu, sigma):
    """
    Compute dNdlogD using dask.array for lazy evaluation.
    """
    log_term = (da.log10(x) - da.log10(mu * 2)) ** 2
    denom = 2 * (da.log10(sigma) ** 2)
    exponent = -log_term / denom
    return N * da.exp(exponent) / (da.log10(sigma) * np.sqrt(2 * np.pi))
    
def erf_function(r, R, sigma):
    """Dask-safe erf term for lognormal integration."""
    return xr.apply_ufunc(
        erf,
        np.log(r / R) / (np.sqrt(2) * np.log(sigma)),
        dask="parallelized",
        output_dtypes=[float]
    )

def NorERF(Nor_ds, radius_values):
    """
    Compute CCN concentration for one or more radii using Dask + xarray.
    
    Parameters
    ----------
    Nor_ds : xarray.Dataset
        Dataset containing NCONCxx, NMRxx, SIGMAxx fields.
    radius_values : array-like
        Radii (same units as NMR variables).
    """
    mode_indices = [1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 14]
    radius_values = np.atleast_1d(radius_values)
    
    CCN_list = []
    
    for r in radius_values:
        # Use xarray + Dask for lazy computation
        CCN_r = sum(
            Nor_ds[f'NCONC{i:02d}'] / 2 * (1 - erf_function(r, Nor_ds[f'NMR{i:02d}'], Nor_ds[f'SIGMA{i:02d}']))
            for i in mode_indices
        )
        CCN_list.append(CCN_r)
    
    # Combine along new 'radius' dimension
    CCN_ds = xr.concat(CCN_list, dim=xr.DataArray(radius_values, dims='radius', name='radius'))
    CCN_ds.name = 'CCN'
    
    return CCN_ds



def ECEarthERF(EC_ds, radius_values):
    """
    Compute CCN concentration for one or more radii using Dask + xarray.
    
    Parameters
    ----------
    EC_ds : xarray.Dataset
        Dataset containing Number and Radii fields of the 7 different modes.
    radius_values : array-like
        Radii (same units as Radii variables).
    """
    mode_indices = [0,1,2,3,4,5,6]
    radius_values = np.atleast_1d(radius_values)
    Radii = ['RDRY_NUS', 'RDRY_AIS', 'RDRY_ACS', 'RWET_AII', 'RDRY_COS','RWET_ACI','RWET_COI',]
    Number = ['N_NUS','N_AIS','N_ACS','N_AII','N_COS','N_ACI','N_COI',]
    Sigma = [1.59,1.59,1.59,2,1.59,1.59,2]
    CCN_list = []
    
    for r in radius_values:
        # Use xarray + Dask for lazy computation
        CCN_r = sum(
            EC_ds[Number[i]] / 2 * (
                1 - xr.apply_ufunc(
                    erf,
                    np.log(r / EC_ds[Radii[i]]) / np.sqrt(np.log(Sigma[i])),
                    dask='parallelized',
                    output_dtypes=[float]
                )
            )
            for i in mode_indices
        )
        CCN_list.append(CCN_r)
    
    # Combine along new 'radius' dimension
    CCN_ds = xr.concat(CCN_list, dim=xr.DataArray(radius_values, dims='radius', name='radius'))
    CCN_ds.name = 'CCN'
    
    return CCN_ds    

def plot_hexbin_regression_multi(
    x, y, fits=None, lims=(1, 1e4),
    x_label='CCN [cm$^{-3}$]', y_label='Nd [cm$^{-3}$]',
    title=None, cmap='inferno', gridsize=50, bins='log'
):
    """
    Create a log–log hexbin plot with one or more regression lines.

    Parameters
    ----------
    x, y : array-like
        Positive data arrays.
    fits : list of dicts
        Each dict should contain:
          - 'slope' (float)
          - 'intercept' (float)
          - 'label' (str)
          - 'style' (str, e.g. 'r-', 'g--', 'b-.')
    lims : tuple
        Axis limits (min, max)
    """

    # Clean data
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[mask], y[mask]

    fig, ax = plt.subplots(figsize=(6, 6))
    hb = ax.hexbin(
        x, y,
        gridsize=gridsize, bins=bins,
        xscale='log', yscale='log',
        cmap=cmap, mincnt=1
    )

    # Axis limits
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Regression lines
    if fits:
        x_fit = np.array(lims)
        for f in fits:
            y_fit = 10 ** (f['intercept'] + f['slope'] * np.log10(x_fit))
            ax.plot(
                x_fit, y_fit,
                f.get('style', '-'),
                lw=2,
                label=f.get('label', f"slope={f['slope']:.2f}")
            )

    # 1:1 reference line
    ax.plot(lims, lims, 'k:', lw=1.5, label='1:1')

    # Labels and styling
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    ax.legend()
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, which='both', ls=':')
    fig.colorbar(hb, ax=ax, label='Counts')

    return fig, ax



# --- Fitting functions ---
def TLS_fit(X, Y):
    """Total Least Squares using eigenvalues and eigenvectors."""
    # Mask invalid values
    mask = np.isfinite(X) & np.isfinite(Y)
    X, Y = X[mask], Y[mask]

    # Handle empty slices
    if X.size < 2 or np.all(X == X[0]):
        return np.nan, np.nan

    # Center data
    Xc, Yc = X - X.mean(), Y - Y.mean()

    # Covariance
    cov = np.cov(Xc, Yc)

    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eig(cov)
    v = eigvecs[:, np.argmax(eigvals)]

    # Slope & intercept
    slope = v[1] / v[0]
    intercept = Y.mean() - slope * X.mean()
    return slope, intercept


def TLS_SVD_fit(X, Y):
    """Total Least Squares using SVD."""
    # Mask invalid values
    mask = np.isfinite(X) & np.isfinite(Y)
    X, Y = X[mask], Y[mask]

    # Handle empty slices
    if X.size < 2 or np.all(X == X[0]):
        return np.nan, np.nan


    Xc, Yc = X - X.mean(), Y - Y.mean()
    Z = np.column_stack((Xc, Yc))
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    v = Vt.T[:, 0]  # principal direction
    slope = v[1] / v[0]
    intercept = Y.mean() - slope * X.mean()
    return slope, intercept

def ODR_fit(X, Y, stretch = False):
    """Orthogonal Distance Regression (ODR)."""

    mask = np.isfinite(X) & np.isfinite(Y)
    X, Y = X[mask], Y[mask]
    if X.size < 2 or np.all(X == X[0]):
        return np.nan, np.nan, None

    def f(B, x): return B[0]*x + B[1]
    model = Model(f)

    if stretch:
        sx, sy = np.std(X)*0.05, np.std(Y)*0.05
        data = RealData(X, Y, sx=sx, sy=sy)
    else:
        data = RealData(X, Y)

    odr = ODR(data, model, beta0=[1.0, 0.0])
    out = odr.run()
    slope, intercept = out.beta
    return slope, intercept, out

def deming_fit(X, Y, lambda_xy=None):
    """Perform Deming regression."""
    
    # Remove NaNs
    mask = np.isfinite(X) & np.isfinite(Y)
    X, Y = X[mask], Y[mask]

    # Handle empty slices
    if X.size < 2 or np.all(X == X[0]):
        return np.nan, np.nan

    # Sample means
    X_mean = np.mean(X)
    Y_mean = np.mean(Y)

    # Centered data
    Xc = X - X_mean
    Yc = Y - Y_mean

    # Sample variances
    Sx = np.var(Xc, ddof=1)
    Sy = np.var(Yc, ddof=1)
    Sxy = np.cov(Xc, Yc, ddof=1)[0, 1]

    # Estimate lambda if not provided
    if lambda_xy is None:
        lambda_xy = Sx / Sy

    # Compute slope (Eq. A8)
    numerator = Sy - lambda_xy * Sx + np.sqrt((Sy - lambda_xy*Sx)**2 + 4 * lambda_xy * Sxy**2)
    slope = numerator / (2 * Sxy)

    # Intercept
    intercept = Y_mean - slope * X_mean

    return  slope, intercept

def PCA_fit(X, Y):
    """Perform Primary Component Analysis regression."""

    mask = np.isfinite(X) & np.isfinite(Y)
    X, Y = X[mask], Y[mask]
    if X.size < 2 or np.all(X == X[0]):
        return np.nan, np.nan

    xbar, ybar = np.mean(X), np.mean(Y)
    cov = np.cov(X, Y, ddof=1)
    Sx, Sxy = cov[0,0], cov[0,1]

    intercept = (Sx*ybar - xbar*Sxy) / (Sx - xbar**2)
    slope     = (-xbar*ybar + Sxy)  / (Sx - xbar**2)
    return slope, intercept




def OLS_fit(X, Y):
    """Ordinary Least Squares regression."""
    # Mask invalid values
    mask = np.isfinite(X) & np.isfinite(Y)
    X, Y = X[mask], Y[mask]

    # Handle empty slices
    if X.size < 2 or np.all(X == X[0]):
        return np.nan, np.nan

    x_mean = X.mean()
    y_mean = Y.mean()
    slope = np.sum((X - x_mean) * (Y - y_mean)) / np.sum((X - x_mean) ** 2)
    intercept = y_mean - slope * x_mean

    # Ensure scalars
    return float(slope), float(intercept)

def OLSGraph(x, y, summary=True, title='OLS Regression: Nd vs CCN'):
    """
    Perform OLS regression in log–log space and plot using shared hexbin function.
    """

    # --- Clean data ---
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[mask], y[mask]

    # --- Log10 transform ---
    X_log = np.log10(x)
    Y_log = np.log10(y)

    # --- OLS regression ---
    X_design = sm.add_constant(X_log)
    model = sm.OLS(Y_log, X_design).fit()
    slope, intercept = model.params[1], model.params[0]

    # --- Optional summary ---
    if summary:
        print(model.summary())

    # --- Plot using shared function ---
    fits = [{
        'slope': slope,
        'intercept': intercept,
        'label': f'OLS: slope={slope:.2f}',
        'style': 'r-'
    }]

    fig, ax = plot_hexbin_regression_multi(
        x, y, fits=fits,
        title=title
    )

    return model, (fig, ax)




