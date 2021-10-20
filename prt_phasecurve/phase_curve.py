import numpy as np
from scipy.interpolate import RBFInterpolator


def wrap_phase_curve(phases, lon, lat, mus, intensity):
    """
    Function to wrap around the phasecurve calculation.

    Parameters
    ----------
    phases (array(P)):
        List of phases at which the phasecurve should be evaluated
    lon (array(M1,M2) or array(M1*M2)):
        Longitude coordinate values. If input is 1D we assume that it has been flattened.
    lat (array(M1,M2) or array(M1*M2)):
        Longitude coordinate values. If input is 1D we assume that it has been flattened.
    mus (array(D)):
        List of mus matching the mus of the calculated intensity
    intensity (array(M1,M2,N,D) or array(M1*M2,N,D)):
        array of intensitities. The order needs to be  Horizontal (1D or 2D), Wavelength, mu

    Returns
    -------
    phase_curve (array (P,N)):
        Array containing the calculated phasecurve. First Dimension is the Phase, second Dimension is the Wavelength
    """

    # Format input data:
    if len(lon.shape) == 2:
        _lon = lon.reshape(lon.shape[0]*lon.shape[1])
    elif len(lon.shape) == 1:
        _lon = lon
    else:
        raise IndexError('Please read the docstring and format your input data accordingly')

    if len(lat.shape) == 2:
        _lat = lat.reshape(lat.shape[0]*lat.shape[1])
    elif len(lat.shape) == 1:
        _lat = lat
    else:
        raise IndexError('Please read the docstring and format your input data accordingly')

    if len(intensity.shape) == 4:
        _intensity = intensity.reshape(intensity.shape[0]*intensity.shape[1], intensity.shape[2], intensity.shape[3])
    elif len(intensity.shape) == 3:
        _intensity = intensity
    else:
        raise IndexError('Please read the docstring and format your input data accordingly')

    # Transform into cartesian coordinates:
    phi = _lon/180.*np.pi
    theta = (_lat+90.)/180.*np.pi
    x = np.cos(phi)*np.sin(theta)
    y = np.sin(phi)*np.sin(theta)
    z = np.cos(theta)

    # Build the Interpolator Instance:
    mu_rad_bas = RBFInterpolator(np.array([x, y, z]).T, _intensity, smoothing=0.1)

    # Carry out the phasecurve calculation:
    phase_curve = np.array([calc_phase_curve(phase, mus, mu_rad_bas) for phase in phases])

    return phase_curve


def calc_phase_curve(phase, mus, mu_rad_bas):
    """
    Function to integrate the intensity to yield the phase curve

    Parameters
    ----------
    phase (float):
        Phase at which the phase curve should be calculated
    mus (1D list):
        array of the mu values for which the intensity has been calculated
    mu_rad_bas (scipy.interpolate.RBFInterpolator):
        Instance of the radial basis function Interpolator

    Returns
    -------
    flux_arr (1D list):
        List of Fluxes for each wavelength bin
    """

    mu_p_grid_bord = np.linspace(0.,1.,11)[::-1]
    mu_p_mean = (mu_p_grid_bord[1:]+mu_p_grid_bord[:-1])/2.
    del_mu_p = -np.diff(mu_p_grid_bord)

    phi_p_grid_bord = np.linspace(0.,2.*np.pi,11)
    phi_p_mean = (phi_p_grid_bord[1:]+phi_p_grid_bord[:-1])/2.
    del_phi_p = np.diff(phi_p_grid_bord)

    i_intps = []
    do_intp = []

    for imu in range(len(mu_p_mean)):
        for jmu in range(len(mus)-1):
            if mu_p_mean[imu] <= mus[0]:
                do_intp.append(False)
                i_intps.append(0)
            elif mu_p_mean[imu] > mus[len(mus)-1]:
                do_intp.append(False)
                i_intps.append(len(mus)-1)
            elif (mu_p_mean[imu] > mus[jmu]) and \
              (mu_p_mean[imu] <= mus[jmu+1]):
                do_intp.append(True)
                i_intps.append(jmu)

    rot = - phase * 2*np.pi
    M = np.matrix([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    M = M.dot([[1,0,0], [0, np.cos(rot), -np.sin(rot)], [0, np.sin(rot), np.cos(rot)]])

    Nlambda = len(mu_rad_bas([[0,0,0]])[0])
    flux_arr = np.zeros(Nlambda)

    for iphi in range(len(phi_p_mean)):
        for itheta in range(len(mu_p_mean)):
            x_p = np.cos(phi_p_mean[iphi])*np.sqrt(1.-mu_p_mean[itheta]**2.)
            y_p = np.sin(phi_p_mean[iphi])*np.sqrt(1.-mu_p_mean[itheta]**2.)
            z_p = mu_p_mean[itheta]

            R = M.dot(np.matrix([[x_p],[y_p],[z_p]]))

            point = np.array([R[0],R[1],R[2]]).reshape(1,3)
            interp = mu_rad_bas(point)

            if do_intp[itheta]:
                I_small = interp[0,:,i_intps[itheta]]
                I_large = interp[0,:,i_intps[itheta]+1]
                I_use = I_small+(I_large-I_small)/(mus[i_intps[itheta]+1]-mus[i_intps[itheta]])* \
                  (mu_p_mean[itheta]-mus[i_intps[itheta]])
            else:
                I_use = interp[0,:,i_intps[itheta]]

            dF = I_use * mu_p_mean[itheta] * del_mu_p[itheta] * del_phi_p[iphi]
            flux_arr = flux_arr + dF

    return flux_arr


def compute_phase_curve(Radtrans, data, phases = np.linspace(0,1,11)):
    """
    Function that calculates the phase curve for

    Parameters
    ----------
    Radtrans (petitRADTRANS.Radtrans like):
        intitialised Radtrans object to be used to calculate the individual emission spectra
    data (xarray.Dataset):
        containing abundancies, containing temperatures, containing lons and lats
    phases (1D list or array):
        Phases for which the phase_curve is calculated

    Returns
    -------

    Ideas:
    1. Script to be called from commandline? Input would be the iternumber. Could be quite useful for the cluster

    Plan:
    1. Compute emission spectrum for each gridcell
    2. SAVE GLOBAL EMISSION SPECTRUM
    3. Compute the Phasecurve for each gridcell
    4. Save global EMISSION SPECTRUM

    """
    raise NotImplementedError('TODO')


if __name__ == "__main__":
    # TODO: remove later on
    import matplotlib.pyplot as plt
    lons = np.linspace(-180, 180, 31)
    lats = np.linspace(-90, 90, 11)
    lon, lat = np.meshgrid(lons, lats)

    mus = np.linspace(-1, 1, 20)
    N = 10
    total_intensity = (np.cos(lon / 180 * np.pi) * np.cos(lat / 180 * np.pi))[:,:,np.newaxis, np.newaxis]
    total_intensity = np.ones((lon.shape[0],lon.shape[1], N, len(mus))) * total_intensity

    phases = np.linspace(0,1,11)

    phase_curve = wrap_phase_curve(phases, lon, lat, mus, total_intensity)

    plt.plot(phases, phase_curve[:,0])
    plt.show()

