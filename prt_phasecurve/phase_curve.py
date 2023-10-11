import numpy as np
from scipy.interpolate import RBFInterpolator
from .mu import mu


def phase_curve(phases, lon, lat, intensity):
    """
    Function to wrap around the phasecurve calculation.

    Parameters
    ----------
    phases (array(P)):
        List of phases at which the phasecurve should be evaluated.
        Phases are from 0 to 1, where 0.5 is the night side and 0.0 is the dayside.
    lon (array(M1,M2) or array(M1*M2)):
        Longitude coordinate values in degrees. If input is 1D we assume that it has been flattened.
    lat (array(M1,M2) or array(M1*M2)):
        Latitude coordinate values in degrees. If input is 1D we assume that it has been flattened.
    mus (array(D)):
        List of mus matching the mus of the calculated intensity
    intensity (array(M1,M2,D,N) or array(M1*M2,D,N)):
        array of intensitities. The order needs to be  Horizontal (1D or 2D), mu, Wavelength

    Returns
    -------
    phase_curve (array (P,N)):
        Array containing the calculated phasecurve. First Dimension is the Phase, second Dimension is the Wavelength
    """

    lon = np.array(lon)
    lat = np.array(lat)
    intensity = np.array(intensity)

    assert phases.min() >= 0.0 and phases.max() <= 1.0, "phases are from 0 to 1, where 0.5 is the night side and 0.0 is the dayside"

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
    phase_curve = np.array([calc_phase_curve(phase, mu, mu_rad_bas) for phase in phases])

    assert (phase_curve >= 0.0).all(), 'we have some negative values here! Use more gridpoints'

    return phase_curve


def calc_phase_curve(phase, mus, mu_rad_bas):
    """
    Function to integrate the intensity to yield the phase curve

    Parameters
    ----------
    phase (float):
        Phase at which the phase curve should be calculated.
        Phases are from 0 to 1, where 0.5 is the night side and 0.0 is the dayside.
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

    rot = phase * 2*np.pi
    M = np.matrix([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    M = M.dot([[1,0,0], [0, np.cos(rot), -np.sin(rot)], [0, np.sin(rot), np.cos(rot)]])

    Nlambda = len(mu_rad_bas([[0,0,0]])[0,0,:])
    flux_arr = np.zeros(Nlambda)

    for iphi in range(len(phi_p_mean)):
        for itheta in range(len(mu_p_mean)):
            x_p = np.cos(phi_p_mean[iphi])*np.sqrt(1.-mu_p_mean[itheta]**2.)
            y_p = np.sin(phi_p_mean[iphi])*np.sqrt(1.-mu_p_mean[itheta]**2.)
            z_p = mu_p_mean[itheta]

            R = M.dot(np.matrix([[x_p],[y_p],[z_p]]))

            point = np.array([R[0],R[1],R[2]]).reshape(1,3)
            interp = mu_rad_bas(point).reshape(len(mus), Nlambda)

            if do_intp[itheta]:
                I_small = interp[i_intps[itheta],:]
                I_large = interp[i_intps[itheta]+1,:]
                I_use = I_small+(I_large-I_small)/(mus[i_intps[itheta]+1]-mus[i_intps[itheta]])* \
                  (mu_p_mean[itheta]-mus[i_intps[itheta]])
            else:
                I_use = interp[i_intps[itheta],:]

            dF = I_use * mu_p_mean[itheta] * del_mu_p[itheta] * del_phi_p[iphi]
            flux_arr = flux_arr + dF

    return flux_arr


