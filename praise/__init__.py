"""PRAISE: Phase Retrieval [and imaging science]."""
import numpy as np

from scipy import fft

# PRAISE is named after Dr. Jim Fienup's research group at the
# University of Rochester.  Its author Brandon Dube is an alumnus.

# PRAISE implements several contemporary Phase Retrieval algorithms, based
# on iterative forward-reverse methods.  Each is only implemented for the two
# image case, one of the pupil and one of the PSF.
# These are simply reference implementations, so that folks new to
# Phase Retrieval can play with them.

# The notation used is after Fienup1982, Phase Retrieval Algorithms: a Comparison
# where
#   g = object estimate
#   f = true object
#   G = Fourier transform of g
#   F = Fourier transform of F
#
# for wavefront sensing, the use here, |f| and |F| are measured.
#
# a small amount of effort is expended in the performance of these algorithms,
# but the code is not muddied by advanced features so that it may be as
# intelligible to the beginner as possible.
#
# one performance-hurting feature is that all algorithms track mean square
# errors as appropriate.
#
#
# These algorithms have great historical significance and are still of utility
# today, however the field of image based wavefront sensing ("phase retrieval")
# has moved on, and the state of the art lies in the use of nonlinear optimizers
# to solve the problem.  Those methods provide a great deal more flexibility and
# indeed are substantially more adaptable to common problems, such as:
#   - undersampled G
#   - polychromatic imaging
#   - unusual diversities (wavelength diversity, translation diversity, weak lens focus diversity, [...])
#
# these benefits only in addition to the optimizer's ability to learn parameters
# other than the phase error, enabling imperfect knowledge of wavelengths, F#s,
# diversity parameters, and so forth.
#
# reading notes:
# the colloquial routine to propagate from g -> G is
# G = fftshift(fft2(ifftshift(g)))
# since we will do this (and the reciprocal) over and over and over again,
# the ifftshift in the middle can be done one time (work with g in fft coordinates)
# and the fftshift neglected
# that is, g -> ifftshift(g); G = fft2(g); g' = ifft2(G')
# the user will have to 'recenter' after, which we provide a utility function for


def recenter(data):
    """Place data in the usual "origin in center of array" arrangement."""
    return fft.fftshift(data)


def _init_iterative_transform(self, psf, pupil_amplitude, phase_guess=None):
    """Initialize an instance of an iterative transform type algorithm."""
    # python teaching moment -- self as an argument name is only a convention,
    # and carries no special meaning.  This function exists to refactor the
    # various iterative transform types without subclassing or inheritance
    if phase_guess is None:
        phase_guess = np.random.rand(*pupil_amplitude.shape)

    absF = np.sqrt(psf)
    absg = pupil_amplitude

    self.absF = fft.ifftshift(absF)
    self.absg = fft.ifftshift(absg)
    phase_guess = fft.ifftshift(phase_guess)

    self.g = self.absg * np.exp(1j*phase_guess)
    self.mse_denom = np.sum((self.absF)**2)
    self.iter = 0
    self.costF = []


def _mean_square_error(a, b, norm=1):
    """Mean square error between a and b, normalized by norm."""
    diff = a - b
    mse = np.sum(diff**2)
    return mse / norm


class GerchbergSaxton:
    """The Gerchberg-Saxton phase retrieval algorithm."""

    def __init__(self, psf, pupil_amplitude, phase_guess=None):
        """Create a new Gerchberg-Saxton problem.

        Parameters
        ----------
        psf : `numpy.ndarray`
            array containing the incoherent PSF, |G|^2, of shape (m,n)
        pupil_amplitude : `numpy.ndarray`
            array containing the amplitude of the pupil, |g|, of shape (m,n)
        phase_guess : `numpy.ndarray`
            array containing the guess for the phase, np.angle(g), of shape (m,n)

        Notes
        -----
        psf and pupil_amplitude should both be centered with the origin in the
        middle of the array.

        """
        _init_iterative_transform(self, psf, pupil_amplitude, phase_guess)

    def step(self):
        """Advance the algorithm one iteration."""
        G = fft.fft2(self.g)
        mse = _mean_square_error(abs(G), self.absF, self.mse_denom)

        phs_G = np.angle(G)
        Gprime = self.absF * np.exp(1j*phs_G)
        gprime = fft.ifft2(Gprime)
        phs_gprime = np.angle(gprime)
        gprimeprime = self.absg * np.exp(1j*phs_gprime)

        self.costF.append(mse)
        self.iter += 1
        self.g = gprimeprime
        return gprimeprime


class ErrorReduction:
    """The Error Reduction phase retieval algorithm."""

    def __init__(self, psf, pupil_amplitude, phase_guess=None):
        """Create a new Error Reduction problem.

        Parameters
        ----------
        psf : `numpy.ndarray`
            array containing the incoherent PSF, |G|^2, of shape (m,n)
        pupil_amplitude : `numpy.ndarray`
            array containing the amplitude of the pupil, |g|, of shape (m,n)
        phase_guess : `numpy.ndarray`
            array containing the guess for the phase, np.angle(g), of shape (m,n)

        Notes
        -----
        psf and pupil_amplitude should both be centered with the origin in the
        middle of the array.

        Error reduction permits the amplitude to change within the support

        """
        _init_iterative_transform(self, psf, pupil_amplitude, phase_guess)
        self.mask = self.absg > 1e-6
        self.invmask = ~self.mask

    def step(self):
        """Advance the algorithm one step."""
        G = fft.fft2(self.g)
        mse = _mean_square_error(abs(G), self.absF, self.mse_denom)

        phs_G = np.angle(G)
        Gprime = self.absF * np.exp(1j*phs_G)
        gprime = fft.ifft2(Gprime)
        # error reduction uses a "minimum update"
        # for G -> G', in this case we use the most
        # common flavor of that; enforce the |F| constraint

        # now apply the ER object domain constraints:
        # positive g, support
        gprimeprime = gprime
        gprimeprime[self.invmask] = 0  # support constraint
        subset = gprimeprime[self.mask]
        subset[subset < 0] = 0  # positivity constraint

        self.costF.append(mse)
        self.iter += 1
        self.g = gprimeprime
        return gprimeprime


class SteepestDescent:
    """The Steepest Descent phase retrieval algorithm."""

    def __init__(self, psf, pupil_amplitude, phase_guess=None, doublestep=True):
        """Create a new SteepestDescent problem.

        Parameters
        ----------
        psf : `numpy.ndarray`
            array containing the incoherent PSF, |G|^2, of shape (m,n)
        pupil_amplitude : `numpy.ndarray`
            array containing the amplitude of the pupil, |g|, of shape (m,n)
        phase_guess : `numpy.ndarray`
            array containing the guess for the phase, np.angle(g), of shape (m,n)
        doublestep : `bool`, optional
            if True, uses double length steps

        Notes
        -----
        psf and pupil_amplitude should both be centered with the origin in the
        middle of the array.

        """
        _init_iterative_transform(self, psf, pupil_amplitude, phase_guess)
        self.doublestep = doublestep

    def step(self):
        """Advance the algorithm one iteration."""
        G = fft.fft2(self.g)
        mse = _mean_square_error(abs(G), self.absF, self.mse_denom)

        phs_G = np.angle(G)
        Gprime = self.absF * np.exp(1j*phs_G)
        gprime = fft.ifft2(Gprime)

        # steepest descent is the same as GS until the time to form
        # g'' ...
        #  g'' - g = -1/4 partial_g B = 1/2 (g' - g)
        # move g out of the LHS
        # -> g'' = 1/2 (g' - g) + g
        # if doublestep g'' = (g' - g) + g => g'' = g'
        if self.doublestep:
            gprimeprime = gprime
        else:
            gprimeprime = 0.5 * (gprime - self.g) + self.g

        # finally, apply the object domain constraint
        phs_gprime = np.angle(gprime)
        gprimeprime = self.absg * np.exp(1j*phs_gprime)

        self.costF.append(mse)
        self.iter += 1
        self.g = gprimeprime
        return gprimeprime


class ConjugateGradient:
    """The Conjugate Gradient phase retrieval algorithm."""

    def __init__(self, psf, pupil_amplitude, phase_guess=None, hk=1):
        """Create a new ConjugateGradient problem.

        Parameters
        ----------
        psf : `numpy.ndarray`
            array containing the incoherent PSF, |G|^2, of shape (m,n)
        pupil_amplitude : `numpy.ndarray`
            array containing the amplitude of the pupil, |g|, of shape (m,n)
        phase_guess : `numpy.ndarray`
            array containing the guess for the phase, np.angle(g), of shape (m,n)
        hk : `bool`, optional
            gain parameter for iteration k.  It is sub-optimal to keep this as
            a constant.  if cg = ConjugateGradient(...); modulating cg.hk in
            lockstep with iterations is superior, and constitutes an algorithm
            similar to ADAM, gradient descent that self-learns epsilon (==h)
            on each iteration

        Notes
        -----
        psf and pupil_amplitude should both be centered with the origin in the
        middle of the array.

        """
        _init_iterative_transform(self, psf, pupil_amplitude, phase_guess)
        self.gprimekm1 = self.g
        self.hk = hk

    def step(self):
        """Advance the algorithm one iteration."""
        G = fft.fft2(self.g)
        mse = _mean_square_error(abs(G), self.absF, self.mse_denom)
        Bk = mse
        phs_G = np.angle(G)
        Gprime = self.absF * np.exp(1j*phs_G)
        gprime = fft.ifft2(Gprime)

        # this is the update described in Fienup1982 Eq. 36
        # if self.iter == 0:
        #     D = gprime - self.g
        # else:
        #     D = (gprime - self.g) + (Bk/self.Bkm1) * self.Dkm1

        # gprimeprime = self.g + self.hk * D

        gprimeprime = gprime + self.hk * (gprime - self.gprimekm1)

        # finally, apply the object domain constraint
        phs_gprime = np.angle(gprimeprime)
        gprimeprime = self.absg * np.exp(1j*phs_gprime)

        self.costF.append(mse)
        self.iter += 1
        self.Bkm1 = Bk  # bkm1 = "B_{k-1}"; B for iter k-1
        # self.Dkm1 = D
        self.gprimekm1 = gprime
        self.g = gprimeprime
        return gprimeprime


class InputOutput:
    """The Input-Output phase retrieval algorithm."""

    def __init__(self, psf, pupil_amplitude, phase_guess=None, beta=1):
        """Create a new Input-Output problem.

        Parameters
        ----------
        psf : `numpy.ndarray`
            array containing the incoherent PSF, |G|^2, of shape (m,n)
        pupil_amplitude : `numpy.ndarray`
            array containing the amplitude of the pupil, |g|, of shape (m,n)
        phase_guess : `numpy.ndarray`
            array containing the guess for the phase, np.angle(g), of shape (m,n)
        beta : `bool`, optional
            gain parameter

        Notes
        -----
        psf and pupil_amplitude should both be centered with the origin in the
        middle of the array.

        """
        if phase_guess is None:
            phase_guess = np.random.rand(*pupil_amplitude.shape)

        absF = np.sqrt(psf)
        absg = pupil_amplitude

        self.absF = fft.ifftshift(absF)
        self.absg = fft.ifftshift(absg)
        phase_guess = fft.ifftshift(phase_guess)

        self.g = self.absg * np.exp(1j*phase_guess)
        self.supportmask = absg < 1e-6
        self.iter = 0
        self.beta = beta
        self.costF = []
        self.costf = []

    def step(self):
        """Advance the algorithm one iteration."""
        G = fft.fft2(self.g)
        mseF = np.sum((abs(G) - self.absF) ** 2) / G.size
        phs_G = np.angle(G)
        Gprime = self.absF * np.exp(1j*phs_G)
        gprime = fft.ifft2(Gprime)

        msef = np.sum((abs(gprime) - self.absg) ** 2) / G.size
        gprimeprime = gprime
        # update g'' where the constraints are violated
        mask = abs(gprime) < 0
        mask |= self.supportmask

        gprimeprime[~mask] = self.g[~mask]
        gprimeprime[mask] = self.g[mask] - self.beta * gprime[mask]

        # finally, apply the object domain constraint
        # phs_gprime = np.angle(gprimeprime)
        # gprimeprime = self.absg * np.exp(1j*phs_gprime)

        self.costF.append(mseF)
        self.costf.append(msef)
        self.iter += 1
        self.g = gprimeprime
        return gprimeprime


class OutputOutput:
    """The Output-Output phase retrieval algorithm."""

    def __init__(self, psf, pupil_amplitude, phase_guess=None, beta=1):
        """Create a new Output-Output problem.

        Parameters
        ----------
        psf : `numpy.ndarray`
            array containing the incoherent PSF, |G|^2, of shape (m,n)
        pupil_amplitude : `numpy.ndarray`
            array containing the amplitude of the pupil, |g|, of shape (m,n)
        phase_guess : `numpy.ndarray`
            array containing the guess for the phase, np.angle(g), of shape (m,n)
        beta : `bool`, optional
            gain parameter

        Notes
        -----
        psf and pupil_amplitude should both be centered with the origin in the
        middle of the array.

        """
        if phase_guess is None:
            phase_guess = np.random.rand(*pupil_amplitude.shape)

        absF = np.sqrt(psf)
        absg = pupil_amplitude

        self.absF = fft.ifftshift(absF)
        self.absg = fft.ifftshift(absg)
        phase_guess = fft.ifftshift(phase_guess)

        self.g = self.absg * np.exp(1j*phase_guess)
        self.supportmask = absg < 1e-6
        self.iter = 0
        self.beta = beta
        self.costF = []
        self.costf = []

    def step(self):
        """Advance the algorithm one iteration."""
        G = fft.fft2(self.g)
        mseF = np.sum((abs(G) - self.absF) ** 2) / G.size
        phs_G = np.angle(G)
        Gprime = self.absF * np.exp(1j*phs_G)
        gprime = fft.ifft2(Gprime)

        msef = np.sum((abs(gprime) - self.absg) ** 2) / G.size
        gprimeprime = gprime
        # update g'' where the constraints are violated
        mask = abs(gprime) < 0
        mask |= self.supportmask

        gprimeprime[mask] = gprime[mask] - self.beta * gprime[mask]

        self.costF.append(mseF)
        self.costf.append(msef)
        self.iter += 1
        self.g = gprimeprime
        return gprimeprime


class HybridInputOutput:
    """The Hybrid Input-Output phase retrieval algorithm."""

    def __init__(self, psf, pupil_amplitude, phase_guess=None, beta=1):
        """Create a new Hybrid Input-Output problem.

        Parameters
        ----------
        psf : `numpy.ndarray`
            array containing the incoherent PSF, |G|^2, of shape (m,n)
        pupil_amplitude : `numpy.ndarray`
            array containing the amplitude of the pupil, |g|, of shape (m,n)
        phase_guess : `numpy.ndarray`
            array containing the guess for the phase, np.angle(g), of shape (m,n)
        beta : `bool`, optional
            gain parameter

        Notes
        -----
        psf and pupil_amplitude should both be centered with the origin in the
        middle of the array.

        """
        if phase_guess is None:
            phase_guess = np.random.rand(*pupil_amplitude.shape)

        absF = np.sqrt(psf)
        absg = pupil_amplitude

        self.absF = fft.ifftshift(absF)
        self.absg = fft.ifftshift(absg)
        phase_guess = fft.ifftshift(phase_guess)

        self.g = self.absg * np.exp(1j*phase_guess)
        self.supportmask = absg < 1e-6
        self.iter = 0
        self.beta = beta
        self.costF = []
        self.costf = []

    def step(self):
        """Advance the algorithm one iteration."""
        G = fft.fft2(self.g)
        mseF = np.sum((abs(G) - self.absF) ** 2) / G.size
        phs_G = np.angle(G)
        Gprime = self.absF * np.exp(1j*phs_G)
        gprime = fft.ifft2(Gprime)

        msef = np.sum((abs(gprime) - self.absg) ** 2) / G.size
        gprimeprime = gprime
        # update g'' where the constraints are violated
        mask = abs(gprime) < 0
        mask |= self.supportmask

        # no need to do the ~mask update, gprimeprime is a copy of gprime
        gprimeprime[mask] = self.g[mask] - self.beta * gprime[mask]

        self.costF.append(mseF)
        self.costf.append(msef)
        self.iter += 1
        self.g = gprimeprime
        return gprimeprime


class MiselTwoPSF:
    """Misel's two PSF focus diversity Phase Retrieval algorithm."""

    def __init__(self, psf0, psf1, wvl, dx, dz, fno, phase_guess=None):
        """Create a new MiselTwoPSF problem.

        Parameters
        ----------
        psf0 : `numpy.ndarray`
            array containing the incoherent PSF, |F_0|^2, of shape (m,n)
        psf1 : `numpy.ndarray`
            array containing the defocused PSF, |F_1|^2, of shape (m,n)
        wvl : `float`
            wavelength, microns
        dx : `float`
            inter-sample spacing for both psf0 and psf1, microns
        dz : `float`
            longitudinal separation between psf0 and psf1, microns
        fno : `float`
            f number, dimmensionless
        phase_guess : `numpy.ndarray`
            array containing the guess for the phase, np.angle(g), of shape (m,n)

        """
        # this code is basically identical to _init_iterative_transform
        if phase_guess is None:
            phase_guess = np.random.rand(*psf0.shape)

        absF0 = np.sqrt(psf0)
        absF1 = np.sqrt(psf1)

        self.absF0 = fft.ifftshift(absF0)
        self.absF1 = fft.ifftshift(absF1)
        phase_guess = fft.ifftshift(phase_guess)

        self.G0 = self.absF0 * np.exp(1j*phase_guess)
        self.G1 = self.absF1  # per Misel, initialization of defocused plane is zero phase

        # only compute the transfer function between 0 -> 1 and 1 -> 0 one time
        # for efficiency
        self.tf0to1 = _angular_spectrum_transfer_function(psf0.shape, wvl, dx, dz)
        self.tf1to0 = _angular_spectrum_transfer_function(psf0.shape, wvl, dx, -dz)

        self.mse_denom0 = np.sum((self.absF0)**2)
        self.mse_denom1 = np.sum((self.absF1)**2)
        self.iter = 0
        self.costF0 = []
        self.costF1 = []

    def step(self):
        """Advance the algorithm one step."""
        G1 = _angular_spectrum_prop(self.G0, self.tf0to1)

        phs_G1 = np.angle(G1)
        G1prime = self.absF1 * np.exp(1j*phs_G1)

        G0prime = _angular_spectrum_prop(G1prime, self.tf1to0)

        phs_G0prime = np.angle(G0prime)
        G0primeprime = self.absF0 * np.exp(1j*phs_G0prime)

        mse1 = _mean_square_error(abs(G1), self.absF1, self.mse_denom1)
        mse0 = _mean_square_error(abs(G0prime), self.absF0, self.mse_denom0)
        self.costF0.append(mse0)
        self.costF1.append(mse1)
        self.iter += 1
        self.G0 = G0primeprime
        return G0primeprime


# angular spectrum code adapted from prysm, see prysm-LICENSE.md for compliance
def _angular_spectrum_transfer_function(shape, wvl, dx, z):
    # transfer function of free space propagation
    # units;
    #   wvl / um
    #   dx / um
    #   z / um
    ky, kx = (fft.fftfreq(s, dx) for s in shape)
    ky = np.broadcast_to(ky, shape).swapaxes(0, 1)
    kx = np.broadcast_to(kx, shape)

    coef = np.pi * wvl * z
    transfer_function = np.exp(-1j * coef * (kx**2 + ky**2))
    return transfer_function


def _angular_spectrum_prop(field, transfer_function):
    # this code is copied from prysm with some modification
    forward = fft.fft2(field)
    return fft.ifft2(forward*transfer_function)
