# Mantid Repository : https://github.com/mantidproject/mantid
#
# Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
#   NScD Oak Ridge National Laboratory, European Spallation Source,
#   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
# SPDX - License - Identifier: GPL - 3.0 +
"""
This module is a wrapper around a set of instrument parameters (to be read from a YAML file)
and methods which then call either Chop.py or MulpyRep.py to do the resolution calculations.
"""

import numpy as np
import yaml
import warnings
import copy
from . import Chop, MulpyRep
from scipy.interpolate import interp1d
from scipy.special import erf
from scipy import constants

# Some global constants
SIGMA2FWHM = 2 * np.sqrt(2 * np.log(2))
SIGMA2FWHMSQ = SIGMA2FWHM ** 2
E2V = np.sqrt((constants.e / 1000) * 2 / constants.neutron_mass) # v = E2V * sqrt(E)    veloc in m/s, E in meV
E2L = 1.e23 * constants.h**2 / (2 * constants.m_n * constants.e) # lam = sqrt(E2L / E)  lam in Angst, E in meV
E2K = constants.e * 2 * constants.m_n / constants.hbar**2 / 1e23 # k = sqrt(E2K * E)    k in 1/Angst, E in meV

def colorline(x, y, z=None, cmap=None, norm=None, linewidth=2, alpha=1.0, ax=None, expand=10):
    """ https://stackoverflow.com/a/25941474 """
    import matplotlib.pyplot, matplotlib.collections
    if cmap is None:
        cmap = 'inferno'
    if isinstance(cmap, str):
        cmap = matplotlib.pyplot.get_cmap(cmap).copy()
        cmap.set_under('gray')
        cmap.set_over('white')

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    def do_expansion(q, n):
        return np.concatenate([np.linspace(a, b, n) for a, b in zip(q[:-1], q[1:])])

    if expand and expand > 0:
        x = do_expansion(x, expand)
        y = do_expansion(y, expand)
        z = do_expansion(z, expand)

    if norm is None:
        norm = matplotlib.pyplot.Normalize(np.min(z[z > 0]), np.max(z[z < np.max(z)]))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = matplotlib.collections.LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)

    if ax is None:
        ax = matplotlib.pyplot.gca()
    ax.add_collection(lc)

    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

def window_paths(window, sample_distance, detector_distance, gain, loss, max_x):
    l_n = window[0][1]
    l_d = window[0][0]
    r_n = window[1][1]
    r_d = window[1][0]

    edge_x, edge_y, edge_z, mean_x, mean_y, mean_z, diff_x, diff_y, diff_z = [], [], [], [], [], [], [], [], []

    l_moderator_t = -l_n / l_d
    r_moderator_t = -r_n / r_d
    moderator_mean_t = (l_moderator_t + r_moderator_t) / 2.0

    if moderator_mean_t < max_x:
        l_sample_t = (sample_distance - l_n) / l_d
        r_sample_t = (sample_distance - r_n) / r_d
        l_detector_t = (detector_distance - l_n) / l_d
        r_detector_t = (detector_distance - r_n) / r_d

        sample_mean_t = (l_sample_t + r_sample_t) / 2.0
        detector_mean_t = (l_detector_t + r_detector_t) / 2.0

        mean_velocity = detector_distance / (detector_mean_t - moderator_mean_t)
        gain_velocity = mean_velocity * np.sqrt(1 + gain)
        loss_velocity = mean_velocity * np.sqrt(1 - loss)
        detector_gain_t = (detector_distance - sample_distance) / gain_velocity + sample_mean_t
        detector_loss_t = (detector_distance - sample_distance) / loss_velocity + sample_mean_t

        edge_x = [l_detector_t, l_sample_t, l_moderator_t, r_moderator_t, r_sample_t, r_detector_t]
        edge_y = [detector_distance, sample_distance, 0, 0, sample_distance, detector_distance]
        edge_z = [mean_velocity] * 6
        mean_x = [moderator_mean_t, sample_mean_t, detector_mean_t]
        mean_y = [0, sample_distance, detector_distance]
        mean_z = [mean_velocity] * 3
        diff_x = [detector_gain_t, sample_mean_t, sample_mean_t, detector_loss_t]
        diff_y = [detector_distance, sample_distance, sample_distance, detector_distance]
        diff_z = [gain_velocity, gain_velocity, loss_velocity, loss_velocity]

    return edge_x, edge_y, edge_z, mean_x, mean_y, mean_z, diff_x, diff_y, diff_z

def join_windows_paths(windows, sample_distance, detector_distance, gain, loss, max_x):
    edge_x, edge_y, edge_z = [], [], []
    mean_x, mean_y, mean_z = [], [], []
    diff_x, diff_y, diff_z = [], [], []
    avg_x = []
    first = True
    for window in windows:
        e_x, e_y, e_z, m_x, m_y, m_z, d_x, d_y, d_z = window_paths(window, sample_distance, detector_distance, gain, loss, max_x)
        if len(e_x) > 0 and len(m_x) > 0 and len(d_x) > 0:
            if not first:
                # add the end points again with zero z-value
                edge_x.append(edge_x[len(edge_x)-1])
                edge_y.append(edge_y[len(edge_y)-1])
                edge_z.append(0)
                mean_x.append(mean_x[len(mean_x)-1])
                mean_y.append(mean_y[len(mean_y)-1])
                mean_z.append(0)
                diff_x.append(diff_x[len(diff_x)-1])
                diff_y.append(diff_y[len(diff_y)-1])
                diff_z.append(0)
                # so we can move to the start points with still-zero z-value
                edge_x.append(e_x[0])
                edge_y.append(e_y[0])
                edge_z.append(0)
                mean_x.append(m_x[0])
                mean_y.append(m_y[0])
                mean_z.append(0)
                diff_x.append(d_x[0])
                diff_y.append(d_y[0])
                diff_z.append(0)
            else:
                first = False
            # insert all points and their actual z-values:
            edge_x[len(edge_x):] = e_x
            edge_y[len(edge_y):] = e_y
            edge_z[len(edge_z):] = e_z
            mean_x[len(edge_x):] = m_x
            mean_y[len(edge_y):] = m_y
            mean_z[len(edge_z):] = m_z
            diff_x[len(edge_x):] = d_x
            diff_y[len(edge_y):] = d_y
            diff_z[len(edge_z):] = d_z
            avg_x.append(m_x[len(m_x)-1])
        else:
            avg_x.append(None)
    return edge_x, edge_y, edge_z, mean_x, mean_y, mean_z, diff_x, diff_y, diff_z, avg_x

def plot_windows(windows, sample_distance, detector_distance, ax=None, gain=0.9, loss=0.9, cmap=None, linewidth=1,
                 max_x=None):
    import matplotlib.pyplot
    if cmap is None:
        cmap = 'coolwarm'
    if isinstance(cmap, str):
        cmap = matplotlib.pyplot.get_cmap(cmap).copy()
        cmap.set_under(color='white')
        cmap.set_over(color='black')

    edge_x, edge_y, edge_z, mean_x, mean_y, mean_z, diff_x, diff_y, diff_z, avg_x = join_windows_paths(windows, sample_distance, detector_distance, gain, loss, max_x)
    all_z = np.concatenate([edge_z, mean_z, diff_z]).flatten()
    min_z = np.min(all_z[all_z > 0])
    max_z = np.max(all_z[np.isfinite(all_z)])
    norm = matplotlib.pyplot.Normalize(min_z, max_z)

    if ax is None:
        ax = matplotlib.pyplot.gca()

    #colorline(mean_x, mean_y, mean_z, cmap=cmap, norm=norm, linewidth=linewidth, ax=ax, expand=None)
    colorline(edge_x, edge_y, edge_z, cmap=cmap, norm=norm, linewidth=linewidth, ax=ax, expand=None)
    colorline(diff_x, diff_y, diff_z, cmap=cmap, norm=norm, linewidth=linewidth, ax=ax, expand=None)
    return avg_x

def plot_window(window, sample_distance, detector_distance, ax=None, gain=0.9, loss=0.9, color='b'):
    import matplotlib.pyplot
    if ax is None:
        ax = matplotlib.pyplot.gca()

    path_x, path_y, _, mean_x, mean_y, _, diff_x, diff_y, _= window_paths(window, sample_distance, detector_distance, gain, loss)

    # draw the full window
    ax.plot(path_x, path_y, color='k')
    # draw the average path
    ax.plot(mean_x, mean_y, color=color)
    # draw the energy transfer window
    ax.plot(diff_x, diff_y, color='r')

    return detector_mean_t

def wrap_attributes(obj, inval, allowed_var_names):
    for key in allowed_var_names:
        if inval:
            if hasattr(inval, key):
                setattr(obj, key, getattr(inval, key))
            elif hasattr(inval, '__getitem__') and key in inval:
                setattr(obj, key, inval[key])


def arg_parser(args, kwargs, arg_names, defaults=None):
    """Parse positional and keyword arguments into a single dictionary

    The number of positional arguments (provided as a packed tuple in the first argument) must not exceed the number of
    'arg_names' dictionary keys.
    Optional default values for any 'arg_names' keys can be provided.
    """
    arg_dict = {key: val for key, val in zip(arg_names, defaults if defaults else [None] * len(arg_names))}
    for key in kwargs:
        arg_dict[key] = kwargs[key]
    for idx in range(len(args)):
        arg_dict[arg_names[idx]] = args[idx]
    return arg_dict


def _check_input(self, *args):
    vtype = ['Incident energy', 'Frequency']
    defval = [self.ei, self.frequency]
    retval = [defval[i] if args[i] is None else args[i] for i in range(min([len(defval), len(args)]))]
    if [v for v in retval if v is None]:
        raise ValueError('%s has not been specified.' % (vtype[[i for i in range(len(retval)) if retval[i] is None][0]]))
    return tuple(retval) if len(retval) > 1 else retval[0]


def soft_hat(x, p):
    """
    ! Soft hat function, from Herbert subroutine library.
    ! For rescaling t-mod at low energy to account for broader moderator term
    """
    x = np.array(x)
    sig2fwhh = np.sqrt(8*np.log(2))
    height, grad, x1, x2 = tuple(p[:4])
    sig1, sig2 = tuple(np.abs(p[4:6]/sig2fwhh))
    # linearly interpolate sig for x1<x<x2
    sig = ((x2-x)*sig1-(x1-x)*sig2)/(x2-x1)
    if np.shape(sig):
        sig[x < x1] = sig1
        sig[x > x2] = sig2
    # calculate blurred hat function with gradient
    e1 = (x1-x) / (np.sqrt(2)*sig)
    e2 = (x2-x) / (np.sqrt(2)*sig)
    y = (erf(e2)-erf(e1)) * ((height+grad*(x-(x2+x1)/2))/2)
    y = y + 1
    return y


class FermiChopper(object):
    """
    Class which represents a Fermi chopper package
    """

    __allowed_var_names = ['name', 'pslit', 'pslat', 'radius', 'rho', 'tjit', 'fluxcorr', 'isPi']

    def __init__(self, inval=None):
        wrap_attributes(self, inval, self.__allowed_var_names)

    def __repr__(self):
        return self.name if self.name else 'Undefined Fermi chopper package'

    def getWidthSquared(self, Ei, freq):
        return Chop.tchop(freq, Ei, self.pslit / 1000., self.radius / 1000., self.rho / 1000.)

    def getWidth(self, *args):
        """ Calculates the chopper time width in seconds for a given neutron energy (Ei) """
        return np.sqrt(self.getWidthSquared(*args))

    def getTransmission(self, Ei, freq):
        """ Calculates the chopper transmission """
        dslat = (self.pslit + self.pslat) / 1000
        return Chop.achop(Ei, freq, dslat, self.pslit / 1000., self.radius / 1000., self.rho / 1000.) / dslat


class ChopperSystem(object):
    """
    Class which represents a set (list) of choppers in a line
    """

    __allowed_var_names = ['name', 'chop_sam', 'sam_det', 'aperture_width', 'aperture_height', 'choppers', 'variants',
                           'frequency_matrix', 'constant_frequencies', 'max_frequencies', 'default_frequencies',
                           'source_rep', 'n_frame', 'emission_time', 'overlap_ei_frac', 'ei_limits',
                           'flux_ref_slot', 'flux_ref_freq', 'frequency_names']

    def __init__(self, inval=None):
        # Default values
        self.source_rep = 50
        self.emission_time = 0
        self.overlap_ei_frac = 0.9
        self.n_frame = 1
        self._ei = None
        # Parse input values (if any)
        wrap_attributes(self, inval, self.__allowed_var_names)
        self._parse_choppers()
        self._parse_variants()
        self.phase = self.phase_shift
        self.slot = self.phase_slot
        self.frequency = self.default_frequencies
        self.gain_ei_frac = self.overlap_ei_frac
        self.loss_ei_frac = self.overlap_ei_frac

    def __repr__(self):
        return self.name if self.name else 'Undefined disk chopper system'

    def _parse_choppers(self):
        """Parses the choppers list to determine how to handle resolution and flux calculations"""
        if not self.choppers:
            return
        self.distance = []
        self.nslot = []
        self.slot_ang_pos = []
        self.slot_width = []
        self.guide_width = []
        self.radius = []
        self.numDisk = []
        self.isFermi = False
        self.phase_to = []
        self.phase_shift = []
        self.phase_names = []
        self.phase_slot = []
        self.slot_names = []
        for idx, chopper in enumerate(self.choppers):
            self.distance.append(chopper['distance'])
            if 'packages' in chopper:
                self.isFermi = True
                self.idxFermi = idx
                self.packages = {key: FermiChopper(val) for key, val in list(chopper['packages'].items())}
                self.nslot.append(1)    # Assume Fermi chopper is curved, will not transmit PI pulse.
                self.slot_ang_pos.append(None)
                self.slot_width.append(10.)
                self.guide_width.append(10.)
                self.radius.append(290.)
                self.numDisk.append(1)
            else:
                self.nslot.append(chopper.get('nslot', len(chopper.get('slot_ang_pos', [0, ]))))
                self.slot_ang_pos.append(chopper.get('slot_ang_pos', None))
                self.slot_width.append(chopper['slot_width'])
                self.guide_width.append(chopper['guide_width'])
                self.radius.append(chopper['radius'])
                self.numDisk.append(2 if chopper.get('isDouble', False) else 1)
            self.phase_to.append(chopper.get('phase_to', 'Ei').lower())
            self.phase_shift.append(chopper.get('phase_shift', None))
            self.phase_slot.append(chopper.get('phase_slot', None))
            slot_label = 'Chopper {:d} {:s}'.format(idx, 'slot index')
            phase_label = 'Chopper {:d} phase{:s} (µs)'.format(idx, '' if 'source' in self.phase_to[-1] else ' offset')
            self.phase_names.append(chopper.get('phase_shift_name', phase_label))
            self.slot_names.append(chopper.get('phase_slot_name', slot_label))
        if not any(self.slot_ang_pos):
            self.slot_ang_pos = None
        source_rep = self.source_rep if (not hasattr(self, 'n_frame') or self.n_frame == 1) else [self.source_rep, self.n_frame]
        self._instpar = [self.distance, self.nslot, self.slot_ang_pos, self.slot_width, self.guide_width, self.radius,
                         self.numDisk, self.sam_det, self.chop_sam, source_rep, self.emission_time,
                         self.overlap_ei_frac, self.phase_to]
        if self.isFermi:
            self.package = list(self.packages.keys())[0]

    def _parse_variants(self):
        if 'variants' not in self.__dict__:
            return
        variant_keys = []
        for var in self.variants:
            if ('default' in self.variants[var].keys() and self.variants[var]['default']) or var is None:
                self._default_variant = self._variant = var
            if var:
                [variant_keys.append(key) for key in self.variants[var].keys() if 'default' not in key]
        self._variant_defaults = {}
        for key in set(variant_keys):
            self._variant_defaults[key] = copy.deepcopy(getattr(self, key))
        if '_variant' not in self.__dict__:
            self._default_variant = list(self.variants.keys())[0]
            warnings.warn('No default variants defined. Using ''%s'' as default' % (self._default_variant), SyntaxWarning)
            self.variant = self._default_variant

    # Define getters/setters here to be backwards compatible with old PyChop2. Actually use properties underneath
    def setChopper(self, *args, **kwargs):
        """
        Set the chopper package type (Fermi instruments) or variant (LET).

        maps = Instrument('MAPS')
        maps.setChopper('A', 400)                     # Sets package A at 400 Hz.
        maps.setChopper(package='A', freq=400)        # Explicit keywords
        let = Instrument('LET')
        let.setChopper('High resolution', [280, 140]) # Change to the high resolution variant at 280 Hz
        let.setChopper(variant='High resolution')
        """
        argdict = arg_parser(args, kwargs, ['package' if self.isFermi else 'variant', 'freq'])
        if self.isFermi:
            self.package = argdict['package']
            if hasattr(self, 'variants') and argdict['package'] in self.variants:
                self.variant = argdict['package']
        elif argdict['variant']:
            self.variant = argdict['variant']
        if argdict['freq']:
            self.frequency = argdict['freq']

    def getChopper(self):
        return self.package if self.isFermi else self.variant

    def getChopperNames(self):
        choppers = list(self.packages.keys()) if self.isFermi else []
        return sorted(set(choppers + (list(self.variants.keys()) if hasattr(self, 'variants') else [])))

    def setFrequency(self, *args, **kwargs):
        """
        Set the chopper frequency(ies) and (optionally) phase(s).

        maps = Instrument('MAPS', 'A')
        maps.setFrequency(400)                        # Sets frequency to 400 Hz.
        maps.setFrequency([400, 100], 1)              # Sets Fermi to 400Hz, disk to 100Hz, and multi-rep mode
        maps.setFrequency(freq=[400, 100], phase=1)
        let = Instrument('LET')
        let.setFrequency([240, 120])                  # Sets resolution chopper to 240Hz, pulse removal to 120Hz
        let.setFrequency([240, 120], phase=-20000)    # Additionally sets the frame overlap phase to -20000us
        """
        arg_dict = arg_parser(args, kwargs, ['freq', 'phase', 'slot'])
        if arg_dict['freq']:
            self.frequency = arg_dict['freq']
        if arg_dict['phase']:
            # this sets only the phases visible in the GUI!
            self.phase = arg_dict['phase']
        if arg_dict['slot']:
            self.slot = arg_dict['slot']

    def getFrequency(self):
        return self.frequency

    def setEi(self, Ei):
        """Sets the (focussed) incident energy"""
        self.ei = Ei

    def getEi(self):
        return self.ei

    def getAllowedEi(self, Ei_in=None):
        return set(np.round(self._MulpyRepDriver(Ei_in, calc_res=False)[0], decimals=4))

    def plotMultiRepFrame(self, h_plt=None, Ei_in=None, frequency=None, first_rep=False):
        """
        Plots the time-distance diagram into a given Matplotlib axes, h_plt
        for a give focused incident energy (in meV) and chopper frequencies (in Hz).
        """
        if h_plt is None:
            try:
                from matplotlib import pyplot
            except ImportError:
                raise RuntimeError('plotMultiRepFrame: Cannot import matplotlib')
            plt = pyplot
        else:
            plt = h_plt
        _check_input(self, Ei_in)
        if frequency:
            freq_stash = self.freq
            self.setFrequency(frequency)
        # eis, _,  _, lines, chop_times = tuple(self._MulpyRepDriver(Ei_in, calc_res=False))
        eis, lines, chop_times = self._NewMulpyRepDriver(Ei_in)
        if frequency:
            self.setFrequency(freq_stash)
        d_sample = self.distance[-1] + self.chop_sam
        d_detector = d_sample + self.sam_det
        source_period_us = 1.e6/self.source_rep
        max_time = source_period_us
        if hasattr(self, 'n_frame') and self.n_frame > 1:
            max_time *= self.n_frame
            for i in range(1, self.n_frame):
                plt.plot([i * source_period_us] * 2, [0, d_detector], color='gray', linewidth=2.)
        t_range = [-source_period_us, max_time]
        # plot the chopper blocked (black) and open (white) times
        for t_chopper_windows, d_chopper in zip(chop_times, self.distance):
            n = len(t_chopper_windows)
            x = np.array(t_chopper_windows).flatten()
            y = np.tile(np.array([d_chopper]*4), (n, 1)).flatten()
            z = np.tile(np.array((0, 1, 1, 0)), (n, 1)).flatten()
            if x[0] > t_range[0]:
                x = np.append(t_range[0], x)
                y = np.append(y[0], y)
                z = np.append(z[0], z)
            if x[-1] < t_range[1]:
                x = np.append(x, t_range[1])
                y = np.append(y, y[-1])
                z = np.append(z, z[-1])
            colorline(x, y, z, ax=plt, expand=20)
        t_detectors = plot_windows(lines, d_sample, d_detector, ax=plt, gain=self.gain_ei_frac, loss=self.loss_ei_frac,
                                   max_x= self.emission_time if first_rep else np.inf)
        # indicate the average incident energy on the plot above the detector line:
        for ei, t_detector in zip(eis, t_detectors):
            plt.text(t_detector, d_detector+0.5, '{:4.2f}'.format(ei))
        # # plot the detector
        plt.plot(t_range, [d_detector]*2, color='black')
        if h_plt is None:
            plt.xlim(0, max_time)
            plt.xlabel(r'TOF ($\mu$sec)')
            plt.ylabel('Distance (m)')
            plt.show()
        else:
            plt.set_xlim(0, max_time)
            plt.set_xlabel(r'TOF ($\mu$sec)')
            plt.set_ylabel(r'Distance (m)')

    def getWidthSquared(self, Ei_in=None):
        return self.getWidth(Ei_in, squared=True)

    def getWidth(self, Ei_in=None, squared=False):
        """Returns the chopper time width (FWHM) at the (final) chopper in microseconds"""
        if self.isFermi:
            return self._ChopDriver(Ei_in, squared), None
        else:
            chop_times = self._MulpyRepDriver(Ei_in, calc_res=False)[1]
            # Output of MulpyRep is FWHM in us - want it in seconds for later calculations
            wd = ((chop_times[1][1] - chop_times[1][0]) / 2. / 1.e6, (chop_times[0][1] - chop_times[0][0]) / 2. / 1.e6)
            return (wd[0]**2, wd[1]**2) if squared else wd

    def getDistances(self):
        """ Returns the (mod->final_chop, aperture->final, chop->sam, sam->det, mod-first_chop) distances for instrument """
        mod_chop = self.choppers[-1]['distance']
        ap_chop = self.choppers[-1]['aperture_distance'] if ('aperture_distance' in self.choppers[-1]) else mod_chop
        return (mod_chop, ap_chop, self.chop_sam, self.sam_det, self.choppers[0]['distance'])

    def getTransmission(self, Ei_in=None, frequency=None, hires=False):
        """ Calculates the flux transmission fraction through the chopper system at specified Ei and frequency """
        Ei = _check_input(self, Ei_in)
        freq = frequency if frequency is not None else self._long_frequency[-1]
        if self.isFermi:
            x0, x1 = (self.choppers[-1]['distance'], self.chop_sam)
            magic = (84403.06 / x0 / (x1 + x0))          # Some magical conversion factor (??)
            fudge = self.packages[self.package].fluxcorr # A chopper package dependent fudge factor
            return self.packages[self.package].getTransmission(Ei, freq) * magic / fudge
        else:
            # For disk choppers, transmission goes quadratic with freq at high resolution, linear at low
            freqdep = (self.flux_ref_freq / freq)**2 if hires else  (self.flux_ref_freq / freq)
            return (self.slot_width[-1] / self.flux_ref_slot) * freqdep

    def setNFrame(self, value):
        self.n_frame = value
        self._instpar[9] = [self.source_rep, value]

    def _get_state(self, Ei_in=None):
        return hash((self.variant, self.package, tuple(self.frequency), tuple(self.phase), tuple(self.slot), Ei_in if Ei_in else self.ei, self.n_frame))

    def _removeLowIntensityReps(self, Eis, lines, Ei=None):
        # Removes reps with Ei where there are no neutrons
        if not hasattr(self, 'ei_limits') or not self.ei_limits:
            return Eis, lines
        Eis = np.array(Eis)
        idx = (Eis >= self.ei_limits[0]) * (Eis <= self.ei_limits[1])
        # Always keeps desired rep even if outside of range
        if Ei:
            idx += ((np.abs(Eis - Ei) / np.abs(Eis)) < 0.1)
        Eis = Eis[idx]
        lines = np.array(lines)[idx]
        return Eis, lines

    def _MulpyRepDriver(self, Ei_in=None, calc_res=True):
        """Private method to calculate resolution for given Ei from chopper opening times"""
        Ei = _check_input(self, Ei_in)
        if '_saved_state' not in self.__dict__ or (self._saved_state[0] != self._get_state(Ei)):
            Eis, all_times, chop_times, lastChopDist, lines = MulpyRep.calcChopTimes(Ei, self._long_frequency, self._instpar, self.phase, self.slot)
            Eis, lines = self._removeLowIntensityReps(Eis, lines, Ei)
            self._saved_state = [self._get_state(Ei), Eis, chop_times, lastChopDist, lines, all_times]
        else:
            Eis, chop_times, lastChopDist, lines, all_times = tuple(self._saved_state[1:])
        if calc_res:
            res_el, percent, chop_width, mod_width = MulpyRep.calcRes(Eis, chop_times, lastChopDist, self.chop_sam,
                                                                      self.sam_det, self.guide_width[-1], self.slot_width[-1])
            return res_el, percent, chop_width, mod_width
        else:
            return [Eis, chop_times, lastChopDist, lines, all_times]

    def _NewMulpyRepDriver(self, Ei_in=None):
        """Private method to calculate resolution for given Ei from chopper opening times"""
        incident_energy = _check_input(self, Ei_in)
        if not hasattr(self, '_new_saved_state') or (self._new_saved_state[0] != self._get_state(incident_energy)):
            eis, all_times, chop_times, last_chop_dist, lines = MulpyRep.newCalcChopTimes(incident_energy, self._long_frequency, self._instpar, self.phase, self.slot)
            # eis, lines = self._removeLowIntensityReps(eis, lines, incident_energy)
            self._new_saved_state = self._get_state(incident_energy), eis, chop_times, last_chop_dist, lines, all_times
        else:
            eis, chop_times, last_chop_dist, lines, all_times = self._new_saved_state[1:]

        return eis, lines, all_times


    def _ChopDriver(self, Ei_in=None, squared=False):
        """Private method to calculate resolution for given Ei from Fermi chopper"""
        Ei = _check_input(self, Ei_in)
        if squared:
            return self.packages[self.package].getWidthSquared(Ei, self._long_frequency[-1]) * SIGMA2FWHMSQ
        else:
            return self.packages[self.package].getWidth(Ei, self._long_frequency[-1]) * SIGMA2FWHM

    @property
    def frequency(self):
        return self._frequency

    @frequency.setter
    def frequency(self, value):
        freq = self.default_frequencies
        if not hasattr(value, '__len__'):
            value = [value]
        freq = [value[i] if i < len(value) else freq[i] for i in range(len(freq))]
        if self.max_frequencies and not (freq <= self.max_frequencies):
            raise ValueError('Value of frequencies outside maximum allowed')
        self._frequency = freq
        if hasattr(self, 'constant_frequencies') and self.constant_frequencies:
            f0 = self.constant_frequencies
        else:
            f0 = [0] * np.shape(self.frequency_matrix)[0]
        self._long_frequency = np.dot(self.frequency_matrix, freq) + f0

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, values):
        phase = copy.deepcopy(self.phase_shift)
        if not hasattr(values, '__len__'):
            values = [values]
        # only the independent or semi-automatic chopper phases should be updated, so None values should not be replaced
        if len(phase) == len(values):
            # *something* has been specified for every chopper
            phase = [p if p is None else v for v, p in zip(values, phase)]
        else:
            # (probably) only the independent and/or semi-automatic choppers phases have been specified
            ind_or_semi = [i for i, p in enumerate(phase) if p is not None]
            if len(values) != len(ind_or_semi):
                raise RuntimeError('Unexpected number of phases provided to setter')
            for index, value in zip(ind_or_semi, values):
                phase[index] = value
        self._phase = phase

    @property
    def slot(self):
        return self._slot

    @slot.setter
    def slot(self, values):
        slot = copy.deepcopy(self.phase_slot)
        if not hasattr(values, '__len__'):
            values = [values]
        if len(slot) == len(values):
            slot = [s if s is None else v for v, s in zip(values, slot)]
        else:
            changeable = [i for i, s in enumerate(slot) if s is not None]
            if len(values) != len(changeable):
                raise RuntimeError('Unexpected number of slots provided to setter')
            for index, value in zip(changeable, values):
                slot[index] = value
        self._slot = slot

    @property
    def gain_ei_frac(self):
        return self._gain_ei_frac

    @gain_ei_frac.setter
    def gain_ei_frac(self, value):
        self._gain_ei_frac = np.abs(float(value[0])) if hasattr(value, '__len__') else float(value)

    @property
    def loss_ei_frac(self):
        return self._loss_ei_frac

    @loss_ei_frac.setter
    def loss_ei_frac(self, value):
        self._loss_ei_frac = np.abs(float(value[1])) if hasattr(value, '__len__') else float(value)

    @property
    def ei(self):
        return self._ei

    @ei.setter
    def ei(self, value):
        if value < 0:
            raise ValueError('Incident neutron energy cannot be less than zero')
        self._ei = value

    @property
    def package(self):
        return self._package if self.isFermi else None

    @package.setter
    def package(self, value):
        if not self.isFermi:
            raise AttributeError('Cannot set Fermi chopper package on this instrument')
        if value not in self.packages.keys():
            ky = [k for k in self.packages.keys() if value.upper() == k.upper()]
            if not ky:
                raise ValueError('Fermi package ''%s'' not recognised. Allowed values are: %s'
                                 % (value, ', '.join(self.packages.keys())))
            else:
                value = ky[0]
        self._package = value
        # Sets whether to allow pi pulse or not
        idx = [i for i in range(len(self.choppers)) if 'packages' in self.choppers[i]][0]
        self._instpar[1][idx] = 2 if self.packages[value].isPi else 1

    @property
    def variant(self):
        return self._variant if hasattr(self, 'variants') and self.variants else None

    def getAllowedChopper(self):
        return self.packages.keys() if self.isFermi else (self.variants.keys() if self.variants else None)

    @variant.setter
    def variant(self, value):
        if 'variants' not in self.__dict__:
            raise AttributeError('This instrument has no variants to set')
        for prop in self._variant_defaults:
            setattr(self, prop, copy.deepcopy(self._variant_defaults[prop]))
        self._variant = value
        if value not in self.variants.keys():
            ky = [k for k in self.variants.keys() if value.upper() == k.upper()]
            if not ky:
                raise ValueError('Variant ''%s'' not recognised. Allowed values are: %s'
                                 % (value, ', '.join(self.variants.keys())))
            else:
                value = ky[0]
        for prop in self.variants[value]:
            if prop == 'choppers':
                for idx, chopper in enumerate(self.variants[value][prop]):
                    if chopper:
                        for ky in chopper:
                            self.choppers[idx][ky] = chopper[ky]
            elif prop in self.__allowed_var_names:
                setattr(self, prop, self.variants[value][prop])
        self._parse_choppers()

    @property
    def tjit(self):
        return self.packages[self.package].tjit if self.isFermi else 0.

    @tjit.setter
    def tjit(self, value):
        self.packages[self.package].tjit = value

    @property
    def emin(self):
        return self.ei_limits[0] if hasattr(self, 'ei_limits') and self.ei_limits else 0.1

    @property
    def emax(self):
        return self.ei_limits[1] if hasattr(self, 'ei_limits') and self.ei_limits else 1000


class Moderator(object):
    """
    Class which represents a neutron moderator
    """

    __allowed_var_names = ['name', 'imod', 'ch_mod', 'mod_pars', 'mod_scale_fn', 'mod_scale_par', 'theta',
                           'source_rep', 'n_frame', 'emission_time', 'measured_flux', 'measured_width']

    def __init__(self, inval=None):
        wrap_attributes(self, inval, self.__allowed_var_names)
        if hasattr(self, 'measured_flux') and self.measured_flux:
            if 'scale_factor' in self.measured_flux:
                self.measured_flux['flux'] = np.array(self.measured_flux['flux']) * float(self.measured_flux['scale_factor'])
            idx = np.argsort(self.measured_flux['wavelength'])
            wavelength = np.array(self.measured_flux['wavelength'])[idx]
            flux = np.array(self.measured_flux['flux'])[idx]
            self.flux_interp = interp1d(wavelength, flux, kind='cubic')
            self.fmn, self.fmx = (min(wavelength), max(wavelength))
        if hasattr(self, 'measured_width') and self.measured_width:
            idx = np.argsort(self.measured_width['wavelength'])
            wavelength = np.array(self.measured_width['wavelength'])[idx]
            widths = np.array(self.measured_width['width'])[idx]
            self.width_interp = interp1d(wavelength, widths, kind='slinear')
            self.wmn, self.wmx = (min(wavelength), max(wavelength))
            if 'isSigma' not in self.measured_width.keys():
                self.measured_width['isSigma'] = False

    def __repr__(self):
        return self.name if self.name else 'Undefined neutron moderator'

    def getAnalyticWidthsSquared(self, Ei):
        if self.imod == 0:
            # CHOP outputs the Gaussian sigma^2 in s^2, we want FWHM^2 in s^2
            tsqmod = Chop.tchi(self.mod_pars / 1000., Ei) * SIGMA2FWHMSQ
        elif self.imod == 1:
            tsqmod = Chop.tikeda(*tuple(self.mod_pars + [Ei])) * SIGMA2FWHMSQ
        elif self.imod == 2:
            d0 = self.mod_pars[0]
            if hasattr(self, 'mod_scale_fn') and self.mod_scale_fn:
                try:
                    d0 *= globals()[self.mod_scale_fn](Ei, self.mod_scale_par)
                except KeyError:
                    pass
            tsqmod = Chop.tchi_2(d0 / 1000., self.mod_pars[1] / 1000., Ei) * SIGMA2FWHMSQ
        elif self.imod == 3:
            # Mode for LET - output of polynomial is FWHM in us
            tsqmod = np.polyval(self.mod_pars, np.sqrt(E2L / Ei))**2 / 1e12
        else:
            raise RuntimeError('PyChop: Undefined moderator time profile type %d' % (self.imod))
        return tsqmod

    def getWidthSquared(self, Ei):
        """ Returns the squared time gaussian FWHM width due to the sample in s^2 """
        if hasattr(self, 'width_interp'):
            wavelength = np.sqrt(E2L / (Ei if not hasattr(Ei, '__len__') else Ei[0]))
            if wavelength >= self.wmn:
                # Data is obtained from measuring widths of powder Bragg peaks in backscattering
                # At low wavelengths / high energies, the peaks are too close together to discern
                # so there is no measurements, but the analytical expressions should still be good.
                width = self.width_interp(min([wavelength, self.wmx]))**2 / 1e12
                return (width * SIGMA2FWHMSQ) if self.measured_width['isSigma'] else width
        return self.getAnalyticWidthsSquared(Ei)

    def getWidth(self, Ei):
        """ Calculates the moderator time width in seconds for a given neutron energy (Ei) """
        if hasattr(self, 'width_interp'):
            wavelength = np.sqrt(E2L / (Ei if not hasattr(Ei, '__len__') else Ei[0]))
            if wavelength >= self.wmn:
                width = self.width_interp(min([wavelength, self.wmx])) / 1e6  # Table has widths in microseconds
                return width * SIGMA2FWHM if self.measured_width['isSigma'] else width
        if self.imod == 3:
            # Mode for LET - output of polynomial is FWHM in us
            return np.polyval(self.mod_pars, np.sqrt(E2L / Ei)) / 1e6
        else:
            return np.sqrt(self.getAnalyticWidthSquared(Ei))

    def getFlux(self, Ei):
        """ Returns the white beam flux estimate from either measured data (preferred) or analytical model (backup) """
        return self.getMeasuredFlux(Ei) if hasattr(self, 'flux_interp') else self.getAnalyticFlux(Ei)

    def getAnalyticFlux(self, Ei):
        """ Estimate white beam flux from TGP's model of the moderators (ISIS TS1 only) """
        if all([self.name != modtype for modtype in ['AP', 'CH4', 'H2']]):
            raise AttributeError('No analytical model for moderator %s' % (self.name))
        return Chop.flux_calc(np.array(Ei), self.name, self.theta_m * np.pi / 180.)

    def getMeasuredFlux(self, Ei):
        """ Interpolates flux from a table of measured flux """
        if not hasattr(self, 'flux_interp'):
            raise AttributeError('This instrument does not have a table of measured flux')
        wavelength = [min(max(l, self.fmn), self.fmx) for l in np.sqrt(E2L / np.array(Ei if hasattr(Ei, '__len__') else [Ei]))]
        return self.flux_interp(wavelength)

    @property
    def theta_m(self):
        return self.theta if (hasattr(self, 'theta') and self.theta) else 0.


class Sample(object):
    """
    Class which represents a sample shape
    """

    __allowed_var_names = ['name', 'sx', 'sy', 'sz', 'isam', 'gamma']

    def __init__(self, inval=None):
        wrap_attributes(self, inval, self.__allowed_var_names)

    def __repr__(self):
        return self.name if self.name else 'Undefined sample'

    def getWidthSquared(self):
        """ Returns the squared time FWHM due to the sample in s^2 """
        # At the moment this routine only returns a non-zero y (beam-axis) width
        return Chop.sam0(self.sx / 1000., self.sy / 1000., self.sz / 1000., self.isam)[1] * SIGMA2FWHMSQ

    def getWidth(self):
        return np.sqrt(self.getWidthSquared)

    @property
    def gamma_deg(self):
        return self.gamma if (hasattr(self, 'gamma') and self.gamma) else 0.


class Detector(object):
    """
    Class which represents a neutron detector
    """

    __allowed_var_names = ['name', 'idet', 'dd', 'tbin', 'phi', 'tthlims']

    def __init__(self, inval=None):
        wrap_attributes(self, inval, self.__allowed_var_names)

    def __repr__(self):
        return self.name if self.name else 'Undefined detector'

    def getWidthSquared(self, Ei, en=0):
        """ Returns the squared time FWHM due to the detector in s^2 """
        return self.getWidth(Ei, en) ** 2

    def getWidth(self, Ei, en=0):
        return Chop.detect2(1., 1., np.sqrt(E2K * (Ei-en)), self.idet, self.dd)[3] * SIGMA2FWHM

    @property
    def phi_deg(self):
        return self.phi if (hasattr(self, 'phi') and self.phi) else 0.


class Instrument(object):
    """
    Class which represents a direct geometry neutron spectrometer
    """

    __allowed_var_names = ['name', 'sample', 'chopper_system', 'moderator', 'detector']

    __child_methods = ['setChopper', 'getChopper', 'setFrequency', 'getFrequency', 'setEi', 'getEi',
                       'getAllowedEi', 'plotMultiRepFrame', 'getChopperNames', 'isFermi']

    __child_properties = ['package', 'variant', 'frequency', 'phase', 'ei', 'tjit', 'emin', 'emax']

    __known_instruments = ['let', 'maps', 'mari', 'merlin']

    def __init__(self, instrument, chopper=None, freq=None):
        if isinstance(instrument, str):
            # check if it is a file or instrument name we want
            if instrument.lower() in self.__known_instruments:
                import os.path
                import sys
                folder = os.path.dirname(sys.modules[self.__module__].__file__)
                instrument = os.path.join(folder, instrument.lower() + '.yaml')
            try:
                with open(instrument) as f:
                    instrument = yaml.safe_load(f)
            except (OSError, IOError) as e:
                raise RuntimeError('Cannot open file %s . Error is %s' % (instrument, e))
        if ((hasattr(instrument, 'moderator') or hasattr(instrument, 'chopper_system'))
                or ('moderator' in instrument or 'chopper_system' in instrument)):
            wrap_attributes(self, instrument, self.__allowed_var_names)
            if isinstance(self.moderator, dict) and isinstance(self.chopper_system, dict):
                for key in ['source_rep', 'n_frame', 'emission_time']:
                    if key in self.moderator:
                        self.chopper_system[key] = self.moderator[key]
        else:
            raise RuntimeError('Input to Instrument must be an Instrument object, a dictionary or a filename string')
        # If we have just loaded a YAML file or constructed from a dictionary, need to convert to correct class
        for elem_nm, classref in zip(['sample', 'chopper_system', 'moderator', 'detector'],
                                     [Sample, ChopperSystem, Moderator, Detector]):
            try:
                element = getattr(self, elem_nm)
                if isinstance(element, dict):
                    setattr(self, elem_nm, classref(element))
                setattr(self, 'has_' + elem_nm, True)
            except AttributeError:
                setattr(self, 'has_' + elem_nm, False)
        if not self.has_chopper_system or not self.has_moderator:
            raise AttributeError('No chopper system or moderator found in input.')
        for method in self.__child_methods:
            setattr(self, method, getattr(self.chopper_system, method))
        for prop in self.__child_properties:
            setattr(type(self), prop,
                    property(lambda obj, prop=prop, self=self: ChopperSystem.__dict__[prop].__get__(self.chopper_system, ChopperSystem),
                             lambda obj, val, prop=prop, self=self: ChopperSystem.__dict__[prop].__set__(self.chopper_system, val)))
        # Now reset default chopper/variant and frequency
        if chopper or freq:
            self.setChopper(chopper if chopper else self.getChopper(), freq if freq else self.frequency)

    def setInstrument(self, instrument):
        self.__dict__.clear()
        self.__init__(instrument)

    def getFlux(self, Ei_in=None, frequency=None):
        """ Returns the monochromatic flux estimate in n/cm^2/s """
        Ei = _check_input(self.chopper_system, Ei_in)
        isHires = False if (self.isFermi or (self.getResolution(0., Ei) / Ei) > 0.02) else True
        return self.moderator.getFlux(Ei) * self.chopper_system.getTransmission(Ei, frequency, hires=isHires)

    def getMultiRepFlux(self, Ei_in=None, frequency=None):
        Ei, _ = _check_input(self.chopper_system, Ei_in, frequency)
        if frequency:
            oldfreq = self.frequency
            self.frequency = frequency
        fluxes = [self.getFlux(ei, frequency) for ei in self.getAllowedEi(Ei)]
        if frequency:
            self.frequency = oldfreq
        return fluxes

    def getResFlux(self, Etrans=None, Ei_in=None, frequency=None):
        """ Returns the resolution and flux as a tuple. """
        return self.getResolution(Etrans, Ei_in, frequency), self.getFlux(Ei_in, frequency)

    def getWidths(self, Ei_in=None, frequency=None):
        """ Returns the time FWHM of different components for one rep (Ei) in microseconds """
        Ei = _check_input(self.chopper_system, Ei_in)
        try:
            widths = self.getVanVar(Ei, frequency)
        except ValueError:
            return None
        widths[1]['Energy'] = (2 * E2V * np.sqrt(Ei**3 * widths[0])) / self.chopper_system.sam_det
        return {k: v if 'Energy' in k else np.sqrt(v)*1e6 for k, v in list(widths[1].items())}

    def getMultiWidths(self, Ei_in=None, frequency=None):
        """ Returns the time FWHM of different components for each possible rep (Ei) in seconds"""
        Ei = _check_input(self.chopper_system, Ei_in)
        Eis = self.getAllowedEi(Ei)
        outdic = {'Eis': Eis}
        widths = [self.getWidths(ei, frequency) for ei in Eis]
        for ky in widths[0]:
            outdic[ky] = np.hstack(np.array([w[ky] for w in widths if w]))
        return outdic

    def getResolution(self, Etrans=None, Ei_in=None, frequency=None):
        """
        Calculates resolution (energy) widths

        van = getResolution()
        van = getResolution(etrans)
        van = getResolution(etrans, ei, omega)

        Inputs:
            etrans - list of numpy array of energy transfers to calculate for (meV) [default: linspace(0.05Ei, 0.95Ei, 19)]
            ei - incident energy in meV [default: preset energy]
            omega - chopper frequency in Hz  [default: preset frequency]

        Output:
            van - the incoherent (Vanadium) energy FWHM at etrans in meV
        """
        Ei = _check_input(self.chopper_system, Ei_in)
        # If not set, sets energy transfers to values to compare exactly to RAE's original implementation.
        if Etrans is None:
            Etrans = np.linspace(0.05*Ei, 0.95*Ei+0.05*0.05*Ei, 19, endpoint=True)
        Etrans = np.array(Etrans if np.shape(Etrans) else [Etrans])
        if len(np.where(Etrans > Ei)[0]) > 0:
            warnings.warn('Cannot calculate for energy transfer greater than Ei (physically negative neutron energies!)')
        Etrans[np.where(Etrans >= Ei)] = np.nan
        v_van, _, _ = self.getVanVar(Ei, frequency, Etrans)
        x2 = self.chopper_system.sam_det
        Ef = Ei - np.array(Etrans)
        van = (2 * E2V * np.sqrt(Ef**3 * v_van)) / x2
        return van

    def getMultiRepResolution(self, Etrans=None, Ei_in=None, frequency=None):
        """ Returns a list of FWHM in meV for all allowed Ei's in multirep mode (in same order as getAllowedEi)
            The input energy transfer is interpreted as fractions of Ei. e.g. linspace(0,0.9,100) """
        Ei = _check_input(self.chopper_system, Ei_in)
        if Etrans is None:
            Etrans = np.linspace(0.05, 0.95, 19, endpoint=True)
        return [self.getResolution(Etrans * ei, ei, frequency) for ei in self.getAllowedEi(Ei)]

    def getVanVar(self, Ei_in=None, frequency=None, Etrans=0):
        """ Calculates the time squared FWHM in s^2 at the sample (Vanadium widths) for different components """
        Ei, _ = _check_input(self.chopper_system, Ei_in, frequency)
        Etrans = np.array(Etrans if np.shape(Etrans) else [Etrans])
        if frequency:
            oldfreq = self.frequency
            self.frequency = frequency
        tsqmod = self.moderator.getWidthSquared(Ei)
        tsqchp = self.chopper_system.getWidthSquared(Ei)
        tsqjit = self.tjit**2
        # Gets distances: x0=mod-final chopper, xa=aperture-final, x1=final-sample, x2=sample-det, xm=mod-first chopper
        x0, xa, x1, x2, xm = self.chopper_system.getDistances()
        # For Disk chopper spectrometers, the opening times of the first chopper can be the effective moderator time
        if tsqchp[1] is not None:
            frac_dist = 1 - (xm / x0)
            tsmeff = tsqmod * frac_dist**2   # Effective moderator time at first chopper
            x0 -= xm                         # Propagate from first chopper, not from moderator (after rescaling tmod)
            tsqmod = tsmeff if (tsqchp[1] > tsmeff) else tsqchp[1]
        tsqchp = tsqchp[0]
        tsqmodchop = np.array([tsqmod, tsqchp, x0])
        # Propagate the time widths to the sample position
        omega = self.frequency[0] * 2 * np.pi
        vi = E2V * np.sqrt(Ei)
        vf = E2V * np.sqrt(Ei - Etrans)
        vratio = (vi / vf)**3
        tanthm = np.tan(self.moderator.theta_m * np.pi / 180.)
        g1, g2 = tuple(1. - ((omega * tanthm / vi) * np.array([xa + x1, x0 - xa])))
        f1, f2 = tuple(1. + (x1 / x0) * np.array([g1, g2]))
        g1, g2, f1, f2 = tuple(np.array([g1, g2, f1, f2]) / (omega * (xa + x1)))
        modfac = (x1 + vratio * x2) / x0
        chpfac = 1. + modfac
        apefac = f1 + ((vratio * x2 / x0) * g1)
        tsqmod *= modfac**2
        tsqchp *= chpfac**2
        tsqjit *= chpfac**2
        tsqape = apefac**2 * (self.aperture_width**2 / 12.) * SIGMA2FWHMSQ
        vsqvan = tsqmod + tsqchp + tsqjit + tsqape
        outdic = {'moderator': tsqmod, 'chopper': tsqchp, 'jitter': tsqjit, 'aperture': tsqape}
        if self.has_detector and hasattr(self.detector, 'idet'):
            phi = self.detector.phi_deg * np.pi / 180.
            tsqdet = (1. / vf)**2 * np.array([self.detector.getWidthSquared(Ei, en) for en in Etrans])
            vsqvan += tsqdet
            outdic['detector'] = tsqdet
        else:
            phi = 0.
        if self.has_sample:
            gam = self.sample.gamma_deg * np.pi / 180.
            bb = (-np.sin(gam) / vi) + (np.sin(gam - phi) / vf) - (f2 * np.cos(gam))
            samfac = bb - ((vratio * x2 / x0) * g2 * np.cos(gam))
            tsqsam = samfac**2 * self.sample.getWidthSquared()
            vsqvan += tsqsam
            outdic['sample'] = tsqsam
        if frequency:
            self.frequency = oldfreq
        return vsqvan, outdic, tsqmodchop

    @property
    def aperture_width(self):
        if hasattr(self.chopper_system, 'aperture_width') and self.chopper_system.aperture_width:
            return self.chopper_system.aperture_width
        return 0.

    @property
    def aperture_height(self):
        if hasattr(self.chopper_system, 'aperture_height') and self.chopper_system.aperture_height:
            return self.chopper_system.aperture_height
        return 0.

    @property
    def instname(self):
        return self.name

    @property
    def n_frame(self):
        return self.chopper_system.n_frame

    @n_frame.setter
    def n_frame(self, value):
        self.moderator.n_frame = value
        self.chopper_system.setNFrame(value)

    @classmethod
    def calculate(cls, *args, **kwargs):
        """
        ! Calculates the resolution and flux directly (without setting up a PyChop2 object)
        !
        ! PyChop2.calculate('mari', 's', 250., 55.)      # Instname, Chopper Type, Freq, Ei in order
        ! PyChop2.calculate('let', 180, 2.2)             # For LET, chopper type is not needed.
        ! PyChop2.calculate('let', [160., 80.], 1.)      # For LET, specify resolution and pulse remover freq
        ! PyChop2.calculate('let', 'High flux', 80, 2.2) # LET default is medium flux configuration
        ! PyChop2.calculate(inst='mari', package='s', freq=250., ei=55.) # With keyword arguments
        ! PyChop2.calculate(inst='let', variant='High resolution', freq=[160., 80.], ei=2.2)
        !
        ! For LET, the allowed variant names are:
        !   'High resolution'
        !   'High flux'
        !   'Intermediate'
        ! You have to use these strings exactly.
        !
        ! By default this function returns the elastic resolution and flux only.
        ! If you want the inelastic resolution, specify the inelastic energy transfer
        ! as either the last positional argument, or as a keyword argument, e.g.:
        !
        ! PyChop2.calculate('merlin', 'g', 450., 60., range(55))
        ! PyChop2.calculate('maps', 'a', 450., 600., etrans=np.linspace(0,550,55))
        !
        ! The results are returned as tuple: (resolution, flux)
        """
        argdict = arg_parser(args, kwargs, ['inst', 'package', 'frequency', 'ei', 'etrans', 'variant'])
        if argdict['inst'] is None:
            raise RuntimeError('You must specify the instrument name')
        obj = cls(argdict['inst'])
        obj.setChopper(argdict['package'], argdict['frequency'])
        obj.ei = argdict['ei']
        if argdict['variant']:
            obj.variant = argdict['variant']
        return obj.getResolution(argdict['etrans'] if argdict['etrans'] else 0.), obj.getFlux()

    def __repr__(self):
        return self.name if self.name else 'Undefined instrument'
