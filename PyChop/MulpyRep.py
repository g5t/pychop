# -*- coding: utf-8 -*-# Mantid Repository : https://github.com/mantidproject/mantid
#
# Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
#   NScD Oak Ridge National Laboratory, European Spallation Source,
#   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
# SPDX - License - Identifier: GPL - 3.0 +

# pylint: disable=line-too-long, invalid-name, too-many-locals, unused-variable

"""
Contains a class to calculate the possible reps, resolution and flux for a direct geometry disk chopper
spectrometer. Python implementation by D J Voneshen based on the original Matlab program of R I Bewley.
"""
from typing import Union

import numpy as np
import copy

from numpy.core._multiarray_umath import ndarray


def findLine(chop_times, chopDist, moderator_limits):
    """
    Calculates the lines on the limit of each chopper

    findline(chop_times, chopDist, moderator_limits)

    chop_times: a list of the opening and closing times of the chopper within the time frame
    chopDist: a list of the distance from moderator to chopper in meters
    moderator_limits: the earliest and latest times that neutrons can leave the the moderator in microseconds

    returns:
        the slope and t=0 intercept for any causally consistent bounding neutron paths
        of the form
            d(t) = v * t + d(0)
        as a list of two-element lists with two elements:
        [ [[v, d(0)]_left, [v, d(0)]_right]_0, [[v, d(0)]_left, [v, d(0)]_right]_1, ... ]
    """
    lines = []
    for i in range(len(chop_times)):
        # final chopper openings
        leftM = (-chopDist) / (moderator_limits[0]-chop_times[i][0])
        rightM = (-chopDist) / (moderator_limits[1]-chop_times[i][1])
        leftC = -leftM*moderator_limits[0]
        rightC = -rightM*moderator_limits[1]
        if leftM > 0 and rightM > 0:
            lines.append([[leftM, leftC], [rightM, rightC]])
    return lines


def checkPath(chop_times, lines, chopDist, chop5Dist):
    """
    A recursive function to check for lines which can satisfy a window in the next chopper
    """
    if len(chop_times) > 1:
        # recursive bit
        lines = checkPath(chop_times[1:], lines, chopDist[1:], chop5Dist)
    newLines = []
    for line in lines:
        # for each line check to see if there is an opening in the right time window
        # fast first
        earlyT = (chopDist[0]-line[0][1]) / line[0][0]
        # then slow
        lateT = (chopDist[0]-line[1][1]) / line[1][0]

        # then compare this time window to when this chopper is open, keep the range if it is possible
        for i in range(len(chop_times[0])):
            if (chop_times[0][i][0] < earlyT) and ((chop_times[0][i][1] > lateT)):
                # the chopper window is larger than the maximum possible spread, change nothing
                newLines.append(line)
            elif (chop_times[0][i][0] > earlyT) and ((chop_times[0][i][1] < lateT)):
                # both are within the window, draw a new box
                chop5_open = (chop5Dist-line[0][1]) / line[0][0]
                leftM = (chopDist[0]-chop5Dist) / (chop_times[0][i][0]-chop5_open)
                leftC = chop5Dist - leftM*chop5_open
                chop5_close = (chop5Dist-line[1][1]) / line[1][0]
                rightM = (chopDist[0]-chop5Dist) / (chop_times[0][i][1]-chop5_close)
                rightC = chop5Dist - rightM*chop5_close
                newLines.append([[leftM, leftC], [rightM, rightC]])
            elif ((chop_times[0][i][1] < lateT) and (chop_times[0][i][1] > earlyT)) and (chop_times[0][i][0] < earlyT):
                # the left most range is fine but the right most is outside the window. Redefine it
                chop5_close = (chop5Dist-line[1][1]) / line[1][0]
                rightM = (chopDist[0]-chop5Dist) / (chop_times[0][i][1]-chop5_close)
                rightC = chop5Dist - rightM*chop5_close
                newLines.append([line[0], [rightM, rightC]])
            elif (chop_times[0][i][1] > lateT) and ((chop_times[0][i][0] > earlyT) and (chop_times[0][i][0] < lateT)):
                # the leftmost range is outside the chopper window
                chop5_open = (chop5Dist-line[0][1]) / line[0][0]
                leftM = (chopDist[0]-chop5Dist) / (chop_times[0][i][0]-chop5_open)
                leftC = chop5Dist - leftM*chop5_open
                newLines.append([[leftM, leftC], line[1]])
    return newLines


def calcEnergy(lines, samDist):
    """
    Calculates the energies of neutrons which can pass through choppering openings.
    """
    Ei = np.zeros(len(lines))
    massN = 1.674927e-27
    for i in range(len(lines)):
        # look at the middle of the time window
        x0 = -lines[i][0][1] / lines[i][0][0]
        x1 = ((samDist-lines[i][0][1]) / lines[i][0][0] + (samDist-lines[i][1][1]) / lines[i][1][0]) / 2.
        v = samDist / (x1 - x0)
        Ei[i] = (v*1e6)**2 * massN / 2. / 1.60217662e-22
    return Ei


def calcRes(ei, chop_times, lastChopDist, samDist, detDist, guide, slot):
    """
    # for each incident energy work out the moderator and chopper component of the resolution
    """
    res = []
    percent = []
    chwid = []
    modwid = []
    # IMPORTANT POINT
    # The chopper opening times are the full opening, for the resolution we want FWHM
    # consequently divide each by a factor of 2 here
    # END IMPORTANT POINT
    #IMPORTANT POINT 2
    #important point 1 is only valid when guide>=slot
    #if slot>guide the transmission function is a trapezium and it is more complex
    #END  IMPORTANT POINT2
    if guide>=slot:
        chop_width=[(chop_times[0][1]-chop_times[0][0])/2.,(chop_times[1][1]-chop_times[1][0])/2.]
    else:
        totalOpen=(chop_times[1][1]-chop_times[1][0])
        flat_time=(slot-guide)*totalOpen/slot
        triangleTime=guide*totalOpen/slot/2. #/2 for FWHM of the triangles
        chop_width=[(chop_times[0][1]-chop_times[0][0])/2.,(flat_time+triangleTime)]
    for energy in ei:
        lamba = np.sqrt(81.81/energy)
        # this is the experimentally determined FWHM of moderator
        mod_FWHM = -3.143*lamba**2 + 49.28*lamba + 0.535
        # the effective width at chopper 1
        mod_eff = 0.6666*mod_FWHM
        # when running chopper 1 slowly the moderator is smaller than the chopper speed so use that
        if chop_width[0] > mod_eff:
            mod_width = mod_eff
        else:
            mod_width = chop_width[0]
        t_mod_chop = 252.82*lastChopDist*lamba
        chopRes = (2*chop_width[1]/t_mod_chop) * ((detDist+samDist+lastChopDist) / detDist)
        modRes = (2*mod_width/t_mod_chop) * (1 + (samDist/detDist))
        res.append(np.sqrt(chopRes**2 + modRes**2) * energy)
        percent.append(np.sqrt(chopRes**2 + modRes**2))
        chwid.append(chop_width[1])
        modwid.append(mod_width)
    return res, percent, chwid, modwid


def calcFlux(Ei, freq1, percent, slot):
    """
    Looks up flux at a give Ei and resolution (frequency) from a table of measured flux
    """
    lamba = np.sqrt(81.8042/Ei)
    # here are some constants (hahaha) relating to the instrument
    intRef = 0.885 # the flux at 5meV
    freqRef = 150. # the frequency this corresponds to
    refSlot = 20.  # reference disk slot width in mm
    fluxProf = [0.0889, 0.1003, 0.1125, 0.1213, 0.1274, 0.1358, 0.1455, 0.1562, 0.1702,
                0.1902, 0.2149, 0.2496, 0.2938, 0.3537, 0.4315, 0.5244, 0.6415, 0.7856,
                0.9341, 1.0551, 1.1437, 1.1955, 1.2004, 1.1903, 1.1662, 1.1428, 1.1176,
                1.0875, 1.0641, 1.0562, 1.0242, 0.9876, 0.9586, 0.9415, 0.924, 0.8856,
                0.8865, 0.8727, 0.842, 0.8125, 0.7849, 0.7596, 0.7417, 0.7143, 0.6869,
                0.6608, 0.6341, 0.6073, 0.581, 0.5548, 0.5304, 0.507, 0.4849, 0.4639,
                0.4445, 0.425, 0.407, 0.3902, 0.3737, 0.3579, 0.3427, 0.3274, 0.3129,
                0.2989, 0.2854, 0.2724, 0.2601, 0.2483, 0.2371, 0.2267, 0.2167, 0.2072,
                0.1984, 0.19, 0.1821, 0.1743, 0.1669, 0.1599, 0.1532, 0.1467, 0.1404,
                0.1346, 0.1291, 0.1238, 0.1189, 0.1141, 0.1097, 0.1053, 0.1014, 0.0975,
                0.0938, 0.0902, 0.0866, 0.0834, 0.0801, 0.077, 0.0741, 0.0712, 0.0686,
                0.066, 0.0637, 0.0614, 0.0593, 0.0571, 0.0551, 0.0532, 0.0512, 0.0494,
                0.0477, 0.0461, 0.0445, 0.043, 0.0415, 0.0401, 0.0387]
    fluxLamba = np.linspace(0.5, 11.9, num=len(fluxProf))
    flux = []
    lamba = lamba if hasattr(lamba, '__len__') else [lamba]
    for j in range(len(lamba)):
        i = (abs(fluxLamba-lamba[j])).argmin()
        intensity = fluxProf[i]
        if percent[j] < 0.02:
            flux.append(5.6e4*intensity/intRef*(slot/refSlot)*(freqRef/freq1)**2)
        else:
            flux.append(5.6e4*intensity/intRef*(slot/refSlot)*(freqRef/freq1))
    return flux


def calcChopTimes(efocus, frequencies, instrumentpars, independent_phase_or_slot_string=5):
    """
    A method to calculate the various possible incident energies with a given chopper setup on a multi-chopper
    instrument like LET.
    efocus: The incident energy that all choppers are focussed on
    frequencies: The rotation frequency of each chopper
    instrumentpars: a list of instrument parameters [see Instruments.py]
    independent_phase_or_slot_string: the phase(s) of any choppers which are *not* focused on the provided energy
                                      or a string representing the integer number of the slot to use

    Original Matlab code R. Bewley STFC
    Rewritten in Python, D Voneshen STFC 2015
    Updated to remove unnecessary code paths, G S Tucker ESS 2021
    """
    # conversion factors
    lam2TOF = 252.7784            # the conversion from wavelength to TOF at 1m, multiply by distance
    uSec = 1e6                    # seconds to microseconds
    lam = np.sqrt(81.8042/efocus) # convert from energy to wavelength

    # extracts the instrument parameters
    distance_per_chopper, n_slots_per_chopper, slot_angle_centers_per_chopper = tuple(instrumentpars[:3])
    # slot_width, guide_width, radius, numDisk, samp_det = tuple(instrumentpars[3:8])
    chop_samp, rep_pack, latest_moderator_emission_time = tuple(instrumentpars[8:11])
    # frac_ei = instrumentpars[11]
    independent_choppers = instrumentpars[12]

    all_chopper_times = [[] for _ in distance_per_chopper] # *unique* empty lists to hold each chopper opening period

    # Ensure provided independent chopper index(es) match the number of independent phase/slot information
    if not hasattr(independent_choppers, '__len__'):
        independent_choppers = [independent_choppers]
    if not hasattr(independent_phase_or_slot_string, '__len__'):
        independent_phase_or_slot_string = [independent_phase_or_slot_string]
    if not len(independent_phase_or_slot_string) == len(independent_choppers):
        raise RuntimeError('The number of independent chopper indices and their phase/slot information must match!')
    # separate the passed commingled information into independent phase or selected slot information:
    chopper_phase = np.full(len(distance_per_chopper), None)
    chopper_slot = np.full(len(distance_per_chopper), 0)
    for i, chopper in enumerate(independent_choppers):
        if isinstance(independent_phase_or_slot_string[i], str):
            # *any* string should be an integer to select the slot
            chopper_slot[chopper] = int(independent_phase_or_slot_string[i]) % n_slots_per_chopper[chopper]
        else:
            # otherwise this should be an independent phase (modifier)
            chopper_phase[chopper] = independent_phase_or_slot_string[i]

    # do we want multiple frames?
    source_rep, nframe = tuple(rep_pack[:2]) if (hasattr(rep_pack, '__len__') and len(rep_pack) > 1) else (rep_pack, 1)
    p_frames = source_rep / nframe
    maximum_t = uSec * nframe / source_rep

    # protect against slot_angle_centers_per_chopper being None or [..., None, ...]
    if not slot_angle_centers_per_chopper:
        slot_angle_centers_per_chopper = [np.linspace(0, 360*(n-1)/n, n) for n in n_slots_per_chopper]
    else:
        for i, n in enumerate(n_slots_per_chopper):
            if not slot_angle_centers_per_chopper[i]:
                slot_angle_centers_per_chopper[i] = np.linspace(0, 360*(n-1)/n, n)

    chopper_it = zip(chopper_phase, chopper_slot, frequencies, all_chopper_times, distance_per_chopper,
                     slot_angle_centers_per_chopper, *tuple(instrumentpars[3:7]))
    for phase, slot, frequency, times, distance, slot_centers, slot_width, guide_width, radius, n_disks in chopper_it:
        # # the slot index for choppers with asymmetric slots (zero by default):
        # # the int() call converts a string to an integer
        # slot = int(phase)%n_slots if (is_independent and isinstance(phase, str)) else 0
        # the effective chopper edge velocity (double disks are effectively twice as fast)
        edge_velocity = 2*np.pi * radius * n_disks * frequency
        # the duration over which the chopper is open
        t_open_close = np.array((0., uSec * (slot_width + guide_width)/edge_velocity))
        # the time of flight for the specified wavelength to reach this chopper
        t_focus = lam2TOF * lam * distance
        if phase and phase < 0:
            # the specified phase is a relative offset from the time-of-flight for the specified wavelength
            t_open_close += t_focus + phase
        elif phase:
            # the specified phase is an absolute microsecond offset from t=0
            t_open_close += phase
        else:
            # centre the open window on the specified wavelength
            t_open_close += t_focus - t_open_close[1]/2
        # the full chopper rotation period:
        period = uSec/frequency
        # find the mid-slot position for each slot, in units of rotations
        rel_pos = np.array(slot_centers) / 360
        # we start with the selected slot and move a partial rotation between slots:
        slot_part_rot = (rel_pos - rel_pos[slot]) % 1
        # make sure we rotate forwards in time far enough to pass t == maximum_t:
        n_rot_fwd = np.ceil(frequency / p_frames - np.min(slot_part_rot) - np.min(t_open_close)/period)
        # make sure we rotate backwards in time far enough to pass t == 0:
        n_rot_bck = np.ceil(np.max(slot_part_rot)+np.max(t_open_close)/period)
        # collect all slot rotations during the full period
        all_rot = (np.arange(-n_rot_bck, n_rot_fwd+1)[:, np.newaxis] + slot_part_rot).flatten()
        # all chopper windows:
        windows = t_open_close + period * all_rot[:, np.newaxis]
        # keep only windows which close after the frame starts and open before the full-period ends
        times[:] = windows[(windows[:, 0] < maximum_t) & (windows[:, 1] > 0)]
    # then we look for what else gets through
    # firstly calculate the bounding box for each window in final chopper
    lines_all = []
    for i in range(nframe):
        t0 = i * uSec / source_rep
        lines = findLine(all_chopper_times[-1], distance_per_chopper[-1], [t0, t0+latest_moderator_emission_time])
        lines = checkPath([np.array(ct)+t0 for ct in all_chopper_times[0:-1]], lines, distance_per_chopper[:-1], distance_per_chopper[-1])
        if lines:
            for line in lines:
                lines_all.append(line)

    # ok, now we know the possible neutron velocities. we now need their energies
    Ei = calcEnergy(lines_all, (distance_per_chopper[-1]+chop_samp))
    return Ei, all_chopper_times, [all_chopper_times[0][0], all_chopper_times[-1][0]], distance_per_chopper[-1]-distance_per_chopper[0], lines_all
