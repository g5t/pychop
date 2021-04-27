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

def newFindLine(chop_times, chopDist, moderator_limits):
    """
    Calculates the lines on the limit of each chopper

    newFindline(chop_times, chopDist, moderator_limits)

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
    for chop in chop_times:
        l_m = (-chopDist) / (moderator_limits[0] - chop[0])
        r_m = (-chopDist) / (moderator_limits[-1] - chop[-1])
        if l_m > 0 and r_m > 0:
            l_c = -l_m * moderator_limits[0]
            r_c = -r_m * moderator_limits[-1]
            lines.append([[l_m, l_c], [r_m, r_c]])
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


def newCheckPath(chop_times, lines, chop_dist, mono_dist):
    """
    A recursive function to check for lines which can satisfy a window in the next chopper
    """
    if len(chop_times) > 1:
        # recursive bit
        lines = newCheckPath(chop_times[1:], lines, chop_dist[1:], mono_dist)
    new_lines = []
    for line in lines:
        l0_m = line[0][0]
        l0_c = line[0][1]
        l1_m = line[1][0]
        l1_c = line[1][1]
        # for each line check to see if there is an opening in the right time window
        # fast first
        early = (chop_dist[0] - l0_c) / l0_m
        # then slow
        late = (chop_dist[0] - l1_c) / l1_m

        # then compare this time window to when this chopper is open, keep the range if it is possible
        for chop in chop_times[0]:
            if chop[0] < early and chop[-1] > late:
                # the chopper window is larger than the maximum possible spread, so the line is unchanged
                new_lines.append(line)
            elif chop[0] > early and chop[-1] < late:
                # both are within the window, draw a new box
                mono_open = (mono_dist - l0_c) / l0_m
                l_m = (chop_dist[0] - mono_dist) / (chop[0] - mono_open)
                l_c = mono_dist - l_m*mono_open
                mono_close = (mono_dist - l1_c) / l1_m
                r_m = (chop_dist[0] - mono_dist) / (chop[-1] - mono_close)
                r_c = mono_dist - r_m*mono_close
                new_lines.append([[l_m, l_c], [r_m, r_c]])
            elif late > chop[-1] > early > chop[0]:
                # the leftmost range is fine but the rightmost is outside of the window. Redefine it
                mono_close = (mono_dist - l1_c) / l1_m
                r_m = (chop_dist[0] - mono_dist) / (chop[-1] - mono_close)
                r_c = mono_dist - r_m * mono_close
                new_lines.append([line[0], [r_m, r_c]])
            elif early < chop[0] < late < chop[-1]:
                # the rightmost range is fine but the leftmost is outside of the window. Redefine it
                mono_open = (mono_dist - l0_c) / l0_m
                l_m = (chop_dist[0] - mono_dist) / (chop[0] - mono_open)
                l_c = mono_dist - l_m * mono_open
                new_lines.append([[l_m, l_c], line[1]])
    return new_lines


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


def calcChopTimes(efocus, frequencies, instrumentpars, chopper_phases, chopper_slots):
    """
    A method to calculate the various possible incident energies with a given chopper setup on a multi-chopper
    instrument like LET.
    efocus: The incident energy that all choppers are focussed on
    frequencies: The rotation frequency of each chopper
    instrumentpars: a list of instrument parameters [see Instruments.py]
    phase_slot_none: for every chopper one of {phase: float, slot: string, none: None} representing the phase offset,
                     the slot number to pass through, or neither for each chopper.

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
    phase_chopper_to = instrumentpars[12]

    all_chopper_times = [[] for _ in distance_per_chopper] # *unique* empty lists to hold each chopper opening period
    n_choppers = len(distance_per_chopper)

    # Ensure provided independent chopper index(es) match the number of independent phase/slot information
    if not hasattr(phase_chopper_to, '__len__'):
        phase_chopper_to = [phase_chopper_to]
    if not hasattr(chopper_phases, '__len__'):
        chopper_phases = [chopper_phases]
    if not hasattr(chopper_slots, '__len__'):
        chopper_slots = [chopper_slots]
    if not (len(chopper_phases) == n_choppers and len(chopper_slots) == n_choppers and len(phase_chopper_to) == n_choppers):
        raise RuntimeError('The independent chopper and phase/slot information must match the number of choppers!')
    # remove any passed None values from the phases and slot indices
    phase_to_source = ['source' in pct for pct in phase_chopper_to]
    chopper_phases = [p if p else 0. for p in chopper_phases]
    chopper_slots = [s % n if s else 0 for s, n in zip(chopper_slots, n_slots_per_chopper)]

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

    chopper_it = zip(phase_to_source, chopper_phases, chopper_slots, frequencies, all_chopper_times,
                     distance_per_chopper, slot_angle_centers_per_chopper, *tuple(instrumentpars[3:7]))
    for p2s, phase, slot, freq, times, distance, slot_centers, slot_width, guide_width, radius, n_disks in chopper_it:
        # the effective chopper edge velocity (double disks are effectively twice as fast)
        edge_velocity = 2*np.pi * radius * n_disks * freq
        # the duration over which the chopper is open
        t_open_close = np.array((0., uSec * (slot_width + guide_width)/edge_velocity))
        # If this chopper is phased to source, the phase is relative to t=0. Otherwise it is relative to the chopper
        # midpoint at the time-of-flight for the focus wavelength arriving at the chopper
        t_reference = 0. if p2s else (lam2TOF * lam * distance - t_open_close[1]/2)
        # we then adjust the opening and closing time of the chopper by the reference time and the desired phase
        t_open_close += t_reference + phase
        # the full chopper rotation period:
        period = uSec/freq
        # find the mid-slot position for each slot, in units of rotations
        rel_pos = np.array(slot_centers) / 360
        # we start with the selected slot and move a partial rotation between slots:
        slot_part_rot = (rel_pos - rel_pos[slot]) % 1
        # make sure we rotate forwards in time far enough to pass t == maximum_t:
        n_rot_fwd = np.ceil(freq / p_frames - np.min(slot_part_rot) - np.min(t_open_close)/period)
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


def newCalcChopTimes(efocus, frequencies, instrumentpars, chopper_phases, chopper_slots):
    """
    A method to calculate the various possible incident energies with a given chopper setup on a multi-chopper
    instrument like LET.
    efocus: The incident energy that all choppers are focussed on
    frequencies: The rotation frequency of each chopper
    instrumentpars: a list of instrument parameters [see Instruments.py]
    phase_slot_none: for every chopper one of {phase: float, slot: string, none: None} representing the phase offset,
                     the slot number to pass through, or neither for each chopper.

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
    phase_chopper_to = instrumentpars[12]

    all_chopper_times = [[] for _ in distance_per_chopper] # *unique* empty lists to hold each chopper opening period
    n_choppers = len(distance_per_chopper)

    # Ensure provided independent chopper index(es) match the number of independent phase/slot information
    if not hasattr(phase_chopper_to, '__len__'):
        phase_chopper_to = [phase_chopper_to]
    if not hasattr(chopper_phases, '__len__'):
        chopper_phases = [chopper_phases]
    if not hasattr(chopper_slots, '__len__'):
        chopper_slots = [chopper_slots]
    if not (len(chopper_phases) == n_choppers and len(chopper_slots) == n_choppers and len(phase_chopper_to) == n_choppers):
        raise RuntimeError('The independent chopper and phase/slot information must match the number of choppers!')
    # remove any passed None values from the phases and slot indices
    phase_to_source = ['source' in pct for pct in phase_chopper_to]
    phase_to_first = ['first' in pct for pct in phase_chopper_to]
    chopper_phases = [p if p else 0. for p in chopper_phases]
    chopper_slots = [s % n if s else 0 for s, n in zip(chopper_slots, n_slots_per_chopper)]

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

    chopper_it = zip(phase_to_source, phase_to_first, chopper_phases, chopper_slots, frequencies, all_chopper_times,
                     distance_per_chopper, slot_angle_centers_per_chopper, *tuple(instrumentpars[3:7]))
    for p2s, p2f, phase, slot, freq, times, distance, slot_centers, slot_width, guide_width, radius, n_disks in chopper_it:
        # the effective chopper edge velocity (double disks are effectively twice as fast)
        edge_velocity = 2*np.pi * radius * n_disks * freq
        # the duration over which the chopper is open
        x_states = (0, guide_width, slot_width, guide_width+slot_width)
        # y_states = (0, 1, 1, 0)  # the transmission profile trapezoidal function __/---\__
        t_states = np.array(x_states) * uSec/edge_velocity
        # If this chopper is phased to source, the phase is relative to t=0. Otherwise it is relative to the chopper
        # midpoint at the time-of-flight for the focus wavelength arriving at the chopper
        t_reference = 0. if p2s else (lam2TOF * lam * distance - t_states[-1]/2)
        if p2f:
            t_reference += (t_states[2] - t_states[1])/2.
        # we then adjust the opening and closing time of the chopper by the reference time and the desired phase
        t_states += t_reference + phase
        # the full chopper rotation period:
        period = uSec/freq
        # find the mid-slot position for each slot, in units of rotations
        rel_pos = np.array(slot_centers) / 360
        # we start with the selected slot and move a partial rotation between slots:
        slot_part_rot = (rel_pos - rel_pos[slot]) % 1
        # make sure we rotate forwards in time far enough to pass t == maximum_t:
        n_rot_fwd = np.ceil(freq / p_frames - np.min(slot_part_rot) - np.min(t_states)/period)
        # make sure we rotate backwards in time far enough to pass t == 0:
        n_rot_bck = np.ceil(np.max(slot_part_rot)+np.max(t_states)/period)
        # collect all slot rotations during the full period
        all_rot = (np.arange(-n_rot_bck, n_rot_fwd+1)[:, np.newaxis] + slot_part_rot).flatten()
        # all chopper windows:
        windows = t_states + period * all_rot[:, np.newaxis]
        # keep only windows which close after the frame starts and open before the full-period ends
        times[:] = windows[(windows[:, 0] < maximum_t) & (windows[:, -1] > 0)]
    # then we look for what else gets through
    # firstly calculate the bounding box for each window in final chopper
    lines_all = []
    fully_open = []
    for i in range(nframe):
        t0 = i * uSec / source_rep
        # including opening and closing times
        lines = newFindLine(all_chopper_times[-1], distance_per_chopper[-1], [t0, t0+latest_moderator_emission_time])
        lines = newCheckPath([np.array(ct)+t0 for ct in all_chopper_times[0:-1]], lines, distance_per_chopper[:-1], distance_per_chopper[-1])
        if lines:
            for line in lines:
                lines_all.append(line)

    # ok, now we know the possible neutron velocities. we now need their energies
    Ei = calcEnergy(lines_all, (distance_per_chopper[-1]+chop_samp))
    return Ei, \
           all_chopper_times, [all_chopper_times[0][0], all_chopper_times[-1][0]], \
           distance_per_chopper[-1]-distance_per_chopper[0],\
           lines_all
