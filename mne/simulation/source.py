# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np

from ..dipole import Dipole
from ..fixes import rng_uniform
from ..label import Label
from ..source_estimate import SourceEstimate, VolSourceEstimate
from ..source_space._source_space import _ensure_src
from ..surface import _compute_nearest
from ..utils import (
    _check_option,
    _ensure_events,
    _ensure_int,
    _validate_type,
    check_random_state,
    fill_doc,
    warn,
)
from .signals import NoiseGenerator


@fill_doc
def select_source_in_label(
    src,
    label,
    random_state=None,
    location="random",
    subject=None,
    subjects_dir=None,
    surf="sphere",
):
    """Select source positions using a label.

    Parameters
    ----------
    src : list of dict
        The source space.
    label : Label
        The label.
    %(random_state)s
    location : str
        The label location to choose. Can be 'random' (default) or 'center'
        to use :func:`mne.Label.center_of_mass` (restricting to vertices
        both in the label and in the source space). Note that for 'center'
        mode the label values are used as weights.

        .. versionadded:: 0.13
    subject : str | None
        The subject the label is defined for.
        Only used with ``location='center'``.

        .. versionadded:: 0.13
    %(subjects_dir)s

        .. versionadded:: 0.13
    surf : str
        The surface to use for Euclidean distance center of mass
        finding. The default here is "sphere", which finds the center
        of mass on the spherical surface to help avoid potential issues
        with cortical folding.

        .. versionadded:: 0.13

    Returns
    -------
    lh_vertno : list
        Selected source coefficients on the left hemisphere.
    rh_vertno : list
        Selected source coefficients on the right hemisphere.
    """
    lh_vertno = list()
    rh_vertno = list()
    _check_option("location", location, ["random", "center"])

    rng = check_random_state(random_state)
    if label.hemi == "lh":
        vertno = lh_vertno
        hemi_idx = 0
    else:
        vertno = rh_vertno
        hemi_idx = 1
    src_sel = np.intersect1d(src[hemi_idx]["vertno"], label.vertices)
    if location == "random":
        idx = src_sel[rng_uniform(rng)(0, len(src_sel), 1)[0]]
    else:  # 'center'
        idx = label.center_of_mass(
            subject, restrict_vertices=src_sel, subjects_dir=subjects_dir, surf=surf
        )
    vertno.append(idx)
    return lh_vertno, rh_vertno


@fill_doc
def simulate_sparse_stc(
    src,
    n_dipoles,
    times,
    data_fun=lambda t: 1e-7 * np.sin(20 * np.pi * t),
    labels=None,
    random_state=None,
    location="random",
    subject=None,
    subjects_dir=None,
    surf="sphere",
):
    """Generate sparse (n_dipoles) sources time courses from data_fun.

    This function randomly selects ``n_dipoles`` vertices in the whole
    cortex or one single vertex (randomly in or in the center of) each
    label if ``labels is not None``. It uses ``data_fun`` to generate
    waveforms for each vertex.

    Parameters
    ----------
    src : instance of SourceSpaces
        The source space.
    n_dipoles : int
        Number of dipoles to simulate.
    times : array
        Time array.
    data_fun : callable
        Function to generate the waveforms. The default is a 100 nAm, 10 Hz
        sinusoid as ``1e-7 * np.sin(20 * pi * t)``. The function should take
        as input the array of time samples in seconds and return an array of
        the same length containing the time courses.
    labels : None | list of Label
        The labels. The default is None, otherwise its size must be n_dipoles.
    %(random_state)s
    location : str
        The label location to choose. Can be ``'random'`` (default) or
        ``'center'`` to use :func:`mne.Label.center_of_mass`. Note that for
        ``'center'`` mode the label values are used as weights.

        .. versionadded:: 0.13
    subject : str | None
        The subject the label is defined for.
        Only used with ``location='center'``.

        .. versionadded:: 0.13
    %(subjects_dir)s

        .. versionadded:: 0.13
    surf : str
        The surface to use for Euclidean distance center of mass
        finding. The default here is "sphere", which finds the center
        of mass on the spherical surface to help avoid potential issues
        with cortical folding.

        .. versionadded:: 0.13

    Returns
    -------
    stc : SourceEstimate
        The generated source time courses.

    See Also
    --------
    simulate_raw
    simulate_evoked
    simulate_stc

    Notes
    -----
    .. versionadded:: 0.10.0
    """
    rng = check_random_state(random_state)
    src = _ensure_src(src, verbose=False)
    subject_src = src._subject
    if subject is None:
        subject = subject_src
    elif subject_src is not None and subject != subject_src:
        raise ValueError(
            f"subject argument ({subject}) did not match the source "
            f"space subject_his_id ({subject_src})"
        )
    data = np.zeros((n_dipoles, len(times)))
    for i_dip in range(n_dipoles):
        data[i_dip, :] = data_fun(times)

    if labels is None:
        # can be vol or surface source space
        offsets = np.linspace(0, n_dipoles, len(src) + 1).astype(int)
        n_dipoles_ss = np.diff(offsets)
        # don't use .choice b/c not on old numpy
        vs = [
            s["vertno"][np.sort(rng.permutation(np.arange(s["nuse"]))[:n])]
            for n, s in zip(n_dipoles_ss, src)
        ]
        datas = data
    elif n_dipoles > len(labels):
        raise ValueError(
            f"Number of labels ({len(labels)}) smaller than n_dipoles ({n_dipoles:d}) "
            "is not allowed."
        )
    else:
        if n_dipoles != len(labels):
            warn(
                "The number of labels is different from the number of "
                f"dipoles. {min(n_dipoles, len(labels))} dipole(s) will be generated."
            )
        labels = labels[:n_dipoles] if n_dipoles < len(labels) else labels

        vertno = [[], []]
        lh_data = [np.empty((0, data.shape[1]))]
        rh_data = [np.empty((0, data.shape[1]))]
        for i, label in enumerate(labels):
            lh_vertno, rh_vertno = select_source_in_label(
                src, label, rng, location, subject, subjects_dir, surf
            )
            vertno[0] += lh_vertno
            vertno[1] += rh_vertno
            if len(lh_vertno) != 0:
                lh_data.append(data[i][np.newaxis])
            elif len(rh_vertno) != 0:
                rh_data.append(data[i][np.newaxis])
            else:
                raise ValueError("No vertno found.")
        vs = [np.array(v) for v in vertno]
        datas = [np.concatenate(d) for d in [lh_data, rh_data]]
        # need to sort each hemi by vertex number
        for ii in range(2):
            order = np.argsort(vs[ii])
            vs[ii] = vs[ii][order]
            if len(order) > 0:  # fix for old numpy
                datas[ii] = datas[ii][order]
        datas = np.concatenate(datas)

    tmin, tstep = times[0], np.diff(times[:2])[0]
    assert datas.shape == data.shape
    cls = SourceEstimate if len(vs) == 2 else VolSourceEstimate
    stc = cls(datas, vertices=vs, tmin=tmin, tstep=tstep, subject=subject)
    return stc


def simulate_stc(
    src, labels, stc_data, tmin, tstep, value_fun=None, allow_overlap=False
):
    """Simulate sources time courses from waveforms and labels.

    This function generates a source estimate with extended sources by
    filling the labels with the waveforms given in stc_data.

    Parameters
    ----------
    src : instance of SourceSpaces
        The source space.
    labels : list of Label
        The labels.
    stc_data : array, shape (n_labels, n_times)
        The waveforms.
    tmin : float
        The beginning of the timeseries.
    tstep : float
        The time step (1 / sampling frequency).
    value_fun : callable | None
        Function to apply to the label values to obtain the waveform
        scaling for each vertex in the label. If None (default), uniform
        scaling is used.
    allow_overlap : bool
        Allow overlapping labels or not. Default value is False.

        .. versionadded:: 0.18

    Returns
    -------
    stc : SourceEstimate
        The generated source time courses.

    See Also
    --------
    simulate_raw
    simulate_evoked
    simulate_sparse_stc
    """
    if len(labels) != len(stc_data):
        raise ValueError("labels and stc_data must have the same length")

    vertno = [[], []]
    stc_data_extended = [[], []]
    hemi_to_ind = {"lh": 0, "rh": 1}
    for i, label in enumerate(labels):
        hemi_ind = hemi_to_ind[label.hemi]
        src_sel = np.intersect1d(src[hemi_ind]["vertno"], label.vertices)
        if len(src_sel) == 0:
            idx = src[hemi_ind]["inuse"].astype("bool")
            xhs = src[hemi_ind]["rr"][idx]
            rr = src[hemi_ind]["rr"][label.vertices]
            closest_src = _compute_nearest(xhs, rr)
            src_sel = src[hemi_ind]["vertno"][np.unique(closest_src)]

        if value_fun is not None:
            idx_sel = np.searchsorted(label.vertices, src_sel)
            values_sel = np.array([value_fun(v) for v in label.values[idx_sel]])

            data = np.outer(values_sel, stc_data[i])
        else:
            data = np.tile(stc_data[i], (len(src_sel), 1))
        # If overlaps are allowed, deal with them
        if allow_overlap:
            # Search for duplicate vertex indices
            # in the existing vertex matrix vertex.
            duplicates = []
            for src_ind, vertex_ind in enumerate(src_sel):
                ind = np.where(vertex_ind == vertno[hemi_ind])[0]
                if len(ind) > 0:
                    assert len(ind) == 1
                    # Add the new data to the existing one
                    stc_data_extended[hemi_ind][ind[0]] += data[src_ind]
                    duplicates.append(src_ind)
            # Remove the duplicates from both data and selected vertices
            data = np.delete(data, duplicates, axis=0)
            src_sel = list(np.delete(np.array(src_sel), duplicates))
        # Extend the existing list instead of appending it so that we can
        # index its elements
        vertno[hemi_ind].extend(src_sel)
        stc_data_extended[hemi_ind].extend(np.atleast_2d(data))

    vertno = [np.array(v) for v in vertno]
    if not allow_overlap:
        for v, hemi in zip(vertno, ("left", "right")):
            d = len(v) - len(np.unique(v))
            if d > 0:
                raise RuntimeError(
                    f"Labels had {d} overlaps in the {hemi} "
                    "hemisphere, they must be non-overlapping"
                )
    # the data is in the order left, right
    data = list()
    for i in range(2):
        if len(stc_data_extended[i]) != 0:
            stc_data_extended[i] = np.vstack(stc_data_extended[i])
            # Order the indices of each hemisphere
            idx = np.argsort(vertno[i])
            data.append(stc_data_extended[i][idx])
            vertno[i] = vertno[i][idx]

    stc = SourceEstimate(
        np.concatenate(data),
        vertices=vertno,
        tmin=tmin,
        tstep=tstep,
        subject=src._subject,
    )
    return stc


class SourceSimulator:
    """Class to generate simulated Source Estimates.

    Parameters
    ----------
    src : instance of SourceSpaces
        Source space.
    tstep : float
        Time step between successive samples in data. Default is 0.001 s.
    duration : float | None
        Time interval during which the simulation takes place in seconds.
        If None, it is computed using existing events and waveform lengths.
    first_samp : int
        First sample from which the simulation takes place, as an integer.
        Comparable to the :term:`first_samp` property of `~mne.io.Raw` objects.
        Default is 0.

    Attributes
    ----------
    duration : float
        The duration of the simulation in seconds.
    n_times : int
        The number of time samples of the simulation.
    """

    def __init__(self, src, tstep=1e-3, duration=None, first_samp=0):
        if duration is not None and duration < tstep:
            raise ValueError("duration must be None or >= tstep.")
        self.first_samp = _ensure_int(first_samp, "first_samp")
        self._src = src
        self._tstep = tstep
        self._labels = []
        self._waveforms = []
        self._events = np.empty((0, 3), dtype=int)
        self._duration = duration  # if not None, sets # samples
        self._last_samples = []
        self._chk_duration = 1000
        self._waveform_ids = []

    @property
    def duration(self):
        """Duration of the simulation in same units as tstep."""
        if self._duration is not None:
            return self._duration
        return self.n_times * self._tstep

    @property
    def n_times(self):
        """Number of time samples in the simulation."""
        if self._duration is not None:
            return int(self._duration / self._tstep)
        ls = self.first_samp
        if len(self._last_samples) > 0:
            ls = np.max(self._last_samples)
        return ls - self.first_samp + 1  # >= 1

    @property
    def last_samp(self):
        return self.first_samp + self.n_times - 1

    @property
    def all_IDs(self):
        return set(self._waveform_ids)

    def number_of_sources(self, desired_ids=None):
        """
        Return the number of sources with ID in desired_ids.

        :param desired_ids:
        :return:
        """
        if desired_ids is None:
            return len(self._labels)
        else:
            if not isinstance(desired_ids, list):
                desired_ids = [desired_ids]
            return sum(self._waveform_ids.count(id_) for id_ in desired_ids)

    def add_data(self, label, waveform, events, waveform_id=None, allow_overlap=False):
        """Add data to the simulation.

        Data should be added in the form of a triplet of
        Label (Where) - Waveform(s) (What) - Event(s) (When)

        Parameters
        ----------
        label : instance of Label
            The label (as created for example by mne.read_label). If the label
            does not match any sources in the SourceEstimate, a ValueError is
            raised.
        waveform : array, shape (n_times,) or (n_events, n_times) | list
            The waveform(s) describing the activity on the label vertices.
            If list, it must have the same length as events.
        events : array of int, shape (n_events, 3)
            Events associated to the waveform(s) to specify when the activity
            should occur.
        """
        if not isinstance(label, Label):
            raise ValueError(f"label must be a Label, not {type(label)}")

        label = label.restrict(self._src)
        if len(label.vertices) == 0:
            # todo: test this error
            raise ValueError(
                "The provided label does not include any vertex of the source space."
            )

        # Check if the vertices of the input label are included in any of self._labels
        for lbl in self._labels:
            mask_lbl = (
                np.isin(label.vertices, lbl.vertices) if lbl.hemi == label.hemi else []
            )
            if np.any(mask_lbl) and not allow_overlap:
                raise RuntimeError(
                    "The input label has overlap with previously added labels. "
                    "The source labels must be non-overlapping"
                )

        _validate_type(label, Label, "label")

        # If it is not a list then make it one
        if not isinstance(waveform, list) and np.ndim(waveform) == 2:
            waveform = list(waveform)
        if not isinstance(waveform, list) and np.ndim(waveform) == 1:
            waveform = [waveform]
        if len(waveform) == 1:
            waveform = waveform * len(events)
        # The length is either equal to the length of events, or 1
        if len(waveform) != len(events):
            raise ValueError(
                "Number of waveforms and events should match or "
                "there should be a single waveform (%d != %d)."
                % (len(waveform), len(events))
            )

        if waveform_id is None:
            waveform_id = ["unknown"] * len(waveform)
        if isinstance(waveform_id, int):
            waveform_id = str(waveform_id)

        if isinstance(waveform_id, str):
            waveform_id = [waveform_id] * len(waveform)
        elif isinstance(waveform_id, list) and len(waveform_id) == 1:
            waveform_id *= len(waveform)

        if len(waveform) != len(
            waveform_id
        ):  # if no one-to-one mapping between IDs and waveforms
            raise ValueError(
                "For each waveform one ID should be passed to the function."
            )
        # waveform_id = [str(id1) for id1 in waveform_id]

        events = _ensure_events(events).astype(np.int64)
        # Update the last sample possible based on events + waveforms
        self._labels.extend([label] * len(events))
        self._waveforms.extend(waveform)
        self._waveform_ids.extend(waveform_id)
        self._events = np.concatenate([self._events, events])
        assert self._events.dtype == np.int64
        # First sample per waveform is the first column of events
        # Last is computed below
        self._last_samples = np.array(
            [self._events[i, 0] + len(w) - 1 for i, w in enumerate(self._waveforms)]
        )

    def add_random_noise(self, n_noise=200, color="pink"):
        """
        Add random noise at the random vertices on the cortex.

        :param n_noise:
        :param color:
        :return:
        """
        labels_noise = random_dipole_lbl(
            self._labels, self._src, n_dipole=n_noise, subject=None
        )
        noise_generator = NoiseGenerator(color)

        for lbl_noise in labels_noise:
            self.add_data(
                label=lbl_noise,
                waveform=noise_generator.noise_signal(self.n_times),
                events=np.zeros((1, 3), int),
                waveform_id=f"{color} noise",
            )

    def get_stim_channel(self, start_sample=0, stop_sample=None):
        """Get the stim channel from the provided data.

        Returns the stim channel data according to the simulation parameters
        which should be added through the add_data method. If both start_sample
        and stop_sample are not specified, the entire duration is used.

        Parameters
        ----------
        start_sample : int
            First sample in chunk. Default is the value of the ``first_samp``
            attribute.
        stop_sample : int | None
            The final sample of the returned stc. If None, then all samples
            from start_sample onward are returned.

        Returns
        -------
        stim_data : ndarray of int, shape (n_samples,)
            The stimulation channel data.
        """
        if start_sample is None:
            start_sample = self.first_samp
        if stop_sample is None:
            stop_sample = start_sample + self.n_times - 1
        elif stop_sample < start_sample:
            raise ValueError("Argument start_sample must be >= stop_sample.")
        n_samples = stop_sample - start_sample + 1

        # Initialize the stim data array
        stim_data = np.zeros(n_samples, dtype=np.int64)

        # Select only events in the time chunk
        stim_ind = np.where(
            np.logical_and(
                self._events[:, 0] >= start_sample, self._events[:, 0] < stop_sample
            )
        )[0]

        if len(stim_ind) > 0:
            relative_ind = self._events[stim_ind, 0] - start_sample
            stim_data[relative_ind] = self._events[stim_ind, 2]

        return stim_data

    def get_stc(self, start_sample=None, stop_sample=None):
        """Simulate a SourceEstimate from the provided data.

        Returns a SourceEstimate object constructed according to the simulation
        parameters which should be added through function add_data. If both
        start_sample and stop_sample are not specified, the entire duration is
        used.

        Parameters
        ----------
        start_sample : int | None
            First sample in chunk. If ``None`` the value of the ``first_samp``
            attribute is used. Defaults to ``None``.
        stop_sample : int | None
            The final sample of the returned STC. If ``None``, then all samples
            past ``start_sample`` are returned.

        Returns
        -------
        stc : SourceEstimate object
            The generated source time courses.
        """
        if len(self._labels) == 0:
            raise ValueError(
                "No simulation parameters were found. Please use "
                "function add_data to add simulation parameters."
            )
        if start_sample is None:
            start_sample = self.first_samp
        if stop_sample is None:
            stop_sample = start_sample + self.n_times - 1
        elif stop_sample < start_sample:
            raise ValueError("start_sample must be >= stop_sample.")
        n_samples = stop_sample - start_sample + 1

        # Initialize the stc_data array to span all possible samples
        stc_data = np.zeros((len(self._labels), n_samples))

        # Select only the events that fall within the span
        ind = np.where(
            np.logical_and(
                self._last_samples >= start_sample, self._events[:, 0] <= stop_sample
            )
        )[0]

        # Loop only over the items that are in the time span
        subset_waveforms = [self._waveforms[i] for i in ind]
        for i, (waveform, event) in enumerate(zip(subset_waveforms, self._events[ind])):
            # We retrieve the first and last sample of each waveform
            # According to the corresponding event
            wf_start = event[0]
            wf_stop = self._last_samples[ind[i]]

            # Recover the indices of the event that should be in the chunk
            waveform_ind = np.isin(
                np.arange(wf_start, wf_stop + 1),
                np.arange(start_sample, stop_sample + 1),
            )

            # Recover the indices that correspond to the overlap
            stc_ind = np.isin(
                np.arange(start_sample, stop_sample + 1),
                np.arange(wf_start, wf_stop + 1),
            )

            # add the resulting waveform chunk to the corresponding label
            stc_data[ind[i]][stc_ind] += waveform[waveform_ind]

        start_sample -= self.first_samp  # STC sample ref is 0
        stc = simulate_stc(
            self._src,
            self._labels,
            stc_data,
            start_sample * self._tstep,
            self._tstep,
            allow_overlap=True,
        )

        return stc

    def __iter__(self):
        """Iterate over 1 second STCs."""
        # Arbitrary chunk size, can be modified later to something else.
        # Loop over chunks of 1 second - or, maximum sample size.
        # Can be modified to a different value.
        last_sample = self.last_samp
        for start_sample in range(self.first_samp, last_sample + 1, self._chk_duration):
            stop_sample = min(start_sample + self._chk_duration - 1, last_sample)
            yield (
                self.get_stc(start_sample, stop_sample),
                self.get_stim_channel(start_sample, stop_sample),
            )

    def restrict_sources(self, desired_ids, return_indices=False):
        """
        Return a subset of the sources defined by the desired source.

        :param desired_ids:
        :param return_indices:
        :return:
        """
        if isinstance(desired_ids, str):
            desired_ids = [desired_ids]
        elif not isinstance(desired_ids, list):
            raise ValueError("The desired IDs should be a (list of) string(s). ")

        mask_restrict = np.isin(self._waveform_ids, desired_ids)
        desired_indices = np.where(mask_restrict)[0]
        if not np.any(mask_restrict):
            return None
        simulator_new = SourceSimulator(
            self._src, self._tstep, self._duration, self.first_samp
        )
        simulator_new._labels = [
            lbl for i, lbl in enumerate(self._labels) if i in desired_indices
        ]
        simulator_new._waveforms = [
            wave for i, wave in enumerate(self._waveforms) if i in desired_indices
        ]
        simulator_new._events = self._events[desired_indices]
        simulator_new._last_samples = self._last_samples[desired_indices]
        simulator_new._chk_duration = self._chk_duration
        simulator_new._waveform_ids = [
            wv_id for i, wv_id in enumerate(self._waveform_ids) if i in desired_indices
        ]
        return (simulator_new, desired_indices) if return_indices else simulator_new

    def calculate_snr(
        self,
        signal_id,
        info,
        fwd,
        freq_band,
        noise_ids="pink noise",
        iir_params=None,
    ):
        """
        Calculate the SNR of the given source IDs.

        :param signal_id:
        :param info:
        :param fwd:
        :param noise_ids:
        :param freq_band:
        :param iir_params:
        :return:
        """
        from .raw import simulate_raw

        if not isinstance(
            signal_id, str
        ):  # the function only calculates the SNR of one type of source
            raise ValueError("the waveform ID should be a string.")
        if signal_id not in set(self._waveform_ids):
            raise ValueError(
                f"The provided signal ID {signal_id} does not exist. "
                f"Use the summary method to get an idea of the available IDs."
            )
        # todo: check that the noise IDs. Also the noise term should be in the ID

        simulator_signal, ind_signals = self.restrict_sources(
            signal_id, return_indices=True
        )
        simulator_noise = self.restrict_sources(noise_ids)

        raw_signal = simulate_raw(info, simulator_signal, forward=fwd)
        raw_noise = simulate_raw(info, simulator_noise, forward=fwd)
        if freq_band is not None:
            iir_params = (
                dict(order=2, ftype="butter") if iir_params is None else iir_params
            )
            raw_signal.filter(
                l_freq=freq_band[0],
                h_freq=freq_band[1],
                method="iir",
                iir_params=iir_params,
            )
            raw_noise.filter(
                l_freq=freq_band[0],
                h_freq=freq_band[1],
                method="iir",
                iir_params=iir_params,
            )

        power_signal = np.sum(np.var(raw_signal.get_data(), axis=-1))
        power_noise = np.sum(np.var(raw_noise.get_data(), axis=-1))
        snr_current = power_signal / power_noise
        # todo: do we need to return ind_signals?
        return snr_current, ind_signals

    def adjust_snr(
        self,
        signal_id,
        desired_snr,
        info,
        fwd,
        freq_band,
        noise_ids="pink noise",
        iir_params=None,
    ):
        """
        Adjust the SNR of desired sources at the given frequency band.

        :param signal_id:
        :param desired_snr:
        :param info:
        :param fwd:
        :param freq_band:
        :param noise_ids:
        :param iir_params:
        :return:
        """
        snr_before, ind_signals = self.calculate_snr(
            signal_id, info, fwd, noise_ids, freq_band, iir_params
        )
        correction_factor = np.sqrt(desired_snr / snr_before)
        self.rescale_sources(correction_factor, ind_signals)
        snr_after, _ = self.calculate_snr(
            signal_id, info, fwd, noise_ids, freq_band, iir_params
        )
        print(
            f"SNR before adjustment was {snr_before}, ",
            "after adjustment it is {snr_after}",
        )

    def rescale_sources(self, factor, ind_signals=None):
        """
        Rescale the source signals with the given factor.

        :param factor:
        :param ind_signals:
        :return:
        """
        if ind_signals is None:
            ind_signals = range(len(self._waveforms))
        for i_sig in ind_signals:
            self._waveforms[i_sig] = self._waveforms[i_sig] * factor

    def summary(self, verbose=True):
        """
        Give a summary of the sources in the class.

        :param verbose:
        :return:
        """
        from collections import defaultdict

        result = defaultdict(int)
        for id1 in set(self._waveform_ids):
            result[id1] += self.number_of_sources(id1)
        if verbose:
            for sig_id, n_sig in result.items():
                print(
                    f"The source simulator includes {n_sig} source(s)",
                    ' with ID "{sig_id}".',
                )
        return result

    def plot_dipole_locations(
        self,
        trans_fname,
        subject,
        subjects_dir,
        desired_ids="all",
        noise_scale=2e-3,
        source_scale=7e-3,
        noise_color=(0, 0, 255),
        source_color=(255, 0, 0),
    ):
        """
        Plot dipole locations of the sources.

        :param trans_fname:
        :param subject:
        :param subjects_dir:
        :param desired_ids:
        :param noise_scale:
        :param source_scale:
        :param noise_color:
        :param source_color:
        :return:
        """
        from ..transforms import read_trans
        from ..viz import create_3d_figure, plot_alignment, plot_dipole_locations

        desired_ids = set(self._waveform_ids) if desired_ids == "all" else desired_ids

        trans = read_trans(trans_fname)

        fig = create_3d_figure(size=(600, 400), bgcolor=(1.0, 1.0, 1.0))
        plot_alignment(
            subject=subject,
            subjects_dir=subjects_dir,
            trans=trans,
            surfaces="white",
            coord_frame="mri",
            fig=fig,
        )

        lbl_hemi = {"lh": 0, "rh": 1}
        for id1 in desired_ids:
            _, ind_id1 = self.restrict_sources(desired_ids=id1, return_indices=True)
            dip_pos = np.concatenate(
                tuple(
                    [
                        self._src[lbl_hemi[self._labels[i].hemi]]["rr"][
                            self._labels[i].vertices
                        ]
                        for i in ind_id1
                    ]
                ),
                axis=0,
            )
            dip_ori = np.zeros_like(dip_pos)  # misc orientation

            dip_len = len(dip_pos)
            dip_times = [0]

            actual_amp = np.ones(dip_len)  # misc amp
            actual_gof = np.ones(dip_len)  # misc GOF
            dipoles = Dipole(dip_times, dip_pos, actual_amp, dip_ori, actual_gof)

            scale = noise_scale if "noise" in id1 else source_scale
            color = noise_color if "noise" in id1 else source_color

            plot_dipole_locations(
                dipoles=dipoles,
                trans=trans,
                mode="sphere",
                subject=subject,
                subjects_dir=subjects_dir,
                coord_frame="mri",
                scale=scale,
                fig=fig,
                color=color,
            )
        return fig


def random_dipole_lbl(exclude_labels, src, n_dipole=200, subject=None):
    """
    Generate labels of randomly selected dipoles.

    :param source_labels:
    :return:
    """
    if subject is None:
        if exclude_labels:
            subject = exclude_labels[0].subject
        else:
            subject = "sample"

    # extract the vertices of exclude_labels and exclude them
    # from the vertices of the src
    src_vertno = [src[0]["vertno"], src[1]["vertno"]]
    for lbl in exclude_labels:
        this_hemi = 0 if (lbl.hemi == "lh") else 1
        _, idx_del_src, _ = np.intersect1d(
            src_vertno[this_hemi], lbl.vertices, return_indices=True
        )
        src_vertno[this_hemi] = np.delete(src_vertno[this_hemi], idx_del_src)

    # select random vertices from the remaining vertices
    n_vert_lh, n_vert_rh = len(src_vertno[0]), len(src_vertno[1])
    ind_dipole = np.random.choice(
        np.arange(n_vert_lh + n_vert_rh), size=n_dipole, replace=False
    )
    ind_lh = ind_dipole[ind_dipole < n_vert_lh]
    ind_rh = ind_dipole[ind_dipole >= n_vert_lh] - n_vert_lh
    dipole_vertno = [sorted(src_vertno[0][ind_lh]), sorted(src_vertno[1][ind_rh])]

    # convert the vertices to labels in the following
    dipole_labels = []
    for hemi, vertices in enumerate(dipole_vertno):
        this_hemi = "lh" if hemi == 0 else "rh"
        for vert in vertices:
            lbl = Label(
                [vert],
                src[hemi]["rr"][vert][None, :] * 1e3,
                None,
                this_hemi,
                "Label from vertex",
                subject=subject,
            )
            dipole_labels.append(lbl)
    return dipole_labels
