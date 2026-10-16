"""
A rolling history of a psrdada ring buffer, read as it fills, and a `Your`
format that reads candidates' chunks from it.

To use `psrdada` you would first need to install
[psrdada-python](https://github.com/TRASAL/psrdada-python).
"""

import logging
import math
import mmap
import multiprocessing
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)

# Bytes of each sample for the bit depths a DADA ring may carry
DTYPES = {8: np.uint8, 16: np.uint16, 32: np.float32}


class Overwritten(RuntimeError):
    """The samples asked for have already left the history."""


def _number(header, *keys):
    for key in keys:
        if header.get(key) not in (None, ""):
            return float(header[key])
    return None


def _angle(value, unit):
    if value in (None, ""):
        return None
    from astropy.coordinates import Angle

    try:
        return float(value)
    except ValueError:
        return Angle(value, unit=unit).degree


def parse_header(header):
    """
    The observation parameters in a DADA header.

    `YOUR_FCH1`, `YOUR_FOFF`, `YOUR_TSAMP`, `YOUR_RA_DEG` and `YOUR_DEC_DEG`,
    which `your.writer.Writer` writes at full precision, are preferred where
    present; otherwise the
    channel layout comes from `FREQ` (centre) and `BW` (negative when the
    highest frequency comes first), and the sampling time from `TSAMP` (µs).

    Args:
        header (dict): DADA header, keys and values as strings

    Returns:
        dict: nchans, nbits, tsamp (s), fch1 and foff (MHz), tstart (MJD),
        source_name, ra_deg and dec_deg (None when absent), npol
    """
    nchans = int(_number(header, "NCHAN"))
    nbits = int(_number(header, "NBIT"))
    if nbits not in DTYPES:
        raise ValueError(f"NBIT {nbits} is not supported; expected 8, 16 or 32")
    npol = int(_number(header, "NPOL") or 1)
    if npol != 1:
        raise ValueError(f"NPOL {npol}: only total intensity rings are supported")
    # DADA's defaults are real samples; complex ones or voltages are not
    # filterbank data
    ndim = int(_number(header, "NDIM") or 1)
    if ndim != 1:
        raise ValueError(f"NDIM {ndim}: only real (detected) samples are supported")
    state = header.get("STATE") or "Intensity"
    if state.lower() != "intensity":
        raise ValueError(f"STATE {state}: only Intensity rings are supported")

    tsamp = _number(header, "YOUR_TSAMP")
    if tsamp is None:
        tsamp = _number(header, "TSAMP") / 1e6
    foff = _number(header, "YOUR_FOFF")
    fch1 = _number(header, "YOUR_FCH1")
    if foff is None or fch1 is None:
        bw = _number(header, "BW")
        foff = bw / nchans
        fch1 = _number(header, "FREQ") - bw / 2 + foff / 2

    tstart = _number(header, "MJD_START")
    if tstart is None:
        from astropy.time import Time

        utc = header["UTC_START"]
        # DADA writes YYYY-MM-DD-hh:mm:ss
        tstart = Time(utc[:10] + "T" + utc[11:], format="isot", scale="utc").mjd
    # the ring's first byte is OBS_OFFSET bytes into the observation
    offset = _number(header, "OBS_OFFSET") or 0
    if offset:
        sample_bytes = nchans * nbits // 8
        tstart += offset / sample_bytes * tsamp / 86400

    return {
        "nchans": nchans,
        "nbits": nbits,
        "tsamp": tsamp,
        "fch1": fch1,
        "foff": foff,
        "tstart": tstart,
        "source_name": header.get("SOURCE") or "",
        "ra_deg": _number(header, "YOUR_RA_DEG")
        if header.get("YOUR_RA_DEG")
        else _angle(header.get("RA"), "hourangle"),
        "dec_deg": _number(header, "YOUR_DEC_DEG")
        if header.get("YOUR_DEC_DEG")
        else _angle(header.get("DEC"), "deg"),
        "npol": npol,
    }


def history_samples(params, max_dm, latency):
    """
    Samples a history must hold to make any candidate up to `max_dm`
    `latency` seconds after its last sample arrived: its chunk spans twice
    the dispersion delay across the band.

    Args:
        params (dict): from `parse_header`
        max_dm (float): highest DM a candidate may have
        latency (float): seconds between a sample arriving and a candidate
            that needs it being reported

    Returns:
        int: samples
    """
    f = params["fch1"] + np.arange(params["nchans"]) * params["foff"]
    delay = 4148808.0 * max_dm * (1 / f.min() ** 2 - 1 / f.max() ** 2) / 1000
    return math.ceil((2 * delay + latency) / params["tsamp"])


class DadaHistory:
    """
    The last `capacity` samples read from a DADA ring, in anonymous shared
    memory so that processes forked after it is made can read it.

    `written` counts every sample read since the start of the observation;
    the history holds samples `max(0, written - capacity) .. written`, less
    the page being copied in. One process adds pages (`add_page`, or `run`
    on a connected reader); any number read (`copy_out`).

    Args:
        header (dict): the ring's DADA header
        capacity (int): samples to keep
    """

    def __init__(self, header, capacity):
        self.header = dict(header)
        self.params = parse_header(header)
        self.nchans = self.params["nchans"]
        self.dtype = np.dtype(DTYPES[self.params["nbits"]])
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("capacity must be at least one sample")
        self.sample_bytes = self.nchans * self.dtype.itemsize
        self._memory = mmap.mmap(-1, self.capacity * self.sample_bytes)
        self.ring = np.frombuffer(self._memory, dtype=self.dtype).reshape(
            self.capacity, self.nchans
        )
        # written and the largest page so far are only ever raised, by the one
        # process adding pages; readers check them before and after copying
        self._written = multiprocessing.RawValue("q", 0)
        self._page = multiprocessing.RawValue("q", 0)
        self._eod = multiprocessing.RawValue("b", 0)
        self._partial = b""
        self.name = f"dada_{self.header.get('UTC_START', 'stream')}"
        # samples that must stay readable: `run` holds the ring rather than
        # take a page that would overwrite `hold_from`, or the first sample of
        # any `pending` (first, end) range whose data have all arrived
        self.hold_from = None
        self.pending = {}
        self.held_seconds = 0.0

    @classmethod
    def connect(cls, key, max_dm, latency, reader=None):
        """
        Connect a reader to the ring with hex `key`, wait for its header and
        make a history long enough for `max_dm` and `latency`.

        Args:
            key (str): ring key, such as "dada"
            max_dm (float): highest DM a candidate may have
            latency (float): seconds candidates may arrive after their data
            reader: a `psrdada.Reader`, made if None

        Returns:
            DadaHistory: with the connected reader kept for `run`
        """
        if reader is None:
            from psrdada import Reader

            reader = Reader()
        reader.connect(int(str(key), 16))
        header = reader.getHeader()
        params = parse_header(header)
        capacity = history_samples(params, max_dm, latency)
        history = cls(header, capacity)
        history.reader = reader
        logger.info(
            f"Keeping {capacity} samples ({capacity * params['tsamp']:.1f} s, "
            f"{history.ring.nbytes / 2**30:.2f} GiB) of ring {key}"
        )
        return history

    @property
    def written(self):
        return self._written.value

    @property
    def eod(self):
        return bool(self._eod.value)

    def oldest(self):
        """
        Returns:
            int: first sample still safe to read, allowing for a page being
            copied in over the oldest ones
        """
        return max(0, self._written.value + self._page.value - self.capacity)

    def add_page(self, page):
        """
        Append a page of the ring. Pages need not hold whole samples; a
        partial sample is kept for the next page.

        Raises:
            ValueError: if the page is over half the history

        Args:
            page: bytes-like page as read from the ring
        """
        raw = memoryview(page).cast("B")
        if self._partial:
            raw = memoryview(self._partial + bytes(raw))
        whole = len(raw) // self.sample_bytes * self.sample_bytes
        self._partial = bytes(raw[whole:])
        data = np.frombuffer(raw[:whole], dtype=self.dtype).reshape(-1, self.nchans)
        n = len(data)
        if 2 * n > self.capacity:
            # the page being copied in is off limits to readers, so a history
            # under two pages long could never be read safely
            raise ValueError(
                f"a page of {n} samples needs a history of at least {2 * n}; "
                f"this one holds {self.capacity}. Raise --latency."
            )
        if n > self._page.value:
            self._page.value = n
        start = self._written.value % self.capacity
        first = min(n, self.capacity - start)
        self.ring[start : start + first] = data[:first]
        self.ring[: n - first] = data[first:]
        self._written.value += n

    def mark_end(self):
        """No more pages will come."""
        self._eod.value = 1

    def room_for(self, n):
        """
        Whether a page of `n` samples can be added without overwriting
        `hold_from` or a pending range whose data have all arrived.
        """
        written = self._written.value
        # ranges still arriving are not held, so the ring cannot wait on
        # itself; the history is long enough for them
        firsts = [f for f, end in list(self.pending.values()) if end <= written]
        if self.hold_from is not None:
            firsts.append(self.hold_from)
        if not firsts:
            return True
        hold = min(firsts)
        page = max(self._page.value, n)
        return written + n + page - self.capacity <= hold

    def run(self, poll=0.01, hold_timeout=None):
        """
        Copy every page of the connected ring into the history until the end
        of data, then disconnect. A page is cleared as soon as it is copied,
        unless taking it would overwrite samples still needed (`room_for`):
        then the page is held, which holds the writer, until there is room.

        Args:
            poll (float): seconds between looks for room while holding
            hold_timeout (float): seconds to hold one page before taking it
                anyway, so a wrong guess at what is needed cannot stall the
                ring for good; None to hold for as long as it takes
        """
        reader = self.reader
        try:
            for page in reader:
                n = (len(page) + len(self._partial)) // self.sample_bytes
                held = None
                while not self.room_for(n):
                    if held is None:
                        held = time.time()
                        logger.debug(f"Holding the ring at sample {self.written}")
                    elif hold_timeout is not None and time.time() - held > hold_timeout:
                        logger.warning(
                            f"Held the ring {hold_timeout} s at sample "
                            f"{self.written}; taking the page, which may lose "
                            "candidates"
                        )
                        break
                    time.sleep(poll)
                if held is not None:
                    self.held_seconds += time.time() - held
                self.add_page(page)
        finally:
            self.mark_end()
            logger.info(f"End of data after {self.written} samples")
            try:
                reader.disconnect()
            except Exception:
                logger.debug("Could not disconnect the reader", exc_info=True)

    def start(self, hold_timeout=None):
        """
        Run `run` in a daemon thread of this process.

        Args:
            hold_timeout (float): as for `run`

        Returns:
            threading.Thread: the reading thread
        """
        thread = threading.Thread(
            target=self.run,
            kwargs={"hold_timeout": hold_timeout},
            name="dada reader",
            daemon=True,
        )
        thread.start()
        return thread

    def wait_for(self, end, poll=0.01):
        """
        Wait until sample `end` has been read, or the data have ended.

        Args:
            end (int): one past the last sample needed
            poll (float): seconds between checks
        """
        while self._written.value < end and not self._eod.value:
            time.sleep(poll)

    def copy_out(self, start, out):
        """
        Copy samples `start .. start + len(out)` into `out`.

        Args:
            start (int): first sample
            out (numpy.ndarray): (n, nchans) array of the history's dtype

        Raises:
            Overwritten: if any of them left the history before or while
                they were copied
            ValueError: if any have not been read yet
        """
        n = len(out)
        if start < 0 or start + n > self._written.value:
            raise ValueError(
                f"samples {start}..{start + n} are not in the {self.written} read"
            )
        if start < self.oldest():
            raise Overwritten(
                f"sample {start} has left the history (oldest {self.oldest()})"
            )
        first_index = start % self.capacity
        first = min(n, self.capacity - first_index)
        out[:first] = self.ring[first_index : first_index + first]
        out[first:] = self.ring[: n - first]
        # a page added meanwhile may have overwritten the start
        if start < self.oldest():
            raise Overwritten(
                f"sample {start} was overwritten while it was copied "
                f"(oldest {self.oldest()})"
            )


def wait_for_chunk(cand, history=None):
    """
    Wait until a candidate's chunk is in its history, and bring the
    candidate's spectra count up to date, so `get_chunk` pads only past the
    end of the data.

    Args:
        cand: Candidate read from a `DadaHistory`
        history (DadaHistory): the history; the candidate's own if None

    Raises:
        Overwritten: if the chunk has started to leave the history
    """
    history = history if history is not None else cand.your_file
    nstart, nsamp = cand.chunk_bounds()
    first = max(0, nstart)
    if first < history.oldest():
        raise Overwritten(
            f"candidate at {cand.tcand} s needs sample {first}, but the history "
            f"starts at {history.oldest()}"
        )
    history.wait_for(nstart + nsamp)
    cand.your_header.native_nspectra = history.written


class DadaFile:
    """
    `Your` format for a `DadaHistory`: the attributes and methods `Your` and
    its `Header` take from `SigprocFile` and `PsrfitsFile`.

    Args:
        history (DadaHistory): the history to read
    """

    def __init__(self, history):
        p = history.params
        self.history = history
        self.nchans = p["nchans"]
        self.nbits = p["nbits"]
        self.nifs = 1
        self.poln_order = "I"
        self.tsamp = p["tsamp"]
        self.fch1 = p["fch1"]
        self.foff = p["foff"]
        self.tstart = p["tstart"]
        self.source_name = p["source_name"]
        self.ra_deg = p["ra_deg"]
        self.dec_deg = p["dec_deg"]
        self.bw = self.nchans * self.foff
        self.cfreq = self.fch1 + (self.bw / 2) - (self.foff / 2)

    def native_tsamp(self):
        return self.tsamp

    def native_foff(self):
        return self.foff

    def native_nchans(self):
        return self.nchans

    def native_nspectra(self):
        return self.history.written

    def get_data(self, nstart, nsamp, offset=0, pol=0, npoln=1):
        """
        Samples `nstart .. nstart + nsamp` of the history, clipped to those
        read so far, as (nsamp, 1, nchans). With a `read_buffer` they are
        copied into it, as `SigprocFile` reads into it.

        Args:
            nstart (int): first sample
            nsamp (int): samples
            offset (int): added to nstart
            pol (int): only 0, total intensity
            npoln (int): only 1

        Returns:
            numpy.ndarray: data
        """
        if pol != 0 or npoln != 1:
            raise ValueError("DADA rings carry total intensity only")
        history = self.history
        start = max(0, int(nstart) + int(offset))
        stop = min(int(nstart) + int(offset) + int(nsamp), history.written)
        n = max(0, stop - start)
        size = n * history.sample_bytes
        read_buffer = getattr(self, "read_buffer", None)
        if read_buffer is None:
            raw = np.empty(size, dtype=np.uint8)
        else:
            raw = read_buffer(size)[:size]
        out = raw.view(history.dtype).reshape(n, history.nchans)
        history.copy_out(start, out)
        return out.reshape(n, 1, history.nchans)
