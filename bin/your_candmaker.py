#!/usr/bin/env python3

import argparse
import glob
import logging
import os

from rich.logging import RichHandler

from your.utils.math import normalise

os.environ["OPENBLAS_NUM_THREADS"] = (
    "1"  # stop numpy multithreading regardless of the backend
)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import textwrap
import time
from datetime import datetime
from multiprocessing import Pool, Value
from multiprocessing.util import Finalize

import numpy as np
import pandas as pd

from your.candidate import Candidate, crop, set_median_threads
from your.formats.dadastream import Overwritten, wait_for_chunk
from your.utils.gpu import (
    PinnedReadBuffer,
    check_compiler,
    check_devices,
    gpu_dedisp_and_dmt_crop,
    gpu_errors,
    release_memory,
    use_device,
)
from your.utils.misc import ReadBuffer, YourArgparseFormatter

logger = logging.getLogger()

_worker_candidate = None
_worker_candidate_key = None
_worker_candidate_finalizer = None
# this process's page-locked read buffer, made on its first GPU candidate
_read_buffer = None
# this process's reused read buffer for the CPU path
_cpu_read_buffer = None
# the GPU this worker makes all its candidates on, None on the CPU
_worker_gpu = None


def cpu_dedisp_dmt(cand, args):
    pulse_width = cand.width
    if pulse_width == 1:
        time_decimate_factor = 1
    else:
        time_decimate_factor = pulse_width // 2
    logger.debug(f"Time decimation factor {time_decimate_factor}")

    # both planes at their cropped size in one pass, where the data allow it
    cropped = (
        cand.crop_planes(
            time_decimate_factor,
            args.time_size,
            upstream_rounding=args.upstream_rounding,
            threads=worker_threads(args.nproc),
        )
        is not None
    )
    if cropped:
        logger.info("Made DMT and dedispersed profile at their cropped size")
        if args.opt_dm:
            logger.info("Optimising DM")
            logger.warning("This feature is experimental!")
            cand.optimize_dm()
        else:
            cand.dm_opt = -1
            cand.snr_opt = -1
        return cand

    # Plan the same crop on the padded, decimated time axis. Only bypass full
    # arrays when every requested bin lies inside the unpadded shifted data.
    nt = cand.data.shape[0]
    time_range = None
    if time_decimate_factor > 0 and args.time_size > 0:
        decimated_nt = (nt + time_decimate_factor - 1) // time_decimate_factor
        crop_start = decimated_nt // 2 - args.time_size // 2
        crop_stop = crop_start + args.time_size
        if (
            crop_start >= 0
            and (crop_stop < decimated_nt or args.time_size == decimated_nt)
            and crop_stop * time_decimate_factor <= nt
        ):
            time_range = (
                crop_start * time_decimate_factor,
                crop_stop * time_decimate_factor,
            )
    # ponytail: crops needing median padding retain the full-array path;
    # add bounded median handling only if those cases dominate real workloads.
    cand.dmtime(time_range=time_range)
    logger.info("Made DMT")
    if args.opt_dm:
        logger.info("Optimising DM")
        logger.warning("This feature is experimental!")
        cand.optimize_dm()
    else:
        cand.dm_opt = -1
        cand.snr_opt = -1
    cand.dedisperse(time_range=time_range)
    logger.info("Made Dedispersed profile")

    # Frequency - Time reshaping
    if time_decimate_factor != 1:
        cand.decimate(
            key="ft",
            axis=0,
            pad=True,
            decimate_factor=time_decimate_factor,
            mode="median",
        )
    if time_range is None:
        crop_start_sample_ft = cand.dedispersed.shape[0] // 2 - args.time_size // 2
        cand.dedispersed = crop(
            cand.dedispersed, crop_start_sample_ft, args.time_size, 0
        )
    logger.info(f"Decimated Time axis of FT to tsize: {cand.dedispersed.shape[0]}")
    # DM-time reshaping
    if time_decimate_factor != 1:
        cand.decimate(
            key="dmt",
            axis=1,
            pad=True,
            decimate_factor=time_decimate_factor,
            mode="median",
        )
    if time_range is None:
        crop_start_sample_dmt = cand.dmt.shape[1] // 2 - args.time_size // 2
        cand.dmt = crop(cand.dmt, crop_start_sample_dmt, args.time_size, 1)
    logger.info(
        f"Decimated DM-Time to dmsize: {cand.dmt.shape[0]} and tsize: {cand.dmt.shape[1]}"
    )
    return cand


def worker_threads(nproc):
    """
    Threads for one of `nproc` candmaker workers: an equal share of the cores
    this process may run on, so the workers do not oversubscribe them.
    """
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count() or 1
    return max(1, cores // nproc)


def _input_files(filename, num_files):
    fname, ext = os.path.splitext(filename)
    if ext == ".fits" or ext == ".sf":
        if num_files == 1:
            return [filename]
        files = glob.glob(fname[:-5] + "*fits")
        if len(files) != num_files:
            raise ValueError(
                "Number of fits files found was not equal to num_files in cand csv."
            )
        return sorted(files)
    if ext == ".fil":
        return [filename]
    raise TypeError("Can only work with list of fits file or filterbanks")


def _reader_key(files):
    return tuple(
        (path, stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_size)
        for path in files
        for stat in [os.stat(path)]
    )


def _clear_worker_candidate():
    global _worker_candidate, _worker_candidate_key
    if _worker_candidate is not None:
        try:
            _worker_candidate.fits.close()
        except Exception:
            logger.debug("Could not close cached FITS reader", exc_info=True)
    _worker_candidate = None
    _worker_candidate_key = None


def _init_cand_worker(gpu_ids=None, started=None):
    """
    Set up a candmaker worker. With GPUs, each worker takes the next of them
    in turn and makes all its candidates there, so it holds a CUDA context,
    compiled kernels and memory pool on one GPU instead of on every one.

    Args:
        gpu_ids (list): GPU IDs to share among the workers
        started (multiprocessing.Value): workers started so far
    """
    global _worker_candidate_finalizer, _worker_gpu
    if gpu_ids is not None and -1 not in gpu_ids:
        with started.get_lock():
            _worker_gpu = gpu_ids[started.value % len(gpu_ids)]
            started.value += 1
        logger.debug(f"Worker on GPU {_worker_gpu}")
    _clear_worker_candidate()
    _worker_candidate_finalizer = Finalize(
        None, _clear_worker_candidate, exitpriority=10
    )


def cand2h5(cand_val):
    return _cand2h5(cand_val)


def _cand2h5(cand_val, *, candidate=None, files=None):
    """
    TODO: Add option to use cand.resize for reshaping FT and DMT
    Generates h5 file of candidate with resized frequency-time and DM-time arrays
    :param cand_val: List of candidate parameters (filename, snr, width, dm, label, tcand(s))
    :type cand_val: Candidate
    :return: None
    """
    (
        filename,
        snr,
        width,
        dm,
        label,
        tcand,
        kill_mask_path,
        num_files,
        args,
        gpu_id,
    ) = cand_val
    # the median's threads, an equal share of the cores per worker
    set_median_threads(worker_threads(args.nproc))
    if os.path.exists(str(kill_mask_path)):
        logger.info(f"Using mask {kill_mask_path}")
        kill_chans = np.loadtxt(kill_mask_path, dtype=np.int32)
    else:
        logger.debug("No Kill Mask")

    if files is None:
        files = _input_files(filename, num_files)

    logger.debug(f"Source file list: {files}")
    candidate_kwargs = dict(
        snr=snr,
        width=width,
        dm=dm,
        label=label,
        tcand=tcand,
        device=gpu_id,
        spectral_kurtosis_sigma=args.spectral_kurtosis_sigma,
        savgol_frequency_window=args.savgol_frequency_window,
        savgol_sigma=args.savgol_sigma,
        flag_rfi=args.flag_rfi,
    )
    if candidate is None:
        cand = Candidate(files, **candidate_kwargs)
    else:
        cand = candidate._reset_candidate(**candidate_kwargs)
    if os.path.exists(str(kill_mask_path)):
        kill_mask = np.zeros(cand.nchans, dtype=np.bool_)
        kill_mask[kill_chans] = True
        cand.kill_mask = kill_mask
    if gpu_id >= 0:
        # current, so the GPU code's device blocks need not switch back to
        # GPU 0 afterwards, which would make a context there too
        use_device(gpu_id)
        # this worker makes one candidate at a time, so every chunk can be
        # read into the same page-locked buffer and uploaded from there, to
        # whichever GPU the candidate goes to
        global _read_buffer
        if _read_buffer is None:
            _read_buffer = PinnedReadBuffer(gpu_id)
        cand.read_buffer = _read_buffer
    else:
        # likewise one ordinary buffer, so each chunk is not a fresh
        # allocation the kernel has to fault in and zero
        global _cpu_read_buffer
        if _cpu_read_buffer is None:
            _cpu_read_buffer = ReadBuffer()
        cand.read_buffer = _cpu_read_buffer
    if cand.format == "dada":
        # live: wait until the ring has delivered the chunk
        wait_for_chunk(cand)
    cand.get_chunk(for_preprocessing=True)
    if cand.format == "fil":
        cand.fp.close()

    logger.info("Got Chunk")

    if gpu_id >= 0:
        logger.debug(f"Using the GPU {gpu_id}")
        try:
            cand = gpu_dedisp_and_dmt_crop(cand, device=gpu_id)
        except gpu_errors() as error:
            logger.info(
                f"Ran into a CUDA error ({error}), using the CPU version for this "
                "candidate"
            )
            gpu_failed = True
        else:
            gpu_failed = False
        if gpu_failed:
            # the failed attempt's arrays are free only now the exception is
            # gone; hand their memory back before the CPU takes over
            release_memory(gpu_id)
            cand = cpu_dedisp_dmt(cand, args)
    else:
        cand = cpu_dedisp_dmt(cand, args)

    cand.resize(
        key="ft", size=args.frequency_size, axis=1, anti_aliasing=True, mode="constant"
    )
    logger.info(f"Resized Frequency axis of FT to fsize: {cand.dedispersed.shape[1]}")
    cand.dmt = normalise(cand.dmt)
    cand.dedispersed = normalise(cand.dedispersed)
    fout = cand.save_h5(out_dir=args.fout)
    logger.debug(f"Filesize of {fout} is {os.path.getsize(fout)}")
    if not os.path.isfile(fout):
        raise IOError(f"File with {cand.id} not written")
    if os.path.getsize(fout) < 100 * 1024:
        raise ValueError(f"File with id: {cand.id} has issues! Its size is too less.")
    logger.info(fout)
    if candidate is None:
        del cand
    else:
        cand.data = cand.dedispersed = cand.dmt = None
    return fout


def _on_worker_gpu(cand_val):
    """The candidate's parameters, sent to this worker's GPU if it has one."""
    if _worker_gpu is None:
        return cand_val
    return [*cand_val[:9], _worker_gpu]


def cached_cand2h5(cand_val):
    """Reuse one FITS Candidate within each serial Pool worker."""
    global _worker_candidate, _worker_candidate_key
    cand_val = _on_worker_gpu(cand_val)
    try:
        files = _input_files(cand_val[0], cand_val[7])
        if os.path.splitext(cand_val[0])[1] not in (".fits", ".sf"):
            _clear_worker_candidate()
            return _cand2h5(cand_val, files=files)

        key = _reader_key(files)
        if key != _worker_candidate_key:
            _clear_worker_candidate()
            _worker_candidate = Candidate(files, dm=cand_val[3])
            _worker_candidate_key = key
        return _cand2h5(cand_val, candidate=_worker_candidate, files=files)
    except BaseException:
        _clear_worker_candidate()
        raise


# the history live workers read candidates from, made before they fork
_live_history = None


def live_cand2h5(cand_val):
    """
    Make one candidate from the live history.

    Returns:
        tuple: (status, h5 file or reason, seconds taken), status being
        "made", "lost" (its data had left the history) or "failed"
    """
    start = time.time()
    cand_val = _on_worker_gpu(cand_val)
    try:
        fout = _cand2h5(cand_val, files=_live_history)
    except Overwritten as error:
        logger.warning(f"Lost candidate at {cand_val[5]} s, DM {cand_val[3]}: {error}")
        result = "lost", str(error)
    except Exception as error:
        # one bad candidate must not stop a live run
        logger.exception(f"Failed candidate at {cand_val[5]} s, DM {cand_val[3]}")
        result = "failed", repr(error)
    else:
        return "made", fout, time.time() - start
    # a candidate that failed on the GPU leaves its memory in CuPy's pool,
    # which Heimdall, on the same GPU, may need
    if cand_val[9] >= 0:
        release_memory(cand_val[9])
    return *result, time.time() - start


def live_cuts(cands, values):
    """
    The Heimdall candidates worth making.

    Args:
        cands (pandas.DataFrame): from `CandWatcher.poll`
        values: parsed arguments

    Returns:
        pandas.DataFrame: candidates passing every cut
    """
    keep = (
        (cands.snr >= values.snr_min)
        & (cands.dm >= values.dm_min)
        & (cands.dm <= values.max_dm)
        & (cands.members >= values.min_members)
    )
    if values.max_boxcar is not None:
        keep &= cands.boxcar <= values.max_boxcar
    return cands[keep]


def live_hand_out(waiting, history, submit):
    """
    Submit the waiting candidates whose samples have all been read, in order.
    Workers never wait for data: were every worker waiting on samples the
    reader has not taken, none would make the candidates the reader holds the
    ring for, and the ring would stall until the hold timed out.

    Args:
        waiting (list): (key, cand_val, stop) of candidates not yet submitted;
            those submitted are removed
        history (DadaHistory): the history
        submit (callable): submits a cand_val, returning its AsyncResult

    Returns:
        list: (key, AsyncResult) of the candidates submitted
    """
    eod = history.eod
    written = history.written
    ready = [entry for entry in waiting if eod or entry[2] <= written]
    waiting[:] = [entry for entry in waiting if not (eod or entry[2] <= written)]
    return [(key, submit(cand_val)) for key, cand_val, _ in ready]


def live_finished(values, finished, eod, reported, written, idle):
    """
    Whether the live candmaker has every candidate Heimdall will report.

    Args:
        values: parsed arguments
        finished (bool): the --heimdall_done file exists
        eod (bool): the ring has ended
        reported (int): samples Heimdall's .cand files cover so far
        written (int): samples read from the ring
        idle (float): seconds since the data ended with no new .cand file

    Returns:
        bool: True to stop
    """
    if finished:
        return True
    # told when Heimdall is done, wait for that: Heimdall can be many gulps
    # behind the writer, so a quiet spell after the data end proves nothing
    if not eod or values.heimdall_done:
        return False
    return reported >= written or idle > values.end_wait


def live_main(values):
    """
    Make candidates from a psrdada ring as Heimdall reports them: read the
    ring into a rolling history, watch Heimdall's .cand files, and make each
    candidate passing the cuts from the history.
    """
    global _live_history
    import multiprocessing

    from your.formats.dadastream import DadaHistory
    from your.utils.heimdall import CandWatcher

    _live_history = DadaHistory.connect(
        values.dada_key, max_dm=values.max_dm, latency=values.latency
    )
    kill_mask_path = values.kill_mask or ""
    # one candidate kept only to work out each candidate's samples
    probe = Candidate(_live_history, dm=1.0)
    # the most samples before its time any candidate can need: Heimdall's
    # unreported gulps hold candidates from their first sample less this
    tsamp = _live_history.params["tsamp"]
    widest = 2 ** (values.max_boxcar if values.max_boxcar is not None else 12)
    tfar = 4 * _live_history.capacity * tsamp
    probe._reset_candidate(dm=values.max_dm, tcand=tfar, width=widest)
    reach = int(tfar / tsamp) - probe.chunk_bounds()[0] + 1
    watcher = CandWatcher(values.cand_dir, settle=values.settle)
    waiting = []
    jobs = []
    counts = {"cut": 0, "made": 0, "lost": 0, "failed": 0}
    # fork, so the workers share the history's memory; the reader thread
    # starts after, in this process only
    context = multiprocessing.get_context("fork")
    with context.Pool(
        processes=values.nproc,
        initializer=_init_cand_worker,
        initargs=(values.gpu_id, context.Value("i", 0)),
    ) as pool:
        _live_history.start(hold_timeout=values.hold_timeout)

        def submit(cand_val):
            return pool.apply_async(live_cand2h5, (cand_val,))

        ending = None
        while True:
            # once Heimdall has exited every .cand file is whole and no more
            # will come: take them all now rather than wait out end_wait
            finished = bool(values.heimdall_done) and os.path.exists(
                values.heimdall_done
            )
            cands = watcher.poll(final=finished)
            kept = live_cuts(cands, values)
            counts["cut"] += len(cands) - len(kept)
            for row in kept.itertuples():
                probe._reset_candidate(
                    dm=row.dm, tcand=row.tcand, width=2**row.boxcar, snr=row.snr
                )
                first, nsamp = probe.chunk_bounds()
                cand_val = [
                    _live_history.name,
                    row.snr,
                    2**row.boxcar,
                    row.dm,
                    -1,
                    row.tcand,
                    kill_mask_path,
                    1,
                    values,
                    # each worker puts it on its own GPU
                    values.gpu_id[0],
                ]
                key = len(jobs) + len(waiting)
                waiting.append((key, cand_val, first + nsamp))
                if not values.drop_when_behind:
                    # hold the ring for it once its data are in
                    _live_history.pending[key] = (max(0, first), first + nsamp)
            if not values.drop_when_behind:
                # Heimdall writes one .cand file per gulp, in order: keep what
                # the candidates of gulps not yet reported may need. Raised
                # only now, once the reported gulps' candidates are pending,
                # so their data cannot slip away in between.
                reported = len(watcher.done) * values.heimdall_gulp
                # with Heimdall done there are no unreported gulps to keep
                _live_history.hold_from = None if finished else reported - reach
            jobs += live_hand_out(waiting, _live_history, submit)
            for key, job in jobs:
                if key in _live_history.pending and job.ready():
                    del _live_history.pending[key]
            if len(cands) or not _live_history.eod:
                ending = None
            else:
                ending = ending or time.time()
            if live_finished(
                values,
                finished=finished,
                eod=_live_history.eod,
                reported=len(watcher.done) * values.heimdall_gulp,
                written=_live_history.written,
                idle=0.0 if ending is None else time.time() - ending,
            ):
                break
            time.sleep(values.poll)
        while waiting:
            jobs += live_hand_out(waiting, _live_history, submit)
            for key, job in jobs:
                if key in _live_history.pending and job.ready():
                    del _live_history.pending[key]
            time.sleep(values.poll)
        for _, job in jobs:
            status, _, _ = job.get()
            counts[status] += 1
    logging.info(
        f"Live candmaker done: {counts['made']} made, {counts['lost']} lost, "
        f"{counts['failed']} failed, {counts['cut']} cut, from "
        f"{_live_history.written} samples; held the ring "
        f"{_live_history.held_seconds:.1f} s"
    )
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="your_candmaker.py",
        description="Your candmaker! Make h5 candidates from the candidate csv files",
        formatter_class=YourArgparseFormatter,
        epilog=textwrap.dedent(
            """\
        `your_candmaker.py` can be used to make candidate cutout files. Some additional notes for this script: 
        - These files are generated in HDF file format. 
        - The output candidates have been preprocessed and consists of Dedispersed Frequency-Time and DM-Time information of the candidate. 
        - The input should be a csv file containing the parameters of the candidates. The input csv file should contain the following fields: 
                - file: Filterbank or PSRFITs file containing the data. In case of multiple files, this should contain the name of first file. 
                - snr: Signal to Noise of the candidate.
                - width: Width of candidate as log2(number of samples). 
                - dm: DM of candidate
                - label: Label of candidate (can be just set to 0, if not known)
                - stime: Start time (seconds) of the candidate.
                - chan_mask_path: Path of the channel mask file. 
                - num_files: Number of files. 
            """
        ),
    )
    parser.add_argument("-v", "--verbose", help="Be verbose", action="store_true")
    parser.add_argument(
        "-fs",
        "--frequency_size",
        type=int,
        help="Frequency size after rebinning",
        default=256,
    )
    parser.add_argument(
        "-g",
        "--gpu_id",
        help="GPU ID (use -1 for CPU). To use multiple GPUs (say with id 2 and 3 "
        "use -g 2 3; the workers (-n) take them in turn, so give at least one "
        "per GPU)",
        nargs="+",
        required=False,
        default=[-1],
        type=int,
    )
    parser.add_argument(
        "-ts", "--time_size", type=int, help="Time length after rebinning", default=256
    )
    parser.add_argument(
        "-c",
        "--cand_param_file",
        help="csv file with candidate parameters (or use --dada_key)",
        type=str,
        required=False,
    )
    parser.add_argument(
        "-n",
        "--nproc",
        type=int,
        help="number of processors to use in parallel (default: 2)",
        default=2,
    )
    parser.add_argument(
        "-o",
        "--fout",
        help="Output file directory for candidate h5",
        type=str,
        default=".",
    )
    parser.add_argument(
        "-r",
        "--flag_rfi",
        help="Turn on RFI flagging",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-sksig",
        "--spectral_kurtosis_sigma",
        help="Sigma for spectral kurtosis filter",
        type=float,
        default=4,
        required=False,
    )
    parser.add_argument(
        "-sgsig",
        "--savgol_sigma",
        help="Sigma for savgol filter",
        type=float,
        default=4,
        required=False,
    )
    parser.add_argument(
        "-sgfw",
        "--savgol_frequency_window",
        help="Filter window for savgol filter (MHz)",
        type=float,
        default=15,
        required=False,
    )
    parser.add_argument(
        "-opt",
        "--opt_dm",
        dest="opt_dm",
        help="Optimise DM",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--no_log_file", help="Do not write a log file", action="store_true"
    )
    parser.add_argument(
        "--upstream_rounding",
        help="On the CPU, round the DM-time plane in float32 as earlier versions "
        "did, for bit-identical output, instead of summing it exactly (slower)",
        action="store_true",
    )
    live = parser.add_argument_group(
        "live mode",
        "Make candidates from a psrdada ring buffer as Heimdall reports them, "
        "instead of from -c. The candmaker must attach to the ring before its "
        "data start: make the ring with one reader more than Heimdall "
        "(dada_db -r 2), or use your_heimdall.py --make_candidates.",
    )
    live.add_argument("--dada_key", help="hex key of the ring, e.g. dada", type=str)
    live.add_argument("--cand_dir", help="directory Heimdall writes .cand files to")
    live.add_argument(
        "--max_dm",
        help="highest DM to make; sets how much of the ring is kept",
        type=float,
    )
    live.add_argument(
        "--latency",
        help="seconds Heimdall may take to report a candidate after its data "
        "arrive: at least its gulp plus processing time",
        type=float,
        default=120.0,
    )
    live.add_argument("--snr_min", help="lowest S/N to make", type=float, default=6.0)
    live.add_argument("--dm_min", help="lowest DM to make", type=float, default=0.0)
    live.add_argument(
        "--max_boxcar",
        help="largest Heimdall boxcar (log2 of the width in samples) to make",
        type=int,
        default=None,
    )
    live.add_argument(
        "--min_members", help="fewest cluster members to make", type=int, default=1
    )
    live.add_argument("--kill_mask", help="channel mask file for every candidate")
    live.add_argument(
        "--settle",
        help="seconds a .cand file must be unchanged before it is read",
        type=float,
        default=1.0,
    )
    live.add_argument(
        "--end_wait",
        help="without --heimdall_done: seconds after the data end with no new "
        ".cand file before stopping, unless Heimdall's gulps already cover every "
        "sample; must be longer than Heimdall takes over a gulp",
        type=float,
        default=30.0,
    )
    live.add_argument(
        "--poll", help="seconds between looks for .cand files", type=float, default=0.5
    )
    live.add_argument(
        "--heimdall_done",
        help="file whose appearance means Heimdall has exited and written all "
        "its .cand files: the candmaker then makes the rest and stops without "
        "waiting --end_wait (your_heimdall.py --make_candidates writes it)",
        type=str,
        default=None,
    )
    live.add_argument(
        "--heimdall_gulp",
        help="Heimdall's -nsamps_gulp; with one .cand file per gulp it tells "
        "how far Heimdall has got",
        type=int,
        default=262144,
    )
    live.add_argument(
        "--hold_timeout",
        help="seconds to hold the ring for one page before giving up on the "
        "candidates that need it",
        type=float,
        default=300.0,
    )
    live.add_argument(
        "--drop_when_behind",
        help="Never hold the ring for candidates not yet made: lose them "
        "instead. By default the ring waits for Heimdall to report and for "
        "queued candidates, which holds its writer; use this where the writer "
        "must not wait, as with a live capture that would drop data.",
        action="store_true",
    )
    values = parser.parse_args()
    if (values.cand_param_file is None) == (values.dada_key is None):
        parser.error("give exactly one of -c/--cand_param_file and --dada_key")
    if values.dada_key is not None and (
        values.cand_dir is None or values.max_dm is None
    ):
        parser.error("--dada_key needs --cand_dir and --max_dm")
    if -1 in values.gpu_id and len(values.gpu_id) > 1:
        parser.error("-g -1 (the CPU) cannot be mixed with GPU IDs")

    logging_format = (
        "%(asctime)s - %(funcName)s -%(name)s - %(levelname)s - %(message)s"
    )
    log_filename = (
        values.fout
        + "/"
        + datetime.utcnow().strftime("your_candmaker_%Y_%m_%d_%H_%M_%S_%f.log")
    )

    if not values.no_log_file:
        if values.verbose:
            logging.basicConfig(
                filename=log_filename,
                level=logging.DEBUG,
                format=logging_format,
            )
        else:
            logging.basicConfig(
                filename=log_filename, level=logging.INFO, format=logging_format
            )
    else:
        if values.verbose:
            logging.basicConfig(
                level=logging.DEBUG,
                format=logging_format,
                handlers=[RichHandler(rich_tracebacks=True)],
            )
        else:
            logging.basicConfig(
                level=logging.INFO,
                format=logging_format,
                handlers=[RichHandler(rich_tracebacks=True)],
            )

    logging.info("Input Arguments:-")
    for arg, value in sorted(vars(values).items()):
        logging.info("%s: %r", arg, value)

    if -1 not in values.gpu_id:
        # every candidate would fail without the kernel compiler or the GPU,
        # each falling back to the CPU; say so once, before any worker forks
        check_compiler()
        try:
            check_devices(values.gpu_id)
        except ValueError as error:
            parser.error(str(error))
        for gpu_ids in values.gpu_id:
            logger.info(f"Using the GPU {gpu_ids}")
        if values.nproc < len(set(values.gpu_id)):
            logger.warning(
                f"{values.nproc} worker(s) for {len(set(values.gpu_id))} GPUs: "
                f"only GPU {', '.join(map(str, sorted(set(values.gpu_id[: values.nproc]))))} "
                "will be used; raise -n"
            )
    else:
        logger.info("Using CPUs only")

    if values.dada_key is not None:
        live_main(values)
        raise SystemExit(0)

    cand_pars = pd.read_csv(values.cand_param_file)
    # Randomly shuffle the candidates, this is so that the high DM candidates are spread through out
    # Else they will clog the GPU memory at once
    cand_pars.sample(frac=1).reset_index(drop=True)
    process_list = []
    # each worker puts its candidates on its own GPU
    gpu_id = values.gpu_id[0]
    for index, row in cand_pars.iterrows():
        process_list.append(
            [
                row["file"],
                row["snr"],
                2 ** row["width"],
                row["dm"],
                row["label"],
                row["stime"],
                row["chan_mask_path"],
                row["num_files"],
                values,
                gpu_id,
            ]
        )

    with Pool(
        processes=values.nproc,
        initializer=_init_cand_worker,
        initargs=(values.gpu_id, Value("i", 0)),
    ) as pool:
        pool.map(cached_cand2h5, process_list, chunksize=1)
