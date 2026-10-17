#!/usr/bin/env python3
"""
CLI to run Heimdall on fits or fils using Your.
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import textwrap
import time
from datetime import datetime
from multiprocessing import Process

import numpy as np
from rich.logging import RichHandler

from your import Your
from your.utils.heimdall import HeimdallManager
from your.utils.misc import MyEncoder, YourArgparseFormatter
from your.writer import Writer


def page_samples(gulp, most):
    """
    The largest whole divisor of the writer's `gulp` that is at most `most`
    samples, so every page of the ring is full.
    """
    best = 1
    for d in range(1, int(gulp**0.5) + 1):
        if gulp % d == 0:
            for c in (d, gulp // d):
                if best < c <= most:
                    best = c
    return best


def run_in_own_group(run):
    """
    Call `run` as the leader of a new process group, so that stopping the
    group also stops any program it started, such as heimdall.
    """
    os.setpgrp()
    run()


def tell_candmaker_heimdall_is_done(candmaker, done_file):
    """
    Write `done_file`, the candmaker's sign that Heimdall has exited and
    written every .cand file, if there is a candmaker and it is not written.
    """
    if candmaker is not None and not os.path.exists(done_file):
        with open(done_file, "w"):
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="your_heimdall.py",
        description="Your Heimdall Fetch FRB",
        formatter_class=YourArgparseFormatter,
        epilog=textwrap.dedent(
            """\
        `your_heimdall.py` runs Heimdall on Dada buffers for given file(s). Here are some additional notes for this script.
         - To use data present in multiple contiguous PSRFITS format files, just use `-f *.fits`.
         - Use the RFI mitigation algorithms provided in `your` by adding `-flag_rfi` to the command.
         - Do sub-banded search with `--channel_start` and `--channel_end` to specify the channel range to use.
         - Give a channel mask as a text file using `--mask`.
         - All the relevant Heimdall inputs can be set using various command line arguments.
         """
        ),
    )
    parser.add_argument("-v", "--verbose", help="Be verbose", action="count", default=0)
    parser.add_argument("-f", "--files", help="filterbank or psrfits", nargs="+")
    parser.add_argument(
        "-dm",
        "--dm",
        help="DM (eg -dm 10 1000)",
        type=float,
        nargs=2,
        default=[10, 1000],
    )
    parser.add_argument(
        "-g",
        "--gpu_id",
        help="GPU ID to run heimdall on",
        type=int,
        required=False,
        default=0,
    )
    parser.add_argument(
        "-flag_rfi",
        "--flag_rfi",
        help="Use your to flag RFI",
        required=False,
        action="store_true",
    )
    parser.add_argument(
        "-sk_sigma",
        "--spectral_kurtosis_sigma",
        help="Sigma for spectral kurtosis based RFI mitigation, only applied if -flag_rfi is used.",
        required=False,
        default=4,
        type=float,
    )
    parser.add_argument(
        "-sg_sigma",
        "--savgol_sigma",
        help="Sigma for Savgol filter for RFI mitigation, only applied if -flag_rfi is used.",
        default=4,
        type=float,
        required=False,
    )
    parser.add_argument(
        "-sg_frequency",
        "--savgol_frequency_window",
        help="Filter window for savgol filter (in MHz), only applied if -flag_rfi is used.",
        default=15,
        required=False,
        type=float,
    )
    parser.add_argument(
        "-dm_tol",
        "--dm_tol",
        help="SNR loss tolerance between DM trials",
        required=False,
        type=float,
        default=1.25,
    )
    parser.add_argument(
        "-nsamps_gulp",
        "--nsamps_gulp",
        help="samples Heimdall reads at a time; by default the next power of two "
        "above the largest dispersion delay, and at least 2**18",
        required=False,
        type=int,
        default=None,
    )
    for name, what in (
        ("time", "samples"),
        ("filter", "boxcar filters"),
        ("dm_trial", "DM trials"),
    ):
        parser.add_argument(
            f"-cand_sep_{name}",
            f"--cand_sep_{name}",
            help=f"{what} apart that Heimdall keeps candidates in separate "
            "clusters; Heimdall's default if not given",
            required=False,
            type=int,
            default=None,
        )
    parser.add_argument(
        "-rfi_no_narrow",
        "--rfi_no_narrow",
        help="disable narrow band RFI excision",
        required=False,
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-rfi_no_broad",
        "--rfi_no_broad",
        help="disable 0-DM RFI excision",
        required=False,
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-boxcar_max",
        "--boxcar_max",
        help="maximum boxcar width in seconds",
        required=False,
        type=float,
        default=50e-3,
    )
    parser.add_argument(
        "-mask",
        "--mask",
        help="File containing channel numbers to flag",
        required=False,
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        help="Output dir for heimdall candidates",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "-fs",
        "--channel_start",
        help="Start channel for sub band search",
        type=int,
        required=False,
        default=0,
    )
    parser.add_argument(
        "-fe",
        "--channel_end",
        help="End channel for sub band search",
        type=int,
        required=False,
        default=-1,
    )
    parser.add_argument(
        "-no_scrunching",
        "--no_scrunching",
        help="Disable adaptive scrunching in heimdall",
        required=False,
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--no_progress",
        help="Do not show the tqdm bar",
        action="store_false",
    )
    parser.add_argument(
        "--no_log_file", help="Do not write a log file", action="store_true"
    )
    parser.add_argument(
        "--make_candidates",
        help="Also run your_candmaker.py on the ring as a second reader, making "
        "h5 candidates from memory as Heimdall reports them",
        action="store_true",
    )
    parser.add_argument(
        "--candmaker_args",
        help="Extra arguments for your_candmaker.py, e.g. '-n 4 -g 0 --snr_min 7'",
        type=str,
        default="",
    )
    values = parser.parse_args()

    if values.output_dir is None:
        values.output_dir = (
            f"{os.getcwd()}/"
            + f"{('.').join(os.path.basename(values.files[0]).split('.')[:-1])}"
        )
        os.makedirs(values.output_dir, exist_ok=True)

    LOGGING_FORMAT = (
        "%(asctime)s - %(funcName)s -%(name)s - %(levelname)s - %(message)s"
    )

    log_filename = (
        values.output_dir
        + "/"
        + datetime.utcnow().strftime("your_heimdall_%Y_%m_%d_%H_%M_%S_%f.log")
    )

    if not values.no_log_file:
        if values.verbose:
            logging.basicConfig(
                filename=log_filename,
                level=logging.DEBUG,
                format=LOGGING_FORMAT,
            )
        else:
            logging.basicConfig(
                filename=log_filename, level=logging.INFO, format=LOGGING_FORMAT
            )
    else:
        if values.verbose:
            logging.basicConfig(
                level=logging.DEBUG,
                format=LOGGING_FORMAT,
                handlers=[RichHandler(rich_tracebacks=True)],
            )
        else:
            logging.basicConfig(
                level=logging.INFO,
                format=LOGGING_FORMAT,
                handlers=[RichHandler(rich_tracebacks=True)],
            )

    logging.info("Input Arguments:-")
    for arg, value in sorted(vars(values).items()):
        logging.info("%s: %r", arg, value)

    your_object = Your(file=values.files)
    max_delay = your_object.dispersion_delay(dms=np.max(values.dm))
    dispersion_delay_samples = np.ceil(max_delay / your_object.your_header.tsamp)
    logging.info("Max Dispersion delay = %f s", max_delay)
    logging.info("Max Dispersion delay = %i samples", dispersion_delay_samples)

    if values.nsamps_gulp is not None:
        nsamps_gulp = min(values.nsamps_gulp, your_object.your_header.nspectra)
    elif your_object.your_header.nspectra < 2**18:
        nsamps_gulp = your_object.your_header.nspectra
    else:
        nsamps_gulp = int(
            np.max([(2 ** np.ceil(np.log2(dispersion_delay_samples))), 2**18])
        )

    if values.channel_end > 0:
        c_max = values.channel_end
    else:
        c_max = None

    your_dada_writer = Writer(
        your_object=your_object,
        progress=~values.no_progress,
        c_min=values.channel_start,
        c_max=c_max,
        highest_frequency_first=True,
        savgol_sigma=values.savgol_sigma,
        spectral_kurtosis_sigma=values.spectral_kurtosis_sigma,
        savgol_frequency_window=values.savgol_frequency_window,
        flag_rfi=values.flag_rfi,
    )
    if values.make_candidates:
        # pages a quarter of Heimdall's gulp, so the writer can run at most
        # the ring's 8 pages, two gulps, ahead of Heimdall, and the
        # candmaker's history need only cover three gulps past the delay
        data_step = page_samples(your_dada_writer.gulp, nsamps_gulp // 4)
        your_dada_writer.setup_dada(data_step=data_step, n_readers=2)
    else:
        # need to run the set up inorder to get the dada key
        your_dada_writer.setup_dada()

    if values.mask:
        logging.info("Reading RFI mask from %s", values.mask)
        mask = np.loadtxt(values.mask)
        if len(mask.shape) == 1:
            bad_chans = list(mask)
        else:
            logging.warning(
                "RFI mask not understood, can only be 1D. Not using RFI flagging."
            )
            bad_chans = None
    else:
        logging.info("No RFI flagging done.")
        bad_chans = None

    if values.verbose <= 1:
        HEIMDALL_VERBOSITY = "v"
    elif values.verbose == 2:
        HEIMDALL_VERBOSITY = "V"
    elif values.verbose == 3:
        HEIMDALL_VERBOSITY = "g"
    else:
        HEIMDALL_VERBOSITY = "G"

    HM = HeimdallManager(
        dm=values.dm,
        dada_key=your_dada_writer.dada_key,
        boxcar_max=int(values.boxcar_max / your_object.your_header.tsamp),
        verbosity=HEIMDALL_VERBOSITY,
        nsamps_gulp=nsamps_gulp,
        gpu_id=values.gpu_id,
        output_dir=values.output_dir,
        no_scrunching=values.no_scrunching,
        zap_chans=bad_chans,
        rfi_no_broad=values.rfi_no_broad,
        rfi_no_narrow=values.rfi_no_narrow,
        dm_tol=values.dm_tol,
        cand_sep_time=values.cand_sep_time,
        cand_sep_filter=values.cand_sep_filter,
        cand_sep_dm_trial=values.cand_sep_dm_trial,
    )

    with open(
        values.output_dir
        + "/"
        + your_object.your_header.basename
        + "_heimdall_inputs.json",
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(HM.__dict__, fp, cls=MyEncoder, indent=4)

    dada_process = Process(name="To dada", target=your_dada_writer.to_dada)
    heimdall_process = Process(name="Heimdall", target=run_in_own_group, args=(HM.run,))

    candmaker = None
    # written once Heimdall has exited, so the candmaker need not wait to be
    # sure no more .cand files are coming
    heimdall_done = os.path.join(values.output_dir, "heimdall.done")
    if values.make_candidates:
        if os.path.exists(heimdall_done):
            os.remove(heimdall_done)
        latency = 3 * nsamps_gulp * your_object.your_header.tsamp + 30
        candmaker_cmd = [
            sys.executable,
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "your_candmaker.py"
            ),
            "--dada_key",
            str(your_dada_writer.dada_key),
            "--cand_dir",
            values.output_dir,
            "--max_dm",
            str(float(np.max(values.dm))),
            "--latency",
            str(latency),
            "--heimdall_gulp",
            str(nsamps_gulp),
            "--max_boxcar",
            str(int(np.ceil(np.log2(max(HM.boxcar_max, 1))))),
            "--heimdall_done",
            heimdall_done,
            "-o",
            values.output_dir,
            *values.candmaker_args.split(),
        ]
        logging.info("Starting the candmaker: %s", " ".join(candmaker_cmd))
        candmaker = subprocess.Popen(candmaker_cmd)

    dada_process.start()
    heimdall_process.start()

    stopped = False
    try:
        # the writer waits for every reader, so a reader that dies (Heimdall
        # or the candmaker) would stall the others; stop everything if one does
        while heimdall_process.is_alive() or dada_process.is_alive():
            if candmaker is not None and candmaker.poll() not in (None, 0):
                raise RuntimeError(
                    f"your_candmaker.py exited with {candmaker.returncode}"
                )
            for process in (heimdall_process, dada_process):
                if process.exitcode not in (None, 0):
                    raise RuntimeError(f"{process.name} exited with {process.exitcode}")
            if heimdall_process.exitcode == 0:
                tell_candmaker_heimdall_is_done(candmaker, heimdall_done)
            time.sleep(1)
        heimdall_process.join()
        dada_process.join()
        if candmaker is not None:
            tell_candmaker_heimdall_is_done(candmaker, heimdall_done)
            candmaker.wait()
            os.remove(heimdall_done)
    except (KeyboardInterrupt, RuntimeError) as error:
        logging.error("Stopping: %r", error)
        stopped = True
        # heimdall itself as well as the process that started it
        try:
            os.killpg(heimdall_process.pid, signal.SIGTERM)
        except (ProcessLookupError, TypeError):
            pass
        heimdall_process.terminate()
        dada_process.terminate()
        if candmaker is not None:
            candmaker.terminate()
        heimdall_process.join()
        dada_process.join()

    # disconnecting waits on the readers, which may be gone after a stop
    your_dada_writer.DM.teardown(disconnect=not stopped)
    logging.info("Destroyed dada buffers")
    if stopped:
        raise SystemExit(1)
