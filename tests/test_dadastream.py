import importlib.util
import os
import threading
import time

import numpy as np
import pandas as pd
import pytest

from your import Your
from your.candidate import Candidate
from your.formats.dadastream import (
    DadaHistory,
    Overwritten,
    history_samples,
    parse_header,
    wait_for_chunk,
)
from your.utils.heimdall import CandWatcher, read_cand_file
from your.writer import Writer

_install_dir = os.path.abspath(os.path.dirname(__file__))
FIL = os.path.join(_install_dir, "data/28.fil")


def _header(nchans=4, nbits=8, **extra):
    header = {
        "NCHAN": str(nchans),
        "NBIT": str(nbits),
        "TSAMP": "64.0",
        "FREQ": "1400.0",
        "BW": "-400.0",
        "MJD_START": "60000.5",
        "UTC_START": "2023-02-25-12:00:00",
        "NPOL": "1",
    }
    header.update(extra)
    return header


def _filled(history, start, n):
    """Pages whose sample s holds s in every channel."""
    data = np.repeat(np.arange(start, start + n)[:, None], history.nchans, axis=1)
    return data.astype(history.dtype)


def test_parse_header_band_from_freq_and_bw():
    p = parse_header(_header())
    assert p["foff"] == -100.0
    assert p["fch1"] == 1550.0
    assert p["tsamp"] == pytest.approx(64e-6)
    assert p["ra_deg"] is None and p["dec_deg"] is None


def test_parse_header_prefers_exact_keys_and_reads_positions():
    p = parse_header(
        _header(
            YOUR_FCH1=repr(1549.9999999),
            YOUR_FOFF=repr(-100.0000001),
            YOUR_TSAMP=repr(6.4e-05),
            RA="19:39:38.56",
            DEC="+21:34:59.1",
            SOURCE="B1937+21",
        )
    )
    assert p["fch1"] == 1549.9999999 and p["foff"] == -100.0000001
    assert p["tsamp"] == 6.4e-05
    assert p["ra_deg"] == pytest.approx(294.91067, abs=1e-4)
    assert p["dec_deg"] == pytest.approx(21.58308, abs=1e-4)
    assert p["source_name"] == "B1937+21"


def test_parse_header_utc_start_without_mjd():
    header = _header()
    del header["MJD_START"]
    assert parse_header(header)["tstart"] == pytest.approx(60000.5)


def test_parse_header_starts_obs_offset_bytes_in():
    header = _header(OBS_OFFSET=str(1000 * 4))
    tstart = parse_header(header)["tstart"]
    assert tstart == pytest.approx(60000.5 + 1000 * 64e-6 / 86400, abs=1e-12)


@pytest.mark.parametrize(
    "bad", [{"NBIT": "4"}, {"NPOL": "2"}, {"NDIM": "2"}, {"STATE": "Nyquist"}]
)
def test_parse_header_refuses_what_it_cannot_read(bad):
    with pytest.raises(ValueError):
        parse_header(_header(**bad))


def test_history_samples_cover_twice_the_delay_and_latency():
    p = parse_header(_header())
    delay = 4148808.0 * 100 * (1 / 1250.0**2 - 1 / 1550.0**2) / 1000
    assert history_samples(p, 100, 2.0) == int(np.ceil((2 * delay + 2) / 64e-6))


@pytest.mark.parametrize("nbits", [8, 16, 32])
def test_history_wraps_and_copies_out(nbits):
    history = DadaHistory(_header(nbits=nbits), capacity=10)
    for start in range(0, 23, 3):
        history.add_page(_filled(history, start, min(3, 23 - start)).tobytes())
    assert history.written == 23
    out = np.empty((7, 4), dtype=history.dtype)
    history.copy_out(16, out)
    np.testing.assert_array_equal(out, _filled(history, 16, 7))
    with pytest.raises(Overwritten):
        history.copy_out(12, np.empty((2, 4), dtype=history.dtype))
    with pytest.raises(ValueError):
        history.copy_out(20, np.empty((4, 4), dtype=history.dtype))


def test_history_keeps_partial_samples_between_pages():
    history = DadaHistory(_header(nbits=16), capacity=8)
    raw = _filled(history, 0, 5).tobytes()
    for cut in (3, 11, 17, 40):
        history.add_page(raw[:cut])
        raw = raw[cut:]
    assert history.written == 5
    out = np.empty((5, 4), dtype=history.dtype)
    history.copy_out(0, out)
    np.testing.assert_array_equal(out, _filled(history, 0, 5))


def test_history_refuses_pages_over_half_its_length():
    history = DadaHistory(_header(), capacity=8)
    history.add_page(_filled(history, 0, 4).tobytes())
    with pytest.raises(ValueError, match="--latency"):
        history.add_page(_filled(history, 4, 5).tobytes())


def test_copy_out_notices_a_page_landing_mid_copy():
    history = DadaHistory(_header(), capacity=6)
    history.add_page(_filled(history, 0, 3).tobytes())
    history.add_page(_filled(history, 3, 3).tobytes())

    class Racing(np.ndarray):
        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            if history.written == 6:
                history.add_page(_filled(history, 6, 3).tobytes())

    out = np.empty((3, 4), dtype=history.dtype).view(Racing)
    # samples 3.. were the oldest safe ones when the copy began
    with pytest.raises(Overwritten):
        history.copy_out(3, out)


def test_wait_for_returns_on_data_or_end():
    history = DadaHistory(_header(), capacity=16)
    threading.Timer(
        0.05, lambda: history.add_page(_filled(history, 0, 4).tobytes())
    ).start()
    t = time.time()
    history.wait_for(4)
    assert history.written == 4 and time.time() - t < 5
    threading.Timer(0.05, history.mark_end).start()
    history.wait_for(100)
    assert history.eod


def _history_of(path, capacity=None):
    """A history fed every sample of a filterbank, as the ring would carry it."""
    your = Your(path)
    header = Writer(your).dada_header()
    data = your.get_data(0, your.your_header.nspectra)
    # one page more than the data, as a page in flight is never read
    history = DadaHistory(header, capacity or len(data) + 1000)
    for start in range(0, len(data), 1000):
        history.add_page(data[start : start + 1000].tobytes())
    history.mark_end()
    return your, history


def test_candidates_from_history_match_the_file():
    your, history = _history_of(FIL)
    assert history.params["fch1"] == your.your_header.fch1
    assert history.params["tsamp"] == your.your_header.tsamp
    assert history.params["ra_deg"] == your.your_header.ra_deg
    assert history.params["dec_deg"] == your.your_header.dec_deg
    kwargs = dict(dm=475.284, tcand=2.02888, width=32, label=-1, snr=16.8, min_samp=256)
    live = Candidate(history, **kwargs)
    wait_for_chunk(live)
    live.get_chunk()
    offline = Candidate(FIL, **kwargs)
    offline.get_chunk()
    assert live.chunk_bounds() == offline.chunk_bounds()
    np.testing.assert_array_equal(live.data, offline.data)
    for cand in (live, offline):
        assert cand.crop_planes(16, 256) is not None
    np.testing.assert_array_equal(live.dmt, offline.dmt)
    np.testing.assert_array_equal(live.dedispersed, offline.dedispersed)
    assert live.id == offline.id


def test_candidate_reads_into_a_read_buffer():
    _, history = _history_of(FIL)
    cand = Candidate(history, dm=475.284, tcand=2.02888, width=2, snr=10.0)
    buffer = np.empty(2**24, dtype=np.uint8)
    cand.read_buffer = lambda nbytes: buffer
    wait_for_chunk(cand)
    cand.get_chunk()
    reference = Candidate(FIL, dm=475.284, tcand=2.02888, width=2, snr=10.0)
    reference.get_chunk()
    np.testing.assert_array_equal(cand.data, reference.data)


def test_lost_candidate_is_reported():
    your, history = _history_of(FIL, capacity=2000)
    cand = Candidate(history, dm=475.284, tcand=0.5, width=2, snr=10.0)
    with pytest.raises(Overwritten):
        wait_for_chunk(cand)


SNIPPET = """6.34084\t159526\t40.8387\t1\t47\t23.703\t2\t159526\t159528\t
10.2453\t211617\t54.174\t1\t0\t10\t2\t162145\t162146\t
24.8437\t185100\t47.3856\t7\t258\t85.7907\t64242\t30388\t40652\t
"""


def test_read_cand_file(tmp_path):
    path = tmp_path / "a.cand"
    path.write_text(SNIPPET)
    frame = read_cand_file(str(path))
    assert list(frame.columns)[:3] == ["snr", "sample", "tcand"]
    assert frame.boxcar.tolist() == [1, 1, 7]
    assert frame.dm.tolist() == [23.703, 10.0, 85.7907]
    empty = tmp_path / "b.cand"
    empty.write_text("")
    assert len(read_cand_file(str(empty))) == 0


def test_cand_watcher_reads_each_file_once_after_it_settles(tmp_path):
    watcher = CandWatcher(str(tmp_path), settle=1.0)
    (tmp_path / "a.cand").write_text(SNIPPET)
    assert len(watcher.poll(now=100.0)) == 0  # first sight
    assert len(watcher.poll(now=100.5)) == 0  # not settled yet
    got = watcher.poll(now=101.2)
    assert len(got) == 3 and got.file.iloc[0].endswith("a.cand")
    assert len(watcher.poll(now=110.0)) == 0  # never again
    # a file still being written restarts its clock
    growing = tmp_path / "b.cand"
    growing.write_text(SNIPPET[:40])
    watcher.poll(now=200.0)
    growing.write_text(SNIPPET)
    os.utime(growing, (1, 2))
    assert len(watcher.poll(now=201.5)) == 0
    assert len(watcher.poll(now=202.6)) == 3


def test_cand_watcher_final_poll_reads_unsettled_files_once(tmp_path):
    watcher = CandWatcher(str(tmp_path), settle=1.0)
    (tmp_path / "a.cand").write_text(SNIPPET)
    (tmp_path / "b.cand").write_text("")
    watcher.poll(now=100.0)
    got = watcher.poll(now=100.1, final=True)
    assert len(got) == 3 and len(watcher.done) == 2
    assert len(watcher.poll(now=200.0, final=True)) == 0


def _your_heimdall():
    spec = importlib.util.spec_from_file_location(
        "your_heimdall", os.path.join(_install_dir, "../bin/your_heimdall.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_heimdall_done_is_written_once_and_only_for_a_candmaker(tmp_path):
    tell = _your_heimdall().tell_candmaker_heimdall_is_done
    done = tmp_path / "heimdall.done"
    tell(None, str(done))
    assert not done.exists()
    tell(object(), str(done))
    assert done.exists()
    done.write_text("kept")
    tell(object(), str(done))
    assert done.read_text() == "kept"


def _candmaker():
    spec = importlib.util.spec_from_file_location(
        "your_candmaker", os.path.join(_install_dir, "../bin/your_candmaker.py")
    )
    candmaker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candmaker)
    return candmaker


def test_live_run_waits_for_a_slow_heimdall():
    from types import SimpleNamespace

    finished = _candmaker().live_finished
    told = SimpleNamespace(heimdall_done="heimdall.done", end_wait=30.0)
    untold = SimpleNamespace(heimdall_done=None, end_wait=30.0)
    # the writer has finished but Heimdall has reported one gulp of four:
    # a long quiet spell must not end the run when Heimdall will say it is done
    state = dict(eod=True, reported=524288, written=2097152, idle=300.0)
    assert not finished(told, finished=False, **state)
    assert finished(told, finished=True, **state)
    # without the done file only the quiet spell is left to go by
    assert finished(untold, finished=False, **state)
    assert not finished(untold, finished=False, **{**state, "idle": 10.0})
    # ...or Heimdall's gulps covering every sample
    covered = {**state, "reported": 2097152, "idle": 0.0}
    assert finished(untold, finished=False, **covered)
    assert not finished(untold, finished=False, **{**covered, "eod": False})


def test_live_cuts():
    from types import SimpleNamespace

    cands = pd.DataFrame(
        {
            "snr": [5.0, 8.0, 9.0, 12.0, 7.0],
            "dm": [50.0, 2.0, 60.0, 70.0, 900.0],
            "boxcar": [1, 2, 9, 3, 2],
            "members": [5, 5, 5, 1, 5],
        }
    )
    values = SimpleNamespace(
        snr_min=6.0, dm_min=5.0, max_dm=500.0, max_boxcar=8, min_members=2
    )
    assert _candmaker().live_cuts(cands, values).index.tolist() == []
    values.max_boxcar, values.min_members = None, 1
    assert _candmaker().live_cuts(cands, values).index.tolist() == [2, 3]


class FakeReader:
    """Pages of a ring, as `psrdada.Reader` iterates them."""

    def __init__(self, pages):
        self.pages = pages
        self.disconnected = False
        self.taken = 0

    def __iter__(self):
        for page in self.pages:
            self.taken += 1
            yield page

    def disconnect(self):
        self.disconnected = True


def test_run_reads_every_page_then_disconnects():
    history = DadaHistory(_header(), capacity=12)
    pages = [_filled(history, s, 3).tobytes() for s in range(0, 15, 3)]
    history.reader = FakeReader(pages)
    history.run()
    assert history.written == 15 and history.eod
    assert history.reader.disconnected


def test_run_holds_the_ring_for_a_queued_candidate():
    history = DadaHistory(_header(), capacity=9)
    pages = [_filled(history, s, 3).tobytes() for s in range(0, 18, 3)]
    history.reader = FakeReader(pages)
    # samples 1..4 have arrived by the second page and must stay readable
    history.pending["cand"] = (1, 4)
    thread = threading.Thread(target=history.run, kwargs={"poll": 0.001}, daemon=True)
    thread.start()
    try:
        time.sleep(0.2)
        # a third page would leave sample 6 + 3 + 3 - 9 = 3 the oldest
        # readable one, past sample 1: so the ring stops after two pages
        assert history.written == 6 and not history.eod
        out = np.empty((3, 4), dtype=history.dtype)
        history.copy_out(1, out)
        np.testing.assert_array_equal(out, _filled(history, 1, 3))
    finally:
        del history.pending["cand"]
    thread.join(5)
    assert history.written == 18 and history.eod


def test_run_does_not_hold_for_data_still_arriving():
    history = DadaHistory(_header(), capacity=9)
    pages = [_filled(history, s, 3).tobytes() for s in range(0, 18, 3)]
    history.reader = FakeReader(pages)
    # a range ending past the data never holds the ring
    history.pending["cand"] = (1, 100)
    history.run()
    assert history.written == 18


def test_run_takes_the_page_after_the_hold_timeout(caplog):
    history = DadaHistory(_header(), capacity=9)
    pages = [_filled(history, s, 3).tobytes() for s in range(0, 18, 3)]
    history.reader = FakeReader(pages)
    # a hold that is never released, as from a wrong guess at Heimdall's gulp
    history.hold_from = 0
    t = time.time()
    with caplog.at_level("WARNING"):
        history.run(poll=0.001, hold_timeout=0.05)
    assert history.written == 18 and time.time() - t < 5
    assert history.held_seconds > 0
    assert "may lose candidates" in caplog.text


@pytest.mark.parametrize(
    "error, status", [(Overwritten("gone"), "lost"), (RuntimeError("bad"), "failed")]
)
def test_live_candidate_that_fails_releases_gpu_memory(monkeypatch, error, status):
    candmaker = _candmaker()

    def fail(cand_val, files=None):
        raise error

    released = []
    monkeypatch.setattr(candmaker, "_cand2h5", fail)
    monkeypatch.setattr(candmaker, "release_memory", released.append)
    cand_val = ["dada", 9.0, 4, 70.0, -1, 1.5, "", 1, None, 0]
    result, reason, seconds = candmaker.live_cand2h5(cand_val)
    assert result == status and reason
    assert seconds >= 0
    assert released == [0]
    # on its worker's GPU, released there; nothing to release on the CPU
    monkeypatch.setattr(candmaker, "_worker_gpu", 3)
    candmaker.live_cand2h5(cand_val)
    monkeypatch.setattr(candmaker, "_worker_gpu", None)
    candmaker.live_cand2h5(cand_val[:9] + [-1])
    assert released == [0, 3]


def test_workers_take_the_gpus_in_turn():
    import multiprocessing

    candmaker = _candmaker()
    started = multiprocessing.Value("i", 0)
    gpus = []
    for _ in range(5):
        candmaker._init_cand_worker([2, 5], started)
        gpus.append(candmaker._worker_gpu)
    assert gpus == [2, 5, 2, 5, 2]
    cand_val = ["f.fil", 9.0, 4, 70.0, -1, 1.5, "", 1, None, 2]
    assert candmaker._on_worker_gpu(cand_val)[9] == 2
    assert candmaker._on_worker_gpu(cand_val)[:9] == cand_val[:9]

    candmaker = _candmaker()
    candmaker._init_cand_worker([-1], multiprocessing.Value("i", 0))
    assert candmaker._worker_gpu is None
    assert candmaker._on_worker_gpu(cand_val) is cand_val


def test_each_worker_uses_its_own_gpu(monkeypatch):
    """A worker sends every candidate to its GPU, whatever the list says."""
    candmaker = _candmaker()
    seen = []
    monkeypatch.setattr(
        candmaker, "_cand2h5", lambda cand_val, files=None: seen.append(cand_val[9])
    )
    monkeypatch.setattr(candmaker, "_worker_gpu", 1)
    candmaker.cached_cand2h5(["x.fil", 9.0, 4, 70.0, -1, 1.5, "", 1, None, 0])
    assert seen == [1]
