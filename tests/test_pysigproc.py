import os

import numpy as np
import pytest

from your.formats.pysigproc import SigprocFile

_install_dir = os.path.abspath(os.path.dirname(__file__))


def test_pysigproc_obj():
    fil_file = os.path.join(_install_dir, "data/28.fil")
    fil_obj = SigprocFile(fil_file)
    assert fil_obj.nchans == 336


def test_get_data_fil():
    fil_file = os.path.join(_install_dir, "data/28.fil")
    fil_obj = SigprocFile(fil_file)
    data = fil_obj.get_data(0, 10)
    assert np.isclose(np.mean(data), 128, atol=1)


def test_pol():
    fil_file = os.path.join(_install_dir, "data/28.fil")
    fil_obj = SigprocFile(fil_file)
    assert fil_obj.poln_order == "I"
    with pytest.raises(AssertionError):
        d = fil_obj.get_data(0, 10, npoln=3)


def test_long_string_value_keeps_header_size(tmp_path):
    fil_obj = SigprocFile(os.path.join(_install_dir, "data/28.fil"))
    data = fil_obj.get_data(0, 10)
    fil_obj.source_name = fil_obj.source_name.decode()
    fil_obj.rawdatafile = "/" + "d" * 120 + ".fil"
    out = tmp_path / "long.fil"
    with open(out, "wb") as f:
        fil_obj.filterbank_header(fout=f)
        hdrbytes = f.tell()
        data.tofile(f)

    read = SigprocFile(str(out))
    assert read.hdrbytes == hdrbytes
    assert read.rawdatafile == fil_obj.rawdatafile.encode()
    assert read.nchans == 336
    np.testing.assert_array_equal(read.get_data(0, 10), data)
