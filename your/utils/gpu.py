import functools
import logging
import math
import subprocess

import numpy as np

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _cupy():
    """
    CuPy, imported the first time a GPU path runs, so the package imports and
    the CPU paths run without it.

    Returns:
        module: cupy
    """
    try:
        import cupy
    except ImportError as error:
        raise ImportError(
            "The GPU paths need CuPy: pip install your[gpu], or the cupy "
            "package matching your CUDA version"
        ) from error
    return cupy


def available():
    """
    Whether CuPy is installed and sees at least one CUDA device.

    Returns:
        bool: True if the GPU paths can run
    """
    return device_count() > 0


def device_count():
    """
    CUDA devices CuPy sees: none without CuPy, a driver or a device.

    Returns:
        int: number of devices
    """
    try:
        return _cupy().cuda.runtime.getDeviceCount()
    except Exception:
        return 0


def check_devices(device_ids):
    """
    Check every GPU ID names a CUDA device, counting the devices in a child
    process: counting them here would start CUDA in this one, and workers
    forked from it afterwards could not use the GPU.

    Args:
        device_ids (list): GPU IDs, as CUDA numbers them (so within
            CUDA_VISIBLE_DEVICES)

    Raises:
        ValueError: an ID is not a device
    """
    import sys

    count = subprocess.run(
        [
            sys.executable,
            "-c",
            "import cupy; print(cupy.cuda.runtime.getDeviceCount())",
        ],
        capture_output=True,
        text=True,
    )
    try:
        why = ""
        count = int(count.stdout)
    except ValueError:
        why = f" ({(count.stderr.strip().splitlines() or ['no output'])[-1]})"
        count = 0
    missing = [i for i in device_ids if not 0 <= i < count]
    if missing:
        raise ValueError(
            f"No GPU {', '.join(map(str, missing))}: CuPy sees {count} "
            f"device(s){why}, numbered from 0 within CUDA_VISIBLE_DEVICES"
        )


def check_compiler():
    """
    Load NVRTC, which compiles the kernels, without making a CUDA context, so
    a process can check it before forking workers that use the GPU.

    Raises:
        RuntimeError: if NVRTC cannot be loaded
    """
    try:
        _cupy().cuda.nvrtc.getVersion()
    except Exception as error:
        raise RuntimeError(
            f"CuPy cannot load NVRTC to compile the GPU kernels ({error}). "
            "Install the NVRTC matching CuPy's CUDA (e.g. pip install "
            "nvidia-cuda-nvrtc-cu12) or use the CPU (-g -1)."
        ) from error


def use_device(device):
    """
    Make `device` this thread's current GPU. CuPy's `with Device(n)` blocks
    switch back to the previous device as they end, and since CUDA 12
    switching to a device makes a context there: a process working on GPU 1
    that never made it current would hold a context on GPU 0 as well.

    Args:
        device (int): GPU ID
    """
    _cupy().cuda.Device(device).use()


def release_memory(device=None):
    """
    Return the memory CuPy's pool keeps to the device, as after a candidate
    that failed part way. Does nothing if CuPy was never imported.

    Args:
        device (int): GPU whose pool to empty; the pool frees only the
            current device's memory, and the default is the current device
    """
    import sys

    if "cupy" not in sys.modules:
        return
    cupy = sys.modules["cupy"]
    try:
        if device is None:
            cupy.get_default_memory_pool().free_all_blocks()
        else:
            with cupy.cuda.Device(device):
                cupy.get_default_memory_pool().free_all_blocks()
    except Exception:
        # a device that has failed may refuse even this
        logger.debug(f"Could not release GPU {device}'s memory", exc_info=True)


def gpu_errors():
    """
    The exceptions a CUDA call raises when the device fails or runs out of
    memory, for callers that fall back to the CPU.

    Returns:
        tuple: exception classes
    """
    cupy = _cupy()
    return (
        cupy.cuda.runtime.CUDARuntimeError,
        cupy.cuda.driver.CUDADriverError,
        cupy.cuda.memory.OutOfMemoryError,
    )


# one thread per output cell throughout, each summing in a register. Sizes
# and offsets are 64-bit, as numba's were, so a chunk may pass 2**31 bytes.
# Modulo of a possibly negative value wraps as Python's does, which these
# kernels relied on under numba.
_SOURCE = r"""
typedef long long i64;

__device__ __forceinline__ i64 pymod(i64 x, i64 n) {
    i64 m = x % n;
    return m < 0 ? m + n : m;
}

// Rows `row0 .. row0 + rows` of a (nsamples, nchans) chunk into columns of
// the (nchans, nsamples) chunk, through a shared-memory tile so both the read
// and the write are coalesced. Launch with 32x8 threads.
extern "C" __global__ void transpose_u8(
    const unsigned char* src, i64 rows, i64 nchans,
    unsigned char* dst, i64 nsamples, i64 row0)
{
    __shared__ unsigned char tile[32][33];
    const i64 tx = threadIdx.x, ty = threadIdx.y;
    const i64 bx = (i64)blockIdx.x * 32, by = (i64)blockIdx.y * 32;
    for (int k = 0; k < 32; k += 8) {
        const i64 r = by + ty + k, c = bx + tx;
        if (r < rows && c < nchans) tile[ty + k][tx] = src[r * nchans + c];
    }
    __syncthreads();
    for (int k = 0; k < 32; k += 8) {
        const i64 r = bx + ty + k, c = by + tx;
        if (r < nchans && c < rows) dst[r * nsamples + row0 + c] = tile[tx][ty + k];
    }
}

// Dedispersed frequency-time plane, decimated and cropped in one pass:
// `fdf` channels by `tdf` samples per kept output cell. `delays` are wrapped
// into [0, nsamples); `col0` is the first decimated column kept, wrapping
// when negative.
template <typename T, typename A>
__global__ void dedisp_window(
    const T* data, i64 nchans, i64 nsamples, const int* delays, i64 fdf,
    i64 tdf, i64 col0, float* out, i64 rows, i64 width)
{
    const i64 col = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    const i64 row = (i64)blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows || col >= width) return;
    const i64 base = pymod(col0 + col, nsamples / tdf) * tdf;
    A acc = 0;
    for (i64 ch = row * fdf; ch < row * fdf + fdf; ch++) {
        const T* channel = data + ch * nsamples;
        const i64 d = delays[ch];
        for (i64 s = base; s < base + tdf; s++) {
            i64 idx = s + d;
            if (idx >= nsamples) idx -= nsamples;
            acc += channel[idx];
        }
    }
    out[row * width + col] = (float)acc;
}

// DM-time plane, decimated and cropped in one pass: every channel over `tdf`
// samples per kept (DM, column) cell. Threads of a warp share a column across
// neighbouring DMs, whose delays barely differ, so their reads land on the
// same few cache lines. `delays` is (nchans, ndm).
template <typename T, typename A>
__global__ void dmt_window(
    const T* data, i64 nchans, i64 nsamples, const int* delays, i64 ndm,
    i64 tdf, i64 col0, float* out, i64 width)
{
    const i64 kk = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    const i64 col = (i64)blockIdx.y * blockDim.y + threadIdx.y;
    if (kk >= ndm || col >= width) return;
    const i64 base = pymod(col0 + col, nsamples / tdf) * tdf;
    A acc = 0;
    for (i64 ch = 0; ch < nchans; ch++) {
        const T* channel = data + ch * nsamples;
        const i64 d = delays[ch * ndm + kk];
        for (i64 s = base; s < base + tdf; s++) {
            i64 idx = s + d;
            if (idx >= nsamples) idx -= nsamples;
            acc += channel[idx];
        }
    }
    out[kk * width + col] = (float)acc;
}

#define BLOCK 32

// Sum of each channel over each block of BLOCK samples, written one column
// right so `block_prefix` can turn each row into exclusive prefix sums.
// `csum` is (nchans, nblocks + 1).
template <typename T, typename A>
__global__ void block_sums(const T* data, i64 nchans, i64 nsamples, A* csum, i64 stride)
{
    const i64 b = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    const i64 ch = (i64)blockIdx.y * blockDim.y + threadIdx.y;
    if (ch >= nchans || b >= stride - 1) return;
    const T* channel = data + ch * nsamples;
    A acc = 0;
    for (i64 s = b * BLOCK; s < b * BLOCK + BLOCK; s++) acc += channel[s];
    csum[ch * stride + b + 1] = acc;
}

// Running sum along each channel's blocks, so `csum[ch, b]` is the sum of the
// channel's first `b * BLOCK` samples.
template <typename A>
__global__ void block_prefix(A* csum, i64 nchans, i64 stride)
{
    const i64 ch = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= nchans) return;
    A* row = csum + ch * stride;
    A acc = 0;
    row[0] = acc;
    for (i64 b = 1; b < stride; b++) {
        acc += row[b];
        row[b] = acc;
    }
}

// Add samples `lo .. hi` of one channel, 0 <= lo <= hi <= nsamples: the
// blocks wholly inside the span as one difference of the prefix sums, and the
// at most 2 * BLOCK - 2 samples either side of them one by one.
template <typename T, typename C>
__device__ __forceinline__ i64 span_sum(
    const T* channel, const C* csum, i64 lo, i64 hi, i64 acc)
{
    const i64 bl = (lo + BLOCK - 1) / BLOCK, bh = hi / BLOCK;
    if (bl < bh) {
        for (i64 s = lo; s < bl * BLOCK; s++) acc += channel[s];
        acc += csum[bh] - csum[bl];
        for (i64 s = bh * BLOCK; s < hi; s++) acc += channel[s];
    } else {
        for (i64 s = lo; s < hi; s++) acc += channel[s];
    }
    return acc;
}

// Add `tdf` samples of one channel from `start`, 0 <= start < 2 * nsamples,
// wrapping at the end of the chunk as the per-sample kernels do.
template <typename T, typename C>
__device__ __forceinline__ i64 wrapped_span_sum(
    const T* channel, const C* csum, i64 nsamples, i64 start, i64 tdf, i64 acc)
{
    if (start >= nsamples) start -= nsamples;
    const i64 stop = start + tdf;
    if (stop <= nsamples) return span_sum(channel, csum, start, stop, acc);
    acc = span_sum(channel, csum, start, nsamples, acc);
    return span_sum(channel, csum, (i64)0, stop - nsamples, acc);
}

// `dedisp_window`, adding each channel's `tdf` samples through the block
// prefix sums `csum` of type C: the same integer sums in about 2 * BLOCK
// reads, not `tdf`. Integer data only, summed in int64.
template <typename T, typename C>
__global__ void dedisp_window_blocks(
    const T* data, const C* csum, i64 stride, i64 nchans, i64 nsamples,
    const int* delays, i64 fdf, i64 tdf, i64 col0, float* out, i64 rows, i64 width)
{
    const i64 col = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    const i64 row = (i64)blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows || col >= width) return;
    const i64 base = pymod(col0 + col, nsamples / tdf) * tdf;
    i64 acc = 0;
    for (i64 ch = row * fdf; ch < row * fdf + fdf; ch++) {
        acc = wrapped_span_sum(
            data + ch * nsamples, csum + ch * stride, nsamples,
            base + delays[ch], tdf, acc);
    }
    out[row * width + col] = (float)acc;
}

// `dmt_window`, adding each channel's `tdf` samples through the block prefix
// sums `csum` of type C: the same integer sums in about 2 * BLOCK reads, not
// `tdf`. Integer data only, summed in int64.
template <typename T, typename C>
__global__ void dmt_window_blocks(
    const T* data, const C* csum, i64 stride, i64 nchans, i64 nsamples,
    const int* delays, i64 ndm, i64 tdf, i64 col0, float* out, i64 width)
{
    const i64 kk = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    const i64 col = (i64)blockIdx.y * blockDim.y + threadIdx.y;
    if (kk >= ndm || col >= width) return;
    const i64 base = pymod(col0 + col, nsamples / tdf) * tdf;
    i64 acc = 0;
    for (i64 ch = 0; ch < nchans; ch++) {
        acc = wrapped_span_sum(
            data + ch * nsamples, csum + ch * stride, nsamples,
            base + delays[ch * ndm + kk], tdf, acc);
    }
    out[kk * width + col] = (float)acc;
}

// Each channel of a (nchans, nsamples) chunk rolled back by its delay,
// wrapped into [0, nsamples): `out[ch, s] = data[ch, (s + delay) % nsamples]`.
template <typename T>
__global__ void roll_channels(
    const T* data, i64 nchans, i64 nsamples, const i64* delays, T* out)
{
    const i64 s = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    const i64 ch = (i64)blockIdx.y * blockDim.y + threadIdx.y;
    if (ch >= nchans || s >= nsamples) return;
    i64 idx = s + delays[ch];
    if (idx >= nsamples) idx -= nsamples;
    out[ch * nsamples + s] = data[ch * nsamples + idx];
}

// One thread per (DM, sample) of the full DM-time plane, summing the channels
// in a register. `delays` is (ndm, nchans) and may be negative.
template <typename T, typename A>
__global__ void dmt_channels(
    const T* data, i64 nchans, i64 nsamples, const int* delays, i64 ndm, float* out)
{
    const i64 jj = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    const i64 kk = (i64)blockIdx.y * blockDim.y + threadIdx.y;
    if (jj >= nsamples || kk >= ndm) return;
    A acc = 0;
    for (i64 ii = 0; ii < nchans; ii++) {
        acc += data[ii * nsamples + pymod(jj + delays[kk * nchans + ii], nsamples)];
    }
    out[kk * nsamples + jj] = (float)acc;
}

// Cumulative sum down the channel axis, one thread per sample, with a leading
// zero row already in place, so any span of channels sums as one subtraction.
template <typename T, typename A>
__global__ void band_cumsum(const T* data, i64 nchans, i64 nsamples, A* cumsum)
{
    const i64 jj = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (jj >= nsamples) return;
    A acc = 0;
    for (i64 ii = 0; ii < nchans; ii++) {
        acc += data[ii * nsamples + jj];
        cumsum[(ii + 1) * nsamples + jj] = acc;
    }
}

// As `dmt_channels`, but each span of channels sharing a delay bin is one
// subtraction of the band cumulative sum. The span tables are (ndm, width).
template <typename A>
__global__ void dmt_runs(
    const A* cumsum, i64 nsamples, const int* starts, const int* stops,
    const int* delays, const int* nruns, i64 width, i64 ndm, float* out)
{
    const i64 jj = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    const i64 kk = (i64)blockIdx.y * blockDim.y + threadIdx.y;
    if (jj >= nsamples || kk >= ndm) return;
    i64 acc = 0;
    for (i64 rr = 0; rr < nruns[kk]; rr++) {
        const i64 idx = pymod(jj + delays[kk * width + rr], nsamples);
        acc += cumsum[(i64)stops[kk * width + rr] * nsamples + idx]
             - cumsum[(i64)starts[kk * width + rr] * nsamples + idx];
    }
    out[kk * nsamples + jj] = (float)acc;
}
"""

_C_TYPES = {
    np.dtype(np.uint8): "unsigned char",
    np.dtype(np.int8): "signed char",
    np.dtype(np.uint16): "unsigned short",
    np.dtype(np.int16): "short",
    np.dtype(np.uint32): "unsigned int",
    np.dtype(np.int32): "int",
    np.dtype(np.int64): "long long",
    np.dtype(np.float32): "float",
    np.dtype(np.float64): "double",
}

# the templates each (data, accumulator) pair is compiled for
_TEMPLATES = (
    "dedisp_window<{t}, {a}>",
    "dmt_window<{t}, {a}>",
    "block_sums<{t}, {a}>",
    "block_prefix<{a}>",
    "dedisp_window_blocks<{t}, {a}>",
    "dmt_window_blocks<{t}, {a}>",
    "roll_channels<{t}>",
    "dmt_channels<{t}, {a}>",
    "band_cumsum<{t}, {a}>",
    "dmt_runs<{a}>",
)


def _c_type(dtype):
    try:
        return _C_TYPES[np.dtype(dtype)]
    except KeyError:
        raise TypeError(f"no GPU kernels for {np.dtype(dtype)} data") from None


@functools.lru_cache(maxsize=None)
def _module(data_dtype, acc_dtype):
    """
    The kernels for one pair of data and accumulator dtypes. CuPy compiles
    them on first use and keeps the binary in its on-disk cache, so later
    processes load it instead of compiling.
    """
    t, a = _c_type(data_dtype), _c_type(acc_dtype)
    names = tuple(name.format(t=t, a=a) for name in _TEMPLATES)
    module = _cupy().RawModule(
        code=_SOURCE, options=("-std=c++14",), name_expressions=names
    )
    return module, t, a


def _kernel(name, data_dtype, acc_dtype=np.int64):
    """
    A templated kernel, as `name` instantiated for these dtypes.

    Args:
        name (str): one of the templates in `_TEMPLATES`, without arguments
        data_dtype: dtype of the data it reads
        acc_dtype: dtype it sums in

    Returns:
        cupy.RawKernel: the kernel
    """
    module, t, a = _module(np.dtype(data_dtype), np.dtype(acc_dtype))
    template = next(x for x in _TEMPLATES if x.startswith(name + "<"))
    return module.get_function(template.format(t=t, a=a))


@functools.lru_cache(maxsize=1)
def _transpose_u8():
    return (
        _cupy()
        .RawModule(code=_SOURCE, options=("-std=c++14",))
        .get_function("transpose_u8")
    )


def _i64(*values):
    return tuple(np.int64(v) for v in values)


def _wide(dtype):
    """float64 for float data, int64 otherwise: what numba summed in."""
    return np.float64 if np.issubdtype(dtype, np.floating) else np.int64


def ft_delays(chan_freqs, dm, tsamp):
    """
    Per-channel shift at the candidate's DM, rounded as the atomic
    dedispersion kernels rounded it in-line: float32 frequencies squared in
    float32, the rest in float64, rounded half to even.

    Args:
        chan_freqs (numpy.ndarray): channel frequencies, MHz
        dm (float): dispersion measure
        tsamp (float): sampling time, seconds

    Returns:
        numpy.ndarray: (nchans,) int64 shift in samples, unwrapped
    """
    f = np.asarray(chan_freqs, dtype=np.float32)
    squares = (f * f).astype(np.float64)
    shift = -4148808.0 * float(dm) * (1 / squares[0] - 1 / squares) / 1000
    return np.round(shift / float(tsamp)).astype(np.int64)


def gpu_dedisperse(cand, device=0):
    """

    GPU dedispersion (by rolling the array)

    Args:
        cand: Candidate instance
        device (int): GPU ID

    Returns:
        candidate object

    """
    cupy = _cupy()
    data = np.ascontiguousarray(np.asarray(cand.data).T, dtype=cand.your_header.dtype)
    nchans, nsamples = data.shape
    delays = ft_delays(cand.chan_freqs, cand.dm, cand.your_header.tsamp) % nsamples
    with cupy.cuda.Device(device):
        data_in = cupy.asarray(data)
        out = cupy.empty_like(data_in)
        _kernel("roll_channels", data.dtype)(
            (math.ceil(nsamples / 32), math.ceil(nchans / 32)),
            (32, 32),
            (data_in, *_i64(nchans, nsamples), cupy.asarray(delays), out),
        )
        cand.dedispersed = out.get().T
        del data_in, out
        cupy.get_default_memory_pool().free_all_blocks()

    return cand


def delay_table(chan_freqs, tsamp, dms):
    """
    Sample shift per DM and channel. float64, so it rounds the way the CPU path
    does; the delays used to be worked out in the kernel in float32, where a few
    channels per plane land in a neighbouring bin.

    Args:
        chan_freqs (numpy.ndarray): channel frequencies, MHz
        tsamp (float): sampling time, seconds
        dms (numpy.ndarray): DMs to make the plane over

    Returns:
        numpy.ndarray: (ndm, nchans) shift in samples

    """
    delay = (
        4148808.0
        * dms[:, None]
        * (1 / chan_freqs[0] ** 2 - 1 / chan_freqs[None, :] ** 2)
        / 1000
        / tsamp
    )
    return -np.round(delay).astype(np.int32)


def run_edges(delays):
    """
    Where each span of equal delay starts, and how many spans each DM has.

    Args:
        delays (numpy.ndarray): (ndm, nchans) shift in samples

    Returns:
        tuple: (ndm, nchans) bool of span starts, (ndm,) span counts

    """
    edge = np.empty(delays.shape, dtype=bool)
    edge[:, 0] = True
    np.not_equal(delays[:, 1:], delays[:, :-1], out=edge[:, 1:])
    return edge, edge.sum(axis=1).astype(np.int32)


def run_table(delays, edge, nruns):
    """
    Lay the spans out per DM, padded to the DM with the most of them.

    Args:
        delays (numpy.ndarray): (ndm, nchans) shift in samples
        edge (numpy.ndarray): (ndm, nchans) bool of span starts
        nruns (numpy.ndarray): (ndm,) span counts

    Returns:
        tuple: starts, stops, span delays, each (ndm, max spans)

    """
    ndm, nchans = delays.shape
    width = int(nruns.max())
    kk, ii = np.nonzero(edge)
    rr = np.arange(kk.size) - np.repeat(np.cumsum(nruns) - nruns, nruns)

    starts = np.zeros((ndm, width), dtype=np.int32)
    run_delays = np.zeros((ndm, width), dtype=np.int32)
    starts[kk, rr] = ii
    run_delays[kk, rr] = delays[kk, ii]

    stops = np.zeros((ndm, width), dtype=np.int32)
    stops[:, :-1] = starts[:, 1:]
    stops[np.arange(ndm), nruns - 1] = nchans
    return starts, stops, run_delays


def gpu_dmt(cand, device=0, dmsteps=256, max_run_fraction=0.6):
    """

    GPU DM-Time bow-tie

    Two kernels, identical output. `dmt_runs` sums each span of channels sharing
    a delay bin as one subtraction of a band cumulative sum, so it wins while the
    band still collapses; once more than `max_run_fraction` of the channels earn
    their own bin it reads four bytes per span where `dmt_channels` reads one
    byte per channel, and loses. The default crossover was measured on a T4.

    Args:
        cand: Candidate instance
        device (int): GPU ID
        dmsteps (int): rows in the DM-time plane
        max_run_fraction (float): spans per channel above which to take
            `dmt_channels`

    Returns:
        candidate object

    """
    cupy = _cupy()
    nsamples, nchans = cand.data.shape
    tsamp = float(cand.your_header.tsamp)
    chan_freqs = np.asarray(cand.chan_freqs, dtype=np.float64)
    # the same expression the CPU path uses, so the DM axes are bit identical
    dms = cand.dm + np.linspace(-cand.dm, cand.dm, dmsteps)

    delays = delay_table(chan_freqs, tsamp, dms)
    # counting the spans is cheap; only lay them out if we are going to use them
    edge, nruns = run_edges(delays)
    fraction = float(nruns.mean()) / nchans

    data = np.ascontiguousarray(cand.data.T)
    threads = 128
    blocks = math.ceil(nsamples / threads)

    with cupy.cuda.Device(device):
        cand_data_in = cupy.asarray(data)
        dmt_return = cupy.empty((dmsteps, nsamples), dtype=np.float32)
        if fraction <= max_run_fraction:
            logger.debug(f"{fraction:.2f} spans per channel, summing runs")
            # int32 holds the whole band for any real filterbank; widen if it cannot
            info = np.iinfo(cand.data.dtype)
            bound = nchans * max(abs(int(info.min)), int(info.max))
            acc_dtype = np.int32 if bound <= np.iinfo(np.int32).max else np.int64
            starts, stops, run_delays = run_table(delays, edge, nruns)
            cumsum = cupy.empty((nchans + 1, nsamples), dtype=acc_dtype)
            cumsum[0] = 0
            _kernel("band_cumsum", data.dtype, acc_dtype)(
                (blocks,), (threads,), (cand_data_in, *_i64(nchans, nsamples), cumsum)
            )
            _kernel("dmt_runs", data.dtype, acc_dtype)(
                (blocks, dmsteps),
                (threads, 1),
                (
                    cumsum,
                    np.int64(nsamples),
                    cupy.asarray(starts),
                    cupy.asarray(stops),
                    cupy.asarray(run_delays),
                    cupy.asarray(nruns),
                    *_i64(starts.shape[1], dmsteps),
                    dmt_return,
                ),
            )
        else:
            logger.debug(f"{fraction:.2f} spans per channel, summing channels")
            _kernel("dmt_channels", data.dtype, _wide(data.dtype))(
                (blocks, dmsteps),
                (threads, 1),
                (
                    cand_data_in,
                    *_i64(nchans, nsamples),
                    cupy.asarray(np.ascontiguousarray(delays)),
                    np.int64(dmsteps),
                    dmt_return,
                ),
            )

        cand.dmt = dmt_return.get()
        del cand_data_in, dmt_return
        cupy.get_default_memory_pool().free_all_blocks()

    return cand


def portable_pinned_array(size):
    """
    A uint8 host array page-locked for every CUDA context in the process, not
    only the current one, so a worker can upload from it to any GPU.

    Args:
        size (int): bytes

    Returns:
        numpy.ndarray: page-locked uint8 array
    """
    cupy = _cupy()
    memory = cupy.cuda.PinnedMemory(size, cupy.cuda.runtime.hostAllocPortable)
    # the array's base holds the pointer, which holds the memory
    return np.frombuffer(cupy.cuda.PinnedMemoryPointer(memory, 0), np.uint8, size)


class PinnedReadBuffer:
    """
    One page-locked host buffer, reused for every chunk read in this process
    and grown when a chunk is larger than any before it. Reading a chunk into
    it lets the upload to the GPU go straight over the bus, where from
    pageable memory the driver first copies each slice into its own staging
    buffer. It is locked for every GPU, so a worker whose candidates go to
    different GPUs keeps the one buffer.

    Each read overwrites the last, so a chunk read into it is only good until
    the next read: meant for a worker that makes one candidate at a time.

    Args:
        device (int): GPU whose context allocates the buffer
        granule (int): allocation size is rounded up to a multiple of this
    """

    def __init__(self, device=0, granule=64 * 2**20):
        self.device = device
        self.granule = granule
        self.buffer = None

    def __call__(self, nbytes):
        if self.buffer is None or self.buffer.size < nbytes:
            self.buffer = None
            size = -(-nbytes // self.granule) * self.granule
            try:
                with _cupy().cuda.Device(self.device):
                    self.buffer = portable_pinned_array(size)
            except gpu_errors():
                # more than the host will lock: read this chunk into ordinary
                # memory, as without a buffer, and try again next time
                logger.warning(f"Could not pin {size} bytes, reading unpinned")
                return np.empty(nbytes, dtype=np.uint8)
        return self.buffer


def to_device_channel_major(data, stream, staging_bytes=128 * 2**20):
    """
    Put a (nsamples, nchans) uint8 chunk on the device as (nchans, nsamples),
    transposing there a slice at a time. The device holds the chunk once plus
    one `staging_bytes` slice, where a whole-chunk transpose would hold it
    twice: four candmaker workers on 1 GB chunks then fit in 8 GB.

    Args:
        data: (nsamples, nchans) uint8 chunk
        stream: cupy.cuda.Stream to copy and transpose on, current
        staging_bytes (int): size of the reused slice buffer

    Returns:
        cupy.ndarray: (nchans, nsamples) chunk
    """
    cupy = _cupy()
    nsamples, nchans = data.shape
    out = cupy.empty((nchans, nsamples), dtype=np.uint8)
    step = max(32, staging_bytes // nchans // 32 * 32)
    staging = cupy.empty((min(step, nsamples), nchans), dtype=np.uint8)
    for row0 in range(0, nsamples, step):
        rows = min(step, nsamples - row0)
        part = staging[:rows]
        part.set(np.ascontiguousarray(data[row0 : row0 + rows]), stream=stream)
        _transpose_u8()(
            (math.ceil(nchans / 32), math.ceil(rows / 32)),
            (32, 8),
            (part, *_i64(rows, nchans), out, *_i64(nsamples, row0)),
        )
    # stream order already keeps each slice's copy behind the previous
    # transpose; wait here so the staging buffer is free to release
    stream.synchronize()
    return out


def dmt_delays(cand, dms):
    """
    Sample shift per channel for each DM of the bow-tie, rounded exactly as
    before (float64 frequencies, numpy rounding).

    Args:
        cand: Candidate instance
        dms (numpy.ndarray): DMs, float64

    Returns:
        numpy.ndarray: (nchans, ndm) int64 shift in samples, unwrapped
    """
    freqs = np.asarray(cand.chan_freqs)
    return np.round(
        -1
        * 4148808.0
        * dms[None, :]
        * (1 / (freqs[0]) ** 2 - 1 / (freqs[:, None]) ** 2)
        / 1000
        / cand.your_header.tsamp
    ).astype(np.int64)


# samples per block of the block prefix sums; BLOCK in the kernels
BLOCK = 32


def accumulator(dtype, count):
    """
    Narrowest exact accumulator for `count` samples of `dtype`: int32 while the
    sum cannot overflow it, int64 past that, float64 for float data.

    Args:
        dtype: data dtype
        count (int): samples summed into one output cell

    Returns:
        numpy.dtype: the accumulator dtype
    """
    if not np.issubdtype(dtype, np.integer):
        return np.dtype(np.float64)
    info = np.iinfo(dtype)
    bound = count * max(abs(int(info.min)), int(info.max))
    return np.dtype(np.int32 if bound <= np.iinfo(np.int32).max else np.int64)


def gpu_dedisp_and_dmt_crop(cand, device=0, width=256, block_from=4 * BLOCK):
    """

    GPU dedispersion and DM-time bow-tie, decimated to 256 channels and by
    half the pulse width in time, and cropped to the central `width` columns.

    Each output cell is summed by one thread, so there are no atomic adds, and
    only the kept window is computed. The result is the same as decimating the
    whole chunk and then cropping it, as this function used to.

    From `block_from` samples per column, integer data is summed through
    per-channel prefix sums over blocks of `BLOCK` samples, so a cell costs
    about `2 * BLOCK` reads per channel rather than one per sample. Integer
    sums either way, so the planes are identical.

    Args:
        cand: Candidate instance
        device (int): GPU ID
        width (int): columns kept around the centre
        block_from (int): samples per column from which to use the block
            prefix sums

    Returns:
        candidate object

    """
    if cand.width < 3:
        time_decimation_factor = 1
    else:
        time_decimation_factor = cand.width // 2

    nsamples, nchans = cand.data.shape
    if nchans < 256:
        raise IndexError("GPU candmaker will not work if nchans < 256.")

    frequency_decimation_factor = nchans // 256
    fdf, tdf = int(frequency_decimation_factor), int(time_decimation_factor)
    rows = nchans // fdf
    col0 = nsamples // tdf // 2 - width // 2

    logger.debug(f"Freq decimation factor: {fdf}")
    logger.debug(f"Time decimation factor: {tdf}")

    # the chunk arrives (nsamples, nchans) in whatever dtype get_chunk left it
    # (float64 where it padded with the median); cast as before
    data = np.asarray(cand.data).astype(cand.your_header.dtype, copy=False)
    # delays on the host, wrapped so the kernels need one compare, not a
    # 64-bit modulo
    ft_host = ft_delays(cand.chan_freqs, cand.dm, cand.your_header.tsamp)
    ft_host = (ft_host % nsamples).astype(np.int32)
    dmt_host = dmt_delays(cand, np.linspace(0, 2 * cand.dm, 256))
    dmt_host = np.ascontiguousarray(dmt_host % nsamples).astype(np.int32)
    ft_acc = accumulator(data.dtype, fdf * tdf)
    dmt_acc = accumulator(data.dtype, nchans * tdf)
    nblocks = nsamples // BLOCK
    blocks = tdf >= block_from and nblocks and np.issubdtype(data.dtype, np.integer)

    cupy = _cupy()
    with cupy.cuda.Device(device):
        stream = cupy.cuda.Stream(non_blocking=True)
        with stream:
            # channel major, transposed on the device where that is quick
            if data.dtype == np.uint8:
                cand_data_in = to_device_channel_major(data, stream)
            else:
                cand_data_in = cupy.empty((nchans, nsamples), dtype=data.dtype)
                cand_data_in.set(np.ascontiguousarray(data.T), stream=stream)
            ft_delay_table = cupy.asarray(ft_host)
            dmt_delay_table = cupy.asarray(dmt_host)
            ft_out = cupy.empty((rows, width), dtype=np.float32)
            dmt_out = cupy.empty((256, width), dtype=np.float32)

            logger.debug("Allocated arrays on the GPU")

            ft_grid = (math.ceil(width / 32), math.ceil(rows / 8)), (32, 8)
            dmt_grid = (math.ceil(256 / 32), math.ceil(width / 4)), (32, 4)
            shape = _i64(nchans, nsamples)
            if blocks:
                logger.debug(f"Summing blocks of {BLOCK} samples")
                # int32 prefix sums while a whole channel cannot overflow them
                csum_acc = accumulator(data.dtype, nsamples)
                csum = cupy.empty((nchans, nblocks + 1), dtype=csum_acc)
                stride = np.int64(nblocks + 1)
                _kernel("block_sums", data.dtype, csum_acc)(
                    (math.ceil(nblocks / 32), math.ceil(nchans / 4)),
                    (32, 4),
                    (cand_data_in, *shape, csum, stride),
                )
                _kernel("block_prefix", data.dtype, csum_acc)(
                    (math.ceil(nchans / 128),),
                    (128,),
                    (csum, np.int64(nchans), stride),
                )
                _kernel("dedisp_window_blocks", data.dtype, csum_acc)(
                    *ft_grid,
                    (
                        cand_data_in,
                        csum,
                        stride,
                        *shape,
                        ft_delay_table,
                        *_i64(fdf, tdf, col0),
                        ft_out,
                        *_i64(rows, width),
                    ),
                )
                _kernel("dmt_window_blocks", data.dtype, csum_acc)(
                    *dmt_grid,
                    (
                        cand_data_in,
                        csum,
                        stride,
                        *shape,
                        dmt_delay_table,
                        *_i64(256, tdf, col0),
                        dmt_out,
                        np.int64(width),
                    ),
                )
            else:
                csum = None
                _kernel("dedisp_window", data.dtype, ft_acc)(
                    *ft_grid,
                    (
                        cand_data_in,
                        *shape,
                        ft_delay_table,
                        *_i64(fdf, tdf, col0),
                        ft_out,
                        *_i64(rows, width),
                    ),
                )
                _kernel("dmt_window", data.dtype, dmt_acc)(
                    *dmt_grid,
                    (
                        cand_data_in,
                        *shape,
                        dmt_delay_table,
                        *_i64(256, tdf, col0),
                        dmt_out,
                        np.int64(width),
                    ),
                )
            cand.dedispersed = ft_out.get(stream=stream).T
            cand.dmt = dmt_out.get(stream=stream)
            stream.synchronize()

        logger.debug("cand.dedispersed and cand.dmt set!")

        # CuPy's pool would keep the chunk's memory for this process, and
        # candmaker's workers share the card, so two chunks per worker could
        # run it out of memory. Return it to the driver now.
        del (
            cand_data_in,
            csum,
            ft_delay_table,
            dmt_delay_table,
            ft_out,
            dmt_out,
        )
        cupy.get_default_memory_pool().free_all_blocks()
    return cand


def get_gpu_memory_map(gpu_id):
    """
    Get the current gpu free memory

    Args:
        gpu_id (int): GPU id, as CUDA numbers them (nvidia-smi's numbers can
            differ, and ignore CUDA_VISIBLE_DEVICES)

    Returns:
        int: free GPU RAM in MiB
    """
    cupy = _cupy()
    with cupy.cuda.Device(gpu_id):
        free, _ = cupy.cuda.runtime.memGetInfo()
    return free // 2**20
