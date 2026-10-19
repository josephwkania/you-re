# Replacing numba with CuPy: our branch vs `devanshkv/cupy-candidates`

Both branches remove numba from the candmaker and start from the same commit,
`fb863f8` ("build(rust)!: require native extension").

- **Ours:** `cupy-gpu` (`58efd67`), carried on in `dada-candmaker` (`4c9d910`,
  the version tested). It is a port: CuPy kernels without atomic adds, Rust CPU
  kernels and a reused page-locked read buffer. It changes about 3,300 lines,
  and its outputs match the numba version exactly.
- **Theirs:** <https://github.com/devanshkv/you-re/tree/cupy-candidates>
  (`4db0d60`, 7 commits, 17 Oct 2026). It is a rewrite of candidate processing
  in about 10,000 lines, half of them tests and benchmarks. It adds a planner,
  a GPU scheduler with a resident input cache, tiled processing for
  candidates too large for GPU memory, and atomic h5 writes. Its outputs are
  new, and the commit marks this as a `BREAKING CHANGE`.

**Update, 18 Oct 2026:** theirs moved on to `3349081`, which ports our CPU
methods and moves their resize to the GPU. The CPU gap is gone, and ours is
still about 3× faster on the GPU. See "Retimed: their `3349081`" first. The
rest of this document describes `4db0d60`.

## Retimed: their `3349081`

Their two new commits are:

- `6be5d48` "perf(candidates)!: speed CPU and CuPy paths";
- `3349081` "perf(gpu): cut single-GPU scheduling overhead".

Their `docs/cupy-performance.md` ("Lessons from main") says what they took from
ours:

- running sums, so the CPU cost no longer grows with pulse width;
- uint8 input without the float64 copy;
- 128-channel bands with tiled byte transposes;
- residue tables;
- releasing the GIL during native processing.

They also do their anti-aliased frequency resize on the GPU now, instead of
our channel averaging, and they keep one process per GPU. The distribution
is renamed `you-re`; Python still imports `your`.

The setup, data and candidates are unchanged. This was Slurm job 1400, with
the A4000 to itself and nothing else on it during any run. Ours was rerun
alongside as a control.

### Whole runs

All runs use the 380 candidates unless noted.

| run | ours | theirs `3349081` | theirs `4db0d60` |
|---|---|---|---|
| GPU, `-n 11` (their default 2 in flight) | 12.0, 11.3 s | 43.6, 34.5 s | 63.4, 64.2 s |
| GPU, FP16 output (their default) | — | 34.8, 35.7 s | 64.5, 64.1 s |
| CPU, `-n 11` | 8.7, 8.7 s | 9.4, 9.6 s | 528.7, 281.6 s |
| CPU, one process, 66-candidate subset | 14.9 s | 12.9 s | 799.2 s |
| CPU, one core, 66-candidate subset | 18.2 s | 18.2 s | — |
| GPU, single stream (ours `-n 1`, theirs 1 in flight) | 33.5 s | **27.2 s** | 72.1 s |
| GPU, theirs 2 / 8 in flight | — | 36.0 / 38.4 s | 63–64 / 61.0 s |

### Per candidate, by pulse width

| width (samples) | ours, CPU, 1 core | theirs, CPU, 1 core | ours, GPU, 1 worker | theirs, GPU, 1 in flight | theirs, GPU dedispersion |
|---|---|---|---|---|---|
| 4 | 0.170 s | 0.200 s | 0.073 s | 0.069 s | 0.014 s |
| 8 | 0.195 s | 0.205 s | 0.075 s | 0.078 s | 0.020 s |
| 16 | 0.224 s | 0.229 s | 0.078 s | 0.102 s | 0.038 s |
| 32 | 0.277 s | 0.260 s | 0.089 s | 0.153 s | 0.070 s |
| 64 | 0.362 s | 0.336 s | 0.103 s | 0.221 s | 0.134 s |
| 128 | 0.594 s | 0.474 s | 0.138 s | 0.406 s | 0.267 s |

### What changed

- **CPU: level.** Per core, their CPU path now matches ours: about 0.2 s per
  candidate, and a little faster than ours at width 64 and above. Across a
  whole run with `-n 11` it is within 10% of ours (9.5 s against 8.7 s).
- **GPU, one candidate at a time: theirs is now faster.** A single stream runs
  27.2 s against our 33.5 s, mainly because the resize moved to the GPU. Their
  CPU-side finishing fell from 0.069 s to about 0.006 s per candidate.
- **GPU dedispersion still grows with width on theirs.** It doubles with each
  doubling of pulse width, from 0.014 s to 0.267 s. They put running sums in
  the CPU path but not the GPU kernels. Ours stays nearly flat, from 0.073 s to
  0.138 s for the whole candidate.
- **GPU, whole runs: ours is still about 3× faster** (11–12 s against 35–44 s).
  Theirs still runs everything in one process per GPU:
  - Saving adds up to 0.038 s per candidate with gzip level 9, but it runs in
    a writer thread alongside the next candidate. Turning it off saves only
    4.3 s of the 27.7 s single-stream run (see "Why theirs is slower on the
    GPU").
  - More candidates in flight makes it slower, not faster: 27.2 s with one,
    36.0 s with two (the default), 38.4 s with eight. The summed stage times
    grow from 32 s to 61 s to 203 s as candidates queue behind the same
    process.
  - Ours shares the GPU between `-n` worker processes, so saving and the other
    CPU work run in parallel.

### Output

- **Their CPU output is unchanged:** byte-identical to `4db0d60` on all 380
  candidates.
- **Their GPU output is close but not identical:** DM–time images are
  identical to their CPU's, and frequency–time images now differ by at most
  about 6×10⁻⁵, from the resize moving to the GPU. Their notes describe this
  rounding difference.
- **Compared with numba, nothing changed:** frequency–time images correlate at
  a median of 0.866 (worst 0.143), DM–time images at a median of 0.976 (worst
  0.834). The file-name hash and the other output changes in "Output
  equivalence" and "File names" still apply.

### Why theirs is slower on the GPU

Measured in Slurm jobs 1419–1421, with the A4000 to itself and their `3349081`
on the same 380 candidates. GPU busy is the mean of `nvidia-smi`'s
utilisation, sampled every 100 ms over the run.

| run | wall | GPU busy |
|---|---|---|
| ours, `-n 11` | 11.6 s | 16% |
| ours, `-n 1` | 32.7 s | 6% |
| theirs, 1 in flight | 27.7 s | 36% |
| theirs, 1 in flight, saves uncompressed | 28.5 s | 35% |
| theirs, 1 in flight, no saves | 23.4 s | 44% |
| theirs, 2 in flight (their default) | 36.4 s | 31% |
| theirs, 2 in flight, no saves | 32.6 s | 28% |
| theirs as 4 separate processes, 1 in flight each | 17.4 s | 60% |
| theirs as 8 separate processes | 17.2 s | 62% |
| theirs as 11 separate processes | 16.9 s | 63% |

The 25 s gap between their default (2 in flight, 36.4 s) and ours with
`-n 11` (11.6 s) splits roughly as follows:

1. **About 10 s comes from doing all host work in one process per GPU.**
   - Their scheduler reads, launches, finishes and saves every candidate from
     one process, so the GPU waits on the host most of the time (36% busy).
   - Running their own candmaker as 4 processes on the same GPU, each on a
     quarter of the candidates, takes 27.7 s down to 17.4 s with no code
     change. More processes don't help beyond that.
   - Ours gets the same effect from its `-n` worker processes.
2. **About 6 s comes from GPU work that grows with pulse width.**
   - Their GPU kernels make both images at full time resolution (256 × width/2
     samples), then average down. The DM–time kernel sums 256 DMs × native
     samples × 4,096 channels: about 0.5 G additions at width 4 and 17 G at
     width 128.
   - Ours takes running sums along time per channel and reads each output bin
     as one difference, a constant 256 × 256 × 4,096.
   - Their GPU time adds up to about 11 s per run (8.7 s of it dedispersion),
     which is why 4 or more processes level off at 17 s with the GPU 60%
     busy. Ours is 16% busy at 11.6 s.
3. **About 9 s comes from their default of two in flight.**
   - Their planner pairs candidates into batches, and each batch loads the
     whole stretch of file the pair spans. That reads and uploads 4.66 GB
     instead of 1.96 GB.
   - The two threads compete for the GIL and the GPU. Per-candidate processing
     doubles (median 32 ms to 74 ms), file reads go from 0.95 s to 4.9 s, and
     CPU-side finishing from 2.2 s to 6.8 s.
   - Each batch also waits for its slower candidate before the next one's
     read starts.
4. **Saving is a small part of item 1, about 4 s.**
   - Turning saves off entirely takes 27.7 s to 23.4 s. Uncompressed saves are
     no faster than gzip-9 ones (28.5 s).
   - The saves do suffer from sharing a process: alone, one candidate's gzip-9
     save takes 17 ms. In a thread beside another thread busy in Python, it
     takes 115 ms, because the save's Python parts wait for the GIL. In their
     runs saves average 38 ms.
   - Ours saves with the same gzip level 9.

Theirs is faster than ours only in a single stream (27.7 s against 32.7 s),
because their resident read cache and GPU resize leave less host work per
candidate.

### Trying the fixes on theirs

Each fix was added to a copy of their `3349081` behind a switch, then run on
the same 380 candidates (Slurm job 1423, A4000 to itself, page cache warm,
float32 output, gzip-9 saves):

- **Processes** (`FIX_PROCS=k`): `run_jobs` starts k worker processes per GPU,
  each with a contiguous share of the candidates and 1/k of the memory budget.
- **Running sums** (`FIX_RUNSUM=1`): for each group of 256 channels, one kernel
  cleans the samples into floats, CuPy takes float64 running sums along time,
  and two small kernels gather the binned frequency–time and DM–time images
  from their differences. The work no longer grows with width. At width 4 it
  was slightly slower than theirs (16.8 ms against 13.4 ms per candidate), so
  it applies only from width 8 up. With uint8 data the sums are exact, so the
  images are unchanged.
- **Pipelining** (`FIX_PIPE=1`): candidates are planned one per batch, as with
  one in flight, and the reading for the next candidate overlaps the GPU work
  of the previous ones. There is no batch barrier and no reading of the span
  between paired candidates.

| run | wall | GPU busy | summed GPU kernel time |
|---|---|---|---|
| theirs, 1 in flight | 27.4 s | 34% | 8.8 s |
| theirs, 2 in flight (default) | 35.3 s | 29% | 13.1 s |
| 2 processes | 23.0 s | 44% | — |
| 4 processes | 19.5 s | 58% | 24.6 s |
| 8 processes | 19.7 s | 55% | — |
| running sums, 1 in flight | 25.1 s | 26% | 7.0 s |
| running sums, 2 in flight | 36.0 s | 18% | — |
| pipelined, 2 in flight | 25.3 s | 37% | 9.5 s |
| pipelined, 4 in flight | 29.3 s | 33% | — |
| running sums + pipelined, 2 in flight | 23.8 s | 24% | — |
| running sums + pipelined, 4 in flight | 38.4 s | 17% | — |
| 4 processes + running sums | 15.6 s | 40% | 10.8 s |
| 8 processes + running sums | 15.3 s | 36% | — |
| **all three, 4 processes, 2 in flight** | **15.0 s** | 43% | 17.4 s |
| ours, `-n 11` | 11.5 s | 17% | — |

Every run's images were identical to the unmodified 1-in-flight run, for all
380 candidates.

- **Processes are the biggest fix,** 27.4 s down to 19.5 s. Beyond 4 they stop
  helping, because the GPU becomes the limit: summed kernel time nearly triples,
  from 8.8 s to 24.6 s, as four processes contend for it.
- **Running sums pay off once the GPU is shared.** In one process they save
  2.3 s. With four processes they take the summed kernel time from 24.6 s to
  10.8 s and the wall time from 19.5 s to 15.6 s.
- **Pipelining fixes their default of 2 in flight** (35.3 s down to 25.3 s, and
  reads down from 4.66 GB to 1.96 GB). More than 2 in flight is worse again,
  because the threads compete for the GIL.
- **With all three, theirs is 15.0 s against our 11.5 s.**

### Where the rest of the gap is

A py-spy profile (100 Hz, every thread and child process) of their
one-in-flight run shows the CPU time per run:

- **20 s saving h5 files** in the writer thread, 11 s of it in `h5py`'s
  `close`, where the gzip compression happens. Running alongside the main
  thread, it costs the main thread GIL time.
- **9 s in CuPy's `nan_to_num`,** called by `_resize_binned_ft`. Its argument
  check runs `cupy.isnan` on each Python scalar, which makes a GPU array and
  reads it back, so the host waits there for the kernels queued before it.
  Most of this is GPU time seen from the host, but it also stops the host
  queueing the resize until the kernels finish.
- **7 s in `_finite_delay_bins`,** the per-candidate plan, which is NumPy work
  on the CPU.
- **10 s finishing on the CPU:** detrending, normalising, and the resize
  fallback check.

The cheap next steps for theirs are therefore a small elementwise kernel in
place of `nan_to_num`, so nothing waits on the GPU before the resize; caching
the delay plan per DM; and spreading saving over the processes.

## Summary (of `4db0d60`)

- **Speed:** ours is about 5× faster on the GPU and 30–60× faster on the CPU.
  Theirs is slower than the numba version it replaces.
- **Output:**
  - Our CPU output is byte-identical to numba's.
  - Our GPU output is identical to numba's GPU output.
  - Theirs differs from numba on every candidate: frequency–time images
    correlate at a median of 0.87 (worst 0.14), DM–time images at a median of
    0.976 (worst 0.83).
  - Their CPU and GPU agree with each other exactly. Ours, like numba, do not.
- **Why ours is faster:**
  1. Our DM–time algorithm costs the same whatever the pulse width.
  2. Our CPU path stays in 8-bit integers.
  3. Our GPU runs spread the CPU-side work over many processes.

  Each is covered under "Why ours is faster" below.

## Setup

- **Data:** J1705 `data_2021-12-31_15-16-54`, cleaned with jess using
  greenburst's settings (`-test jarque_bera -spb 4096 -mtz 1.5`). Heimdall
  (rebuilt, `-O3`, all-major) was run with greenburst's flags, then greenburst's
  cuts were applied. That gives 451 candidates, of which 380 were kept. Their
  pulse widths, in samples: 189 of 4, 165 of 8, 7 of 16, 8 of 32, 8 of 64 and
  3 of 128.
- **Environments:** three, all Python 3.12 with numpy 2.3.5, built from exact
  commits with `git archive`:
  - **base:** `fb863f8`, with numba 0.68 and numba-cuda 0.30.4;
  - **ours:** `4c9d910`, with CuPy 13.6;
  - **theirs:** `4db0d60`, with CuPy 13.6.
- **Hardware:** one RTX A4000, with the node to itself (`--exclusive`), and
  36 CPU cores.
- **The cleaned file was given a short name, `c.fil`.** jess records the output
  path in the header. Theirs and the base lack our fix for header strings over
  80 characters, so a long path would make them misread the data.

## Speed

### Whole run: 380 candidates, `-n 11`, two runs each

| version | GPU | CPU |
|---|---|---|
| numba base (`fb863f8`) | 26.1, 23.7 s * | 18.7, 18.6 s |
| **ours (`4c9d910`)** | **12.7, 11.8 s** | **9.1, 9.2 s** |
| theirs, float32 output | 63.4, 64.2 s | 528.7, 281.6 s |
| theirs, default FP16 output | 64.5, 64.1 s | — |

\* The base's GPU time is mostly CPU time. In each worker, numba hit
`CUDA_ERROR_CONTEXT_IS_DESTROYED` after its first candidate and fell back to
the CPU. That happened for 369 of the 380 candidates in both runs, so only 11
(one per worker) ran on the GPU.

### Per candidate, single process

These come from the log time between successive saves (ours) and from
`--profile_json` (theirs). "1 core" means pinned with `taskset`. Theirs is
single-threaded either way.

| width (samples) | ours, CPU, 1 core | ours, CPU, 36 threads | theirs, CPU | ours, GPU, 1 worker | theirs, GPU, 1 in flight |
|---|---|---|---|---|---|
| 4 | 0.18 s | 0.12 s | 1.75 s | 0.076 s | 0.200 s |
| 8 | 0.20 s | 0.13 s | 3.58 s | 0.078 s | 0.199 s |
| 16 | 0.23 s | 0.13 s | 9.13 s | 0.082 s | 0.212 s |
| 32 | 0.28 s | 0.18 s | 17.5 s | 0.093 s | 0.256 s |
| 64 | 0.38 s | 0.21 s | 34.5 s | 0.104 s | 0.287 s |
| 128 | 0.56 s | 0.26 s | 68.0 s | 0.144 s | 0.483 s |

### How the GPU runs scale (380 candidates)

| version | setting | wall |
|---|---|---|
| ours | `-n 1` | 34.7 s |
| ours | `-n 2` | 19.4 s |
| ours | `-n 4` | 13.8 s |
| ours | `-n 11` | 11.8–12.7 s |
| theirs | `--gpu_inflight 1` | 72.1 s |
| theirs | `--gpu_inflight 2` (default) | 63.4–64.5 s |
| theirs | `--gpu_inflight 8` | 61.0 s |

## Why ours is faster

### 1. Their DM–time cost doubles with every doubling of pulse width

This accounts for most of the CPU gap.

- **Theirs:** for each of the 256 DMs, they sum all 4,096 channels at every
  native sample in the window, and only then average down to 256 time bins.
  The work is DMs × channels × 256 × the averaging factor, which is half the
  pulse width. This is `process_cpu` in `your/utils/candidate_pipeline.py`,
  and `dmt_tiled_*` followed by `bin_dmt` in `your/utils/gpu_kernels.cu`.
  - Their CPU dedispersion step: 1.3 s at width 4, 2.9 s at 8, 8.3 s at 16,
    16.3 s at 32, 32.5 s at 64 and 64.3 s at 128.
  - Their GPU step: 0.038 s rising to 0.259 s.
- **Ours:** each channel's integer samples get a running sum, so the total over
  any run of samples is the difference of two of its values. Summed over the
  channels, those differences telescope. Each output point therefore costs one
  read per channel, whatever the pulse width, and the work is DMs × channels ×
  256. The CPU code is in `rust/src/crop.rs`, and the GPU has the same
  running-sum path in `gpu_dedisp_and_dmt_crop`. What still grows with width
  is reading the longer stretch of input and building the running sums, which
  is why ours rises from 0.18 s to 0.56 s on one core.

### 2. Their CPU path is float64; ours stays in 8-bit integers

At the narrowest widths the two algorithms cost about the same, yet theirs is
still 10× slower. This is why.

- **Theirs:** `process_cpu` converts the input to float64 (8 bytes per sample),
  takes medians and fills padding in float64, then calls `_rust_dedispersets`
  256 times, once per DM. Each call walks across all 4,096 channels of an array
  laid out by time.
- **Ours:** reads the uint8 samples, lays them out by channel with 8×8-byte
  register transposes, and sums them exactly in 64-bit integers. It also gives
  each worker an equal share of the cores, through `worker_threads`.

### 3. On the GPU, their CPU-side work runs in one process

- **One process per GPU.** `run_jobs` starts one worker per GPU, and `-n` does
  not apply to GPU runs. For each candidate, that one Python process also:
  - reads the input from the file (about 0.006 s);
  - resizes the frequency–time image with skimage (0.049 s);
  - detrends and normalises it (0.007 s);
  - saves it with gzip level 9 (0.069 s).

  Together that is about 0.13 of their 0.20 s per candidate.
- **More concurrency doesn't help.** Raising `--gpu_inflight` from 1 to 8 only
  takes 72 s to 61 s, and the summed stage times grow about fourfold as
  candidates queue behind the same process. The GPU is not the limit.
- **Ours shares the GPU between `-n` worker processes,** so the reading,
  resizing and saving run in parallel (34.7 s with one worker, about 12 s
  with 11).
- **Theirs brings back the full image.** Their GPU returns the 4,096-channel
  frequency–time image and resizes it on the CPU. Ours averages blocks of 16
  channels into 256 on the GPU, as the numba version did, so the CPU resize
  has almost nothing to do.

### Not the cause

- **Compression:** both write h5 files with `compression="gzip"` and
  `compression_opts=9`.
- **Output precision:** their FP16 and float32 runs take the same time.

## Output equivalence

Candidates were matched by `(tcand, dm, snr)` from the h5 attributes. All runs
produced 256 × 256 images.

| comparison | frequency–time identical | DM–time identical |
|---|---|---|
| base CPU, run 2 vs run 1 | 380 / 380 | 380 / 380 |
| ours CPU vs base CPU | 380 / 380 | 380 / 380 |
| ours GPU vs base GPU, over the 11 candidates numba ran on the GPU | 11 / 11 | — |
| ours GPU vs base GPU, over all 380 | — | 373 / 380 |
| ours GPU vs base CPU | 0 / 380 | 374 / 380 |
| theirs vs base CPU (CPU, GPU, FP16) | 0 / 380 | 0 / 380 |
| theirs CPU vs theirs GPU | 380 / 380 | 380 / 380 |

- **Our GPU and CPU images differ, as numba's did.** The GPU path sums blocks
  of 16 channels, while the CPU path smooths and resizes from 4,096 channels
  with skimage (`anti_aliasing=True`, `mode="constant"`). The constant-mode
  edge drags the CPU image's first channel down: a mean of about −10 after
  normalisation in the example checked.
- **How far theirs is from numba:** each image was correlated with numba's
  CPU image, after detrending and normalising numba's frequency–time image so
  like is compared with like.
  - Frequency–time images: median 0.866, worst 0.143.
  - DM–time images: median 0.976, worst 0.834.
- **What their branch changes on purpose** (from `docs/cupy-candidates.md`):
  - samples past the chunk edges come from the file instead of wrapping around;
  - the frequency–time image is detrended inside the candmaker;
  - the DM–time image averages channels instead of summing them;
  - images and h5 datasets are FP16 by default;
  - file names gain a hash.

  FETCH's classifications on their images should be checked against the old
  ones before adopting them, since its models were trained on the old images.

## File names

Theirs appends a 24-character BLAKE2b hash to the old name:
`cand_tstart_<mjd>_tcand_<s>_dm_<dm>_snr_<snr>_<hash>.h5`.

- **What the hash covers:** the resolved input paths, `tstart`, `tcand`, `dm`,
  `snr`, `width`, `label`, `min_samp`, the kill mask and the RFI settings
  (`Candidate._update_id`).
- **What it's for:** stopping rows that differ only in width, label or mask,
  or come from different files with the same start time, from overwriting
  each other.
- **Their CPU path also appends the CSV row number** (`_<hash>_0.h5`), and
  their GPU path does not. The same candidate is therefore named differently on
  CPU and GPU.
- **The names aren't portable.** The hash includes the real path of the input
  file, so a moved or copied file gives different names.
- **It breaks greenburst's database tools** in `~/greenburst_db`, which parse
  the old names:
  - `cutout_store.py:108` and `lost_h5.py:70` anchor on `_snr_([0-9.]+)$`;
  - `bad_h5_report.py:17` expects `.h5` immediately after the S/N;
  - `lost_h5.py:77` rebuilds names from CSV fields and can't compute the hash.

  `find_h5.py:42` and greenburst's `pulsar_plot_limit.py:41` don't anchor the
  end of the name, so they would still match.
- **A gentler fix** would add a suffix only when a name is already taken.

## Worth taking from theirs

- Writing each h5 to a temporary file and renaming it, so a failed save never
  leaves a partial file.
- Reading real samples past the chunk edges instead of wrapping, as an option
  rather than a silent default.
- CPU and GPU giving the same result.
- Tiled processing for candidates too large for GPU memory.
- Per-stage profiling (`--profile_json`).
- The larger test suite, and the lint step in CI.

## What theirs would need to match our speed

In order of payoff:

1. Running sums for the time averaging, on both CPU and GPU.
2. uint8 input with integer sums, instead of float64.
3. Several worker processes per GPU, instead of threads in one process.
4. Averaging channels on the GPU before the image comes back.

Items 1 and 4 change their current output:

- **Running sums** give identical results for integer data. For float data
  they sum in a different order, which changes the last bits.
- **Averaging channels on the GPU,** in place of the CPU's skimage resize,
  changes the frequency axis. That difference is exactly why our GPU and CPU
  images disagree.

## Gaps in their branch

- **No fix for header strings over 80 characters.** That fix is in ours
  (`200556e`). Without it, files cleaned by jess under long paths are read at
  the wrong offset.
- **No live psrdada mode.** Their rewrite of `bin/your_candmaker.py`,
  `your/candidate.py` and `your/utils/gpu.py` overlaps nearly everything
  `dada-candmaker` changed, so merging the two would conflict heavily.

## Caveats

- This is one file of mostly narrow pulses: 354 of the 380 candidates are 4 or
  8 samples wide. Wider pulses favour ours further (the per-width table).
- Their development setup was a V100 with CUDA 12.9. I'd expect the ranking to
  hold on other GPUs, but not the exact ratios.
- The base's GPU numbers reflect numba's fallback to the CPU, not numba's GPU
  speed.

## Reproducing

Everything is in the session scratchpad, under `build/cmpcupy-k2GG/`:

| file | what it is |
|---|---|
| `build.sh` | builds the three environments |
| `env.sh` | runs a command in one of them, with CuPy's pip NVRTC on the library path |
| `run.sh` | the whole-run benchmark, including jess, Heimdall and the cuts |
| `times.txt` | its results |
| `why.sh`, `why.txt`, `why/` | the single-process and scaling runs |
| `pertime.py` | the per-width tables |
| `ret.sh`, `ret.txt`, `ret/`, `pertime2.py` | the retiming of their `3349081` |
| `../whyslow/run.sh`, `split.sh`, `gil.py`, `res.txt`, `out/` | the GPU slowdown measurements; `pkg/` is their package with an `ABL_SAVE` switch (`none`, `skip`) |
| `../fixes/run.sh`, `res.txt`, `out/`, `prof/`, `spy.py` | the fixes tried on theirs and the py-spy profiles; `pkg/` is their package with the `FIX_PROCS`, `FIX_RUNSUM` and `FIX_PIPE` switches |
| `wait_submit.sh` | holds the job until nothing outside Slurm is using the A4000 |
| `cmp_h5.py` | the output comparison |
| `data/c.fil` | the cleaned input, 8 GB |
| `data/cands.csv` | the candidates |
| `data/strat.csv` | the 66-candidate subset covering every width |
