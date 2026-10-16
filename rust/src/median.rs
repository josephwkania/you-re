//! Per-channel value counts of 8-bit data, from which `channel_median` takes
//! its order statistics.

use crate::crop::run_split;

// channels counted by one task: its 256 x 256 table of 32-bit counts is
// 256 KiB, small enough to stay in cache while every sample passes it
const BAND: usize = 256;

/// `counts[ch * 256 + v]`: how many of the `nt` samples of channel `ch` in
/// `(nt, nf)` data are `v`. Bands of channels are split between `threads`
/// threads; each reads whole rows of its band and counts into a private
/// table before storing it.
pub(crate) fn channel_histogram(
    data: &[u8],
    nt: usize,
    nf: usize,
    threads: usize,
    counts: &mut [i64],
) {
    let bands: Vec<(usize, &mut [i64])> = counts
        .chunks_mut(BAND * 256)
        .enumerate()
        .map(|(band, out)| (band * BAND, out))
        .collect();
    run_split(bands, threads, |(c0, out)| {
        let width = out.len() / 256;
        let mut local = vec![0_u32; width * 256];
        for row in data.chunks_exact(nf).take(nt) {
            for (ch, &v) in row[c0..c0 + width].iter().enumerate() {
                local[ch * 256 + v as usize] += 1;
            }
        }
        for (o, &l) in out.iter_mut().zip(&local) {
            *o = i64::from(l);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_every_channel_whatever_the_threads() {
        let (nt, nf) = (37, 600);
        let data: Vec<u8> = (0..nt * nf).map(|i| (i * 7919 % 251) as u8).collect();
        let mut expected = vec![0_i64; nf * 256];
        for s in 0..nt {
            for ch in 0..nf {
                expected[ch * 256 + data[s * nf + ch] as usize] += 1;
            }
        }
        for threads in [1, 2, 3, 8] {
            let mut counts = vec![0_i64; nf * 256];
            channel_histogram(&data, nt, nf, threads, &mut counts);
            assert_eq!(counts, expected);
        }
    }
}
