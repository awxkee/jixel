/*
 * // Copyright (c) Radzivon Bartoshyk 7/2026. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
use std::sync::atomic::{AtomicUsize, Ordering};

pub(crate) fn steal_map<T, F>(len: usize, nthreads: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Sync,
{
    let lanes = nthreads.max(1).min(len);
    if lanes <= 1 {
        return (0..len).map(f).collect();
    }

    let cursor = AtomicUsize::new(0);
    let (f, cursor) = (&f, &cursor);
    // Copy closure: one stealing lane. Each lane collects its own
    // (index, value) pairs — no shared writes.
    let lane = move || {
        let mut out = Vec::new();
        loop {
            let i = cursor.fetch_add(1, Ordering::Relaxed);
            if i >= len {
                break out;
            }
            out.push((i, f(i)));
        }
    };

    // The caller runs a lane itself instead of parking in join; by the time it
    // finishes, the cursor is drained and joins only wait out stragglers.
    let mut chunks: Vec<Vec<(usize, T)>> = std::thread::scope(|s| {
        let handles: Vec<_> = (1..lanes).map(|_| s.spawn(lane)).collect();
        let own = lane();
        let mut all: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        all.push(own);
        all
    });

    // Scatter into index order.
    let mut slots: Vec<Option<T>> = (0..len).map(|_| None).collect();
    for pair in chunks.drain(..).flatten() {
        slots[pair.0] = Some(pair.1);
    }
    slots.into_iter().map(|v| v.unwrap()).collect()
}
