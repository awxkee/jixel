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
use std::any::Any;
use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use crate::coder_scratch::CoderScratch;

struct Completion {
    remaining: AtomicUsize,
    panic: Mutex<Option<Box<dyn Any + Send>>>,
}

impl Completion {
    fn finish(&self, panic: Option<Box<dyn Any + Send>>, shared: &Shared) {
        if let Some(payload) = panic {
            let mut first_panic = self.panic.lock().unwrap();
            if first_panic.is_none() {
                *first_panic = Some(payload);
            }
        }
        let _queue = shared.queue.lock().unwrap();
        self.remaining.fetch_sub(1, Ordering::Release);
        shared.activity.notify_all();
    }
}

struct Task {
    run: Box<dyn FnOnce(&mut CoderScratch) + Send + 'static>,
    completion: Arc<Completion>,
}

impl Task {
    fn execute(self, shared: &Shared, scratch: &mut CoderScratch) {
        let Self { run, completion } = self;
        let result = catch_unwind(AssertUnwindSafe(|| run(scratch)));
        // `run` (and therefore all lifetime-erased borrowed captures) has been
        // consumed and destroyed before completion can reach zero.
        completion.finish(result.err(), shared);
    }
}

struct Shared {
    queue: Mutex<VecDeque<Task>>,
    activity: Condvar,
    shutdown: AtomicBool,
}

impl Shared {
    fn push(&self, task: Task) {
        self.queue.lock().unwrap().push_back(task);
        self.activity.notify_one();
    }

    fn run_one(&self, scratch: &mut CoderScratch) -> bool {
        let task = self.queue.lock().unwrap().pop_front();
        if let Some(task) = task {
            task.execute(self, scratch);
            true
        } else {
            false
        }
    }

    fn wait_for_activity(&self, remaining: &AtomicUsize) {
        let queue = self.queue.lock().unwrap();
        if queue.is_empty()
            && remaining.load(Ordering::Acquire) != 0
            && !self.shutdown.load(Ordering::Acquire)
        {
            drop(self.activity.wait(queue).unwrap());
        }
    }
}

/// Fixed set of workers reused by all parallel phases of one encoding.
///
/// The calling thread is also a worker, so a pool configured for `n` threads
/// owns `n - 1` background threads. Waiting workers help run queued work. That
/// is important for nested maps (AC-strategy bands inside DC-group setup) and
/// prevents them from deadlocking the pool.
pub(crate) struct ThreadPool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    num_threads: usize,
}

impl ThreadPool {
    pub(crate) fn new(num_threads: usize) -> Self {
        let num_threads = num_threads.max(1);
        let shared = Arc::new(Shared {
            queue: Mutex::new(VecDeque::new()),
            activity: Condvar::new(),
            shutdown: AtomicBool::new(false),
        });
        let workers = (1..num_threads)
            .map(|i| {
                let shared = Arc::clone(&shared);
                std::thread::Builder::new()
                    .name(format!("jixel-worker-{i}"))
                    .spawn(move || {
                        let mut scratch = Box::<CoderScratch>::default();
                        worker_loop(&shared, &mut scratch);
                    })
                    .expect("failed to start encoder worker")
            })
            .collect();
        Self {
            shared,
            workers,
            num_threads,
        }
    }

    #[inline]
    pub(crate) fn num_threads(&self) -> usize {
        self.num_threads
    }

    pub(crate) fn steal_map<T, F>(
        &self,
        caller_scratch: &mut CoderScratch,
        len: usize,
        f: F,
    ) -> Vec<T>
    where
        T: Send,
        F: Fn(usize, &mut CoderScratch) -> T + Sync,
    {
        self.steal_map_with_threads(caller_scratch, len, self.num_threads, f)
    }

    pub(crate) fn steal_map_with_threads<T, F>(
        &self,
        caller_scratch: &mut CoderScratch,
        len: usize,
        max_threads: usize,
        f: F,
    ) -> Vec<T>
    where
        T: Send,
        F: Fn(usize, &mut CoderScratch) -> T + Sync,
    {
        let lanes = self.num_threads.min(max_threads.max(1)).min(len);
        if lanes <= 1 {
            return (0..len).map(|i| f(i, caller_scratch)).collect();
        }

        let cursor = AtomicUsize::new(0);
        let chunks = Mutex::new(Vec::<Vec<(usize, T)>>::with_capacity(lanes));
        let completion = Arc::new(Completion {
            remaining: AtomicUsize::new(lanes),
            panic: Mutex::new(None),
        });

        let lane = |scratch: &mut CoderScratch| {
            let mut out = Vec::new();
            loop {
                let i = cursor.fetch_add(1, Ordering::Relaxed);
                if i >= len {
                    break;
                }
                out.push((i, f(i, scratch)));
            }
            chunks.lock().unwrap().push(out);
        };

        for _ in 1..lanes {
            let run: Box<dyn FnOnce(&mut CoderScratch) + Send + '_> = Box::new(&lane);
            // `steal_map_with_threads` does not return until every queued lane
            // has completed, so the borrowed closure and its
            // captures outlive each task. This is the same scoped-lifetime
            // guarantee provided by `std::thread::scope`, applied to persistent
            // workers instead of newly spawned threads.
            let run = unsafe {
                std::mem::transmute::<
                    Box<dyn FnOnce(&mut CoderScratch) + Send + '_>,
                    Box<dyn FnOnce(&mut CoderScratch) + Send + 'static>,
                >(run)
            };
            self.shared.push(Task {
                run,
                completion: Arc::clone(&completion),
            });
        }
        let result = catch_unwind(AssertUnwindSafe(|| lane(caller_scratch)));
        completion.finish(result.err(), &self.shared);

        while completion.remaining.load(Ordering::Acquire) != 0 {
            if !self.shared.run_one(caller_scratch) {
                self.shared.wait_for_activity(&completion.remaining);
            }
        }

        if let Some(payload) = completion.panic.lock().unwrap().take() {
            resume_unwind(payload);
        }

        let mut slots: Vec<Option<T>> = (0..len).map(|_| None).collect();
        for (i, value) in chunks.into_inner().unwrap().into_iter().flatten() {
            slots[i] = Some(value);
        }
        slots.into_iter().map(Option::unwrap).collect()
    }

    /// Apply `f` to every item, allowing workers to borrow distinct mutable
    /// elements without wrapping each element in a lock.
    pub(crate) fn steal_for_each_mut<T, F>(
        &self,
        caller_scratch: &mut CoderScratch,
        items: &mut [T],
        f: F,
    ) where
        T: Send,
        F: Fn(usize, &mut T, &mut CoderScratch) + Sync,
    {
        self.steal_for_each_mut_with_threads(caller_scratch, items, self.num_threads, f)
    }

    pub(crate) fn steal_for_each_mut_with_threads<T, F>(
        &self,
        caller_scratch: &mut CoderScratch,
        items: &mut [T],
        max_threads: usize,
        f: F,
    ) where
        T: Send,
        F: Fn(usize, &mut T, &mut CoderScratch) + Sync,
    {
        let lanes = self.num_threads.min(max_threads.max(1)).min(items.len());
        if lanes <= 1 {
            for (i, item) in items.iter_mut().enumerate() {
                f(i, item, caller_scratch);
            }
            return;
        }

        let cursor = AtomicUsize::new(0);
        let items_ptr = AtomicPtr::new(items.as_mut_ptr());
        let len = items.len();
        let completion = Arc::new(Completion {
            remaining: AtomicUsize::new(lanes),
            panic: Mutex::new(None),
        });

        let lane = |scratch: &mut CoderScratch| {
            let items_ptr = items_ptr.load(Ordering::Relaxed);
            loop {
                let i = cursor.fetch_add(1, Ordering::Relaxed);
                if i >= len {
                    break;
                }
                // Every index is returned exactly once by `cursor`, so
                // concurrent lanes cannot create overlapping mutable
                // references. All lanes complete before this function returns,
                // keeping `items` alive.
                let item = unsafe { &mut *items_ptr.add(i) };
                f(i, item, scratch);
            }
        };

        for _ in 1..lanes {
            let run: Box<dyn FnOnce(&mut CoderScratch) + Send + '_> = Box::new(&lane);
            // See `steal_map_with_threads`: completion scopes the borrowed
            // closure and `items` to this call despite the persistent workers.
            let run = unsafe {
                std::mem::transmute::<
                    Box<dyn FnOnce(&mut CoderScratch) + Send + '_>,
                    Box<dyn FnOnce(&mut CoderScratch) + Send + 'static>,
                >(run)
            };
            self.shared.push(Task {
                run,
                completion: Arc::clone(&completion),
            });
        }
        let result = catch_unwind(AssertUnwindSafe(|| lane(caller_scratch)));
        completion.finish(result.err(), &self.shared);

        while completion.remaining.load(Ordering::Acquire) != 0 {
            if !self.shared.run_one(caller_scratch) {
                self.shared.wait_for_activity(&completion.remaining);
            }
        }

        if let Some(payload) = completion.panic.lock().unwrap().take() {
            resume_unwind(payload);
        }
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        let queue = self.shared.queue.lock().unwrap();
        self.shared.shutdown.store(true, Ordering::Release);
        self.shared.activity.notify_all();
        drop(queue);
        for worker in self.workers.drain(..) {
            worker.join().unwrap();
        }
    }
}

fn worker_loop(shared: &Shared, scratch: &mut CoderScratch) {
    loop {
        let task = {
            let mut queue = shared.queue.lock().unwrap();
            while queue.is_empty() && !shared.shutdown.load(Ordering::Acquire) {
                queue = shared.activity.wait(queue).unwrap();
            }
            if queue.is_empty() {
                return;
            }
            queue.pop_front().unwrap()
        };
        task.execute(shared, scratch);
    }
}

#[cfg(test)]
mod tests {
    use super::{CoderScratch, ThreadPool};

    #[test]
    fn persistent_pool_preserves_index_order() {
        let pool = ThreadPool::new(4);
        let mut scratch = Box::<CoderScratch>::default();
        assert_eq!(
            pool.steal_map(&mut scratch, 257, |i, _scratch| i.wrapping_mul(17)),
            (0usize..257)
                .map(|i| i.wrapping_mul(17))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn nested_maps_share_the_pool() {
        let pool = ThreadPool::new(4);
        let mut scratch = Box::<CoderScratch>::default();
        let result = pool.steal_map(&mut scratch, 16, |outer, scratch| {
            pool.steal_map_with_threads(scratch, 9, 2, |inner, _scratch| outer * 100 + inner)
        });
        for (outer, row) in result.iter().enumerate() {
            assert_eq!(
                row,
                &(0..9).map(|inner| outer * 100 + inner).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn pool_can_be_reused_for_consecutive_maps() {
        let pool = ThreadPool::new(3);
        let mut scratch = Box::<CoderScratch>::default();
        for offset in 0..8 {
            assert_eq!(
                pool.steal_map(&mut scratch, 31, |i, _scratch| offset + i),
                (0..31).map(|i| offset + i).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn mutable_map_visits_each_item_once_without_locks() {
        let pool = ThreadPool::new(4);
        let mut scratch = Box::<CoderScratch>::default();
        let mut values = vec![0usize; 257];
        pool.steal_for_each_mut(&mut scratch, &mut values, |i, value, _scratch| {
            *value = i.wrapping_mul(17);
        });
        assert_eq!(
            values,
            (0usize..257)
                .map(|i| i.wrapping_mul(17))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn worker_panic_is_propagated_without_losing_the_pool() {
        let pool = ThreadPool::new(4);
        let mut scratch = Box::<CoderScratch>::default();
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.steal_map(&mut scratch, 64, |i, _scratch| {
                assert_ne!(i, 17, "worker failure");
                i
            });
        }));
        assert!(panic.is_err());
        assert_eq!(
            pool.steal_map(&mut scratch, 4, |i, _scratch| i),
            vec![0, 1, 2, 3]
        );
    }
}
