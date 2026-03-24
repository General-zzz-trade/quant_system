use pyo3::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::cell::UnsafeCell;

/// Cache-line padded atomic index to prevent false sharing.
#[repr(align(64))]
struct PaddedAtomic {
    val: AtomicUsize,
}

impl PaddedAtomic {
    fn new(v: usize) -> Self {
        Self {
            val: AtomicUsize::new(v),
        }
    }
}

/// Lock-free single-producer single-consumer ring buffer for PyObject events.
///
/// - push() is called from IO/callback thread (producer)
/// - pop() / drain() is called from engine loop thread (consumer)
/// - Zero mutex, zero allocation at runtime
/// - Capacity must be power of 2 (enforced at construction)
struct RingInner {
    buffer: Vec<UnsafeCell<Option<PyObject>>>,
    mask: usize,
    head: PaddedAtomic, // producer writes here
    tail: PaddedAtomic, // consumer reads here
}

// Safety: RingInner is designed for single-producer single-consumer pattern.
// The head index is only modified by the producer (push).
// The tail index is only modified by the consumer (pop/drain).
// PyObject is Send because Python GIL protects actual access.
unsafe impl Send for RingInner {}
unsafe impl Sync for RingInner {}

impl RingInner {
    fn new(capacity: usize) -> Self {
        // Round up to power of 2
        let cap = capacity.next_power_of_two();
        let mut buffer = Vec::with_capacity(cap);
        for _ in 0..cap {
            buffer.push(UnsafeCell::new(None));
        }
        Self {
            buffer,
            mask: cap - 1,
            head: PaddedAtomic::new(0),
            tail: PaddedAtomic::new(0),
        }
    }

    fn capacity(&self) -> usize {
        self.mask + 1
    }

    /// Push an item. Returns false if full.
    fn push(&self, item: PyObject) -> bool {
        let head = self.head.val.load(Ordering::Relaxed);
        let tail = self.tail.val.load(Ordering::Acquire);
        if head.wrapping_sub(tail) >= self.capacity() {
            return false;
        }
        let slot = head & self.mask;
        // Safety: producer exclusively owns this slot (between head and tail)
        unsafe {
            *self.buffer[slot].get() = Some(item);
        }
        self.head.val.store(head.wrapping_add(1), Ordering::Release);
        true
    }

    /// Pop an item. Returns None if empty.
    fn pop(&self) -> Option<PyObject> {
        let tail = self.tail.val.load(Ordering::Relaxed);
        let head = self.head.val.load(Ordering::Acquire);
        if tail == head {
            return None;
        }
        let slot = tail & self.mask;
        // Safety: consumer exclusively owns this slot
        let item = unsafe { (*self.buffer[slot].get()).take() };
        self.tail.val.store(tail.wrapping_add(1), Ordering::Release);
        item
    }

    fn len(&self) -> usize {
        let head = self.head.val.load(Ordering::Acquire);
        let tail = self.tail.val.load(Ordering::Acquire);
        head.wrapping_sub(tail)
    }
}

/// Python-exposed SPSC ring buffer.
///
/// Replaces queue.Queue in EngineLoop for lock-free event passing.
/// Designed for single-producer (IO thread) + single-consumer (engine loop) pattern.
#[pyclass]
pub struct RustSpscRing {
    inner: RingInner,
    drop_count: AtomicUsize,
}

#[pymethods]
impl RustSpscRing {
    #[new]
    #[pyo3(signature = (capacity=131072))]
    fn new(capacity: usize) -> Self {
        Self {
            inner: RingInner::new(capacity),
            drop_count: AtomicUsize::new(0),
        }
    }

    /// Push an event into the ring. Returns True if successful, False if full.
    fn push(&self, event: PyObject) -> bool {
        if self.inner.push(event) {
            true
        } else {
            self.drop_count.fetch_add(1, Ordering::Relaxed);
            false
        }
    }

    /// Pop an event from the ring. Returns None if empty.
    fn pop(&self, py: Python<'_>) -> Option<PyObject> {
        let _ = py;
        self.inner.pop()
    }

    /// Drain up to max_items from the ring into a list.
    fn drain(&self, py: Python<'_>, max_items: usize) -> Vec<PyObject> {
        let _ = py;
        let mut result = Vec::with_capacity(max_items.min(self.inner.len()));
        for _ in 0..max_items {
            match self.inner.pop() {
                Some(item) => result.push(item),
                None => break,
            }
        }
        result
    }

    /// Number of items currently in the ring.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Ring buffer capacity (always power of 2).
    fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Total number of dropped events (push when full).
    fn drop_count(&self) -> usize {
        self.drop_count.load(Ordering::Relaxed)
    }
}
