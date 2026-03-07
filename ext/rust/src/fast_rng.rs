/// xorshift64* PRNG — fast, good quality, no external deps.
/// Internal module, not exported to Python.
pub struct FastRNG {
    state: u64,
}

impl FastRNG {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E3779B97F4A7C15u64,
        }
    }

    pub fn next(&mut self) -> u64 {
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        self.state.wrapping_mul(0x2545F4914F6CDD1Du64)
    }

    pub fn randint(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    pub fn gauss(&mut self, mu: f64, sigma: f64) -> f64 {
        let u1 = (self.next() as f64 + 1.0) / 18446744073709551617.0_f64;
        let u2 = (self.next() as f64 + 1.0) / 18446744073709551617.0_f64;
        let z = (-2.0 * u1.ln()).sqrt() * (6.283185307179586 * u2).cos();
        mu + sigma * z
    }
}
