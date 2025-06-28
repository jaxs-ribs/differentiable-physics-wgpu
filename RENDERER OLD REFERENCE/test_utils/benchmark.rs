use std::time::{Duration, Instant};

pub struct BenchmarkTimer {
    warmup_steps: u32,
    measure_steps: u32,
}

impl BenchmarkTimer {
    pub fn new(warmup_steps: u32, measure_steps: u32) -> Self {
        Self {
            warmup_steps,
            measure_steps,
        }
    }
    
    pub fn measure<F: FnMut()>(&self, mut operation: F) -> BenchmarkResults {
        println!("Warming up with {} steps...", self.warmup_steps);
        for _ in 0..self.warmup_steps {
            operation();
        }
        
        println!("Measuring {} steps...", self.measure_steps);
        let start = Instant::now();
        
        for _ in 0..self.measure_steps {
            operation();
        }
        
        let total_time = start.elapsed();
        
        BenchmarkResults {
            total_time,
            steps: self.measure_steps,
        }
    }
}

impl Default for BenchmarkTimer {
    fn default() -> Self {
        Self::new(10, 100)
    }
}

pub struct BenchmarkResults {
    pub total_time: Duration,
    pub steps: u32,
}

impl BenchmarkResults {
    pub fn average_step_time(&self) -> Duration {
        self.total_time / self.steps
    }
    
    pub fn steps_per_second(&self) -> f64 {
        self.steps as f64 / self.total_time.as_secs_f64()
    }
    
    pub fn print_summary(&self, label: &str) {
        println!("\n{} Benchmark Results:", label);
        println!("  Total time: {:?}", self.total_time);
        println!("  Steps: {}", self.steps);
        println!("  Average per step: {:?}", self.average_step_time());
        println!("  Steps per second: {:.2}", self.steps_per_second());
    }
    
    pub fn print_throughput(&self, label: &str, operations_per_step: usize) {
        let total_operations = self.steps as f64 * operations_per_step as f64;
        let throughput = total_operations / self.total_time.as_secs_f64();
        
        println!("\n{} Throughput:", label);
        println!("  Operations per step: {}", operations_per_step);
        println!("  Total operations: {:.0}", total_operations);
        println!("  Throughput: {:.2} ops/sec", throughput);
        
        if throughput > 1_000_000.0 {
            println!("  Throughput: {:.2} M ops/sec", throughput / 1_000_000.0);
        }
    }
}