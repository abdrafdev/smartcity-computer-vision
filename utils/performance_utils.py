#!/usr/bin/env python3
"""
Smart City Computer Vision - Performance Monitoring Utilities

This module provides performance monitoring, profiling, and optimization
utilities for the smart city computer vision project.
"""

import time
import psutil
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from contextlib import contextmanager


@dataclass
class PerformanceMetrics:
    """Data class to store performance metrics."""
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    gpu_memory_mb: Optional[float] = None
    fps: Optional[float] = None
    throughput: Optional[float] = None


class PerformanceMonitor:
    """Real-time performance monitoring for computer vision operations."""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.metrics_history: List[Dict[str, Any]] = []
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
    
    def start_monitoring(self):
        """Start continuous performance monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
    
    def _monitor_loop(self):
        """Monitoring loop running in separate thread."""
        while self._monitoring:
            metrics = self._collect_metrics()
            self.metrics_history.append({
                'timestamp': time.time(),
                'metrics': metrics
            })
            time.sleep(self.monitoring_interval)
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics."""
        process = psutil.Process()
        
        metrics = {
            'cpu_percent': process.cpu_percent(),
            'memory_mb': process.memory_info().rss / 1024 / 1024,
            'system_cpu_percent': psutil.cpu_percent(),
            'system_memory_percent': psutil.virtual_memory().percent
        }
        
        # GPU metrics if available
        try:
            import torch
            if torch.cuda.is_available():
                metrics['gpu_memory_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
                metrics['gpu_memory_cached_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
        except ImportError:
            pass
        
        return metrics
    
    def get_average_metrics(self) -> Dict[str, float]:
        """Get average metrics from monitoring history."""
        if not self.metrics_history:
            return {}
        
        total_metrics = {}
        count = len(self.metrics_history)
        
        for entry in self.metrics_history:
            for key, value in entry['metrics'].items():
                if key not in total_metrics:
                    total_metrics[key] = 0
                total_metrics[key] += value
        
        return {key: value / count for key, value in total_metrics.items()}


@contextmanager
def performance_timer(operation_name: str = "Operation"):
    """Context manager for timing operations."""
    start_time = time.perf_counter()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024
    
    print(f"Starting {operation_name}...")
    
    try:
        yield
    finally:
        end_time = time.perf_counter()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        execution_time = end_time - start_time
        memory_delta = end_memory - start_memory
        
        print(f"{operation_name} completed:")
        print(f"  Execution time: {execution_time:.3f} seconds")
        print(f"  Memory usage: {end_memory:.1f} MB ({memory_delta:+.1f} MB)")


class FPSCounter:
    """Frame rate counter for video processing."""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.frame_times: List[float] = []
        self.last_time = time.perf_counter()
    
    def tick(self) -> float:
        """Register a new frame and return current FPS."""
        current_time = time.perf_counter()
        frame_time = current_time - self.last_time
        self.last_time = current_time
        
        self.frame_times.append(frame_time)
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
        
        if len(self.frame_times) < 2:
            return 0.0
        
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
    
    def reset(self):
        """Reset the FPS counter."""
        self.frame_times.clear()
        self.last_time = time.perf_counter()


class InferenceProfiler:
    """Profiler specifically for model inference operations."""
    
    def __init__(self):
        self.profiles: Dict[str, List[PerformanceMetrics]] = {}
    
    @contextmanager
    def profile_inference(self, model_name: str, batch_size: int = 1):
        """Profile a single inference operation."""
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        start_cpu = psutil.cpu_percent()
        
        # GPU metrics if available
        gpu_memory_start = None
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory_start = torch.cuda.memory_allocated() / 1024 / 1024
        except ImportError:
            pass
        
        yield
        
        end_time = time.perf_counter()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        end_cpu = psutil.cpu_percent()
        
        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory
        cpu_usage = end_cpu - start_cpu
        
        gpu_memory = None
        if gpu_memory_start is not None:
            try:
                import torch
                gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024 - gpu_memory_start
            except ImportError:
                pass
        
        fps = batch_size / execution_time if execution_time > 0 else 0
        throughput = 1.0 / execution_time if execution_time > 0 else 0
        
        metrics = PerformanceMetrics(
            execution_time=execution_time,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            gpu_memory_mb=gpu_memory,
            fps=fps,
            throughput=throughput
        )
        
        if model_name not in self.profiles:
            self.profiles[model_name] = []
        self.profiles[model_name].append(metrics)
    
    def get_average_metrics(self, model_name: str) -> Optional[PerformanceMetrics]:
        """Get average performance metrics for a model."""
        if model_name not in self.profiles or not self.profiles[model_name]:
            return None
        
        metrics_list = self.profiles[model_name]
        count = len(metrics_list)
        
        avg_execution_time = sum(m.execution_time for m in metrics_list) / count
        avg_memory_usage = sum(m.memory_usage_mb for m in metrics_list) / count
        avg_cpu_usage = sum(m.cpu_usage_percent for m in metrics_list) / count
        
        avg_gpu_memory = None
        gpu_metrics = [m.gpu_memory_mb for m in metrics_list if m.gpu_memory_mb is not None]
        if gpu_metrics:
            avg_gpu_memory = sum(gpu_metrics) / len(gpu_metrics)
        
        avg_fps = sum(m.fps for m in metrics_list if m.fps is not None) / count
        avg_throughput = sum(m.throughput for m in metrics_list if m.throughput is not None) / count
        
        return PerformanceMetrics(
            execution_time=avg_execution_time,
            memory_usage_mb=avg_memory_usage,
            cpu_usage_percent=avg_cpu_usage,
            gpu_memory_mb=avg_gpu_memory,
            fps=avg_fps,
            throughput=avg_throughput
        )
    
    def print_summary(self, model_name: str):
        """Print performance summary for a model."""
        if model_name not in self.profiles:
            print(f"No profile data available for {model_name}")
            return
        
        avg_metrics = self.get_average_metrics(model_name)
        if not avg_metrics:
            return
        
        print(f"\n{model_name} Performance Summary:")
        print("=" * 50)
        print(f"Average execution time: {avg_metrics.execution_time:.4f} seconds")
        print(f"Average FPS: {avg_metrics.fps:.2f}")
        print(f"Average memory usage: {avg_metrics.memory_usage_mb:.1f} MB")
        print(f"Average CPU usage: {avg_metrics.cpu_usage_percent:.1f}%")
        
        if avg_metrics.gpu_memory_mb is not None:
            print(f"Average GPU memory: {avg_metrics.gpu_memory_mb:.1f} MB")
        
        print(f"Total inferences: {len(self.profiles[model_name])}")
        print("=" * 50)


def benchmark_model_inference(model, test_data, num_iterations: int = 100) -> Dict[str, float]:
    """Benchmark model inference performance."""
    profiler = InferenceProfiler()
    
    print(f"Benchmarking model inference ({num_iterations} iterations)...")
    
    for i in range(num_iterations):
        with profiler.profile_inference("benchmark_model", len(test_data)):
            _ = model(test_data)
        
        if (i + 1) % 20 == 0:
            print(f"Completed {i + 1}/{num_iterations} iterations")
    
    avg_metrics = profiler.get_average_metrics("benchmark_model")
    
    if avg_metrics:
        return {
            'avg_execution_time': avg_metrics.execution_time,
            'avg_fps': avg_metrics.fps,
            'avg_memory_usage_mb': avg_metrics.memory_usage_mb,
            'avg_cpu_usage_percent': avg_metrics.cpu_usage_percent,
            'avg_gpu_memory_mb': avg_metrics.gpu_memory_mb or 0.0
        }
    
    return {}


def optimize_inference_parameters(model, test_data, batch_sizes: List[int] = None) -> Dict[int, Dict[str, float]]:
    """Find optimal inference parameters."""
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16]
    
    results = {}
    
    print("Optimizing inference parameters...")
    
    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        
        try:
            # Create batched test data
            if hasattr(test_data, 'repeat'):
                batched_data = test_data.repeat(batch_size, 1, 1, 1)
            else:
                batched_data = test_data
            
            benchmark_results = benchmark_model_inference(model, batched_data, 50)
            
            # Calculate throughput per image
            if benchmark_results.get('avg_fps', 0) > 0:
                throughput_per_image = benchmark_results['avg_fps'] / batch_size
                benchmark_results['throughput_per_image'] = throughput_per_image
            
            results[batch_size] = benchmark_results
            
        except Exception as e:
            print(f"Failed to test batch size {batch_size}: {e}")
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Testing performance utilities...")
    
    # Test performance timer
    with performance_timer("Test Operation"):
        time.sleep(1)
        # Simulate some work
        data = [i ** 2 for i in range(1000000)]
    
    # Test FPS counter
    fps_counter = FPSCounter()
    for i in range(30):
        time.sleep(0.033)  # Simulate ~30 FPS
        fps = fps_counter.tick()
        if i % 10 == 0:
            print(f"Current FPS: {fps:.2f}")
    
    print("Performance utilities test completed.")