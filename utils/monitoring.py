"""
Monitoring utilities for LLM tracing and performance tracking.
"""

import os
import logging
from typing import Dict, Any, Optional
from functools import wraps

logger = logging.getLogger(__name__)


class OpikTracer:
    """Simple Opik integration for LLM call tracing."""

    def __init__(self):
        """Initialize Opik tracer."""
        self.opik_available = False

        if os.getenv('OPIK_API_KEY'):
            try:
                import opik
                self.opik = opik
                self.opik_available = True
                logger.info("Opik monitoring enabled")
            except ImportError:
                logger.warning("Opik package not installed")
            except Exception as e:
                logger.error(f"Error initializing Opik: {e}")
        else:
            logger.info("Opik monitoring disabled (no API key)")

    def trace_llm_call(self, operation_name: str):
        """Decorator for tracing LLM calls."""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                if self.opik_available:
                    # Use Opik tracing
                    with self.opik.trace(name=operation_name) as trace:
                        try:
                            result = await func(*args, **kwargs)
                            trace.log_success()
                            return result
                        except Exception as e:
                            trace.log_error(e)
                            raise
                else:
                    # No tracing
                    return await func(*args, **kwargs)

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                if self.opik_available:
                    # Use Opik tracing
                    with self.opik.trace(name=operation_name) as trace:
                        try:
                            result = func(*args, **kwargs)
                            trace.log_success()
                            return result
                        except Exception as e:
                            trace.log_error(e)
                            raise
                else:
                    # No tracing
                    return func(*args, **kwargs)

            # Return appropriate wrapper based on function type
            import asyncio
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

        return decorator

    def log_metric(self, name: str, value: float, metadata: Optional[Dict[str, Any]] = None):
        """Log a custom metric."""
        if self.opik_available:
            try:
                # Log to Opik if available
                pass  # Opik doesn't have direct metric logging in this version
            except Exception as e:
                logger.warning(f"Error logging metric to Opik: {e}")

        # Always log to local logging
        logger.info(f"Metric: {name} = {value}", extra=metadata or {})


# Global tracer instance
opik_tracer = OpikTracer()


def trace_llm_operation(operation_name: str):
    """Decorator for tracing LLM operations."""
    return opik_tracer.trace_llm_call(operation_name)


class PerformanceMonitor:
    """Simple performance monitoring for operations."""

    def __init__(self):
        """Initialize performance monitor."""
        self.metrics = {}

    def start_timer(self, operation: str) -> str:
        """Start timing an operation."""
        import time
        timer_id = f"{operation}_{time.time()}"
        self.metrics[timer_id] = {
            'operation': operation,
            'start_time': time.time(),
            'status': 'running'
        }
        return timer_id

    def end_timer(self, timer_id: str, success: bool = True, metadata: Optional[Dict[str, Any]] = None):
        """End timing an operation."""
        import time
        if timer_id in self.metrics:
            metric = self.metrics[timer_id]
            metric['end_time'] = time.time()
            metric['duration'] = metric['end_time'] - metric['start_time']
            metric['status'] = 'success' if success else 'error'
            if metadata:
                metric['metadata'] = metadata

            # Log to Opik
            opik_tracer.log_metric(
                f"{metric['operation']}_duration",
                metric['duration'],
                {'status': metric['status']}
            )

            logger.info(f"Operation {metric['operation']} completed in {metric['duration']:.3f}s")
            return metric

        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.metrics:
            return {'message': 'No operations tracked yet'}

        operations = {}
        for timer_id, metric in self.metrics.items():
            op_name = metric['operation']
            if op_name not in operations:
                operations[op_name] = {
                    'count': 0,
                    'total_time': 0,
                    'success_count': 0,
                    'error_count': 0
                }

            operations[op_name]['count'] += 1
            operations[op_name]['total_time'] += metric.get('duration', 0)

            if metric['status'] == 'success':
                operations[op_name]['success_count'] += 1
            else:
                operations[op_name]['error_count'] += 1

        # Calculate averages
        for op_name in operations:
            op = operations[op_name]
            op['avg_time'] = op['total_time'] / op['count']
            op['success_rate'] = op['success_count'] / op['count']

        return {
            'total_operations': len(self.metrics),
            'operations': operations
        }


# Global performance monitor
performance_monitor = PerformanceMonitor()