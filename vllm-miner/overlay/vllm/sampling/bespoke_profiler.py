# profiler.py
import time
import atexit
import functools

# Global registry for all profiled methods
_profile_data: dict[str, dict[str, float]] = {}

def profiler(fn):
    """
    Decorator to profile how many times fn is called and 
    its total/min/max elapsed time.
    
    Attach it to any method with @profiler.
    """
    qual = fn.__qualname__
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        dt = time.perf_counter() - t0
        
        stats = _profile_data.setdefault(qual, {
            "count": 0,
            "total": 0.0,
            "min": float("inf"),
            "max": 0.0,
        })
        stats["count"] += 1
        stats["total"] += dt
        stats["min"] = min(stats["min"], dt)
        stats["max"] = max(stats["max"], dt)
        
        return result
    return wrapper

def _print_profile_summary():
    if not _profile_data:
        print("No profiling data collected.")
        return
    print("\n=== Profiling Summary ===")
    for qual, s in _profile_data.items():
        avg = s["total"] / s["count"]
        print(
            f"{qual:50s} "
            f"calls={s['count']:5d}  "
            f"avg={avg*1e3:6.2f} ms  "
            f"min={s['min']*1e3:6.2f} ms  "
            f"max={s['max']*1e3:6.2f} ms"
        )
    print("=========================\n")

# print automatically on normal exit
atexit.register(_print_profile_summary)
