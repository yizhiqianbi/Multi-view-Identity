import time
import torch

# 模块级的私有字典，用于存储所有计时器实例
_TIMERS = {}

class Timer:
    """
    一个用于测量代码块执行时间的计时器。
    它能自动选择使用 CUDA Events（用于精确 GPU 计时）或标准 time 模块（用于 CPU）。
    """
    def __init__(self, name: str, use_gpu: bool = True):
        self.name = name
        self._use_gpu = use_gpu and torch.cuda.is_available()
        self.reset()

        if self._use_gpu:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)

    def reset(self):
        """重置计时器状态。"""
        self.total_time_ms = 0.0
        self.count = 0

    def start(self):
        """标记计时的开始。"""
        if self._use_gpu:
            # 在 CUDA 流中记录一个事件
            self.start_event.record()
        else:
            self.start_cpu_time = time.perf_counter()

    def stop(self) -> float:
        """
        标记计时的结束，并返回本次计时的耗时（毫秒）。
        这个操作会强制同步 CUDA 流以获得准确时间。
        """
        if self._use_gpu:
            self.end_event.record()
            # 等待直到 end_event 之前的所有 CUDA 核心任务完成
            torch.cuda.synchronize()
            # 计算两个事件之间的毫秒数
            elapsed_ms = self.start_event.elapsed_time(self.end_event)
        else:
            end_cpu_time = time.perf_counter()
            elapsed_ms = (end_cpu_time - self.start_cpu_time) * 1000.0
        
        self.total_time_ms += elapsed_ms
        self.count += 1
        return elapsed_ms

    def average_ms(self) -> float:
        """返回平均耗时（毫秒）。"""
        return self.total_time_ms / self.count if self.count > 0 else 0.0

class _Timers:
    """
    一个管理所有 Timer 实例的单例访问类。
    通过 `__getattr__` 魔法方法，可以在首次访问时动态创建计时器。
    例如 `timers.step_time` 会自动创建或返回名为 'step_time' 的计时器。
    """
    def __init__(self):
        self._use_gpu = torch.cuda.is_available()

    def __getattr__(self, name: str) -> Timer:
        if name not in _TIMERS:
            _TIMERS[name] = Timer(name, use_gpu=self._use_gpu)
        return _TIMERS[name]
    
    def __call__(self, name: str) -> Timer:
        """允许使用函数调用语法获取计时器，如 timers('step_time')。"""
        return self.__getattr__(name)

    def reset(self):
        """重置所有已创建的计时器。"""
        for timer in _TIMERS.values():
            timer.reset()

# 创建一个全局单例
_global_timers_instance = _Timers()

def get_timers() -> _Timers:
    """
    获取全局计时器管理器实例。
    这是从任何文件访问计时器的推荐方式。
    """
    return _global_timers_instance