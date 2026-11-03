"""线程间"最新者赢"单槽缓冲。

管线里不同模块频率不同（采集 30fps / 热图 10fps / 主色 10Hz），消费方永远只关心
最新的那份中间结果，而不是按序消费——所以用单槽覆盖语义，而不是队列。
"""

from __future__ import annotations

import threading
from typing import Any


class LatestSlot:
    """单槽"最新者赢"缓冲：生产者覆盖、消费者取最新。线程安全（CPython 引用读写原子）。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seq = 0
        self._data: Any = None

    def push(self, data: Any) -> int:
        with self._lock:
            self._seq += 1
            self._data = data
            return self._seq

    def get(self) -> tuple[int, Any]:
        """返回 (seq, data)；无数据时 data 为 None。约定：数据不可变（生产者只换新引用）。"""
        with self._lock:
            return self._seq, self._data
