"""
لاگ مرجع شبیه‌ساز (Ground-truth log).

این لاگ کاملاً جدا از چیزی است که روی شبکه رد و بدل می‌شود: فقط
ارکستریتور (origin) و خودِ گره‌ها (receive) به آن می‌نویسند، و شامل
مبدأ واقعی و زمان واقعی است — چیزی که نه گره‌های عادی و نه جاسوس‌ها
از طریق بسته نمی‌بینند.

چند پردازه هم‌زمان روی این فایل می‌نویسند. چون هر رکورد یک خط کوتاه
JSON است، نوشتن با فلگ O_APPEND روی لینوکس برای رکوردهای کوچک (زیر
PIPE_BUF) اتمیک است؛ برای این شبیه‌ساز آموزشی از قفل فایل جداگانه
صرف‌نظر شده است.
"""

import json
import time


def log_event(log_path: str, event: dict, lock=None) -> None:
    event.setdefault("wall_time", time.time())
    line = json.dumps(event, ensure_ascii=False) + "\n"
    
    # اگر قفل ارسال شده بود، ابتدا قفل را بگیر و بعد بنویس
    if lock:
        with lock:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line)
    else:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line)


def read_log(log_path: str):
    events = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return events
