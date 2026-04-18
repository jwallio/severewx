"""Date helpers."""

from __future__ import annotations

from datetime import date, datetime, timedelta


def parse_ymd(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def parse_cycle(value: str) -> int:
    hour = int(value)
    if hour not in {0, 6, 12, 18}:
        raise ValueError("cycle must be one of 00, 06, 12, 18")
    return hour


def cycle_datetime(value_date: str, value_cycle: str) -> datetime:
    d = parse_ymd(value_date)
    return datetime(d.year, d.month, d.day, parse_cycle(value_cycle))


def iter_dates(start: str, end: str) -> list[date]:
    current = parse_ymd(start)
    stop = parse_ymd(end)
    dates: list[date] = []
    while current <= stop:
        dates.append(current)
        current += timedelta(days=1)
    return dates
