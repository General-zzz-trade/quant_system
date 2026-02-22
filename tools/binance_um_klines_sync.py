from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


BINANCE_HOST = "https://data.binance.vision"


def _http_download(url: str, dst: Path, timeout: int = 300) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return
    tmp = dst.with_suffix(dst.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(dst)


def _try_download(url: str, dst: Path) -> bool:
    try:
        _http_download(url, dst)
        return True
    except Exception as e:
        msg = str(e)
        if "HTTP Error 404" in msg or "404" in msg:
            return False
        print(f"[warn] download failed: {url} err={e}")
        return False


def _extract_single_csv(zip_path: Path, out_dir: Path) -> Optional[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            csv_names = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                return None
            name = csv_names[0]
            target = out_dir / Path(name).name
            if target.exists() and target.stat().st_size > 0:
                return target
            with z.open(name) as src, target.open("wb") as dst:
                while True:
                    buf = src.read(1024 * 1024)
                    if not buf:
                        break
                    dst.write(buf)
            return target
    except zipfile.BadZipFile:
        print(f"[warn] bad zip: {zip_path}")
        return None


def _convert_binance_kline_csv_to_ohlcv(inp_csv: Path, out_csv: Path) -> bool:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists() and out_csv.stat().st_size > 0:
        return True

    try:
        with inp_csv.open("r", newline="", encoding="utf-8") as f_in, out_csv.open(
            "w", newline="", encoding="utf-8"
        ) as f_out:
            r = csv.reader(f_in)
            w = csv.writer(f_out)
            w.writerow(["ts", "open", "high", "low", "close", "volume"])
            for row in r:
                if not row or len(row) < 6:
                    continue
                w.writerow([row[0], row[1], row[2], row[3], row[4], row[5]])
        return True
    except Exception as e:
        print(f"[warn] convert failed: {inp_csv} err={e}")
        return False


def _merge_ohlcv_csvs(inputs: Iterable[Path], merged_out: Path) -> None:
    merged_out.parent.mkdir(parents=True, exist_ok=True)
    parts = sorted(set(Path(p) for p in inputs), key=lambda p: p.name)
    if not parts:
        raise RuntimeError("No OHLCV parts to merge.")

    with merged_out.open("w", newline="", encoding="utf-8") as f_out:
        w = csv.writer(f_out)
        w.writerow(["ts", "open", "high", "low", "close", "volume"])
        for p in parts:
            with p.open("r", newline="", encoding="utf-8") as f_in:
                r = csv.reader(f_in)
                first = True
                for row in r:
                    if first:
                        first = False
                        continue
                    if row:
                        w.writerow(row)


def _utc_today() -> dt.date:
    return dt.datetime.now(dt.timezone.utc).date()


def _month_iter(start_ym: str, end_ym: str) -> Iterable[tuple[int, int]]:
    sy, sm = start_ym.split("-")
    ey, em = end_ym.split("-")
    y, m = int(sy), int(sm)
    y2, m2 = int(ey), int(em)
    while (y, m) <= (y2, m2):
        yield y, m
        m += 1
        if m == 13:
            y += 1
            m = 1


def _last_complete_utc_month(today: dt.date) -> str:
    first = dt.date(today.year, today.month, 1)
    last_prev = first - dt.timedelta(days=1)
    return f"{last_prev.year:04d}-{last_prev.month:02d}"


def _days_inclusive(a: dt.date, b: dt.date) -> Iterable[dt.date]:
    cur = a
    while cur <= b:
        yield cur
        cur += dt.timedelta(days=1)


@dataclass(frozen=True)
class Layout:
    root: Path
    symbol: str
    interval: str

    @property
    def base(self) -> Path:
        return self.root / "futures" / "um" / "klines" / self.symbol / self.interval

    @property
    def monthly_zip(self) -> Path:
        return self.base / "monthly" / "zips"

    @property
    def monthly_csv(self) -> Path:
        return self.base / "monthly" / "csv"

    @property
    def monthly_ohlcv(self) -> Path:
        return self.base / "monthly" / "ohlcv"

    @property
    def daily_zip(self) -> Path:
        return self.base / "daily" / "zips"

    @property
    def daily_csv(self) -> Path:
        return self.base / "daily" / "csv"

    @property
    def daily_ohlcv(self) -> Path:
        return self.base / "daily" / "ohlcv"

    @property
    def merged_out(self) -> Path:
        return self.root / "ohlcv" / f"{self.symbol}_{self.interval}_ohlcv.csv"


def sync(
    root: Path,
    symbol: str,
    interval: str,
    start_month: str,
    do_monthly: bool,
    do_daily: bool,
    do_extract: bool,
    do_convert: bool,
    do_merge: bool,
) -> Path:
    lay = Layout(root=root, symbol=symbol, interval=interval)
    parts: List[Path] = []

    today = _utc_today()
    end_month = _last_complete_utc_month(today)

    if do_monthly:
        prefix = f"data/futures/um/monthly/klines/{symbol}/{interval}/"
        for y, m in _month_iter(start_month, end_month):
            fname = f"{symbol}-{interval}-{y:04d}-{m:02d}.zip"
            url = f"{BINANCE_HOST}/{prefix}{fname}"
            dst = lay.monthly_zip / fname
            ok = _try_download(url, dst)
            if not ok:
                continue

            if do_extract:
                csv_path = _extract_single_csv(dst, lay.monthly_csv)
                if csv_path and do_convert:
                    out = lay.monthly_ohlcv / (csv_path.stem + "_ohlcv.csv")
                    if _convert_binance_kline_csv_to_ohlcv(csv_path, out):
                        parts.append(out)

    if do_daily:
        prefix = f"data/futures/um/daily/klines/{symbol}/{interval}/"
        start_day = dt.date(today.year, today.month, 1)
        end_day = today - dt.timedelta(days=1)
        if end_day >= start_day:
            for d in _days_inclusive(start_day, end_day):
                fname = f"{symbol}-{interval}-{d.year:04d}-{d.month:02d}-{d.day:02d}.zip"
                url = f"{BINANCE_HOST}/{prefix}{fname}"
                dst = lay.daily_zip / fname
                ok = _try_download(url, dst)
                if not ok:
                    continue

                if do_extract:
                    csv_path = _extract_single_csv(dst, lay.daily_csv)
                    if csv_path and do_convert:
                        out = lay.daily_ohlcv / (csv_path.stem + "_ohlcv.csv")
                        if _convert_binance_kline_csv_to_ohlcv(csv_path, out):
                            parts.append(out)

    if do_merge:
        if not parts:
            raise RuntimeError("No converted OHLCV parts found. Use --extract --convert first.")
        _merge_ohlcv_csvs(parts, lay.merged_out)

    return lay.merged_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/binance")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="1m")
    ap.add_argument("--start-month", default="2019-09", help="YYYY-MM, default: 2019-09")
    ap.add_argument("--monthly", action="store_true")
    ap.add_argument("--daily", action="store_true")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--convert", action="store_true")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    if not args.monthly and not args.daily:
        print("You must pass --monthly and/or --daily")
        raise SystemExit(2)

    out = sync(
        root=Path(args.root),
        symbol=args.symbol,
        interval=args.interval,
        start_month=args.start_month,
        do_monthly=args.monthly,
        do_daily=args.daily,
        do_extract=args.extract,
        do_convert=args.convert,
        do_merge=args.merge,
    )
    print(f"Done. Merged OHLCV file: {out}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.")
        sys.exit(130)
