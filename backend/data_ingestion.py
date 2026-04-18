"""
Data Ingestion Module
Fetches stock data from Yahoo Finance, news sentiment, and generates synthetic macroeconomic features
"""

import pandas as pd
import numpy as np
import yfinance as yf
from io import BytesIO
from datetime import datetime, timedelta
from typing import Tuple, Dict, List, Optional
from pathlib import Path
import re
import logging
import threading
import warnings

warnings.filterwarnings("ignore")

from .utils import get_ticker_sector

logger = logging.getLogger(__name__)

# NIFTY 50 constituent stocks (NSE tickers)
NIFTY50_STOCKS = {
    "ADANIENT.NS": "Adani Enterprises",
    "ADANIPORTS.NS": "Adani Ports",
    "APOLLOHOSP.NS": "Apollo Hospitals",
    "ASIANPAINT.NS": "Asian Paints",
    "AXISBANK.NS": "Axis Bank",
    "BAJAJ-AUTO.NS": "Bajaj Auto",
    "BAJFINANCE.NS": "Bajaj Finance",
    "BAJAJFINSV.NS": "Bajaj Finserv",
    "BPCL.NS": "Bharat Petroleum",
    "BHARTIARTL.NS": "Bharti Airtel",
    "BRITANNIA.NS": "Britannia",
    "CIPLA.NS": "Cipla",
    "COALINDIA.NS": "Coal India",
    "DIVISLAB.NS": "Divi's Laboratories",
    "DRREDDY.NS": "Dr. Reddy's Laboratories",
    "EICHERMOT.NS": "Eicher Motors",
    "GRASIM.NS": "Grasim Industries",
    "HCLTECH.NS": "HCL Technologies",
    "HDFCBANK.NS": "HDFC Bank",
    "HDFCLIFE.NS": "HDFC Life",
    "HEROMOTOCO.NS": "Hero MotoCorp",
    "HINDALCO.NS": "Hindalco",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "ICICIBANK.NS": "ICICI Bank",
    "INDUSINDBK.NS": "IndusInd Bank",
    "INFY.NS": "Infosys",
    "ITC.NS": "ITC Limited",
    "JSWSTEEL.NS": "JSW Steel",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "LT.NS": "Larsen & Toubro",
    "M&M.NS": "Mahindra & Mahindra",
    "MARUTI.NS": "Maruti Suzuki",
    "NESTLEIND.NS": "Nestle India",
    "NTPC.NS": "NTPC Limited",
    "ONGC.NS": "ONGC",
    "POWERGRID.NS": "Power Grid Corporation",
    "RELIANCE.NS": "Reliance Industries",
    "SBIN.NS": "State Bank of India",
    "SBILIFE.NS": "SBI Life",
    "SUNPHARMA.NS": "Sun Pharmaceutical",
    "TCS.NS": "Tata Consultancy Services",
    "TATACONSUM.NS": "Tata Consumer Products",
    "TATAMOTORS.NS": "Tata Motors",
    "TATASTEEL.NS": "Tata Steel",
    "TECHM.NS": "Tech Mahindra",
    "TITAN.NS": "Titan",
    "ULTRACEMCO.NS": "UltraTech Cement",
    "UPL.NS": "UPL",
    "WIPRO.NS": "Wipro",
}


class DataIngestion:
    """Handles all data ingestion tasks"""

    @staticmethod
    def _unique_paths(paths: List[Path]) -> List[Path]:
        """Return unique paths while preserving order."""
        seen: set[str] = set()
        unique: List[Path] = []
        for path in paths:
            key = str(path.resolve()) if path.exists() else str(path)
            if key in seen:
                continue
            seen.add(key)
            unique.append(path)
        return unique

    @staticmethod
    def _parse_dates_mixed(series: pd.Series) -> pd.Series:
        """Parse mixed date formats without dropping valid ISO rows."""
        first_pass = pd.to_datetime(series, errors="coerce", dayfirst=False)
        missing = first_pass.isna()
        if missing.any():
            second_pass = pd.to_datetime(
                series[missing], errors="coerce", dayfirst=True
            )
            first_pass.loc[missing] = second_pass
        return first_pass

    def __init__(self, lookback_days: int = 500):
        """
        Initialize data ingestion

        Args:
            lookback_days: Number of historical days to fetch
        """
        self.lookback_days = lookback_days
        # Keep enough rows for feature windows + train/test split when possible.
        self.min_training_rows = 180
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=lookback_days)

        # Prefer the project-wide NIFTY dataset first, then stock-specific CSVs, then Yahoo Finance.
        project_root = Path(__file__).resolve().parents[1]
        workspace_root = project_root.parent
        cwd = Path.cwd().resolve()

        root_candidates = self._unique_paths(
            [
                project_root,
                workspace_root,
                workspace_root.parent,
                cwd,
                cwd.parent,
            ]
        )

        project_dataset_candidates = self._unique_paths(
            [
                workspace_root / "dataset",
                workspace_root.parent / "ML" / "dataset",
                project_root / "dataset",
                cwd / "dataset",
            ]
        )
        self.project_dataset_dir = next(
            (
                candidate
                for candidate in project_dataset_candidates
                if candidate.exists()
            ),
            project_dataset_candidates[0],
        )
        self.use_project_dataset = False
        self.local_data_candidates = self._unique_paths(
            [
                project_root / "data" / "external_nifty50",
                project_root / "data" / "raw",
                workspace_root / "data" / "external_nifty50",
                cwd / "data" / "external_nifty50",
            ]
            + [root / "nifty50" / "NIFTY 50" for root in root_candidates]
            + [root / "NIFTY 50" for root in root_candidates]
            + [root / "data" / "raw" for root in root_candidates]
        )
        self._all_tickers_cache: Optional[List[str]] = None
        self._project_symbol_file_map: Optional[Dict[str, List[Path]]] = None
        self._project_file_cache: Dict[Path, pd.DataFrame] = {}
        self._cache_lock = threading.RLock()

    def _target_window_rows(self) -> int:
        """Target row window balancing requested lookback and training viability."""
        return max(int(self.lookback_days), int(self.min_training_rows))

    def _ensure_min_training_window(
        self, selected: pd.DataFrame, full_history: pd.DataFrame
    ) -> pd.DataFrame:
        """Backfill with older rows from full history when selected window is too short."""
        if selected is None or selected.empty:
            return selected
        if full_history is None or full_history.empty:
            return selected

        if len(selected) >= self.min_training_rows:
            return selected

        target_rows = self._target_window_rows()
        if len(full_history) <= len(selected):
            return selected

        return full_history.tail(target_rows).reset_index(drop=True)

    def _clean_ohlcv_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standard OHLCV cleaning used across all data sources."""
        frame = data.copy()
        frame.columns = [str(col).strip().lower() for col in frame.columns]
        frame = frame.loc[
            :, [not re.match(r"^unnamed(:\s*\d+)?$", str(col)) for col in frame.columns]
        ]

        if "date" in frame.columns:
            frame["date"] = self._parse_dates_mixed(frame["date"])

        numeric_cols = ["open", "high", "low", "close", "volume", "adj close"]
        for col in numeric_cols:
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")

        required = [
            c
            for c in ["date", "open", "high", "low", "close", "volume"]
            if c in frame.columns
        ]
        if required:
            frame = frame.dropna(subset=required)

        if "volume" in frame.columns:
            frame["volume"] = frame["volume"].clip(lower=0)

        if "date" in frame.columns:
            # Keep one row per trading date per symbol when symbol data exists.
            # A date-only dedupe collapses multi-symbol chunk files and drops most
            # rows for individual tickers.
            dedupe_cols = ["date", "symbol"] if "symbol" in frame.columns else ["date"]
            frame = frame.sort_values("date").drop_duplicates(
                subset=dedupe_cols, keep="last"
            )

        if "symbol" in frame.columns:
            frame["symbol"] = (
                frame["symbol"]
                .astype(str)
                .str.strip()
                .str.upper()
                .replace({"": np.nan})
            )

        if "adj close" not in frame.columns and "close" in frame.columns:
            frame["adj close"] = frame["close"]

        return frame.reset_index(drop=True)

    @staticmethod
    def _normalize_name(value: str) -> str:
        """Normalize names for robust filename matching."""
        return re.sub(r"[^A-Z0-9]+", "", value.upper())

    def _detect_file_format(self, content: bytes, filename: str) -> str:
        """Infer uploaded file format from filename extension."""
        suffix = Path(filename or "").suffix.lower()
        if suffix in {".csv", ".txt"}:
            return "csv"
        if suffix == ".tsv":
            return "tsv"
        if suffix == ".json":
            return "json"
        if suffix == ".xlsx":
            return "xlsx"
        if suffix == ".parquet":
            return "parquet"
        if suffix == ".feather":
            return "feather"
        return "csv"

    def _parse_any_format(self, content: bytes, filename: str) -> pd.DataFrame:
        """Parse uploaded bytes into a dataframe using detected file format."""
        file_format = self._detect_file_format(content, filename)
        buffer = BytesIO(content)

        if file_format == "csv":
            return pd.read_csv(buffer, sep=None, engine="python")
        if file_format == "tsv":
            return pd.read_csv(buffer, sep="\t")
        if file_format == "json":
            return pd.read_json(buffer)
        if file_format == "xlsx":
            return pd.read_excel(buffer)
        if file_format == "parquet":
            return pd.read_parquet(buffer)
        if file_format == "feather":
            return pd.read_feather(buffer)

        return pd.read_csv(buffer, sep=None, engine="python")

    def _normalize_ohlcv_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize common OHLCV aliases into canonical column names."""
        normalized = df.copy()
        alias_map = {
            "open": "open",
            "openprice": "open",
            "open_price": "open",
            "o": "open",
            "high": "high",
            "highprice": "high",
            "high_price": "high",
            "h": "high",
            "dayhigh": "high",
            "low": "low",
            "lowprice": "low",
            "low_price": "low",
            "l": "low",
            "daylow": "low",
            "close": "close",
            "closeprice": "close",
            "close_price": "close",
            "c": "close",
            "ltp": "close",
            "last": "close",
            "lastprice": "close",
            "adjclose": "adj close",
            "adj_close": "adj close",
            "adjustedclose": "adj close",
            "volume": "volume",
            "vol": "volume",
            "tradedquantity": "volume",
            "tottrdqty": "volume",
            "tot_trd_qty": "volume",
            "quantity": "volume",
            "shares": "volume",
            "date": "date",
            "timestamp": "date",
            "time": "date",
            "datetime": "date",
            "trade_date": "date",
            "tradingdate": "date",
            "symbol": "symbol",
            "ticker": "symbol",
            "tradingsymbol": "symbol",
            "security": "symbol",
            "instrument": "symbol",
            "company": "symbol",
        }

        rename_map: Dict[str, str] = {}
        for col in normalized.columns:
            key = re.sub(r"[^a-z0-9_]+", "", str(col).strip().lower().replace(" ", "_"))
            if key in alias_map:
                rename_map[col] = alias_map[key]
        normalized = normalized.rename(columns=rename_map)
        normalized = self._coalesce_duplicate_columns(normalized)

        if "date" not in normalized.columns and len(normalized.columns) > 0:
            normalized = normalized.rename(columns={normalized.columns[0]: "date"})

        return normalized

    @staticmethod
    def _coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Merge duplicate column names by taking first non-null value across duplicates."""
        if df.columns.is_unique:
            return df

        out = pd.DataFrame(index=df.index)
        for col_name in dict.fromkeys(df.columns):
            col_data = df.loc[:, df.columns == col_name]
            if isinstance(col_data, pd.DataFrame):
                out[col_name] = col_data.bfill(axis=1).iloc[:, 0]
            else:
                out[col_name] = col_data
        return out

    @staticmethod
    def _normalize_price_frame(data: pd.DataFrame) -> pd.DataFrame:
        """Normalize OHLCV frame column names and basic column aliases."""
        normalized = data.copy()
        normalized.columns = [str(col).strip().lower() for col in normalized.columns]
        normalized = normalized.loc[
            :,
            [
                not re.match(r"^unnamed(:\s*\d+)?$", str(col))
                for col in normalized.columns
            ],
        ]
        normalized = normalized.rename(
            columns={"adjclose": "adj close", "adj_close": "adj close"}
        )
        return normalized

    @staticmethod
    def _is_valid_ohlcv(data: pd.DataFrame) -> bool:
        """Check whether a dataframe has the minimum OHLCV columns."""
        required_cols = {"date", "open", "high", "low", "close", "volume"}
        return (
            data is not None
            and not data.empty
            and required_cols.issubset(set(data.columns))
        )

    def _validate_ohlcv(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate parsed OHLCV dataframe and return warnings."""
        warnings_list: List[str] = []
        if df is None or df.empty:
            return False, ["Dataset is empty"]

        if len(df) < 120:
            warnings_list.append(
                "Dataset has fewer than 120 rows; LSTM may be skipped."
            )

        if "close" not in df.columns:
            return False, ["Missing required close column"]
        if not any(col in df.columns for col in ["open", "high", "low"]):
            return False, ["Need at least one of open/high/low columns"]

        missing_ratio = float(df.isna().mean().mean())
        if missing_ratio > 0.05:
            warnings_list.append("More than 5% missing values detected.")

        numeric_cols = [
            c for c in ["open", "high", "low", "close", "volume"] if c in df.columns
        ]
        if numeric_cols:
            z = np.abs(
                (df[numeric_cols] - df[numeric_cols].mean())
                / (df[numeric_cols].std() + 1e-9)
            )
            outlier_ratio = float((z > 3).mean().mean())
            if outlier_ratio > 0.10:
                warnings_list.append("More than 10% outlier values (3-sigma) detected.")

        return True, warnings_list

    def _fill_missing_ohlc(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing OHLC using close-based fallback rules."""
        out = df.copy()
        if "close" in out.columns:
            close_col = out["close"]
            if isinstance(close_col, pd.DataFrame):
                close_col = close_col.bfill(axis=1).iloc[:, 0]
            out["close"] = pd.to_numeric(close_col, errors="coerce").ffill().bfill()
            for col in ["open", "high", "low"]:
                if col not in out.columns:
                    out[col] = out["close"]
                col_values = out[col]
                if isinstance(col_values, pd.DataFrame):
                    col_values = col_values.bfill(axis=1).iloc[:, 0]
                out[col] = pd.to_numeric(col_values, errors="coerce").fillna(
                    out["close"]
                )
        return out

    def _synthesize_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """Synthesize missing volume as ones for feature compatibility."""
        out = df.copy()
        if "volume" not in out.columns:
            out["volume"] = 1.0
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(1.0)
        return out

    def _standardize_ohlcv_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize a loaded dataframe into the expected OHLCV column layout."""
        frame = self._normalize_price_frame(data)

        if "date" not in frame.columns:
            for alias in ("datetime", "timestamp", "trade_date"):
                if alias in frame.columns:
                    frame = frame.rename(columns={alias: "date"})
                    break

        rename_map = {}
        for column in frame.columns:
            normalized = self._normalize_name(column)
            if normalized == "OPENPRICE":
                rename_map[column] = "open"
            elif normalized == "HIGHPRICE":
                rename_map[column] = "high"
            elif normalized == "LOWPRICE":
                rename_map[column] = "low"
            elif normalized == "CLOSEPRICE":
                rename_map[column] = "close"
            elif normalized in {"VOLUME", "VOLUMETRADED", "QTY", "QUANTITY"}:
                rename_map[column] = "volume"
            elif normalized in {
                "SYMBOL",
                "TICKER",
                "TRADINGSYMBOL",
                "SECURITY",
                "INSTRUMENT",
                "COMPANY",
            }:
                rename_map[column] = "symbol"

        if rename_map:
            frame = frame.rename(columns=rename_map)

        if "adj close" not in frame.columns and "close" in frame.columns:
            frame["adj close"] = frame["close"]

        return frame

    def _filter_uploaded_by_ticker(
        self, df: pd.DataFrame, ticker: str, warnings_list: List[str]
    ) -> Tuple[pd.DataFrame, str]:
        """Filter uploaded dataset to one symbol when a symbol column exists."""
        out = df.copy()
        requested = str(ticker or "").strip().upper()

        if "symbol" not in out.columns:
            inferred = requested or "UPLOADED_STOCK"
            return out, inferred

        symbol_series = out["symbol"].astype(str).str.strip()
        non_empty_symbols = symbol_series[symbol_series != ""]
        unique_symbols = list(dict.fromkeys(non_empty_symbols.tolist()))
        unique_norm = set(non_empty_symbols.map(self._normalize_name).tolist())

        if not unique_symbols:
            inferred = requested or "UPLOADED_STOCK"
            warnings_list.append(
                "Symbol column was present but empty; using full dataset as-is."
            )
            return out, inferred

        inferred = requested
        query_keys: set[str] = set()
        if requested:
            query_keys.update(self._normalize_ticker_queries(requested))
            requested_base = requested.split(".")[0]
            query_keys.add(self._normalize_name(requested_base))
            query_keys.add(self._normalize_name(requested))

        # Filter only when the upload contains multiple symbols and a requested ticker was provided.
        if len(unique_norm) > 1 and query_keys:
            symbol_norm = symbol_series.map(self._normalize_name)
            mask = symbol_norm.isin(query_keys)
            if mask.any():
                before_count = int(len(out))
                out = out.loc[mask].copy()
                warnings_list.append(
                    f"Upload contained multiple symbols. Filtered rows from {before_count} to {len(out)} for ticker {requested}."
                )
                return out, requested

        # If no request was provided or no rows matched request, select dominant symbol in multi-symbol uploads.
        if len(unique_norm) > 1:
            dominant = symbol_series.value_counts(dropna=False).idxmax()
            dominant_norm = self._normalize_name(str(dominant))
            out = out.loc[
                symbol_series.map(self._normalize_name) == dominant_norm
            ].copy()
            inferred = str(dominant).strip().upper() or "UPLOADED_STOCK"
            if requested:
                warnings_list.append(
                    f"Requested ticker {requested} not found in uploaded symbols; using dominant symbol {inferred}."
                )
            else:
                warnings_list.append(
                    f"Upload contained multiple symbols; using dominant symbol {inferred}."
                )
            return out, inferred

        # Single-symbol upload.
        if not requested:
            inferred = str(unique_symbols[0]).strip().upper() or "UPLOADED_STOCK"
            warnings_list.append(
                f"Ticker was not provided; inferred ticker {inferred} from upload."
            )
        else:
            inferred = requested

        return out, inferred

    def _build_upload_summary(
        self, df: pd.DataFrame, requested_ticker: str, resolved_ticker: str
    ) -> Dict[str, object]:
        """Create a compact summary for uploaded files with symbol and sector information."""
        summary: Dict[str, object] = {
            "requested_ticker": requested_ticker or "",
            "resolved_ticker": resolved_ticker or "",
            "rows_after_cleaning": int(len(df)),
            "date_range": None,
            "has_symbol_column": bool("symbol" in df.columns),
            "unique_symbols": 1,
            "unique_sectors": 1,
            "top_symbols": [],
            "top_sectors": [],
        }

        if "date" in df.columns and not df.empty:
            start_date = pd.to_datetime(df["date"].iloc[0], errors="coerce")
            end_date = pd.to_datetime(df["date"].iloc[-1], errors="coerce")
            if not pd.isna(start_date) and not pd.isna(end_date):
                summary["date_range"] = {
                    "start": str(start_date.date()),
                    "end": str(end_date.date()),
                }

        if "symbol" in df.columns:
            symbol_series = df["symbol"].astype(str).str.strip()
            symbol_series = symbol_series[symbol_series != ""]
            symbol_counts = symbol_series.value_counts().head(8)
            summary["unique_symbols"] = (
                int(symbol_series.nunique()) if not symbol_series.empty else 0
            )
            summary["top_symbols"] = [
                {"symbol": str(symbol), "rows": int(count)}
                for symbol, count in symbol_counts.items()
            ]

            sector_series = symbol_series.map(get_ticker_sector)
            sector_counts = (
                sector_series[sector_series != "Unknown"].value_counts().head(8)
            )
            summary["unique_sectors"] = (
                int(sector_series[sector_series != "Unknown"].nunique())
                if not sector_series.empty
                else 0
            )
            summary["top_sectors"] = [
                {"sector": str(sector), "rows": int(count)}
                for sector, count in sector_counts.items()
            ]
        else:
            sector_name = get_ticker_sector(resolved_ticker or requested_ticker)
            summary["top_sectors"] = [{"sector": sector_name, "rows": int(len(df))}]

        if not summary["top_sectors"]:
            sector_name = get_ticker_sector(resolved_ticker or requested_ticker)
            summary["top_sectors"] = [{"sector": sector_name, "rows": int(len(df))}]

        return summary

    def _normalize_ticker_queries(self, ticker: str) -> List[str]:
        """Build possible symbol matches for a ticker."""
        ticker_key = ticker.split(".")[0]
        company_name = NIFTY50_STOCKS.get(ticker, "")
        queries = [
            self._normalize_name(ticker),
            self._normalize_name(ticker_key),
            self._normalize_name(company_name),
        ]
        return [query for query in queries if query]

    def _build_project_symbol_file_map(self) -> Dict[str, List[Path]]:
        """Build normalized symbol -> project chunk file list mapping once."""
        with self._cache_lock:
            if self._project_symbol_file_map is not None:
                return self._project_symbol_file_map

        symbol_map: Dict[str, List[Path]] = {}
        if not self.project_dataset_dir.exists():
            with self._cache_lock:
                self._project_symbol_file_map = symbol_map
            return symbol_map

        csv_files = sorted(
            [
                *self.project_dataset_dir.glob("*.csv"),
                *self.project_dataset_dir.glob("*.CSV"),
            ]
        )

        for csv_path in csv_files:
            symbols: List[str] = []
            try:
                sym_df = pd.read_csv(csv_path, usecols=["SYMBOL"])
                symbols = sym_df["SYMBOL"].dropna().astype(str).str.strip().tolist()
            except Exception:
                try:
                    full_df = pd.read_csv(csv_path)
                    symbol_col = None
                    for col in full_df.columns:
                        if str(col).strip().upper() == "SYMBOL":
                            symbol_col = col
                            break
                    if symbol_col is None:
                        continue
                    symbols = (
                        full_df[symbol_col].dropna().astype(str).str.strip().tolist()
                    )
                except Exception:
                    continue

            for sym in symbols:
                norm_sym = self._normalize_name(sym)
                if not norm_sym:
                    continue
                symbol_map.setdefault(norm_sym, []).append(csv_path)

        with self._cache_lock:
            self._project_symbol_file_map = symbol_map
        return symbol_map

    def _get_project_chunk_data(self, csv_path: Path) -> Optional[pd.DataFrame]:
        """Load and cache standardized chunk data for repeated ticker filtering."""
        with self._cache_lock:
            cached = self._project_file_cache.get(csv_path)
        if cached is not None:
            return cached

        try:
            data = self._standardize_ohlcv_frame(pd.read_csv(csv_path))
            if "symbol" not in data.columns:
                return None

            data = self._clean_ohlcv_frame(data)
            data["_symbol_norm"] = data["symbol"].astype(str).map(self._normalize_name)
            with self._cache_lock:
                self._project_file_cache[csv_path] = data
            return data
        except Exception:
            logger.exception("Error reading project dataset chunk: %s", csv_path)
            return None

    def _load_from_project_dataset(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load ticker rows from the chunked project dataset under ML/dataset."""
        if not self.use_project_dataset:
            return None

        if not self.project_dataset_dir.exists():
            return None

        query_keys = self._normalize_ticker_queries(ticker)
        symbol_map = self._build_project_symbol_file_map()
        if not symbol_map:
            return None

        candidate_files: set[Path] = set()
        for key in query_keys:
            for csv_path in symbol_map.get(key, []):
                candidate_files.add(csv_path)
        if not candidate_files:
            return None

        matches = []
        for csv_path in sorted(candidate_files):
            data = self._get_project_chunk_data(csv_path)
            if data is None or "_symbol_norm" not in data.columns:
                continue

            symbol_norm = data["_symbol_norm"]
            if not symbol_norm.isin(query_keys).any():
                continue

            filtered = data.loc[symbol_norm.isin(query_keys)].copy()
            if filtered.empty:
                continue

            matches.append(filtered)

        if not matches:
            return None

        data = pd.concat(matches, ignore_index=True)
        if "_symbol_norm" in data.columns:
            data = data.drop(columns=["_symbol_norm"])
        data = data.sort_values("date").reset_index(drop=True)

        filtered = data[data["date"] >= self.start_date].reset_index(drop=True)
        if filtered.empty:
            filtered = data.tail(self.lookback_days).reset_index(drop=True)

        data = self._ensure_min_training_window(filtered, data)

        if data.empty:
            return None

        if "adj close" not in data.columns:
            data["adj close"] = data["close"]

        logger.debug(
            "Loaded project dataset for %s from %d chunk files", ticker, len(matches)
        )
        return data.tail(self._target_window_rows()).reset_index(drop=True)

    def _find_local_csv_path(self, ticker: str) -> Optional[Path]:
        """Find best-matching local CSV file for a ticker."""
        query_keys = self._normalize_ticker_queries(ticker)
        strong_query_keys = [key for key in query_keys if len(key) >= 4]

        for data_dir in self.local_data_candidates:
            if not data_dir.exists():
                continue

            csv_files = [*data_dir.glob("*.csv"), *data_dir.glob("*.CSV")]
            if not csv_files:
                continue

            # 1) Exact normalized stem match
            for csv_path in csv_files:
                stem_norm = self._normalize_name(csv_path.stem)
                if any(stem_norm == key for key in query_keys):
                    return csv_path

            # 2) Contains match (handles variants with spaces/underscores)
            for csv_path in csv_files:
                stem_norm = self._normalize_name(csv_path.stem)
                if any(key in stem_norm for key in strong_query_keys):
                    return csv_path

        return None

    def _load_from_local_csv(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load OHLCV data from local CSV dataset when available."""
        csv_path = self._find_local_csv_path(ticker)
        if csv_path is None:
            return None

        try:
            data = self._normalize_price_frame(pd.read_csv(csv_path))

            if "date" not in data.columns:
                # If date column is missing, try first column as date.
                first_col = data.columns[0]
                data = data.rename(columns={first_col: "date"})

            if not self._is_valid_ohlcv(data):
                logger.warning(
                    "Local CSV missing required columns for %s: %s", ticker, csv_path
                )
                return None

            data = self._clean_ohlcv_frame(data)

            # Keep only the lookback window to match Yahoo behavior.
            min_date = self.start_date
            data = data[data["date"] >= min_date].reset_index(drop=True)

            # If local data is older than the date window, still use latest rows.
            if data.empty:
                full_data = self._normalize_price_frame(pd.read_csv(csv_path))
                if "date" not in full_data.columns:
                    first_col = full_data.columns[0]
                    full_data = full_data.rename(columns={first_col: "date"})

                if not self._is_valid_ohlcv(full_data):
                    logger.warning(
                        "Local CSV remains invalid after re-read for %s: %s",
                        ticker,
                        csv_path,
                    )
                    return None

                full_data = self._clean_ohlcv_frame(full_data)
                data = full_data.tail(self.lookback_days).reset_index(drop=True)
            else:
                full_data = self._normalize_price_frame(pd.read_csv(csv_path))
                if "date" not in full_data.columns:
                    first_col = full_data.columns[0]
                    full_data = full_data.rename(columns={first_col: "date"})
                if self._is_valid_ohlcv(full_data):
                    full_data = self._clean_ohlcv_frame(full_data)
                    data = self._ensure_min_training_window(data, full_data)

            if data.empty:
                return None

            # Ensure adj close exists for downstream compatibility.
            if "adj close" not in data.columns:
                data["adj close"] = data["close"]

            logger.debug("Loaded local CSV for %s: %s", ticker, csv_path)
            return data.tail(self._target_window_rows()).reset_index(drop=True)
        except Exception:
            logger.exception("Error loading local CSV for %s", ticker)
            return None

    def fetch_stock_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data from Yahoo Finance

        Args:
            ticker: Stock ticker in NSE format (e.g., 'RELIANCE.NS')

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Use local NIFTY 50 CSV files first.
            local_data = self._load_from_local_csv(ticker)
            if local_data is not None:
                return local_data

            logger.info("Fetching data for %s from Yahoo Finance...", ticker)
            data = yf.download(
                ticker,
                start=self.start_date,
                end=self.end_date,
                progress=False,
                auto_adjust=False,
                threads=False,
            )

            if data is None:
                logger.warning("Yahoo Finance returned no frame for %s", ticker)
                return None

            if data.empty:
                logger.warning("No data found for %s", ticker)
                return None

            # Normalize columns to lowercase strings
            if isinstance(data.columns, pd.MultiIndex):
                normalized_columns = []
                for col in data.columns:
                    parts = [
                        str(c).strip()
                        for c in col
                        if str(c).strip() not in {"", "None"}
                    ]
                    first = parts[0].lower() if parts else ""
                    second = parts[1].upper() if len(parts) > 1 else ""

                    if first in {
                        "open",
                        "high",
                        "low",
                        "close",
                        "adj close",
                        "volume",
                        "date",
                    }:
                        normalized_columns.append(first)
                    elif (
                        len(col) == 2
                        and isinstance(col[0], str)
                        and second == ticker.upper()
                    ):
                        normalized_columns.append(str(col[0]).strip().lower())
                    else:
                        normalized_columns.append("_".join(parts).lower())
                data.columns = normalized_columns
            else:
                data.columns = [
                    col.lower() if isinstance(col, str) else str(col).lower()
                    for col in data.columns
                ]

            data = data.reset_index()
            data.columns = [str(col).strip().lower() for col in data.columns]

            if not self._is_valid_ohlcv(data):
                logger.warning(
                    "Yahoo Finance data for %s is missing OHLCV columns", ticker
                )
                return None

            data = self._clean_ohlcv_frame(data)
            data = data.tail(self._target_window_rows()).reset_index(drop=True)

            return data

        except Exception:
            logger.exception("Error fetching %s", ticker)
            return None

    def fetch_nifty_index(self) -> Optional[pd.DataFrame]:
        """Fetch NIFTY 50 index data"""
        try:
            logger.info("Fetching NIFTY 50 index...")
            data = yf.download(
                "^NSEI",  # NIFTY 50 ticker
                start=self.start_date,
                end=self.end_date,
                progress=False,
                auto_adjust=False,
                threads=False,
            )
            if data is None or data.empty:
                logger.warning("No NIFTY 50 index data returned")
                return None
            data.columns = [col.lower() for col in data.columns]
            data = data.reset_index()
            if not self._is_valid_ohlcv(data):
                logger.warning("NIFTY 50 index data missing OHLCV columns")
                return None
            return data
        except Exception:
            logger.exception("Error fetching NIFTY 50 index")
            return None

    def generate_synthetic_sentiment(self, price_data: pd.DataFrame) -> np.ndarray:
        """
        Generate synthetic sentiment scores based on price patterns
        In production, this would use FinBERT or actual news API

        Args:
            price_data: Historical price data

        Returns:
            Array of sentiment scores (-1 to 1)
        """
        # Sentiment based on momentum and volatility
        returns = price_data["close"].pct_change().fillna(0)
        momentum = returns.rolling(window=20).mean()
        volatility = returns.rolling(window=20).std()

        # Normalize sentiment
        sentiment = (momentum / (volatility + 1e-5)).rolling(window=5).mean().fillna(0)
        sentiment = np.clip(sentiment, -1, 1)  # Bound between -1 and 1

        return sentiment

    def generate_macroeconomic_features(self, length: int) -> Dict[str, np.ndarray]:
        """
        Generate synthetic macroeconomic features
        In production, fetch from external sources (RBI, Trading Economics API)

        Args:
            length: Number of time periods

        Returns:
            Dict of macro features
        """
        # Synthetic features - replace with real API calls in production
        rng = np.random.default_rng(42)
        interest_rate = np.sin(np.linspace(0, 10, length)) * 0.5 + 6.5  # 6% ± 0.5%
        inflation_rate = np.cos(np.linspace(0, 8, length)) * 1 + 5.5  # 5.5% ± 1%
        usd_inr = 82 + rng.normal(0, 0.1, length).cumsum()
        global_vix = 15 + rng.normal(0, 0.5, length).cumsum()  # VIX proxy

        return {
            "interest_rate": interest_rate,
            "inflation_rate": inflation_rate,
            "usd_inr": usd_inr,
            "global_vix": np.clip(global_vix, 10, 40),
        }

    def process_stock_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Complete data processing pipeline for a single stock

        Args:
            ticker: Stock ticker

        Returns:
            Processed DataFrame with all features
        """
        # Fetch price data
        raw_data = self.fetch_stock_data(ticker)
        if raw_data is None:
            logger.error("Failed to fetch stock data for %s", ticker)
            return None

        if not self._is_valid_ohlcv(raw_data):
            logger.error("Invalid OHLCV frame for %s", ticker)
            return None

        # Rename to lowercase for consistency
        raw_data.columns = raw_data.columns.str.lower()

        # Add sentiment
        try:
            raw_data["sentiment"] = self.generate_synthetic_sentiment(raw_data)
        except Exception:
            logger.exception("Failed to generate synthetic sentiment for %s", ticker)
            raw_data["sentiment"] = 0.0

        # Add macroeconomic features
        try:
            macro_features = self.generate_macroeconomic_features(len(raw_data))
            for feature_name, feature_values in macro_features.items():
                raw_data[feature_name] = feature_values
        except Exception:
            logger.exception("Failed to generate macro features for %s", ticker)
            raw_data["interest_rate"] = 6.5
            raw_data["inflation_rate"] = 5.5
            raw_data["usd_inr"] = 82.0
            raw_data["global_vix"] = 15.0

        # Add sector information
        sector_map = {
            "RELIANCE.NS": "Energy",
            "TCS.NS": "IT",
            "INFY.NS": "IT",
            "WIPRO.NS": "IT",
            "HDFC.NS": "Banking",
            "ICICIBANK.NS": "Banking",
            "AXISBANK.NS": "Banking",
            "MARUTI.NS": "Auto",
            "TECHM.NS": "IT",
            "SBIN.NS": "Banking",
            "JSWSTEEL.NS": "Steel",
            "TATASTEEL.NS": "Steel",
            "ASIANPAINT.NS": "Paints",
            "LT.NS": "Engineering",
            "BHARTIARTL.NS": "Telecom",
            "DRREDDY.NS": "Pharma",
            "SUNPHARM.NS": "Pharma",
            "M&M.NS": "Auto",
            "ITC.NS": "FMCG",
            "POWERGRID.NS": "Power",
            "NTPC.NS": "Power",
        }
        raw_data["sector"] = sector_map.get(ticker, "Other")

        # Ensure downstream numeric columns are present and finite.
        if "adj close" not in raw_data.columns:
            raw_data["adj close"] = raw_data["close"]

        for numeric_column in [
            "sentiment",
            "interest_rate",
            "inflation_rate",
            "usd_inr",
            "global_vix",
        ]:
            raw_data[numeric_column] = pd.to_numeric(
                raw_data[numeric_column], errors="coerce"
            ).fillna(0.0)

        return raw_data

    def process_uploaded_file(
        self, content: bytes, filename: str, ticker: str
    ) -> Dict[str, object]:
        """Parse and validate an uploaded stock dataset for downstream prediction pipeline."""
        try:
            df = self._parse_any_format(content, filename)
            df = self._normalize_ohlcv_columns(df)
            warnings_list: List[str] = []
            df, resolved_ticker = self._filter_uploaded_by_ticker(
                df, ticker=ticker, warnings_list=warnings_list
            )
            df = self._fill_missing_ohlc(df)
            df = self._synthesize_volume(df)

            is_valid, validation_warnings = self._validate_ohlcv(df)
            warnings_list.extend(validation_warnings)
            if not is_valid:
                return {"ok": False, "warnings": warnings_list, "data": None}

            if "date" in df.columns:
                df["date"] = self._parse_dates_mixed(df["date"])
                df = df.sort_values("date").drop_duplicates(
                    subset=["date"], keep="last"
                )

            df = self._standardize_ohlcv_frame(df)
            df = self._clean_ohlcv_frame(df)

            preview = df.head(5).to_dict(orient="records")
            upload_summary = self._build_upload_summary(
                df, requested_ticker=ticker, resolved_ticker=resolved_ticker
            )
            return {
                "ok": True,
                "warnings": warnings_list,
                "rows_parsed": int(len(df)),
                "columns_detected": list(df.columns),
                "data": df,
                "sample": preview,
                "ticker": resolved_ticker,
                "upload_summary": upload_summary,
            }
        except Exception as exc:
            return {
                "ok": False,
                "warnings": [f"Unable to parse file: {exc}"],
                "data": None,
            }

    def fetch_multiple_stocks(
        self, tickers: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple stocks"""
        if tickers is None:
            tickers = list(NIFTY50_STOCKS.keys())[:10]  # First 10 for demo

        data_dict = {}
        for ticker in tickers:
            data = self.process_stock_data(ticker)
            if data is not None:
                data_dict[ticker] = data

        return data_dict

    def get_all_available_tickers(self) -> List[str]:
        """Return NIFTY-focused ticker universe for UI and screening."""
        with self._cache_lock:
            if self._all_tickers_cache is not None:
                return self._all_tickers_cache

        discovered: set[str] = set()
        for ticker in NIFTY50_STOCKS.keys():
            if self._find_local_csv_path(ticker) is not None:
                discovered.add(str(ticker).strip().upper())

        # Safety fallback when local files are not available.
        if not discovered:
            discovered = set(NIFTY50_STOCKS.keys())

        # Normalize and keep NSE symbols only.
        normalized = []
        for ticker in discovered:
            t = str(ticker).strip().upper()
            if not t:
                continue
            if not t.endswith(".NS"):
                continue
            normalized.append(t)

        with self._cache_lock:
            self._all_tickers_cache = sorted(set(normalized))
            return self._all_tickers_cache

    def warmup_caches(
        self, preload_chunk_files: int = 12, preload_tickers: int = 150
    ) -> Dict[str, int]:
        """Warm symbol/file caches to reduce first-run latency spikes."""
        symbol_map = self._build_project_symbol_file_map()
        all_tickers = self.get_all_available_tickers()

        loaded_chunks = 0
        if preload_chunk_files > 0 and self.project_dataset_dir.exists():
            csv_files = sorted(
                [
                    *self.project_dataset_dir.glob("*.csv"),
                    *self.project_dataset_dir.glob("*.CSV"),
                ]
            )
            for csv_path in csv_files[:preload_chunk_files]:
                if self._get_project_chunk_data(csv_path) is not None:
                    loaded_chunks += 1

        warmed_tickers = 0
        if preload_tickers > 0:
            for ticker in all_tickers[:preload_tickers]:
                data = self._load_from_project_dataset(ticker)
                if data is None or data.empty:
                    data = self._load_from_local_csv(ticker)
                if data is not None and not data.empty:
                    warmed_tickers += 1

        return {
            "symbols_indexed": int(len(symbol_map)),
            "tickers_discovered": int(len(all_tickers)),
            "chunk_files_loaded": int(loaded_chunks),
            "ticker_frames_warmed": int(warmed_tickers),
        }

    def get_data_health_report(self, sample_size: int = 250) -> Dict[str, object]:
        """Report dataset coverage and post-cleaning quality metrics."""
        all_tickers = self.get_all_available_tickers()
        sample_n = max(1, min(int(sample_size), len(all_tickers))) if all_tickers else 0
        if sample_n <= 0:
            sample = []
        elif sample_n >= len(all_tickers):
            sample = list(all_tickers)
        else:
            # Uniformly spread sample across sorted universe to avoid head-of-list bias.
            idx = np.linspace(0, len(all_tickers) - 1, sample_n, dtype=int)
            unique_idx = list(np.unique(idx))
            if len(unique_idx) < sample_n:
                used = set(unique_idx)
                for i in range(len(all_tickers)):
                    if i in used:
                        continue
                    unique_idx.append(i)
                    if len(unique_idx) >= sample_n:
                        break
            sample = [all_tickers[i] for i in unique_idx[:sample_n]]

        project_hits = 0
        local_hits = 0
        missing_hits = 0
        missing_preview: List[str] = []
        valid_cleaned = 0
        total_rows = 0
        duplicate_dates = 0
        non_positive_close = 0
        nan_in_required_rows = 0

        for ticker in sample:
            source = "missing"
            data = self._load_from_project_dataset(ticker)
            if data is not None and not data.empty:
                source = "project"
                project_hits += 1
            else:
                data = self._load_from_local_csv(ticker)
                if data is not None and not data.empty:
                    source = "local"
                    local_hits += 1

            if source == "missing" or data is None or data.empty:
                missing_hits += 1
                if len(missing_preview) < 25:
                    missing_preview.append(ticker)
                continue

            cleaned = self._clean_ohlcv_frame(data)
            if cleaned is None or cleaned.empty:
                continue

            required_cols = ["date", "open", "high", "low", "close", "volume"]
            has_required = all(col in cleaned.columns for col in required_cols)
            if has_required:
                valid_cleaned += 1

            total_rows += int(len(cleaned))
            if "date" in cleaned.columns:
                duplicate_dates += int(cleaned["date"].duplicated().sum())
            if "close" in cleaned.columns:
                non_positive_close += int((cleaned["close"] <= 0).sum())

            subset_cols = [c for c in required_cols if c in cleaned.columns]
            if subset_cols:
                nan_in_required_rows += int(
                    cleaned[subset_cols].isna().any(axis=1).sum()
                )

        coverage_pct = (
            (100.0 * (len(sample) - missing_hits) / len(sample)) if sample else 0.0
        )
        avg_rows = (float(total_rows) / float(valid_cleaned)) if valid_cleaned else 0.0

        return {
            "lookback_days": int(self.lookback_days),
            "discovered_tickers_total": int(len(all_tickers)),
            "sample_tickers_checked": int(len(sample)),
            "coverage": {
                "project_hits": int(project_hits),
                "local_hits": int(local_hits),
                "missing_hits": int(missing_hits),
                "coverage_pct": float(round(coverage_pct, 2)),
            },
            "cleaning_quality": {
                "valid_ohlcv_frames": int(valid_cleaned),
                "avg_rows_per_ticker": float(round(avg_rows, 1)),
                "duplicate_dates_after_cleaning": int(duplicate_dates),
                "rows_with_nan_required_fields_after_cleaning": int(
                    nan_in_required_rows
                ),
                "non_positive_close_rows": int(non_positive_close),
            },
            "missing_tickers_preview": missing_preview,
        }

    def get_missing_tickers(self, limit: int = 200) -> List[str]:
        """Return discovered tickers that currently have no loadable local/project data."""
        missing: List[str] = []
        for ticker in self.get_all_available_tickers():
            project_data = self._load_from_project_dataset(ticker)
            if project_data is not None and not project_data.empty:
                continue
            local_data = self._load_from_local_csv(ticker)
            if local_data is not None and not local_data.empty:
                continue
            missing.append(ticker)
            if len(missing) >= max(1, int(limit)):
                break
        return missing


def main():
    """Quick test of data ingestion"""
    ingestion = DataIngestion(lookback_days=365)

    # Fetch data for a few stocks
    tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
    data_dict = {}

    for ticker in tickers:
        data = ingestion.process_stock_data(ticker)
        if data is not None:
            data_dict[ticker] = data
            print(f"\n{ticker} data shape: {data.shape}")
            print(data.head())

    print("\n✓ Data ingestion successful!")


if __name__ == "__main__":
    main()
