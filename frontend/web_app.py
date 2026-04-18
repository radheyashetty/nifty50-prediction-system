"""
FastAPI + Bootstrap frontend for NIFTY 50 Stock Lab.
Keeps Python ML pipeline in backend modules.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, File, Form, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import asyncio
from anyio.to_thread import run_sync

# Add project root to path so `backend` package resolves when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.data_ingestion import NIFTY50_STOCKS  # noqa: E402
from backend.predictions import PredictionService  # noqa: E402
from backend.screener import StockScreener  # noqa: E402
from backend.sector_analysis import SectorAnalyzer  # noqa: E402
from backend.utils import get_nse_sector_map, get_ticker_sector  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="NIFTY 50 Stock Lab", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
_services: dict[int, PredictionService] = {}


def _get_service(lookback_days: int) -> PredictionService:
    service = _services.get(lookback_days)
    if service is None:
        service = PredictionService(lookback_days=lookback_days)
        _services[lookback_days] = service
    return service


class AnalyzeRequest(BaseModel):
    ticker: str = Field(..., description="NSE ticker, e.g., RELIANCE.NS")
    lookback_days: int = Field(365, ge=90, le=730)
    analysis_mode: str = Field("after_training", description="cache | after_training")


class ScreenerRequest(BaseModel):
    sector: str | None = None
    min_confidence: float = Field(0.55, ge=0.0, le=1.0)
    min_volume_ratio: float = Field(0.0, ge=0.0)
    regime_filter: str | None = None


class SectorAnalysisRequest(BaseModel):
    sector: str | None = None


class CompareRequest(BaseModel):
    tickers: list[str] = Field(default_factory=list, min_length=2, max_length=4)


def _to_jsonable(value: Any) -> Any:
    """Convert numpy/scalar containers to JSON-safe values."""
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _normalize_ticker(ticker: str) -> str:
    t = str(ticker or "").strip().upper()
    if not t:
        return ""
    if t.endswith((".NS", ".BO")):
        return t
    return f"{t}.NS"


def _normalize_upload_ticker(ticker: str) -> str:
    """Normalize free-form upload ticker labels without forcing NSE suffix."""
    t = str(ticker or "").strip().upper()
    if not t:
        return ""
    return t


def _parse_upload_result(parsed: Any) -> dict[str, Any]:
    if isinstance(parsed, dict):
        return parsed
    return {}


def _upload_preview_from_parsed(parsed: dict[str, Any]) -> dict[str, Any]:
    return {
        "rows_parsed": parsed.get("rows_parsed", 0),
        "columns_detected": parsed.get("columns_detected", []),
        "warnings": parsed.get("warnings", []),
        "sample": parsed.get("sample", []),
    }


def _warnings_from_parsed(parsed: dict[str, Any]) -> list[str]:
    raw_warnings = parsed.get("warnings", ["Invalid dataset"])
    if isinstance(raw_warnings, list):
        return [str(w) for w in raw_warnings]
    return [str(raw_warnings)]


def _process_upload_payload(
    service: PredictionService,
    content: bytes,
    filename: str,
    requested_ticker: str,
) -> tuple[pd.DataFrame, str, dict[str, Any], dict[str, Any]]:
    parsed = _parse_upload_result(
        service.data_ingestion.process_uploaded_file(
            content=content,
            filename=filename,
            ticker=requested_ticker,
        )
    )
    if not parsed.get("ok"):
        raise ValueError("; ".join(_warnings_from_parsed(parsed)))

    uploaded_df = parsed.get("data")
    if not isinstance(uploaded_df, pd.DataFrame):
        raise ValueError("Uploaded dataset could not be parsed into a table")

    resolved_ticker = _normalize_upload_ticker(
        str(parsed.get("ticker", requested_ticker))
    )
    upload_preview = _upload_preview_from_parsed(parsed)
    return uploaded_df, resolved_ticker, upload_preview, parsed


@app.get("/", response_class=HTMLResponse)
def home():
    service = _get_service(365)
    all_tickers = service.data_ingestion.get_all_available_tickers()
    html = templates.get_template("index.html").render(
        {
            "title": "NIFTY 50 Stock Lab",
            "stocks": [
                {
                    "ticker": t,
                    "name": NIFTY50_STOCKS.get(t, t.split(".")[0]),
                    "sector": get_ticker_sector(t),
                }
                for t in all_tickers
            ],
        }
    )
    return HTMLResponse(content=html)


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    favicon_path = BASE_DIR / "static" / "favicon.svg"
    if favicon_path.exists():
        return Response(
            content=favicon_path.read_text(encoding="utf-8"), media_type="image/svg+xml"
        )
    return Response(status_code=204)


@app.get("/api/stocks")
def get_stocks():
    service = _get_service(365)
    tickers = service.data_ingestion.get_all_available_tickers()
    sector_map = get_nse_sector_map()

    by_sector: dict[str, list[str]] = {}
    for sector, symbols in sector_map.items():
        filtered = sorted(set([s for s in tickers if get_ticker_sector(s) == sector]))
        by_sector[sector] = filtered

    return {
        "stocks": [
            {
                "ticker": t,
                "name": NIFTY50_STOCKS.get(t, t.split(".")[0]),
                "sector": get_ticker_sector(t),
            }
            for t in tickers
        ],
        "by_sector": by_sector,
        "total_count": len(tickers),
    }


@app.get("/api/data-health")
def get_data_health(sample_size: int = 250, lookback_days: int = 365):
    service = _get_service(lookback_days)
    report = service.data_ingestion.get_data_health_report(sample_size=sample_size)
    return _to_jsonable(report)


@app.get("/api/data-health/missing")
def get_missing_tickers(limit: int = 200, lookback_days: int = 365):
    service = _get_service(lookback_days)
    missing = service.data_ingestion.get_missing_tickers(
        limit=max(1, min(int(limit), 5000))
    )
    return _to_jsonable(
        {
            "lookback_days": int(lookback_days),
            "missing_count_returned": int(len(missing)),
            "missing_tickers": missing,
        }
    )


@app.post("/api/system/warmup")
def warmup_system(
    preload_chunk_files: int = 12, preload_tickers: int = 150, lookback_days: int = 365
):
    service = _get_service(lookback_days)
    stats = service.data_ingestion.warmup_caches(
        preload_chunk_files=max(0, min(int(preload_chunk_files), 200)),
        preload_tickers=max(0, min(int(preload_tickers), 2000)),
    )
    return _to_jsonable(
        {
            "status": "ok",
            "message": "Warmup complete",
            "stats": stats,
        }
    )


@app.post("/api/analyze")
async def analyze_stock(payload: AnalyzeRequest):
    """Analyze a stock with error handling and timeout protection."""
    ticker = _normalize_ticker(payload.ticker)
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker is required")

    # Validate lookback days
    if payload.lookback_days < 90:
        raise HTTPException(status_code=400, detail="Minimum lookback_days is 90")
    if payload.lookback_days > 730:
        raise HTTPException(status_code=400, detail="Maximum lookback_days is 730")

    service = _get_service(payload.lookback_days)

    # Add error handling wrapper
    try:
        result = await asyncio.wait_for(
            run_sync(service.predict_stock, ticker, False, payload.analysis_mode),
            timeout=30.0,
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Models not initialized: {str(e)}. Run 'python backend/train_models.py' first.",
        )
    except TimeoutError as e:
        raise HTTPException(
            status_code=504,
            detail=f"Analysis timeout: {str(e)}. Request took too long.",
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid analysis parameters: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return _to_jsonable(result)


@app.post("/api/upload-train")
async def upload_train(
    ticker: str = Form(""),
    lookback_days: int = Form(365),
    analysis_mode: str = Form("after_training"),
    file: UploadFile | None = File(default=None),
    csv_content: str | None = Form(default=None),
):
    requested_ticker = _normalize_upload_ticker(ticker)
    service = _get_service(lookback_days)

    uploaded_df = None
    upload_preview = None
    resolved_ticker = requested_ticker
    parsed: dict[str, Any] = {}
    try:
        if file is not None:
            file_bytes = await file.read()
            uploaded_df, resolved_ticker, upload_preview, parsed = (
                _process_upload_payload(
                    service=service,
                    content=file_bytes,
                    filename=file.filename or "uploaded.csv",
                    requested_ticker=requested_ticker,
                )
            )
        elif csv_content:
            uploaded_df, resolved_ticker, upload_preview, parsed = (
                _process_upload_payload(
                    service=service,
                    content=csv_content.encode("utf-8"),
                    filename="uploaded.csv",
                    requested_ticker=requested_ticker,
                )
            )
        else:
            raise ValueError("Provide a dataset file or CSV text")
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Unable to parse uploaded dataset: {exc}"
        )

    if not isinstance(uploaded_df, pd.DataFrame):
        raise HTTPException(
            status_code=400, detail="Uploaded dataset could not be parsed into a table"
        )

    if not resolved_ticker:
        resolved_ticker = "UPLOADED_STOCK"

    normalized = None
    try:
        normalized = service.prepare_uploaded_price_data(
            uploaded_df, ticker=resolved_ticker
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid dataset structure: {exc}")

    result = service.predict_from_uploaded_data(
        ticker=resolved_ticker,
        uploaded_data=normalized,
        analysis_mode=analysis_mode,
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    preview_rows = (
        normalized[["date", "open", "high", "low", "close", "volume"]].tail(5).copy()
    )
    preview_rows["date"] = preview_rows["date"].dt.strftime("%Y-%m-%d")

    values = {
        "source_file": file.filename if file is not None else "csv_text_input",
        "source_format": (
            Path(file.filename).suffix.lower().lstrip(".")
            if file and file.filename
            else "csv"
        ),
        "ticker_requested": requested_ticker,
        "ticker_resolved": resolved_ticker,
        "rows": int(len(normalized)),
        "start_date": str(normalized["date"].iloc[0].date()),
        "end_date": str(normalized["date"].iloc[-1].date()),
        "latest_close": float(normalized["close"].iloc[-1]),
        "latest_volume": float(normalized["volume"].iloc[-1]),
        "preview": preview_rows.to_dict(orient="records"),
        "upload_summary": parsed.get("upload_summary", {}),
    }

    if upload_preview is not None:
        values["upload_preview"] = upload_preview
        result["upload_preview"] = upload_preview
        result["ticker"] = resolved_ticker
        upload_summary = cast(dict[str, Any], parsed.get("upload_summary", {}))
        top_sectors = (
            cast(list[dict[str, Any]], upload_summary.get("top_sectors", []))
            if isinstance(upload_summary, dict)
            else []
        )
        sector_name = result.get("sector")
        if not sector_name and top_sectors:
            sector_name = str(top_sectors[0].get("sector", ""))
        result["sector"] = sector_name
        result["data_quality"] = {
            "warnings": upload_preview.get("warnings", []),
            "columns_detected": upload_preview.get("columns_detected", []),
        }

    return _to_jsonable({"result": result, "values": values})


@app.post("/api/screener")
async def run_screener(payload: ScreenerRequest):
    service = _get_service(365)
    screener = StockScreener(service)
    try:
        result = await asyncio.wait_for(
            run_sync(
                screener.run_screener,
                payload.sector,
                payload.min_confidence,
                payload.min_volume_ratio,
                payload.regime_filter,
                True,  # use_cache
                20,  # top_n
            ),
            timeout=60.0,
        )
    except TimeoutError as e:
        raise HTTPException(
            status_code=504,
            detail=f"Screener timeout: {str(e)}. Request took too long.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Screener failed: {str(e)}")
    return _to_jsonable(result)


@app.post("/api/sector-analysis")
async def sector_analysis(payload: SectorAnalysisRequest):
    service = _get_service(365)
    screener = StockScreener(service)
    analyzer = SectorAnalyzer(screener)
    try:
        result = await asyncio.wait_for(
            run_sync(analyzer.analyze_all_sectors), timeout=90.0
        )
    except TimeoutError as e:
        raise HTTPException(
            status_code=504,
            detail=f"Sector analysis timeout: {str(e)}. Request took too long.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sector analysis failed: {str(e)}")

    if payload.sector and payload.sector.lower() != "all":
        filtered_sectors = [
            sector
            for sector in result.get("sectors", [])
            if str(sector.get("sector_name", "")).lower() == payload.sector.lower()
        ]
        result = {
            **result,
            "sectors": filtered_sectors,
            "top_sector": filtered_sectors[0] if filtered_sectors else None,
        }

    return _to_jsonable(result)


@app.post("/api/compare")
async def compare_stocks(payload: CompareRequest):
    service = _get_service(365)

    requested_tickers: list[str] = []
    for raw_ticker in payload.tickers:
        normalized = _normalize_ticker(raw_ticker)
        if not normalized:
            continue
        if normalized not in requested_tickers:
            requested_tickers.append(normalized)

    invalid_tickers = [t for t in requested_tickers if t not in NIFTY50_STOCKS]
    if invalid_tickers:
        raise HTTPException(
            status_code=400,
            detail=(
                "Comparison supports NIFTY 50 stocks only. Invalid tickers: "
                + ", ".join(invalid_tickers)
            ),
        )

    results = []
    for ticker in requested_tickers:
        try:
            result = await asyncio.wait_for(
                run_sync(service.predict_stock, ticker, False, "cache"), timeout=15.0
            )
            if "error" in result:
                continue
            results.append(result)
        except Exception:
            continue

    if len(results) < 2:
        raise HTTPException(
            status_code=400, detail="Unable to compare: need at least 2 valid tickers"
        )

    best_signal = max(
        results,
        key=lambda r: float(
            r.get("confidence") or r.get("predictions", {}).get("confidence", 0.0)
        ),
    ).get("ticker")

    return _to_jsonable(
        {
            "stocks": results,
            "best_signal": best_signal,
        }
    )
