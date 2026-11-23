"""The HTTP routes (FastAPI). Thin: each one calls `archive`, `Store` or `RunManager` and turns
their errors into HTTP answers.

    GET    /api/providers            which providers have a key configured (never the key)
    POST   /api/uploads              multipart file -> `UploadInfo`; 400 with the reason if rejected
    GET    /api/uploads/{id}         -> `UploadInfo`
    DELETE /api/uploads/{id}
    POST   /api/runs                 `RunRequest` -> `RunStatus` (201)
    GET    /api/runs                 -> list of `RunSummary`
    GET    /api/runs/{id}            -> `RunStatus`
    POST   /api/runs/{id}/cancel     -> `RunStatus`
    GET    /api/runs/{id}/report     -> the `Report` JSON (404 until done)
    DELETE /api/runs/{id}
    POST   /api/gpu/reports          multipart gpu_profile.json (+ optional upload_id) -> `GpuReportInfo` (201)
    GET    /api/gpu/reports          -> list of `GpuReportInfo`
    GET    /api/gpu/reports/{id}     -> the `GpuReport` JSON
    DELETE /api/gpu/reports/{id}

A GPU report is built when its profile is uploaded (GPU-5 to GPU-9): assessment, recommendations,
checklist, and costs when `gpu_prices.yaml` sits next to the catalog. Picking an export
(`upload_id`) lets the costs use its token counts for cost per million tokens.

Errors: `FileRejected`, `MissingKeys` and `RequestRejected` become `{"detail": "<message>"}` with the
right status (400, 400, `.status_code`); `NotFound` is 404. An unexpected exception is a plain
500 whose body says nothing about the cause.

An upload is read in chunks and refused as soon as it passes `archive.MAX_ARCHIVE_BYTES`, before
the rest is read. The filename is kept for display only.

There is no CORS setup on purpose: in development the Vite server proxies `/api` here, so the
browser sees one origin, and a built frontend is served by this app itself.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from migration_eval.archive import MAX_ARCHIVE_BYTES, read_archive
from migration_eval.capture.file_checks import FileRejected
from migration_eval.gpu.prices import PriceTable, load_prices
from migration_eval.gpu.profile import parse_profile
from migration_eval.gpu.report import GpuReport, build_gpu_report
from migration_eval.keys import MissingKeys
from migration_eval.models import Report
from migration_eval.server.batch import BatchRuns
from migration_eval.server.jobs import FactoryBuilder, RequestRejected, RunManager, real_factory_builder
from migration_eval.server.models import GpuReportInfo, ProviderStatus, RunRequest, RunStatus, RunSummary, UploadInfo
from migration_eval.server.store import NotFound, Store

CHUNK_BYTES = 1024 * 1024
MAX_GPU_PROFILE_BYTES = 64 * 1024 * 1024  # a long capture on a big machine is a few MB; this is generous
GPU_PRICES_FILENAME = "gpu_prices.yaml"  # looked for next to the catalog


def _detail(message: str, status_code: int) -> JSONResponse:
    return JSONResponse({"detail": message}, status_code=status_code)


async def _read_capped(file: UploadFile, limit: int) -> bytes:
    """The upload's bytes, refused (413) as soon as they pass `limit`, before the rest is read."""
    chunks: list[bytes] = []
    size = 0
    while chunk := await file.read(CHUNK_BYTES):
        size += len(chunk)
        if size > limit:
            raise RequestRejected(f"The file is larger than the {limit // (1024 * 1024)} MB limit.", 413)
        chunks.append(chunk)
    return b"".join(chunks)


def _gpu_prices(catalog_path: Path) -> PriceTable | None:
    """gpu_prices.yaml next to the catalog, or None when there is none (costs are then left out)."""
    path = catalog_path.parent / GPU_PRICES_FILENAME
    return load_prices(path) if path.exists() else None


def create_app(
    data_dir: Path,
    catalog_path: Path,
    factory_builder: FactoryBuilder = real_factory_builder,
    frontend_dist: Path | None = None,
    batch: BatchRuns | None = None,
) -> FastAPI:
    """Build the app: a `Store` in `data_dir`, a `RunManager`, the exception handlers described
    above and the routes below. When `frontend_dist` exists (the output of `npm run build`), serve
    it at `/` with `StaticFiles(..., html=True)` after the `/api` routes, so one process serves both.
    With `batch`, runs execute as AWS Batch jobs (`server/batch.py`) instead of in this process.
    """
    store = Store(data_dir)
    manager = RunManager(store, catalog_path, factory_builder, batch=batch)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        manager.shutdown()

    app = FastAPI(title="migration-eval", docs_url="/api/docs", openapi_url="/api/openapi.json", lifespan=lifespan)

    @app.exception_handler(FileRejected)
    async def file_rejected(_request: Request, exc: FileRejected) -> JSONResponse:
        return _detail(str(exc), 400)

    @app.exception_handler(MissingKeys)
    async def missing_keys(_request: Request, exc: MissingKeys) -> JSONResponse:
        return _detail(str(exc), 400)

    @app.exception_handler(RequestRejected)
    async def request_rejected(_request: Request, exc: RequestRejected) -> JSONResponse:
        return _detail(str(exc), exc.status_code)

    @app.exception_handler(NotFound)
    async def not_found(_request: Request, _exc: NotFound) -> JSONResponse:
        return _detail("Not found.", 404)

    @app.get("/api/providers")
    def providers() -> list[ProviderStatus]:
        return manager.providers()

    @app.post("/api/uploads", status_code=201)
    async def create_upload(file: UploadFile = File(...)) -> UploadInfo:
        """Read the body in chunks (stop at the size cap), `archive.read_archive`, then `Store.save_upload`."""
        chunks: list[bytes] = []
        size = 0
        while chunk := await file.read(CHUNK_BYTES):
            size += len(chunk)
            if size > MAX_ARCHIVE_BYTES:  # stop reading: do not take the rest of an oversized file
                raise RequestRejected(f"The file is larger than the {MAX_ARCHIVE_BYTES // (1024 * 1024)} MB limit.", 413)
            chunks.append(chunk)
        data = b"".join(chunks)
        loaded = await asyncio.to_thread(read_archive, data)  # parsing a big archive must not block the server
        return await asyncio.to_thread(store.save_upload, file.filename or "upload", size, loaded.records)

    @app.get("/api/uploads/{upload_id}")
    def get_upload(upload_id: str) -> UploadInfo:
        return store.get_upload(upload_id)

    @app.delete("/api/uploads/{upload_id}", status_code=204)
    def delete_upload(upload_id: str) -> Response:
        manager.delete_upload(upload_id)
        return Response(status_code=204)

    @app.post("/api/runs", status_code=201)
    async def create_run(request: RunRequest) -> RunStatus:
        return await manager.start(request)

    @app.get("/api/runs")
    def list_runs() -> list[RunSummary]:
        return manager.summaries()

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> RunStatus:
        return manager.status(run_id)

    @app.post("/api/runs/{run_id}/cancel")
    def cancel_run(run_id: str) -> RunStatus:
        return manager.cancel(run_id)

    @app.get("/api/runs/{run_id}/report", response_model=Report)
    def get_report(run_id: str) -> JSONResponse:
        # Sent as the stored JSON, so what the browser reads is exactly what write_report wrote.
        return JSONResponse(manager.report(run_id).model_dump(mode="json", by_alias=True))

    @app.delete("/api/runs/{run_id}", status_code=204)
    def delete_run(run_id: str) -> Response:
        manager.delete_run(run_id)
        return Response(status_code=204)

    @app.post("/api/gpu/reports", status_code=201)
    async def create_gpu_report(file: UploadFile = File(...), upload_id: str | None = Form(None)) -> GpuReportInfo:
        """Validate the profile, build the report (with the export's records when one is picked), store it."""
        data = await _read_capped(file, MAX_GPU_PROFILE_BYTES)
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise FileRejected("The file is not UTF-8 text. Upload the gpu_profile.json that `migration-eval gpu capture` wrote.") from None
        profile = parse_profile(text, "The file")
        records = await asyncio.to_thread(store.load_records, upload_id) if upload_id else None
        report = await asyncio.to_thread(build_gpu_report, profile, _gpu_prices(catalog_path), records)
        return await asyncio.to_thread(store.save_gpu_report, file.filename or "gpu_profile.json", report, upload_id)

    @app.get("/api/gpu/reports")
    def list_gpu_reports() -> list[GpuReportInfo]:
        return store.list_gpu_reports()

    @app.get("/api/gpu/reports/{report_id}")
    def get_gpu_report(report_id: str) -> GpuReport:
        return store.get_gpu_report(report_id)

    @app.delete("/api/gpu/reports/{report_id}", status_code=204)
    def delete_gpu_report(report_id: str) -> Response:
        store.delete_gpu_report(report_id)
        return Response(status_code=204)

    if frontend_dist is not None and (frontend_dist / "index.html").exists():
        app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")  # last, so /api wins

    return app
