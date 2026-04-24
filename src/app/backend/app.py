from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .services import (
    DATA_RUNTIME_DIR,
    FEATURE_MODES,
    FRONTEND_DIR,
    WEB_OUTPUT_DIR,
    IndexStatusOptions,
    RebuildIndexOptions,
    SearchOptions,
    clear_web_outputs,
    get_index_status,
    get_runtime_options,
    get_status,
    rebuild_gallery_index,
    resolve_query_path,
    search_gallery,
)

app = FastAPI(title="Course Retrieval Web API")

app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR), check_dir=False), name="frontend")
app.mount("/outputs-static", StaticFiles(directory=str(WEB_OUTPUT_DIR), check_dir=False), name="outputs_static")
app.mount("/data-runtime-static", StaticFiles(directory=str(DATA_RUNTIME_DIR), check_dir=False), name="data_runtime_static")


class RebuildIndexPayload(BaseModel):
    gallery_path: str = Field(min_length=1)
    feature_mode: str = "face"
    index_name: str | None = None
    device: str | None = None
    sample_fps: float = 1.0
    arcface_weight_path: str | None = None
    yolo_weights: str | None = None
    yolo_conf: float = 0.25
    yolo_iou: float = 0.7
    yolo_max_det: int = 100
    person_model: str = "resnet"
    resnet_backbone: str = "resnet18"
    resnet_pretrained: bool = False
    resnet_weight_path: str | None = None
    person_input_size: int = 224


class IndexStatusPayload(BaseModel):
    feature_mode: str = "face"
    index_name: str | None = None
    gallery_path: str = Field(min_length=1)
    person_model: str = "resnet"
    resnet_backbone: str = "resnet18"


class SearchPayload(BaseModel):
    query_path: str = Field(min_length=1)
    gallery_path: str = Field(min_length=1)
    feature_mode: str = "face"
    index_name: str | None = None
    topk: int = 5
    device: str | None = None
    sample_fps: float = 1.0
    arcface_weight_path: str | None = None
    yolo_weights: str | None = None
    yolo_conf: float = 0.25
    yolo_iou: float = 0.7
    yolo_max_det: int = 100
    person_model: str = "resnet"
    resnet_backbone: str = "resnet18"
    resnet_pretrained: bool = False
    resnet_weight_path: str | None = None
    person_input_size: int = 224


@app.exception_handler(ValueError)
async def value_error_handler(_, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(ImportError)
async def import_error_handler(_, exc: ImportError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(_, exc: FileNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(RuntimeError)
async def runtime_error_handler(_, exc: RuntimeError) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.get("/")
async def index() -> FileResponse:
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(str(index_path))


@app.get("/api/status")
async def api_status() -> dict[str, object]:
    return get_status()


@app.get("/api/runtime/options")
async def api_runtime_options() -> dict[str, object]:
    return get_runtime_options()


@app.post("/api/admin/rebuild-gallery-index")
async def api_rebuild_gallery_index(payload: RebuildIndexPayload) -> dict[str, object]:
    options = RebuildIndexOptions(**payload.model_dump())
    options.feature_mode = options.feature_mode.strip().lower()
    if options.feature_mode not in FEATURE_MODES:
        raise ValueError(f"Unsupported feature_mode: {options.feature_mode}")
    summary = rebuild_gallery_index(options)
    summary["status"] = get_status()
    return summary


@app.post("/api/index/status")
async def api_index_status(payload: IndexStatusPayload) -> dict[str, object]:
    options = IndexStatusOptions(**payload.model_dump())
    return get_index_status(options)


@app.post("/api/search/gallery")
async def api_search_gallery(payload: SearchPayload) -> dict[str, object]:
    query_path = resolve_query_path(payload.query_path)
    options = SearchOptions(
        feature_mode=payload.feature_mode,
        index_name=payload.index_name,
        topk=payload.topk,
        device=payload.device,
        sample_fps=payload.sample_fps,
        arcface_weight_path=payload.arcface_weight_path,
        yolo_weights=payload.yolo_weights,
        yolo_conf=payload.yolo_conf,
        yolo_iou=payload.yolo_iou,
        yolo_max_det=payload.yolo_max_det,
        person_model=payload.person_model,
        resnet_backbone=payload.resnet_backbone,
        resnet_pretrained=payload.resnet_pretrained,
        resnet_weight_path=payload.resnet_weight_path,
        person_input_size=payload.person_input_size,
    )
    return search_gallery(query_path=query_path, gallery_path=payload.gallery_path, options=options)


@app.delete("/api/admin/clear-web-outputs")
async def api_clear_web_outputs() -> dict[str, object]:
    return clear_web_outputs()
