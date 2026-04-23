from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .services import (
    FEATURE_MODES,
    FRONTEND_DIR,
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    WEB_OUTPUT_DIR,
    RebuildIndexOptions,
    SearchOptions,
    clear_web_outputs,
    get_status,
    rebuild_gallery_index,
    save_upload,
    search_gallery,
    search_uploaded_video,
)

app = FastAPI(title="Course Retrieval Web API")

app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR), check_dir=False), name="frontend")
app.mount("/outputs-static", StaticFiles(directory=str(WEB_OUTPUT_DIR), check_dir=False), name="outputs_static")


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
    resnet_backbone: str = "resnet50"
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


@app.post("/api/admin/rebuild-gallery-index")
async def api_rebuild_gallery_index(payload: RebuildIndexPayload) -> dict[str, object]:
    options = RebuildIndexOptions(**payload.model_dump())
    options.feature_mode = options.feature_mode.strip().lower()
    if options.feature_mode not in FEATURE_MODES:
        raise ValueError(f"Unsupported feature_mode: {options.feature_mode}")
    summary = rebuild_gallery_index(options)
    summary["status"] = get_status()
    return summary


@app.post("/api/search/gallery")
async def api_search_gallery(
    query: UploadFile = File(...),
    gallery_path: str = Form(...),
    feature_mode: str = Form("face"),
    index_name: str | None = Form(None),
    topk: int = Form(5),
    device: str | None = Form(None),
    sample_fps: float = Form(1.0),
    arcface_weight_path: str | None = Form(None),
    yolo_weights: str | None = Form(None),
    yolo_conf: float = Form(0.25),
    yolo_iou: float = Form(0.7),
    yolo_max_det: int = Form(100),
    resnet_backbone: str = Form("resnet50"),
    resnet_pretrained: bool = Form(False),
    resnet_weight_path: str | None = Form(None),
    person_input_size: int = Form(224),
) -> dict[str, object]:
    query_path = await save_upload(query, kind="query", allowed_extensions=IMAGE_EXTENSIONS)
    options = SearchOptions(
        feature_mode=feature_mode,
        index_name=index_name,
        topk=topk,
        device=device,
        sample_fps=sample_fps,
        arcface_weight_path=arcface_weight_path,
        yolo_weights=yolo_weights,
        yolo_conf=yolo_conf,
        yolo_iou=yolo_iou,
        yolo_max_det=yolo_max_det,
        resnet_backbone=resnet_backbone,
        resnet_pretrained=resnet_pretrained,
        resnet_weight_path=resnet_weight_path,
        person_input_size=person_input_size,
    )
    return search_gallery(query_path=query_path, gallery_path=gallery_path, options=options)


@app.post("/api/search/uploaded-video")
async def api_search_uploaded_video(
    query: UploadFile = File(...),
    video: UploadFile = File(...),
    feature_mode: str = Form("face"),
    index_name: str | None = Form(None),
    topk: int = Form(5),
    device: str | None = Form(None),
    sample_fps: float = Form(1.0),
    arcface_weight_path: str | None = Form(None),
    yolo_weights: str | None = Form(None),
    yolo_conf: float = Form(0.25),
    yolo_iou: float = Form(0.7),
    yolo_max_det: int = Form(100),
    resnet_backbone: str = Form("resnet50"),
    resnet_pretrained: bool = Form(False),
    resnet_weight_path: str | None = Form(None),
    person_input_size: int = Form(224),
) -> dict[str, object]:
    query_path = await save_upload(query, kind="query", allowed_extensions=IMAGE_EXTENSIONS)
    video_path = await save_upload(video, kind="video", allowed_extensions=VIDEO_EXTENSIONS)
    options = SearchOptions(
        feature_mode=feature_mode,
        index_name=index_name,
        topk=topk,
        device=device,
        sample_fps=sample_fps,
        arcface_weight_path=arcface_weight_path,
        yolo_weights=yolo_weights,
        yolo_conf=yolo_conf,
        yolo_iou=yolo_iou,
        yolo_max_det=yolo_max_det,
        resnet_backbone=resnet_backbone,
        resnet_pretrained=resnet_pretrained,
        resnet_weight_path=resnet_weight_path,
        person_input_size=person_input_size,
    )
    return search_uploaded_video(
        query_path=query_path,
        video_path=video_path,
        video_name=video.filename or video_path.name,
        options=options,
    )


@app.delete("/api/admin/clear-web-outputs")
async def api_clear_web_outputs() -> dict[str, object]:
    return clear_web_outputs()
