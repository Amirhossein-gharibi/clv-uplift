# src/clv_uplift/api/main.py
#
# Entry point for the API service. Application-factory pattern + lifespan context manager
# (NOT the deprecated @app.on_event handler).

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from .routers import health, predict, explain, info
from clv_uplift import __version__

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup warms both caches: the dummy scaffold (always) and the real ServingBundle
    (if the trained artifact exists). If the bundle is absent, the service still starts
    and serves /score; /predict, /explain and /model-info return 503 until trained.
    """
    from .dependencies import get_model, get_bundle
    get_model()
    try:
        get_bundle()
        bundle_status = "bundle loaded"
    except RuntimeError:
        logger.warning("No model bundle found - /predict, /explain, /model-info will return "
                       "503 until notebooks/01_uplift_training.py is run.")
        bundle_status = "no bundle (run training)"
    print(f"CLV Uplift Service v{__version__} started. DummyModel loaded; {bundle_status}.")

    yield

    print("CLV Uplift Service shutting down.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="CLV Uplift Service",
        version=__version__,
        description=(
            "RFM-based engagement scoring (scaffold) and causal CATE estimation "
            "with explanation for retail uplift modeling."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.include_router(health.router,  prefix="/health",  tags=["health"])
    app.include_router(predict.router, prefix="/api/v1",  tags=["prediction"])
    app.include_router(explain.router, prefix="/api/v1",  tags=["explanation"])
    app.include_router(info.router,    prefix="/api/v1",  tags=["model"])

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    from clv_uplift.config import API_HOST, API_PORT
    uvicorn.run(
        "clv_uplift.api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
    )