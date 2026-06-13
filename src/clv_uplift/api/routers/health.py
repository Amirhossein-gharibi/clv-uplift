# src/clv_uplift/api/routers/health.py
#
# The health endpoint is the simplest possible endpoint but one of the
# most important in production. Docker, Kubernetes, and cloud platforms
# all poll GET /health periodically. If it returns anything other than
# a 2xx status code, the platform considers the service unhealthy and
# may restart the container or stop routing traffic to it.

from fastapi import APIRouter, Depends
from ..schemas import HealthResponse        # .. means "go up one level to api/"
from ..dependencies import get_model, DummyModel
from clv_uplift import __version__

# APIRouter is FastAPI's blueprint system. Rather than registering all
# endpoints directly on the app object, you register them on a router,
# then include the router in the app with a prefix. This keeps each
# file focused on one domain of functionality.
router = APIRouter()


@router.get(
    "",                              # path is "" because the prefix "/health"
                                     # is added when the router is included in main.py
    response_model=HealthResponse,   # FastAPI validates the return value
                                     # against this schema before sending
)
async def health_check(
    model: DummyModel = Depends(get_model)   # dependency injection
):
    """
    Liveness probe: confirms the service is running and the model is loaded.
    
    The Depends(get_model) expression tells FastAPI: before calling this
    function, call get_model() and pass its return value as the `model`
    parameter. This is dependency injection — the endpoint declares what
    it needs, and the framework provides it.
    
    If get_model() raises an exception (e.g., model file not found),
    FastAPI catches it and returns a 500 error before this function runs.
    """
    return HealthResponse(
        status="ok",
        model_loaded=True,    # if we got here, get_model() succeeded
        version=__version__,
    )