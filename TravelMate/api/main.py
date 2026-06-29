from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="TravelMate API", version="2.0", description="AI智能出行助手 API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "TravelMate", "version": "2.0"}


# Mount sub-routers
try:
    from api.routes.chat import router as chat_router
    app.include_router(chat_router, prefix="/api")
except ImportError:
    pass

try:
    from api.routes.trips import router as trips_router
    app.include_router(trips_router, prefix="/api")
except ImportError:
    pass
