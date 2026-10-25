import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

# Router imports (Ensure these files exist in your project structure)
from api.routes import health, predictions

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    Handles model warmup and resource allocation.
    """
    # ---- STARTUP LOGIC ----
    logger.info("Initializing FastAPI Application...")
    
    # Simulate or execute ML model warmup here
    logger.info("Warming up Machine Learning models...")
    # Example: global_model_pool.load_models()
    
    logger.info("Application startup complete and ready to accept connections.")
    
    yield  # Application runs here
    
    # ---- SHUTDOWN LOGIC ----
    logger.info("Shutting down Application...")
    logger.info("Cleaning up resources and closing connections.")
    logger.info("Application shutdown complete.")

# Initialize FastAPI instance with Metadata
app = FastAPI(
    title="ML Inference API",
    description="Backend API for managing model predictions and health checks.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware Configuration (Critical for Streamlit integration)
origins = [
    "http://localhost:8501",      # Default Streamlit local port
    "http://127.0.0.1:8501",    # Alternative local address
    # Yahan aap apna production Streamlit URL bhi add kar sakte hain
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],          # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],          # Allows all HTTP headers
)

# Include Routers with /api/v1 Prefix
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(predictions.router, prefix="/api/v1", tags=["Predictions"])

@app.get("/", tags=["Root"])
async def root():
    """
    Root route that redirects users to the interactive API documentation.
    """
    logger.info("Root endpoint hit. Redirecting to API documentation.")
    return RedirectResponse(url="/docs")
