"""
svc -- Service layer (FastAPI, Typer, pydantic-settings, SQLAlchemy, httpx+tenacity).

Framework code lives only in this package; core modules stay stdlib+scipy.
The package exports nothing heavy: do not import mlops.svc to avoid loading FastAPI.
"""
