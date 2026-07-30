# Jojoba Travel Agency — Backend Blueprint

A FastAPI + MongoDB backend with a production-oriented middleware stack: request
correlation, structured logging, centralized error handling, security headers,
rate limiting, and CORS.

## Stack

- **FastAPI** — web framework
- **Motor** — async MongoDB driver
- **Pydantic Settings** — typed, `.env`-driven configuration
- **Uvicorn** — ASGI server

## Project Structure

```
jojoba-travel-agency/
├── .env                        # local environment config (not committed)
├── .env.example                # template for required env vars
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── main.py                 # app factory, lifespan, middleware wiring
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py         # Settings(BaseSettings) + get_settings()
│   │   └── database.py         # Mongo connect/close/get_db
│   ├── middleware/
│   │   ├── __init__.py
│   │   ├── request_context.py  # X-Request-ID + Server-Timing + ContextVar
│   │   ├── logging_middleware.py
│   │   ├── security_headers.py
│   │   ├── error_middleware.py # HTTPException / validation / generic handlers
│   │   └── rate_limit.py       # Depends(check_rate_limit)
│   ├── routers/                # feature routers, mounted in main.py
│   ├── services/                # business logic
│   └── utils/
│       └── time.py             # get_time_now_wib()
└── tests/
```

## Configuration

All config is centralized in `src/config/settings.py` via a single
`Settings(BaseSettings)` class, cached with `@lru_cache` through `get_settings()`.
Never read `os.getenv(...)` directly elsewhere in the app — add a field to
`Settings` instead, so `.env` stays the single source of truth.

| Env var         | Field            | Default                          |
|------------------|------------------|-----------------------------------|
| `MODE`           | `environment`    | `DEV` (normalized to lowercase)  |
| `APP_NAME`       | `app_name`       | `Jojoba Travel Dashboard`        |
| `APP_VERSION`    | `app_version`    | `v1.0.0`                         |
| `DEBUG`          | `debug`          | `False`                          |
| `HOST`           | `api_host`       | `0.0.0.0`                        |
| `PORT`           | `api_port`       | `8001`                           |
| `MONGO_URL`      | `mongo_url_dev`  | `mongodb://127.0.0.1:27017`      |
| `MONGO_URL_2`    | `mongo_url_prod` | `mongodb://127.0.0.1:27017`      |
| `DATABASE_NAME`  | `database_name`  | —                                 |
| `CORS_ORIGINS`   | `cors_origins_raw` | `http://localhost:3000,http://localhost:3001` |
| `RATE_LIMIT`     | `rate_limit`     | `50`                             |
| `RATE_LIMIT_PERIOD` | `rate_limit_period` | `60` (seconds)               |

## Running

From the project root (the folder containing `src/`):

```bash
python -m src.main
```

This reads `HOST`/`PORT`/`DEBUG` from `.env` automatically via `Settings`.

Alternative (CLI, ignores `.env` — pass flags manually):

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8001
```

## Middleware Stack

Starlette runs middleware in **reverse of registration order** (last
registered = outermost = runs first on the way in). Registered in `main.py`
bottom-to-top of this execution order:

1. **`SecurityHeadersMiddleware`** *(outermost)*
   Sets `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`,
   `Permissions-Policy`, `Cross-Origin-Opener-Policy` on every response.
   `Strict-Transport-Security` is opt-in via `hsts=True`, enabled only when
   `environment == "production"`.

2. **`RequestContextMiddleware`**
   Assigns every request an `X-Request-ID` (reused from the client if
   provided, else generated), stores it on `request.state.request_id` and in
   a `ContextVar` (`current_request_id()`) so any code in the call stack —
   services, DB layer, background tasks — can access it without threading it
   through function signatures. Also sets the `Server-Timing` response
   header. This is the backbone that ties logs together across a request.

3. **`LoggingMiddleware`**
   Logs method/path on request-in and status/duration on response-out, tagged
   with the request ID from step 2. Must run after `RequestContextMiddleware`.

4. **`CORSMiddleware`** *(innermost of this group)*
   Origins come from `CORS_ORIGINS` in `.env`. Never combine `allow_origins=["*"]`
   with `allow_credentials=True` — browsers reject that combination outright.

**Error handling** is *not* middleware — it uses FastAPI's native
`add_exception_handler()`, registered separately:

- `HTTPException` → structured JSON with status/detail
- `RequestValidationError` → 422 with field-level detail
- `Exception` (catch-all) → 500, generic message to the client, full
  traceback logged server-side only

**Rate limiting** is a per-route/router dependency, not global middleware:

```python
from fastapi import Depends
from src.middleware.rate_limit import check_rate_limit

router = APIRouter(dependencies=[Depends(check_rate_limit)])
```

> Current implementation is in-memory (`defaultdict`), which only enforces
> correctly with a **single worker process**. Scaling to multiple
> Uvicorn workers or replicas requires moving the counter store to Redis so
> all processes share state — see `rate_limit.py` for the note.

## Health Check

```
GET /health → {"status": "ok", "environment": "dev"}
```

## Known Trade-offs / Follow-ups

- Rate limiting is single-process only until backed by Redis.
- `/docs` and `/redoc` are disabled when `environment == "production"`, since
  `SecurityHeadersMiddleware` does not set a CSP and Swagger UI serves HTML/JS.
- `BaseHTTPMiddleware`-based middleware (all four above) has known Starlette
  edge cases around streaming responses and client disconnects — acceptable
  here since none of them touch the response body, but worth knowing before
  adding a middleware that does.