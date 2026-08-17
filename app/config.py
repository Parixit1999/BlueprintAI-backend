from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    aws_region: str = "us-east-1"
    s3_bucket: str = "blueprintai-files"
    # Set to http://localhost:9000 for local MinIO; leave unset for real S3
    s3_endpoint_url: str | None = None
    # Endpoint browsers can reach, used only for presigned URLs. Needed when
    # the backend runs in docker (it reaches MinIO as http://minio:9000, a
    # hostname the user's browser cannot resolve). Defaults to s3_endpoint_url.
    s3_public_endpoint_url: str | None = None
    s3_access_key: str | None = None
    s3_secret_key: str | None = None

    # Current Claude models on Bedrock require the cross-region inference
    # profile ID (us. prefix) - the bare model ID is rejected for on-demand
    bedrock_vision_model: str = "us.anthropic.claude-opus-4-8"
    bedrock_text_model: str = "us.anthropic.claude-opus-4-8"
    bedrock_embed_model: str = "amazon.titan-embed-text-v2:0"

    # Pipeline concurrency. Embedding calls are small and fast - 8 parallel
    # Titan calls stay well under Bedrock quotas and cut dense-sheet ingest
    # ~8x. Vision calls are heavy (32K-token responses) - 2 parallel pages
    # roughly halves multi-page extraction without tripping throttles.
    embed_concurrency: int = 8
    vision_page_concurrency: int = 2
    # Documents extracted at once per instance. Extraction shares the process
    # with the API: size this with the task's CPU (rasterization is the
    # CPU-heavy part) so background work never starves request handling.
    extract_concurrency: int = 2

    # Hybrid OCR: Amazon Textract reads text at full resolution with
    # pixel-accurate boxes; the vision model uses it as a reference and its
    # boxes snap to matching lines. Best-effort - missing permission or
    # offline development silently falls back to vision-only.
    textract_enabled: bool = True

    # "ollama" (local) or "bedrock" (AWS) — controls embeddings + generation
    ai_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_embed_model: str = "snowflake-arctic-embed:335m"  # 1024-dim, same as Titan v2
    ollama_text_model: str = "llama3.1:8b"
    ollama_vision_model: str = "llama3.2-vision:11b"
    # Vision/generation responses are streamed, so this is the max gap between
    # chunks (covers model load + time-to-first-token), not the whole-response
    # budget. Local vision on a detailed scanned drawing can be slow.
    ollama_read_timeout: float = 300.0

    # DWG support: path to the free ODA File Converter (the amd64 docker
    # image ships it as /usr/local/bin/oda-convert and sets this env var).
    # When unset or missing, conversion falls back to bundled LibreDWG,
    # which cannot read AutoCAD 2018+ (AC1032) files.
    oda_converter_path: str | None = None

    # Streamed to S3/disk (never fully in memory), so the ceiling is about
    # processing sanity, not RAM. Revit models commonly run to 1 GB.
    max_upload_bytes: int = 1024 * 1024 * 1024  # 1 GB

    # Render derivatives served through CloudFront (production): the render
    # URL becomes the /renders/... path on the app's own domain, which a
    # dedicated CloudFront behavior caches at the edge - repeat views load
    # instantly and browsers can cache (presigned URLs change every request,
    # defeating both caches). Off locally: no CloudFront, so presigned URLs.
    render_cdn: bool = False

    # Region-count guard: a massive DXF emitted 98k text entities - unreviewable
    # in any UI and pure noise for retrieval. Extraction keeps the most
    # information-dense regions and discloses the truncation via an advisory.
    max_regions_per_file: int = 5000

    # --- Component detection (services/extraction/components.py) ---
    # One whole-sheet vision pass cannot see small components: an E-size sheet
    # downscaled to the model's 1568px input turns a 1/2" valve symbol into two
    # pixels. The fix is to zoom in - crop suspiciously large boxes, then sweep
    # the sheet in tiles - which costs one vision call per crop. This is the
    # cost ceiling for that zooming, per sheet. 0 disables every extra call and
    # restores single-pass behaviour (the free improvements - tighter boxes,
    # OCR snapping, keynote linking - still apply). Set 0 for AI_PROVIDER=ollama:
    # local vision takes ~60s per call, so 4 extra calls per sheet is minutes.
    component_detail_calls: int = 4
    # A component box covering at least this much of the sheet AND wide/tall in
    # both axes is treated as a GROUP the model failed to break apart (one box
    # around a pump + its base + its piping). It gets cropped and re-read so
    # each element inside comes back with its own box. Measured: genuine single
    # components on archive sheets sit well under 10% of sheet area; the boxes
    # users complained about ran 30-80%.
    component_oversized_area_pct: float = 12.0
    # Tile sweep trigger: the ratio between the sheet's real long side and the
    # 1568px the model actually sees. Below this the overview pass already read
    # the sheet at near-native resolution and tiling would only re-find what it
    # found. 1.6 means "the model lost a third of the detail or more".
    component_tile_min_scale: float = 1.6
    # Components the vision model could not name get one consolidated naming
    # call per sheet: their crops are composed into a numbered contact sheet so
    # a single request names them all. This caps how many ride in that image -
    # beyond ~12 the panels get too small to read.
    component_max_named_per_sheet: int = 12

    # Two documents whose embeddings are at least this cosine-similar are
    # flagged as possible duplicates. Calibrated on real drawings: the same
    # drawing across formats scores ~0.98, the same drawing extracted by
    # DIFFERENT pipeline versions ~0.88, the closest genuinely-different
    # pair (same drawing series) ~0.66. 0.85 catches cross-version copies
    # with a wide margin above real neighbors.
    duplicate_similarity_threshold: float = 0.85

    # Single-user mode for now; auth later just replaces this per-request.
    default_user_id: str = "global"

    # Seed password for the first 'admin' account (used only when the users
    # table is empty). Leave unset to have a random one generated and logged
    # once at startup.
    initial_admin_password: str | None = None

    database_url: str = "postgresql://postgres:postgres@localhost:5432/blueprintai"

    # RDS (Phase B): when both are set, the master credentials are fetched
    # from Secrets Manager at startup and the connection URL is built from
    # them - the password never appears in env files, compose, or logs.
    # Leave unset (or empty) to use database_url directly.
    database_secret_id: str | None = None
    database_host: str | None = None


settings = Settings()
