"""Pydantic models for Selenium browser automation MCP server."""

from __future__ import annotations

from typing import Literal, Sequence

import pydantic


class BaseModel(pydantic.BaseModel):
    """Base model with strict validation - no extra fields, all fields required unless Optional."""

    model_config = pydantic.ConfigDict(extra="forbid", strict=True)


class ChromeProfileEssential(BaseModel):
    """Essential profile metadata (default/concise view)."""

    profile_dir: str  # "Default", "Profile 1", etc.
    name: str  # Display name
    user_name: str | None  # Primary email address
    gaia_name: str | None  # Full name from Google
    gaia_id: str | None  # Google account ID
    avatar_icon: str  # chrome://theme/... or path
    is_managed: bool | None  # Enterprise managed (sometimes missing in Chrome data)
    is_ephemeral: bool  # Guest/temporary profile
    active_time: float | None  # Last active timestamp


class ChromeProfileFull(ChromeProfileEssential):
    """Full profile metadata (verbose view) - extends essential."""

    # Additional Local State fields
    background_apps: bool | None = None
    hosted_domain: str | None = None  # "NO_HOSTED_DOMAIN" or domain
    is_using_default_name: bool | None = None
    is_using_default_avatar: bool | None = None
    is_consented_primary_account: bool | None = None
    gaia_given_name: str | None = None
    gaia_picture_file_name: str | None = None
    last_downloaded_gaia_picture_url_with_size: str | None = None
    profile_color_seed: int | None = None
    profile_highlight_color: int | None = None
    metrics_bucket_index: int | None = None
    first_account_name_hash: int | None = None
    force_signin_profile_locked: bool | None = None
    is_glic_eligible: bool | None = None
    is_using_new_placeholder_avatar_icon: bool | None = None
    user_accepted_account_management: bool | None = None
    default_avatar_fill_color: int | None = None
    default_avatar_stroke_color: int | None = None


class ChromeProfilesResult(BaseModel):
    """Result of list_chrome_profiles tool."""

    profiles: list[ChromeProfileEssential | ChromeProfileFull]
    total_count: int
    default_profile: str | None  # Which profile is default
    chrome_base_path: str  # Base Chrome directory path


# Chrome JSON structure models (for parsing Chrome's Local State and Preferences files)


class ChromeInfoCacheEntry(BaseModel):
    """Profile entry from Local State -> profile.info_cache[profile_dir].

    Strict validation - will fail if Chrome adds unknown fields.
    """

    # Essential fields
    name: str
    avatar_icon: str
    is_managed: int | None = None  # 0 or 1, but sometimes missing (e.g., Profile 4)
    is_ephemeral: bool
    active_time: float
    user_name: str | None = None
    gaia_name: str | None = None
    gaia_id: str | None = None

    # Additional metadata fields
    background_apps: bool | None = None
    hosted_domain: str | None = None
    is_using_default_name: bool | None = None
    is_using_default_avatar: bool | None = None
    is_consented_primary_account: bool | None = None
    gaia_given_name: str | None = None
    gaia_picture_file_name: str | None = None
    last_downloaded_gaia_picture_url_with_size: str | None = None
    profile_color_seed: int | None = None
    profile_highlight_color: int | None = None
    metrics_bucket_index: int | None = None
    first_account_name_hash: int | None = None
    force_signin_profile_locked: bool | None = None
    is_glic_eligible: bool | None = None
    is_using_new_placeholder_avatar_icon: bool | None = None
    user_accepted_account_management: bool | None = None
    default_avatar_fill_color: int | None = None
    default_avatar_stroke_color: int | None = None
    enterprise_label: str | None = None
    has_multiple_account_names: bool | None = None
    managed_user_id: str | None = None
    signin_with_credential_provider: bool | None = pydantic.Field(
        None, alias="signin.with_credential_provider"
    )


class ChromeLocalStateProfile(BaseModel):
    """Profile section from Local State JSON. Strict validation."""

    info_cache: dict[str, ChromeInfoCacheEntry]
    last_active_profiles: list[str] | None = None
    last_used: str | None = None
    metrics: dict | None = None  # e.g., {"next_bucket_index": 6}
    picker_shown: bool | None = None
    profile_counts_reported: str | None = None
    profiles_created: int | None = None
    profiles_order: list[str] | None = None


class ChromeLocalState(BaseModel):
    """Chrome Local State JSON structure. Strict validation - includes ALL fields."""

    profile: ChromeLocalStateProfile

    # Top-level Chrome settings and state fields (alphabetically sorted)
    accessibility: dict | None = None
    app_shims: dict | None = None
    autofill: dict | None = None
    background_tracing: dict | None = None
    breadcrumbs: dict | None = None
    browser: dict | None = None
    hardware_acceleration_mode_previous: bool | None = None
    legacy: dict | None = None
    local: dict | None = None
    management: dict | None = None
    network_time: dict | None = None
    optimization_guide: dict | None = None
    origin_trials: dict | None = None
    password_manager: dict | None = None
    performance_intervention: dict | None = None
    performance_tuning: dict | None = None
    policy: dict | None = None
    privacy_budget: dict | None = None
    profile_network_context_service: dict | None = None
    segmentation_platform: dict | None = None
    session_id_generator_last_value: str | None = None
    signin: dict | None = None
    subresource_filter: dict | None = None
    tab_stats: dict | None = None
    toast: dict | None = None
    tpcd: dict | None = None
    tpcd_experiment: dict | None = None
    ukm: dict | None = None
    uninstall_metrics: dict | None = None
    updateclientdata: dict | None = None
    updateclientlastupdatecheckerror: int | None = None
    updateclientlastupdatecheckerrorcategory: int | None = None
    updateclientlastupdatecheckerrorextracode1: int | None = None
    user_experience_metrics: dict | None = None

    # Variations framework fields (A/B testing)
    variations_compressed_seed: str | None = None
    variations_country: str | None = None
    variations_crash_streak: int | None = None
    variations_failed_to_fetch_seed_streak: int | None = None
    variations_google_groups: dict | None = None
    variations_last_fetch_time: str | None = None
    variations_permanent_consistency_country: list | None = None
    variations_safe_compressed_seed: str | None = None
    variations_safe_seed_date: str | None = None
    variations_safe_seed_fetch_time: str | None = None
    variations_safe_seed_locale: str | None = None
    variations_safe_seed_milestone: int | None = None
    variations_safe_seed_permanent_consistency_country: str | None = None
    variations_safe_seed_session_consistency_country: str | None = None
    variations_safe_seed_signature: str | None = None
    variations_seed_date: str | None = None
    variations_seed_milestone: int | None = None
    variations_seed_serial_number: str | None = None
    variations_seed_signature: str | None = None
    variations_sticky_studies: str | None = None

    # Additional state
    was: dict | None = None


# =============================================================================
# Core Web Vitals Models
# =============================================================================

MetricRating = Literal["good", "needs-improvement", "poor"]


class WebVitalMetric(pydantic.BaseModel):
    """Base model for Web Vital metrics - allows extra fields from JS."""

    model_config = pydantic.ConfigDict(extra="ignore")

    name: str
    value: float  # milliseconds for timing, unitless for CLS
    rating: MetricRating


class FCPMetric(WebVitalMetric):
    """First Contentful Paint metric."""

    name: Literal["FCP"] = "FCP"


class LCPMetric(WebVitalMetric):
    """Largest Contentful Paint metric with element details."""

    name: Literal["LCP"] = "LCP"
    size: int | None = None  # Pixels
    element_id: str | None = None
    url: str | None = None


class TTFBPhases(pydantic.BaseModel):
    """TTFB timing phase breakdown."""

    model_config = pydantic.ConfigDict(extra="ignore")

    dns: float = 0
    tcp: float = 0
    request: float = 0


class TTFBMetric(WebVitalMetric):
    """Time to First Byte with phase breakdown."""

    name: Literal["TTFB"] = "TTFB"
    phases: TTFBPhases | None = None


class LayoutShiftSource(pydantic.BaseModel):
    """Source element that caused layout shift."""

    model_config = pydantic.ConfigDict(extra="ignore")

    node: str | None = None


class LayoutShiftEntry(pydantic.BaseModel):
    """Individual layout shift entry."""

    model_config = pydantic.ConfigDict(extra="ignore")

    value: float
    time: float
    sources: list[LayoutShiftSource] = []


class CLSMetric(WebVitalMetric):
    """Cumulative Layout Shift with session entries."""

    name: Literal["CLS"] = "CLS"
    entries: list[LayoutShiftEntry] = []


class INPDetails(pydantic.BaseModel):
    """INP interaction phase breakdown."""

    model_config = pydantic.ConfigDict(extra="ignore")

    duration: float
    name: str
    start_time: float
    input_delay: float
    processing_time: float
    presentation_delay: float


class INPMetric(WebVitalMetric):
    """Interaction to Next Paint metric."""

    name: Literal["INP"] = "INP"
    details: INPDetails | None = None


class CoreWebVitals(pydantic.BaseModel):
    """Complete Core Web Vitals report."""

    model_config = pydantic.ConfigDict(extra="ignore")

    url: str
    timestamp: float  # Unix timestamp when captured

    # Core metrics
    lcp: LCPMetric | None = None
    cls: CLSMetric | None = None
    inp: INPMetric | None = None

    # Supplementary metrics
    fcp: FCPMetric | None = None
    ttfb: TTFBMetric | None = None

    # Metadata
    collection_duration_ms: float
    errors: list[str] = []


# =============================================================================
# Network Timing / HAR Models
# =============================================================================


class RequestTiming(pydantic.BaseModel):
    """CDP ResourceTiming converted to milliseconds."""

    model_config = pydantic.ConfigDict(extra="ignore")

    blocked: float = 0
    dns: float = 0
    connect: float = 0
    ssl: float = 0
    send: float = 0
    wait: float = 0  # TTFB for this request
    receive: float = 0

    @property
    def total(self) -> float:
        """Total request time in milliseconds."""
        return (
            self.blocked
            + self.dns
            + self.connect
            + self.ssl
            + self.send
            + self.wait
            + self.receive
        )


class NetworkRequest(pydantic.BaseModel):
    """Individual network request with timing data."""

    model_config = pydantic.ConfigDict(extra="ignore")

    request_id: str
    url: str
    method: str
    resource_type: str | None = None
    status: int | None = None
    status_text: str | None = None
    mime_type: str | None = None
    timing: RequestTiming | None = None
    request_headers: dict[str, str] = {}
    response_headers: dict[str, str] = {}
    encoded_data_length: int = 0
    started_at: float = 0  # Wall time
    finished_at: float | None = None
    duration_ms: float | None = None  # Computed total duration
    error: str | None = None


class NetworkCapture(pydantic.BaseModel):
    """Complete network capture result."""

    model_config = pydantic.ConfigDict(extra="ignore")

    url: str
    timestamp: float
    requests: list[NetworkRequest]
    total_requests: int
    total_size_bytes: int
    total_time_ms: float

    # Summary statistics
    slowest_requests: list[dict] = []  # [{url, duration_ms, status}]
    requests_by_type: dict[str, int] = {}  # {document: 1, xhr: 5, ...}

    errors: list[str] = []


# =============================================================================
# JavaScript Execution Models
# =============================================================================

# Constrained types for JavaScript execution results
JavaScriptResultType = Literal[
    "string",
    "number",
    "boolean",
    "object",
    "array",
    "null",
    "undefined",
    "bigint",
    "symbol",
    "function",
    "error",
    "unserializable",
]

JavaScriptErrorType = Literal["timeout", "execution"]


class JavaScriptResult(BaseModel):
    """Result of JavaScript execution in browser context.

    Contains the execution outcome with typed result and error details.
    The result field holds JSON-serializable values; non-serializable values
    (DOM nodes, functions, etc.) return null with result_type explaining why.
    """

    success: bool
    result: str | int | float | bool | dict | list | None = None
    result_type: JavaScriptResultType
    error: str | None = None
    error_type: JavaScriptErrorType | None = None
    error_stack: str | None = None  # JS stack trace for debugging
    note: str | None = None  # Explanation for special cases (e.g., why result is null)


# =============================================================================
# Navigation and Page Extraction Models
# =============================================================================


class CapturedResource(BaseModel):
    """Individual captured resource from page."""

    url: str
    path: str
    absolute_path: str
    type: str
    size_bytes: int
    content_type: str
    status: int


class ResourceCapture(BaseModel):
    """Result of resource capture operation."""

    output_dir: str
    html_path: str
    captured: list[CapturedResource]
    total_size_mb: float
    resource_count: int
    errors: list[dict]


class HARExportResult(BaseModel):
    """Result of HAR export operation."""

    path: str
    entry_count: int
    size_bytes: int
    has_errors: bool = False
    errors: list[str] = []


class NavigationResult(BaseModel):
    """Result of navigation operation."""

    current_url: str
    title: str
    resources: ResourceCapture | None = None


class InteractiveElement(BaseModel):
    """Clickable element with selector for automation."""

    tag: str
    text: str
    selector: str
    cursor: str
    href: str | None
    classes: str


class FocusableElement(BaseModel):
    """Keyboard-navigable element with tab order."""

    tag: str
    text: str
    selector: str
    tab_index: int
    is_tabbable: bool
    classes: str


class SmartExtractionInfo(BaseModel):
    """Metadata about smart extraction decisions. Only present for selector='auto'."""

    fallback_used: bool  # True if no suitable main/article found, fell back to body
    body_character_count: int  # Total body chars for coverage calculation


class PageTextResult(BaseModel):
    """Result of text extraction operation."""

    # Core content (always present)
    title: str
    url: str
    text: str
    character_count: int
    source_element: str  # What was extracted: "main", "article", "body", or CSS selector

    # Smart extraction transparency (only present for selector='auto')
    smart_info: SmartExtractionInfo | None = None


# =============================================================================
# Storage State Models (Playwright-compatible session persistence)
# =============================================================================

SameSitePolicy = Literal["Strict", "Lax", "None"]


class StorageStateCookie(BaseModel):
    """Cookie in Playwright storageState format.

    Matches Playwright's cookie structure for cross-tool compatibility.
    Session cookies use expires=-1, persistent cookies use epoch timestamp.
    """

    name: str
    value: str
    domain: str
    path: str
    expires: float  # Epoch seconds, -1 for session cookies
    httpOnly: bool
    secure: bool
    sameSite: SameSitePolicy


class StorageStateLocalStorageItem(BaseModel):
    """localStorage key-value pair in storageState format."""

    name: str
    value: str


class StorageStateIndexedDBRecord(BaseModel):
    """IndexedDB record - key/value pair.

    Keys can be strings, numbers, dates (as ISO strings), or arrays (compound keys).
    Values are JSON-serializable representations of stored objects.

    Complex types (Date, Map, Set, ArrayBuffer) are serialized with __type markers:
    - Date: {"__type": "Date", "__value": "2024-01-01T00:00:00.000Z"}
    - Map: {"__type": "Map", "__value": [[key, value], ...]}
    - Set: {"__type": "Set", "__value": [item, ...]}
    - ArrayBuffer: {"__type": "ArrayBuffer", "__value": [byte, byte, ...]}
    """

    model_config = pydantic.ConfigDict(extra="forbid", strict=False)  # Allow flexible JSON values

    key: str | int | float | list | None
    value: str | int | float | bool | dict | list | None


class StorageStateIndexedDBIndex(BaseModel):
    """IndexedDB object store index metadata."""

    name: str
    keyPath: str | Sequence[str]
    unique: bool
    multiEntry: bool


class StorageStateIndexedDBObjectStore(BaseModel):
    """IndexedDB object store with schema and data.

    Captures the complete state of an object store including:
    - Schema: keyPath, autoIncrement, indexes
    - Data: all records as key/value pairs
    """

    name: str
    keyPath: str | Sequence[str] | None
    autoIncrement: bool
    indexes: Sequence[StorageStateIndexedDBIndex]
    records: Sequence[StorageStateIndexedDBRecord]


class StorageStateIndexedDB(BaseModel):
    """IndexedDB database with version and object stores.

    Playwright-compatible format for IndexedDB persistence.
    The version number is critical for schema migrations.
    """

    databaseName: str
    version: int
    objectStores: Sequence[StorageStateIndexedDBObjectStore]


class StorageStateOrigin(BaseModel):
    """Origin storage data in storageState format.

    Each origin (scheme + domain + port) has isolated storage.
    sessionStorage and IndexedDB are optional.

    Note: sessionStorage is session-scoped by design. Restored sessionStorage
    persists only for the lifetime of the browser context - closing the browser
    clears it. For cross-session persistence, use localStorage or cookies.
    """

    origin: str  # e.g., "https://www.marriott.com"
    localStorage: Sequence[StorageStateLocalStorageItem]
    sessionStorage: Sequence[StorageStateLocalStorageItem] | None = None
    indexedDB: Sequence[StorageStateIndexedDB] | None = None


class StorageState(BaseModel):
    """Playwright-compatible storage state for session persistence.

    Contains browser storage that maintains authenticated sessions.
    Export after login, import to restore auth in future sessions.

    Format matches Playwright's browserContext.storageState() for portability.
    """

    cookies: Sequence[StorageStateCookie]
    origins: Sequence[StorageStateOrigin]


class SaveStorageStateResult(BaseModel):
    """Result from save_storage_state operation."""

    path: str
    cookies_count: int
    origins_count: int
    current_origin: str
    size_bytes: int
    # IndexedDB stats (only present if include_indexeddb=True)
    indexeddb_databases_count: int | None = None
    indexeddb_records_count: int | None = None
    # All origins visited during session (for multi-origin storage capture)
    tracked_origins: Sequence[str] = []
