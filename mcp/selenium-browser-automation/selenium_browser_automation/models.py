"""Pydantic models for Selenium browser automation MCP server."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import pydantic
import pydantic.alias_generators

# Browser selection for Selenium automation
# Use "chromium" to avoid AppleScript targeting conflicts when personal Chrome is running
type Browser = Literal['chrome', 'chromium']


class BaseModel(pydantic.BaseModel):
    """Base model with strict validation - no extra fields, all fields required unless Optional."""

    model_config = pydantic.ConfigDict(extra='forbid', strict=True)


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
    signin_with_credential_provider: bool | None = pydantic.Field(None, alias='signin.with_credential_provider')


class ChromeLocalStateProfile(BaseModel):
    """Profile section from Local State JSON. Strict validation."""

    info_cache: dict[str, ChromeInfoCacheEntry]
    last_active_profiles: list[str] | None = None
    last_used: str | None = None
    metrics: dict[str, Any] | None = None  # e.g., {"next_bucket_index": 6}
    picker_shown: bool | None = None
    profile_counts_reported: str | None = None
    profiles_created: int | None = None
    profiles_order: list[str] | None = None


class ChromeLocalState(BaseModel):
    """Chrome Local State JSON structure. Strict validation - includes ALL fields."""

    profile: ChromeLocalStateProfile

    # Top-level Chrome settings and state fields (alphabetically sorted)
    accessibility: dict[str, Any] | None = None
    app_shims: dict[str, Any] | None = None
    app_shims_cdhash_hmac_key: str | None = None
    autofill: dict[str, Any] | None = None
    background_tracing: dict[str, Any] | None = None
    breadcrumbs: dict[str, Any] | None = None
    browser: dict[str, Any] | None = None
    cloned_install: dict[str, Any] | None = None  # Chrome 120+ install tracking
    glic: dict[str, Any] | None = None  # Glic multi-instance settings
    hardware_acceleration_mode_previous: bool | None = None
    legacy: dict[str, Any] | None = None
    local: dict[str, Any] | None = None
    management: dict[str, Any] | None = None
    network_time: dict[str, Any] | None = None
    optimization_guide: dict[str, Any] | None = None
    origin_trials: dict[str, Any] | None = None
    password_manager: dict[str, Any] | None = None
    performance_intervention: dict[str, Any] | None = None
    performance_tuning: dict[str, Any] | None = None
    policy: dict[str, Any] | None = None
    privacy_budget: dict[str, Any] | None = None
    profile_network_context_service: dict[str, Any] | None = None
    profiles: dict[str, Any] | None = None  # Note: different from 'profile' section
    restart: dict[str, Any] | None = None  # Session restart state
    segmentation_platform: dict[str, Any] | None = None
    session_id_generator_last_value: str | None = None
    signin: dict[str, Any] | None = None
    sm: dict[str, Any] | None = None  # Chrome sync/management state
    subresource_filter: dict[str, Any] | None = None
    tab_stats: dict[str, Any] | None = None
    task_manager: dict[str, Any] | None = None  # Task manager window state
    toast: dict[str, Any] | None = None
    tpcd: dict[str, Any] | None = None
    tpcd_experiment: dict[str, Any] | None = None
    ukm: dict[str, Any] | None = None
    uninstall_metrics: dict[str, Any] | None = None
    updateclientdata: dict[str, Any] | None = None
    updateclientlastupdatecheckerror: int | None = None
    updateclientlastupdatecheckerrorcategory: int | None = None
    updateclientlastupdatecheckerrorextracode1: int | None = None
    user_experience_metrics: dict[str, Any] | None = None

    # Variations framework fields (A/B testing)
    variations_compressed_seed: str | None = None
    variations_country: str | None = None
    variations_crash_streak: int | None = None
    variations_failed_to_fetch_seed_streak: int | None = None
    variations_google_groups: dict[str, Any] | None = None
    variations_last_fetch_time: str | None = None
    variations_permanent_consistency_country: list[str] | None = None
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
    was: dict[str, Any] | None = None


# =============================================================================
# Core Web Vitals Models
# =============================================================================

MetricRating = Literal['good', 'needs-improvement', 'poor']


class WebVitalMetric(pydantic.BaseModel):
    """Base model for Web Vital metrics - allows extra fields from JS."""

    model_config = pydantic.ConfigDict(extra='ignore')

    name: str
    value: float  # milliseconds for timing, unitless for CLS
    rating: MetricRating


class FCPMetric(WebVitalMetric):
    """First Contentful Paint metric."""

    name: Literal['FCP'] = 'FCP'


class LCPMetric(WebVitalMetric):
    """Largest Contentful Paint metric with element details."""

    name: Literal['LCP'] = 'LCP'
    size: int | None = None  # Pixels
    element_id: str | None = None
    url: str | None = None


class TTFBPhases(pydantic.BaseModel):
    """TTFB timing phase breakdown."""

    model_config = pydantic.ConfigDict(extra='ignore')

    dns: float = 0
    tcp: float = 0
    request: float = 0


class TTFBMetric(WebVitalMetric):
    """Time to First Byte with phase breakdown."""

    name: Literal['TTFB'] = 'TTFB'
    phases: TTFBPhases | None = None


class LayoutShiftSource(pydantic.BaseModel):
    """Source element that caused layout shift."""

    model_config = pydantic.ConfigDict(extra='ignore')

    node: str | None = None


class LayoutShiftEntry(pydantic.BaseModel):
    """Individual layout shift entry."""

    model_config = pydantic.ConfigDict(extra='ignore')

    value: float
    time: float
    sources: list[LayoutShiftSource] = []


class CLSMetric(WebVitalMetric):
    """Cumulative Layout Shift with session entries."""

    name: Literal['CLS'] = 'CLS'
    entries: list[LayoutShiftEntry] = []


class INPDetails(pydantic.BaseModel):
    """INP interaction phase breakdown."""

    model_config = pydantic.ConfigDict(extra='ignore')

    duration: float
    name: str
    start_time: float
    input_delay: float
    processing_time: float
    presentation_delay: float


class INPMetric(WebVitalMetric):
    """Interaction to Next Paint metric."""

    name: Literal['INP'] = 'INP'
    details: INPDetails | None = None


class CoreWebVitals(pydantic.BaseModel):
    """Complete Core Web Vitals report."""

    model_config = pydantic.ConfigDict(extra='ignore')

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

    model_config = pydantic.ConfigDict(extra='ignore')

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
        return self.blocked + self.dns + self.connect + self.ssl + self.send + self.wait + self.receive


class NetworkRequest(pydantic.BaseModel):
    """Individual network request with timing data."""

    model_config = pydantic.ConfigDict(extra='ignore')

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

    model_config = pydantic.ConfigDict(extra='ignore')

    url: str
    timestamp: float
    requests: list[NetworkRequest]
    total_requests: int
    total_size_bytes: int
    total_time_ms: float

    # Summary statistics
    slowest_requests: list[dict[str, Any]] = []  # [{url, duration_ms, status}]
    requests_by_type: dict[str, int] = {}  # {document: 1, xhr: 5, ...}

    errors: list[str] = []


# =============================================================================
# JavaScript Execution Models
# =============================================================================

# Constrained types for JavaScript execution results
JavaScriptResultType = Literal[
    'string',
    'number',
    'boolean',
    'object',
    'array',
    'null',
    'undefined',
    'bigint',
    'symbol',
    'function',
    'error',
    'unserializable',
]

JavaScriptErrorType = Literal['timeout', 'execution']


class JavaScriptResult(BaseModel):
    """Result of JavaScript execution in browser context.

    Contains the execution outcome with typed result and error details.
    The result field holds JSON-serializable values; non-serializable values
    (DOM nodes, functions, etc.) return null with result_type explaining why.
    """

    success: bool
    result: str | int | float | bool | dict[str, Any] | list[Any] | None = None
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
    errors: list[dict[str, Any]]


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
    elapsed_seconds: float | None = None  # Time taken for the operation in seconds


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
    text: str  # Full text if small, or preview (first 2000 chars) if saved to file
    character_count: int  # Original text length (not preview length)
    source_element: str  # What was extracted: "main", "article", "body", or CSS selector

    # Large output handling (when character_count > 25K threshold)
    saved_to_file: bool = False  # True if text was saved to file
    file_path: str | None = None  # Path to saved file (when saved_to_file=True)

    # Smart extraction transparency (only present for selector='auto')
    smart_info: SmartExtractionInfo | None = None


# =============================================================================
# Profile State Models (Browser state persistence)
# =============================================================================

SameSitePolicy = Literal['Strict', 'Lax', 'None']


class ProfileStateCookie(BaseModel):
    """Cookie in profile state format.

    Cookies are stored at the top level of ProfileState (not under origins) because
    they use domain+path scoping, not strict origin scoping. A single cookie for
    ".example.com" applies to multiple origins: https://example.com,
    https://www.example.com, https://api.example.com, etc.

    Session cookies use expires=-1, persistent cookies use epoch timestamp.
    """

    name: str
    value: str
    domain: str
    path: str
    expires: float  # Epoch seconds, -1 for session cookies
    http_only: bool
    secure: bool
    same_site: SameSitePolicy


class ProfileStateIndexedDBRecord(BaseModel):
    """IndexedDB record - key/value pair.

    Keys can be strings, numbers, dates (as ISO strings), or arrays (compound keys).
    Values are JSON-serializable representations of stored objects.

    Complex types (Date, Map, Set, ArrayBuffer) are serialized with __type markers:
    - Date: {"__type": "Date", "__value": "2024-01-01T00:00:00.000Z"}
    - Map: {"__type": "Map", "__value": [[key, value], ...]}
    - Set: {"__type": "Set", "__value": [item, ...]}
    - ArrayBuffer: {"__type": "ArrayBuffer", "__value": [byte, byte, ...]}
    """

    model_config = pydantic.ConfigDict(extra='forbid', strict=False)  # Allow flexible JSON values

    key: str | int | float | list[Any] | None
    value: str | int | float | bool | dict[str, Any] | list[Any] | None


class ProfileStateIndexedDBIndex(BaseModel):
    """IndexedDB object store index metadata.

    Serializes to camelCase for JavaScript compatibility:
    key_path → keyPath, multi_entry → multiEntry
    """

    model_config = pydantic.ConfigDict(
        alias_generator=pydantic.alias_generators.to_camel,
        populate_by_name=True,
    )

    name: str
    key_path: str | Sequence[str]
    unique: bool
    multi_entry: bool


class ProfileStateIndexedDBObjectStore(BaseModel):
    """IndexedDB object store with schema and data.

    Captures the complete state of an object store including:
    - Schema: key_path, auto_increment, indexes
    - Data: all records as key/value pairs

    Serializes to camelCase for JavaScript compatibility:
    key_path → keyPath, auto_increment → autoIncrement
    """

    model_config = pydantic.ConfigDict(
        alias_generator=pydantic.alias_generators.to_camel,
        populate_by_name=True,
    )

    name: str
    key_path: str | Sequence[str] | None
    auto_increment: bool
    indexes: Sequence[ProfileStateIndexedDBIndex]
    records: Sequence[ProfileStateIndexedDBRecord]


class ProfileStateIndexedDB(BaseModel):
    """IndexedDB database with version and object stores.

    The version number is critical for schema migrations.

    Serializes to camelCase for JavaScript compatibility:
    database_name → databaseName, object_stores → objectStores
    """

    model_config = pydantic.ConfigDict(
        alias_generator=pydantic.alias_generators.to_camel,
        populate_by_name=True,
    )

    database_name: str
    version: int
    object_stores: Sequence[ProfileStateIndexedDBObjectStore]


class ProfileStateOriginStorage(BaseModel):
    """Storage data for a single origin.

    Each origin (scheme://host:port) has isolated storage per the same-origin policy.
    All storage types for this origin are grouped together.

    Note: session_storage is session-scoped by design. Restored session_storage
    persists only for the lifetime of the browser context - closing the browser
    clears it. For cross-session persistence, use local_storage or cookies.
    """

    local_storage: dict[str, str]
    session_storage: dict[str, str] | None = None
    indexed_db: Sequence[ProfileStateIndexedDB] | None = None


class ProfileState(BaseModel):
    """Browser profile state for session persistence.

    Captures browser state that maintains authenticated sessions:
    - Cookies (domain-scoped, stored at top level)
    - Per-origin storage (localStorage, sessionStorage, IndexedDB)

    Design Principles:
    - Cookies are top-level because they use domain+path scoping, not origin scoping.
      A single cookie can match multiple origins.
    - Storage is origin-keyed because localStorage/sessionStorage/IndexedDB use strict
      origin scoping (scheme://host:port must match exactly).
    - Format designed for future expansion: extensions, permissions, preferences.

    See docs/browser-data-taxonomy.md for detailed architecture documentation.
    """

    schema_version: str = '1.0'
    captured_at: str | None = None  # ISO 8601 timestamp

    # Cookies: domain+path scoped, top-level (see docstring for rationale)
    cookies: Sequence[ProfileStateCookie]

    # Storage: origin-scoped, keyed by origin string (e.g., "https://example.com")
    origins: dict[str, ProfileStateOriginStorage]

    # Future expansion slots (not yet implemented)
    extensions: dict[str, Any] | None = None
    permissions: dict[str, Any] | None = None
    preferences: dict[str, Any] | None = None


class SaveProfileStateResult(BaseModel):
    """Result from save_profile_state operation."""

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


class ConsoleLogEntry(BaseModel):
    """Individual browser console log entry."""

    level: str  # SEVERE, WARNING, INFO
    message: str
    source: str  # javascript, network, security, etc.
    timestamp: int  # Epoch milliseconds


class ConsoleLogsResult(BaseModel):
    """Result of get_console_logs operation."""

    logs: Sequence[ConsoleLogEntry]
    total_count: int
    # Breakdown by level for quick triage
    severe_count: int = 0
    warning_count: int = 0
    info_count: int = 0


# =============================================================================
# Chrome Profile State Export Models
# =============================================================================


class ChromeProfileStateExportResult(BaseModel):
    """Result of exporting Chrome profile state from profile files.

    Returned by export_chrome_profile_state() which reads profile state directly
    from Chrome's profile directory (cookies, localStorage, sessionStorage,
    IndexedDB) for use in Selenium automation.

    Complements save_profile_state() which exports from a running Selenium browser.
    Use export_chrome_profile_state when you've logged in manually and want to
    capture that authenticated state for automation.

    IndexedDB includes full schema (version, key_path, auto_increment, indexes)
    via dfindexeddb library, enabling complete database restoration.

    sessionStorage Extraction:
        On macOS, attempts to extract LIVE sessionStorage from open Chrome tabs
        via AppleScript (session_storage_source='live'). Falls back to disk
        (session_storage_source='disk') if Chrome isn't running. FAILS if
        Chrome is running but the setting is disabled - user must enable:
        Chrome > View > Developer > Allow JavaScript from Apple Events
    """

    path: str
    cookie_count: int
    origin_count: int
    local_storage_keys: int
    session_storage_keys: int
    indexeddb_origins: int
    session_storage_source: Literal['live', 'disk'] = 'disk'
    warnings: Sequence[str] = []
