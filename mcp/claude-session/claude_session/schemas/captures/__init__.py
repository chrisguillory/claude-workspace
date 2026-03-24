"""
HTTP traffic capture modeling with type-safe validation.

This module provides Pydantic v2 models for capturing, validating, and
analyzing HTTP traffic from the Claude Code observability platform.

Usage:
    from claude_session.schemas.captures import load_capture, CapturedTraffic

    capture = load_capture(Path("captures/req_001_api_anthropic_com_v1_messages.json"))

    if isinstance(capture, MessagesRequestCapture):
        print(f"Model: {capture.body.model}")
        print(f"Messages: {len(capture.body.messages)}")

For the full philosophy and design decisions, see README.md in this directory.
"""

from __future__ import annotations

from claude_session.schemas.captures.anthropic import (
    AccountSettingsRequestCapture,
    AccountSettingsResponseCapture,
    ApiError,
    ApiErrorDetail,
    ClientDataRequestCapture,
    ClientDataResponseCapture,
    CountTokensRequestCapture,
    CountTokensResponseCapture,
    CreateApiKeyRequestCapture,
    CreateApiKeyResponseCapture,
    EvalRequestCapture,
    EvalResponseCapture,
    GroveRequestCapture,
    GroveResponseCapture,
    HelloRequestCapture,
    HelloResponseCapture,
    MessagesJsonResponseCapture,
    MessagesRequestCapture,
    MessagesStreamResponseCapture,
    MetricsEnabledRequestCapture,
    MetricsEnabledResponseCapture,
    MetricsRequestCapture,
    MetricsResponseCapture,
    ModelAccessRequestCapture,
    ModelAccessResponseCapture,
    ProfileRequestCapture,
    ProfileResponseCapture,
    ReferralEligibilityRequestCapture,
    ReferralEligibilityResponseCapture,
    ReferralRedemptionsRequestCapture,
    ReferralRedemptionsResponseCapture,
    RolesRequestCapture,
    RolesResponseCapture,
    SettingsRequestCapture,
    SettingsResponseCapture,
    TelemetryRequestCapture,
    TelemetryResponseCapture,
)
from claude_session.schemas.captures.base import (
    AnthropicRequestCapture,
    AnthropicResponseCapture,
    ClientConnection,
    ClientConnectionTiming,
    IPv4Address,
    IPv6Address,
    NetworkAddress,
    RequestCapture,
    ResponseCapture,
    ServerConnection,
    ServerConnectionTiming,
    StatsigRequestCapture,
    StatsigResponseCapture,
)
from claude_session.schemas.captures.datadog import (
    DatadogApiErrorLogEntry,
    DatadogApiSuccessLogEntry,
    DatadogLogEntry,
    DatadogLogEntryBase,
    DatadogOAuthSuccessLogEntry,
    DatadogRequestCapture,
    DatadogResponseCapture,
    DatadogToolUseSuccessLogEntry,
)
from claude_session.schemas.captures.external import (
    CodeClaudeComDocRequestCapture,
    CodeClaudeComDocResponseCapture,
    DomainInfoRequestCapture,
    DomainInfoResponseCapture,
    OAuthTokenRequestCapture,
    OAuthTokenResponseCapture,
    PlatformClaudeComDocRequestCapture,
    PlatformClaudeComDocResponseCapture,
)
from claude_session.schemas.captures.gcs import (
    GCSVersionRequestCapture,
    GCSVersionResponseCapture,
    RawTextBody,
)
from claude_session.schemas.captures.loader import (
    CapturedTraffic,
    load_capture,
    load_captures_batch,
)
from claude_session.schemas.captures.proxy import (
    ProxyErrorCapture,
    ProxyErrorDetail,
    ProxyErrorRequest,
)
from claude_session.schemas.captures.registry import (
    CAPTURE_REGISTRY,
    extract_endpoint_from_filename,
    get_capture_type,
    normalize_path,
)
from claude_session.schemas.captures.segment import (
    SegmentBatchRequestCapture,
    SegmentBatchResponseCapture,
)
from claude_session.schemas.captures.statsig import (
    StatsigInitializeRequestCapture,
    StatsigInitializeResponseCapture,
    StatsigRegisterRequestCapture,
    StatsigRegisterResponseCapture,
)
from claude_session.schemas.captures.unknown import (
    UnknownRequestCapture,
    UnknownResponseCapture,
)

__all__ = [
    # Main API
    'CapturedTraffic',
    'load_capture',
    'load_captures_batch',
    # Registry
    'CAPTURE_REGISTRY',
    'get_capture_type',
    'normalize_path',
    'extract_endpoint_from_filename',
    # Base classes
    'RequestCapture',
    'ResponseCapture',
    'AnthropicRequestCapture',
    'AnthropicResponseCapture',
    'StatsigRequestCapture',
    'StatsigResponseCapture',
    # Connection metadata
    'ClientConnectionTiming',
    'ServerConnectionTiming',
    'ClientConnection',
    'ServerConnection',
    # Network address types
    'IPv4Address',
    'IPv6Address',
    'NetworkAddress',
    # Anthropic captures - Messages
    'MessagesRequestCapture',
    'MessagesStreamResponseCapture',
    'MessagesJsonResponseCapture',
    'ApiError',
    'ApiErrorDetail',
    # Anthropic captures - Telemetry
    'TelemetryRequestCapture',
    'TelemetryResponseCapture',
    # Anthropic captures - Count Tokens
    'CountTokensRequestCapture',
    'CountTokensResponseCapture',
    # Anthropic captures - Feature Flags
    'EvalRequestCapture',
    'EvalResponseCapture',
    # Anthropic captures - Metrics
    'MetricsRequestCapture',
    'MetricsResponseCapture',
    'MetricsEnabledRequestCapture',
    'MetricsEnabledResponseCapture',
    # Anthropic captures - Health
    'HelloRequestCapture',
    'HelloResponseCapture',
    # Anthropic captures - Grove
    'GroveRequestCapture',
    'GroveResponseCapture',
    # Anthropic captures - Settings
    'SettingsRequestCapture',
    'SettingsResponseCapture',
    # Anthropic captures - OAuth
    'ClientDataRequestCapture',
    'ClientDataResponseCapture',
    'ProfileRequestCapture',
    'ProfileResponseCapture',
    'RolesRequestCapture',
    'RolesResponseCapture',
    'AccountSettingsRequestCapture',
    'AccountSettingsResponseCapture',
    'CreateApiKeyRequestCapture',
    'CreateApiKeyResponseCapture',
    # Anthropic captures - Referral
    'ReferralEligibilityRequestCapture',
    'ReferralEligibilityResponseCapture',
    'ReferralRedemptionsRequestCapture',
    'ReferralRedemptionsResponseCapture',
    # Anthropic captures - Model Access
    'ModelAccessRequestCapture',
    'ModelAccessResponseCapture',
    # Statsig captures
    'StatsigRegisterRequestCapture',
    'StatsigRegisterResponseCapture',
    'StatsigInitializeRequestCapture',
    'StatsigInitializeResponseCapture',
    # Datadog captures
    'DatadogRequestCapture',
    'DatadogResponseCapture',
    'DatadogLogEntryBase',
    'DatadogApiSuccessLogEntry',
    'DatadogApiErrorLogEntry',
    'DatadogToolUseSuccessLogEntry',
    'DatadogOAuthSuccessLogEntry',
    'DatadogLogEntry',
    # GCS captures
    'GCSVersionRequestCapture',
    'GCSVersionResponseCapture',
    'RawTextBody',
    # External captures - OAuth
    'OAuthTokenRequestCapture',
    'OAuthTokenResponseCapture',
    # External captures - Segment
    'SegmentBatchRequestCapture',
    'SegmentBatchResponseCapture',
    # External captures - Domain Info
    'DomainInfoRequestCapture',
    'DomainInfoResponseCapture',
    # External captures - Doc Fetches
    'CodeClaudeComDocRequestCapture',
    'CodeClaudeComDocResponseCapture',
    'PlatformClaudeComDocRequestCapture',
    'PlatformClaudeComDocResponseCapture',
    # Fallback captures (unknown.py - application layer)
    'UnknownRequestCapture',
    'UnknownResponseCapture',
    # Proxy infrastructure (proxy.py - infrastructure layer)
    'ProxyErrorCapture',
    'ProxyErrorDetail',
    'ProxyErrorRequest',
]
