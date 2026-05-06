"""grok-kit: MCP server + CLI mirroring grok.com conversations.

Architecture: hand-authored OpenAPI spec at ../api-spec/openapi.yaml is the
source of truth for the API surface. The Speakeasy-generated grok-kit-sdk/
sub-package handles HTTP transport, typed models, and error mapping. This
package is the consumer — auth bootstrap, transport plumbing, business logic,
MCP tool exposure, CLI commands.
"""

from __future__ import annotations

__all__ = []
