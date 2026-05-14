#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "cc_lib",
#   "httpx",
#   "pydantic>=2.0.0",
#   "typer>=0.9.0",
# ]
#
# [tool.uv.sources]
# cc_lib = { path = "../cc-lib/", editable = true }
# ///
"""Upload a file to Linear, print an embeddable https://uploads.linear.app/... URL.

Linear's GraphQL ``fileUpload`` mutation returns a presigned upload URL plus
the headers that must accompany the PUT. The ``assetUrl`` is the permanent
CDN link suitable for embedding in issue bodies and comments.

Usage:
    LINEAR_API_KEY=lin_api_... linear-upload path/to/file.png [--alt "alt text"]

Prints the assetUrl to stdout (one line, pipe-friendly). Prints a ready-to-paste
markdown image line to stderr.

Auth:
    Personal API keys go in the ``Authorization`` header WITHOUT the
    ``Bearer`` prefix (that's OAuth-only). See:
    https://developers.linear.app/docs/graphql/working-with-the-graphql-api
"""

from __future__ import annotations

import mimetypes
import os
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import httpx
import pydantic
import typer
from cc_lib.cli import add_install_command, create_app, run_app
from cc_lib.error_boundary import ErrorBoundary
from cc_lib.schemas.base import SubsetModel

LINEAR_GRAPHQL_URL = 'https://api.linear.app/graphql'

FILE_UPLOAD_MUTATION = """
mutation FileUpload($size: Int!, $contentType: String!, $filename: String!) {
  fileUpload(size: $size, contentType: $contentType, filename: $filename) {
    success
    uploadFile { uploadUrl assetUrl headers { key value } }
  }
}
"""


# -- Pydantic models -----------------------------------------------------------


class UploadHeader(SubsetModel):
    """A header Linear instructs us to send on the presigned PUT."""

    key: str
    value: str


class UploadFile(SubsetModel):
    """The ``uploadFile`` payload from Linear's fileUpload mutation."""

    upload_url: str
    asset_url: str
    headers: Sequence[UploadHeader] = ()

    model_config = pydantic.ConfigDict(
        populate_by_name=True,
        alias_generator=lambda f: {
            'upload_url': 'uploadUrl',
            'asset_url': 'assetUrl',
            'headers': 'headers',
        }[f],
    )


class FileUploadResult(SubsetModel):
    """The ``fileUpload`` field of the GraphQL response data."""

    success: bool
    upload_file: UploadFile

    model_config = pydantic.ConfigDict(
        populate_by_name=True,
        alias_generator=lambda f: {
            'success': 'success',
            'upload_file': 'uploadFile',
        }[f],
    )


# -- Exceptions ----------------------------------------------------------------


class LinearUploadError(Exception):
    """File upload to Linear failed."""


class MissingApiKeyError(LinearUploadError):
    """LINEAR_API_KEY env var is unset."""


class GraphQLError(LinearUploadError):
    """Linear's GraphQL endpoint returned an ``errors`` array."""


# -- App + boundary ------------------------------------------------------------

app = create_app(help='Upload a file to Linear and print its assetUrl.')
add_install_command(app, script_path=__file__)
boundary = ErrorBoundary(exit_code=1)


@app.command()
@boundary
def upload(
    file: Annotated[Path, typer.Argument(help='Path to the file to upload.')],
    alt: Annotated[str, typer.Option('--alt', help='Markdown alt text (default: file stem).')] = '',
    content_type: Annotated[
        str | None,
        typer.Option('--content-type', help='MIME type (default: guessed from extension).'),
    ] = None,
) -> None:
    """Upload FILE to Linear; print assetUrl to stdout, markdown to stderr."""
    api_key = os.environ.get('LINEAR_API_KEY')
    if not api_key:
        raise MissingApiKeyError('Set LINEAR_API_KEY (personal API key from https://linear.app/settings/api).')
    if not file.is_file():
        raise LinearUploadError(f'Not a regular file: {file}')

    data = file.read_bytes()
    ctype = content_type or mimetypes.guess_type(file.name)[0] or 'application/octet-stream'

    upload_file = _request_presigned_url(api_key, filename=file.name, size=len(data), content_type=ctype)
    _put_to_presigned_url(upload_file, data=data, content_type=ctype)

    asset = upload_file.asset_url
    label = alt or file.stem
    print(asset)
    print(f'![{label}]({asset})', file=sys.stderr)


# -- HTTP steps ----------------------------------------------------------------


def _request_presigned_url(
    api_key: str,
    *,
    filename: str,
    size: int,
    content_type: str,
) -> UploadFile:
    """Step 1: ask Linear for a presigned upload URL.

    Personal API key goes in Authorization WITHOUT ``Bearer`` (OAuth-only prefix).
    """
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            LINEAR_GRAPHQL_URL,
            headers={'Authorization': api_key, 'Content-Type': 'application/json'},
            json={
                'query': FILE_UPLOAD_MUTATION,
                'variables': {
                    'size': size,
                    'contentType': content_type,
                    'filename': filename,
                },
            },
        )
    resp.raise_for_status()
    body = resp.json()
    if body.get('errors'):
        raise GraphQLError(str(body['errors']))
    result = FileUploadResult.model_validate(body['data']['fileUpload'])
    if not result.success:
        raise LinearUploadError('Linear reported fileUpload.success=false')
    return result.upload_file


def _put_to_presigned_url(
    upload_file: UploadFile,
    *,
    data: bytes,
    content_type: str,
) -> None:
    """Step 2: PUT bytes to the presigned URL with headers Linear specified."""
    put_headers = {'Content-Type': content_type}
    for h in upload_file.headers:
        put_headers[h.key] = h.value
    with httpx.Client(timeout=120.0) as client:
        resp = client.put(upload_file.upload_url, content=data, headers=put_headers)
    resp.raise_for_status()


# -- Boundary handlers ---------------------------------------------------------


@boundary.handler(LinearUploadError)
def _handle_upload_error(exc: LinearUploadError) -> None:
    print(f'ERROR: {exc}', file=sys.stderr)


@boundary.handler(httpx.HTTPStatusError)
def _handle_http_status(exc: httpx.HTTPStatusError) -> None:
    body = exc.response.text[:500] if exc.response.content else ''
    print(f'ERROR: HTTP {exc.response.status_code} from {exc.request.url}: {body}', file=sys.stderr)


@boundary.handler(httpx.RequestError)
def _handle_http_request(exc: httpx.RequestError) -> None:
    print(f'ERROR: Network failure ({type(exc).__name__}): {exc}', file=sys.stderr)


@boundary.handler(Exception)
def _handle_crash(exc: Exception) -> None:
    print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
    for frame in traceback.format_tb(exc.__traceback__)[-2:]:
        print(frame.rstrip(), file=sys.stderr)


if __name__ == '__main__':
    run_app(app)
