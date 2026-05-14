"""Channel-multiplexed length-prefixed framing over TCP."""

from __future__ import annotations

import asyncio
import json
import struct

import pydantic

from claude_remote_bash.exceptions import ProtocolError
from claude_remote_bash.schemas.protocol import Message

__all__ = [
    'CONTROL_CHANNEL',
    'parse_message',
    'read_frame',
    'read_message',
    'write_frame',
    'write_message',
]

HEADER_FORMAT = '!IBI'
"""Wire header: ``[uint32 BE length] [uint8 flags] [uint32 BE channel_id]``.

The flags byte is reserved for future per-frame compression.
"""

HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

MAX_PAYLOAD_SIZE = 10 * 1024 * 1024
"""Largest single frame payload accepted. NFS RPCs cap well below 1 MB per frame."""

CONTROL_CHANNEL = 0
"""Reserved channel for JSON control messages.

Channel 0 carries the framed-JSON ``Message`` protocol from
``schemas/protocol.py``; channels with id > 0 carry opaque tunnel payload
(raw NFS RPC bytes), established via control-channel ``TunnelOpen`` /
``TunnelOk`` exchanges. Recipients route frames purely by ``channel_id``
without interpreting the payload as JSON unless the id is zero.
"""

_MESSAGE_ADAPTER: pydantic.TypeAdapter[Message] = pydantic.TypeAdapter(Message)


async def read_frame(reader: asyncio.StreamReader) -> tuple[int, bytes]:
    """Read one wire frame; return ``(channel_id, payload_bytes)``.

    Raises:
        ProtocolError: On framing errors or oversized payloads.
        asyncio.IncompleteReadError: On unexpected connection close.
    """
    header = await reader.readexactly(HEADER_SIZE)
    length, flags, channel_id = struct.unpack(HEADER_FORMAT, header)

    if flags != 0:
        raise ProtocolError(f'unsupported flags byte: 0x{flags:02x}')

    if length > MAX_PAYLOAD_SIZE:
        raise ProtocolError(f'payload too large: {length} bytes (max {MAX_PAYLOAD_SIZE})')

    payload = await reader.readexactly(length)
    return channel_id, payload


async def write_frame(writer: asyncio.StreamWriter, channel_id: int, payload: bytes) -> None:
    """Serialize and write one wire frame to the stream."""
    if len(payload) > MAX_PAYLOAD_SIZE:
        raise ProtocolError(f'payload too large: {len(payload)} bytes (max {MAX_PAYLOAD_SIZE})')
    header = struct.pack(HEADER_FORMAT, len(payload), 0x00, channel_id)
    writer.write(header + payload)
    await writer.drain()


def parse_message(payload: bytes) -> Message:
    """Deserialize a control-channel JSON payload into the discriminated ``Message`` union.

    Raises:
        ProtocolError: On malformed JSON, non-object payload, or schema
            validation failure. Pydantic's ``ValidationError`` is wrapped so
            every caller has a single exception class to handle for any
            protocol-level violation.
    """
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as e:
        raise ProtocolError(f'malformed JSON: {e}') from e

    if not isinstance(data, dict):
        raise ProtocolError(f'expected JSON object, got {type(data).__name__}')

    try:
        return _MESSAGE_ADAPTER.validate_python(data)
    except pydantic.ValidationError as e:
        raise ProtocolError(f'schema validation failed: {e}') from e


async def read_message(reader: asyncio.StreamReader) -> Message:
    """Read and validate a control-channel JSON message.

    Convenience for callers that only deal with the control channel. Reads
    one frame, asserts ``channel_id == CONTROL_CHANNEL``, and dispatches the
    JSON payload through ``parse_message``.

    Raises:
        ProtocolError: On framing errors, off-channel frames, malformed
            JSON, or schema validation failures.
        asyncio.IncompleteReadError: On unexpected connection close.
    """
    channel_id, payload = await read_frame(reader)
    if channel_id != CONTROL_CHANNEL:
        raise ProtocolError(f'expected control-channel frame, got channel_id={channel_id}')
    return parse_message(payload)


async def write_message(writer: asyncio.StreamWriter, msg: Message) -> None:
    """Serialize and write a control-channel JSON message."""
    payload = _MESSAGE_ADAPTER.dump_json(msg, by_alias=True)
    await write_frame(writer, CONTROL_CHANNEL, payload)
