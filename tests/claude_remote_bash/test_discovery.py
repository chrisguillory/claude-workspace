"""Regression tests for the discovery wire format, partition logic, and classifier."""

from __future__ import annotations

import pytest
from claude_remote_bash.discovery import (
    _NETWORKSETUP_BLOCK,
    DiscoveredAddress,
    DiscoveredHost,
    _decode_addresses_txt,
    _encode_addresses_txt,
    _partition,
)
from claude_remote_bash.exceptions import ProtocolError


class TestWireFormat:
    def test_roundtrip_preserves_pairs(self) -> None:
        addrs = [
            DiscoveredAddress(ip='192.168.4.27', kind='wifi'),
            DiscoveredAddress(ip='192.168.4.49', kind='ethernet'),
            DiscoveredAddress(ip='100.64.0.1', kind='vpn'),
        ]
        decoded = _decode_addresses_txt(_encode_addresses_txt(addrs))
        assert decoded == {'192.168.4.27': 'wifi', '192.168.4.49': 'ethernet', '100.64.0.1': 'vpn'}

    def test_decode_rejects_missing_pipe(self) -> None:
        with pytest.raises(ProtocolError, match=r'missing `\|` separator'):
            _decode_addresses_txt('192.168.1.1')

    def test_decode_rejects_unknown_kind(self) -> None:
        with pytest.raises(ProtocolError, match='unknown InterfaceKind'):
            _decode_addresses_txt('192.168.1.1|quantum-entangled')

    def test_encode_rejects_overflow(self) -> None:
        """Regression: encoder must raise before zeroconf's TXT packer hits its 255-byte cap."""
        many = [DiscoveredAddress(ip=f'192.168.{i // 256}.{i % 256}', kind='ethernet') for i in range(15)]
        with pytest.raises(ProtocolError, match='DNS TXT character-strings cap'):
            _encode_addresses_txt(many)


class TestPartition:
    def test_overlapping_hosts_does_not_drop_either(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Two hosts claiming a local IP must not silently drop one.

        Daemon restart can leave a zombie mDNS record overlapping with the new
        advertisement; both rows then match ``local_addresses()`` at browse time.
        The first becomes ``local_daemon`` and the second falls through to
        ``remote_daemons``.
        """
        monkeypatch.setattr(
            'claude_remote_bash.discovery.local_addresses',
            lambda: [DiscoveredAddress(ip='10.0.0.5', kind='ethernet')],
        )
        hosts = [_host('M5-old', ['10.0.0.5']), _host('M5-new', ['10.0.0.5'])]
        result = _partition(hosts)
        assert result.local_daemon is not None
        assert result.local_daemon.alias == 'M5-old'
        assert [h.alias for h in result.remote_daemons] == ['M5-new']


class TestNetworkSetupRegex:
    def test_missing_device_line_does_not_cross_blocks(self) -> None:
        """A block without a ``Device:`` line must not borrow the next block's device.

        The regex must not span the blank separator between adjacent Hardware
        Port stanzas. Cross-pairing silently misclassifies interfaces — pinning
        a Wi-Fi label onto an Ethernet device's BSD name reintroduces the
        Wi-Fi-wins-over-Ethernet failure mode interface typing was meant to fix.
        """
        text = (
            'Hardware Port: Wi-Fi\n'
            'Ethernet Address: aa:bb:cc:dd:ee:ff\n'
            '\n'
            'Hardware Port: Ethernet\n'
            'Device: en1\n'
            'Ethernet Address: 11:22:33:44:55:66\n'
        )
        matches = [(m.group('port'), m.group('device')) for m in _NETWORKSETUP_BLOCK.finditer(text)]
        assert matches == [('Ethernet', 'en1')]


def _host(alias: str, ips: list[str]) -> DiscoveredHost:
    """Minimal DiscoveredHost factory for partition tests."""
    return DiscoveredHost(
        alias=alias,
        hostname=f'{alias}.local.',
        addresses=[DiscoveredAddress(ip=ip, kind='ethernet') for ip in ips],
        port=12345,
        version='0.4.0',
    )
