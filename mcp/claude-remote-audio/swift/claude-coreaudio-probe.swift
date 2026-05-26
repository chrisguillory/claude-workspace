// claude-coreaudio-probe — open a named Core Audio input device step-by-step
// and surface the OSStatus from each step as a symbolic verdict.
//
// Why this exists: SoX/sox_ng's CoreAudio backend collapses every input-open
// failure into one generic message — `roc_sndio: backend dispatcher: failed
// to open source`. When roc-send dies on hub launch, the orchestrator can't
// tell whether the requested device was missing, output-only, format-rejected,
// or wedged. Patching sox_ng to forward OSStatus is option (b) (10 callsites,
// upstream release dependency); building our own probe at the lowest layer
// we own — public CoreAudio HAL APIs — is option (a), composable with (b),
// and follows the same template that retired the 5/20 TCC-misdiagnosis
// (claude-tcc-probe). See CLAUDE.md "Layered architecture": when the cause
// is upstream and unmovable, wrap the foundation in a detect adapter.
//
// Wired as Tier 1.5 in `_diagnose_roc_send_start_failure` — runs after
// Microphone TCC (Tier 1) and before the libsox log-marker scan (Tier 2).
//
// Build:
//   swiftc -O claude-coreaudio-probe.swift -o claude-coreaudio-probe \
//          -framework CoreAudio -framework Foundation
//
// Install: `scripts/bootstrap.sh` handles compile + install to /usr/local/bin/.
//
// Usage:
//   claude-coreaudio-probe device "Device Name"
//
// Output (stdout, key=value lines; final line is `verdict=<symbol>`):
//   op=find-device
//   device-found=true
//   device-id=87
//   op=check-alive
//   device-alive=true
//   op=stream-config
//   input-channels=1
//   op=stream-format
//   input-sample-rate=48000.0
//   input-format-id=lpcm
//   input-bytes-per-frame=4
//   verdict=ok
//
// Verdicts:
//   ok                  every probe step succeeded; sox_ng's open failure is elsewhere
//   device-missing      no Core Audio device matches the name
//   device-not-alive    found by name but `kAudioDevicePropertyDeviceIsAlive == 0`
//   no-input-scope      found and alive but 0 input-scope channels (output-only device)
//   format-unreadable   stream config / format property reads failed with non-noErr OSStatus
//   unknown             usage error or unexpected probe failure (rare)
//
// Exit codes: 0 when the probe ran (verdict is on stdout); 2 on usage error.

import CoreAudio
import Foundation

func formatStatus(_ status: OSStatus) -> String {
    let u = UInt32(bitPattern: status)
    let bytes: [UInt8] = [
        UInt8((u >> 24) & 0xff),
        UInt8((u >> 16) & 0xff),
        UInt8((u >> 8) & 0xff),
        UInt8(u & 0xff),
    ]
    let printable = bytes.allSatisfy { (0x20...0x7e).contains($0) }
    if printable, let chars = String(bytes: bytes, encoding: .ascii) {
        return String(format: "%d (0x%08x '\(chars)')", status, u)
    }
    return String(format: "%d (0x%08x)", status, u)
}

func getStringProperty(_ deviceID: AudioDeviceID, _ selector: AudioObjectPropertySelector) -> String? {
    var addr = AudioObjectPropertyAddress(
        mSelector: selector,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    var size: UInt32 = 0
    guard AudioObjectGetPropertyDataSize(deviceID, &addr, 0, nil, &size) == noErr else { return nil }
    var cfStr: CFString = "" as CFString
    let status = withUnsafeMutablePointer(to: &cfStr) { ptr -> OSStatus in
        var s = size
        return AudioObjectGetPropertyData(deviceID, &addr, 0, nil, &s, ptr)
    }
    guard status == noErr else { return nil }
    return cfStr as String
}

func allDeviceIDs() -> [AudioDeviceID] {
    var addr = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyDevices,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    var size: UInt32 = 0
    guard AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size) == noErr else { return [] }
    let count = Int(size) / MemoryLayout<AudioDeviceID>.size
    var ids = [AudioDeviceID](repeating: 0, count: count)
    let status = ids.withUnsafeMutableBufferPointer { buf -> OSStatus in
        var s = size
        return AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &s, buf.baseAddress!)
    }
    guard status == noErr else { return [] }
    return ids
}

func findDevice(named name: String) -> AudioDeviceID? {
    for id in allDeviceIDs() {
        if let n = getStringProperty(id, kAudioDevicePropertyDeviceNameCFString), n == name { return id }
    }
    return nil
}

func isAlive(_ deviceID: AudioDeviceID) -> (alive: Bool?, status: OSStatus) {
    var addr = AudioObjectPropertyAddress(
        mSelector: kAudioDevicePropertyDeviceIsAlive,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    var alive: UInt32 = 0
    var size = UInt32(MemoryLayout<UInt32>.size)
    let status = AudioObjectGetPropertyData(deviceID, &addr, 0, nil, &size, &alive)
    if status != noErr { return (nil, status) }
    return (alive != 0, status)
}

func inputChannels(_ deviceID: AudioDeviceID) -> (channels: Int?, status: OSStatus) {
    var addr = AudioObjectPropertyAddress(
        mSelector: kAudioDevicePropertyStreamConfiguration,
        mScope: kAudioDevicePropertyScopeInput,
        mElement: kAudioObjectPropertyElementMain
    )
    var size: UInt32 = 0
    let sizeStatus = AudioObjectGetPropertyDataSize(deviceID, &addr, 0, nil, &size)
    if sizeStatus != noErr { return (nil, sizeStatus) }
    let bufList = UnsafeMutableRawPointer.allocate(byteCount: Int(size), alignment: MemoryLayout<AudioBufferList>.alignment)
    defer { bufList.deallocate() }
    var s = size
    let status = AudioObjectGetPropertyData(deviceID, &addr, 0, nil, &s, bufList)
    if status != noErr { return (nil, status) }
    let abl = bufList.assumingMemoryBound(to: AudioBufferList.self)
    let buffers = UnsafeMutableAudioBufferListPointer(abl)
    let total = buffers.reduce(0) { $0 + Int($1.mNumberChannels) }
    return (total, noErr)
}

func inputStreamFormat(_ deviceID: AudioDeviceID) -> (format: AudioStreamBasicDescription?, status: OSStatus) {
    var addr = AudioObjectPropertyAddress(
        mSelector: kAudioDevicePropertyStreamFormat,
        mScope: kAudioDevicePropertyScopeInput,
        mElement: kAudioObjectPropertyElementMain
    )
    var asbd = AudioStreamBasicDescription()
    var size = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
    let status = AudioObjectGetPropertyData(deviceID, &addr, 0, nil, &size, &asbd)
    if status != noErr { return (nil, status) }
    return (asbd, noErr)
}

func fourCC(_ value: UInt32) -> String {
    let bytes: [UInt8] = [
        UInt8((value >> 24) & 0xff),
        UInt8((value >> 16) & 0xff),
        UInt8((value >> 8) & 0xff),
        UInt8(value & 0xff),
    ]
    if bytes.allSatisfy({ (0x20...0x7e).contains($0) }), let s = String(bytes: bytes, encoding: .ascii) {
        return s
    }
    return String(format: "0x%08x", value)
}

func probeDevice(name: String) {
    print("op=find-device")
    guard let deviceID = findDevice(named: name) else {
        print("device-found=false")
        print("verdict=device-missing")
        return
    }
    print("device-found=true")
    print("device-id=\(deviceID)")

    print("op=check-alive")
    let (alive, aliveStatus) = isAlive(deviceID)
    guard let aliveValue = alive else {
        print("device-alive=unknown")
        print("alive-status=\(formatStatus(aliveStatus))")
        print("verdict=format-unreadable")
        return
    }
    print("device-alive=\(aliveValue)")
    if !aliveValue {
        print("verdict=device-not-alive")
        return
    }

    print("op=stream-config")
    let (channels, channelsStatus) = inputChannels(deviceID)
    guard let channelCount = channels else {
        print("input-channels=unknown")
        print("stream-config-status=\(formatStatus(channelsStatus))")
        print("verdict=format-unreadable")
        return
    }
    print("input-channels=\(channelCount)")
    if channelCount == 0 {
        print("verdict=no-input-scope")
        return
    }

    print("op=stream-format")
    let (format, fmtStatus) = inputStreamFormat(deviceID)
    guard let asbd = format else {
        print("stream-format-status=\(formatStatus(fmtStatus))")
        print("verdict=format-unreadable")
        return
    }
    print("input-sample-rate=\(asbd.mSampleRate)")
    print("input-format-id=\(fourCC(asbd.mFormatID))")
    print("input-format-flags=\(asbd.mFormatFlags)")
    print("input-bytes-per-frame=\(asbd.mBytesPerFrame)")
    print("verdict=ok")
}

let args = CommandLine.arguments
guard args.count >= 3 else {
    FileHandle.standardError.write("usage: claude-coreaudio-probe device \"<name>\"\n".data(using: .utf8)!)
    exit(2)
}

switch args[1].lowercased() {
case "device":
    probeDevice(name: args[2])
default:
    FileHandle.standardError.write("unknown subcommand: \(args[1]) (supported: device)\n".data(using: .utf8)!)
    exit(2)
}
