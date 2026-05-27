// claude-coreaudio-volume — set / get / list Core Audio device volume by name
//
// Why this exists: macOS has no stock CLI that addresses an audio device by name
// and reads/writes its volume scalar. `osascript 'set volume input/output volume N'`
// only acts on the system default; `switchaudio-osx` is switching-only. The
// claude-remote-audio orchestrator needs to pin Claude Remote Mic / Claude Remote
// Speaker volumes to max across a multi-Mac mesh — a digital-intermediate policy
// that requires address-by-identity rather than address-by-default. This tool
// fills that gap with ~170 lines of stdlib-only Swift wrapping CoreAudio's
// `kAudioDevicePropertyVolumeScalar`.
//
// Build:
//   swiftc -O claude-coreaudio-volume.swift -o claude-coreaudio-volume \
//          -framework CoreAudio -framework Foundation
//
// Install: `scripts/bootstrap.sh` handles compile + install to /usr/local/bin/.
//
// Usage:
//   claude-coreaudio-volume list
//   claude-coreaudio-volume get "Device Name" <input|output>
//   claude-coreaudio-volume set "Device Name" <input|output> <0.0..1.0>
//
// Exit codes: 0 ok, 1 device not found, 2 usage error, 3 no volume property in scope.

import CoreAudio
import Foundation

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

func channelCount(_ deviceID: AudioDeviceID, scope: AudioObjectPropertyScope) -> Int {
    var addr = AudioObjectPropertyAddress(
        mSelector: kAudioDevicePropertyStreamConfiguration,
        mScope: scope,
        mElement: kAudioObjectPropertyElementMain
    )
    var size: UInt32 = 0
    guard AudioObjectGetPropertyDataSize(deviceID, &addr, 0, nil, &size) == noErr else { return 0 }
    let bufList = UnsafeMutableRawPointer.allocate(byteCount: Int(size), alignment: MemoryLayout<AudioBufferList>.alignment)
    defer { bufList.deallocate() }
    var s = size
    guard AudioObjectGetPropertyData(deviceID, &addr, 0, nil, &s, bufList) == noErr else { return 0 }
    let abl = bufList.assumingMemoryBound(to: AudioBufferList.self)
    let buffers = UnsafeMutableAudioBufferListPointer(abl)
    return buffers.reduce(0) { $0 + Int($1.mNumberChannels) }
}

func findDevice(named name: String) -> AudioDeviceID? {
    for id in allDeviceIDs() {
        if let n = getStringProperty(id, kAudioDevicePropertyDeviceNameCFString), n == name { return id }
    }
    return nil
}

func volumeAddress(scope: AudioObjectPropertyScope, element: AudioObjectPropertyElement) -> AudioObjectPropertyAddress {
    return AudioObjectPropertyAddress(
        mSelector: kAudioDevicePropertyVolumeScalar,
        mScope: scope,
        mElement: element
    )
}

func getVolume(_ deviceID: AudioDeviceID, scope: AudioObjectPropertyScope) -> Float32? {
    // Master element first; fall back to channel 1 (some devices expose only per-channel)
    var addr = volumeAddress(scope: scope, element: kAudioObjectPropertyElementMain)
    if AudioObjectHasProperty(deviceID, &addr) {
        var v: Float32 = 0
        var size = UInt32(MemoryLayout<Float32>.size)
        if AudioObjectGetPropertyData(deviceID, &addr, 0, nil, &size, &v) == noErr { return v }
    }
    addr.mElement = 1
    if AudioObjectHasProperty(deviceID, &addr) {
        var v: Float32 = 0
        var size = UInt32(MemoryLayout<Float32>.size)
        if AudioObjectGetPropertyData(deviceID, &addr, 0, nil, &size, &v) == noErr { return v }
    }
    return nil
}

func setVolume(_ deviceID: AudioDeviceID, scope: AudioObjectPropertyScope, value: Float32) -> Bool {
    // Set master, then every per-channel scalar — stereo balance stays consistent
    // because devices that honor only per-channel get the same value on each.
    var v = value
    let size = UInt32(MemoryLayout<Float32>.size)
    var anySuccess = false

    var addr = volumeAddress(scope: scope, element: kAudioObjectPropertyElementMain)
    if AudioObjectHasProperty(deviceID, &addr) {
        if AudioObjectSetPropertyData(deviceID, &addr, 0, nil, size, &v) == noErr {
            anySuccess = true
        }
    }
    let channels = channelCount(deviceID, scope: scope)
    for ch in 1...max(channels, 2) {
        addr = volumeAddress(scope: scope, element: AudioObjectPropertyElement(ch))
        if AudioObjectHasProperty(deviceID, &addr) {
            if AudioObjectSetPropertyData(deviceID, &addr, 0, nil, size, &v) == noErr {
                anySuccess = true
            }
        }
    }
    return anySuccess
}

func scopeForArg(_ s: String) -> AudioObjectPropertyScope? {
    switch s.lowercased() {
    case "in", "input": return kAudioDevicePropertyScopeInput
    case "out", "output": return kAudioDevicePropertyScopeOutput
    default: return nil
    }
}

let args = CommandLine.arguments
guard args.count >= 2 else {
    FileHandle.standardError.write("usage: claude-coreaudio-volume list|get|set ...\n".data(using: .utf8)!)
    exit(2)
}

switch args[1] {
case "list":
    for id in allDeviceIDs() {
        let name = getStringProperty(id, kAudioDevicePropertyDeviceNameCFString) ?? "<unknown>"
        let inCh = channelCount(id, scope: kAudioDevicePropertyScopeInput)
        let outCh = channelCount(id, scope: kAudioDevicePropertyScopeOutput)
        let inVol = getVolume(id, scope: kAudioDevicePropertyScopeInput).map { String(format: "%.2f", $0) } ?? "—"
        let outVol = getVolume(id, scope: kAudioDevicePropertyScopeOutput).map { String(format: "%.2f", $0) } ?? "—"
        print("id=\(id)  in=\(inCh)ch vol=\(inVol)  out=\(outCh)ch vol=\(outVol)  name=\(name)")
    }
case "get":
    guard args.count == 4, let scope = scopeForArg(args[3]) else { exit(2) }
    guard let id = findDevice(named: args[2]) else {
        FileHandle.standardError.write("device not found: \(args[2])\n".data(using: .utf8)!); exit(1)
    }
    if let v = getVolume(id, scope: scope) {
        print(String(format: "%.4f", v))
    } else {
        FileHandle.standardError.write("device has no volume property in that scope\n".data(using: .utf8)!); exit(3)
    }
case "set":
    guard args.count == 5, let scope = scopeForArg(args[3]), let target = Float32(args[4]) else { exit(2) }
    guard let id = findDevice(named: args[2]) else {
        FileHandle.standardError.write("device not found: \(args[2])\n".data(using: .utf8)!); exit(1)
    }
    let ok = setVolume(id, scope: scope, value: target)
    if !ok {
        FileHandle.standardError.write("no settable volume property in that scope\n".data(using: .utf8)!); exit(3)
    }
    if let v = getVolume(id, scope: scope) { print(String(format: "%.4f", v)) }
default:
    FileHandle.standardError.write("unknown subcommand\n".data(using: .utf8)!); exit(2)
}
