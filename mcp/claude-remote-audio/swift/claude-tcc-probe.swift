// claude-tcc-probe — query the TCC authorization status for a service from this
// process's responsible-app chain on macOS.
//
// Why this exists: macOS gates microphone access via TCC, and the TCC subject
// is the *responsible app* of the access chain — for daemon-spawned grandchildren
// (claude-remote-bash-daemon → roc-send), that's the daemon's parent app (typically
// iTerm.app). When the responsible app has no Microphone consent, every roc-send
// invocation through the chain fails silently — SoX wraps the TCC denial as a
// generic "roc_sndio: backend dispatcher: failed to open source" error, indis-
// tinguishable from a HAL wedge or a device-specific failure. The orchestrator's
// post-mortem diagnostic uses this probe as Tier 1 to disambiguate before falling
// through to a HAL probe; without it, every TCC denial misdiagnoses as a HAL wedge.
//
// `AVCaptureDevice.authorizationStatus(for:)` is a pure read — does NOT trigger a
// prompt, does NOT touch the microphone, side-effect free. Safe to call repeatedly.
//
// Build:
//   swiftc -O claude-tcc-probe.swift -o claude-tcc-probe \
//          -framework AVFoundation -framework Foundation
//
// Install: `scripts/bootstrap.sh` handles compile + install to /usr/local/bin/.
//
// Usage:
//   claude-tcc-probe microphone
//
// Output (stdout): exactly one of `authorized`, `denied`, `notDetermined`,
// `restricted`, or `unknown` — terminated by a newline.
//
// Exit codes: 0 when the probe ran (status is in stdout); 2 on usage error.

import AVFoundation
import Foundation

let args = CommandLine.arguments
guard args.count >= 2 else {
    FileHandle.standardError.write("usage: claude-tcc-probe microphone\n".data(using: .utf8)!)
    exit(2)
}

switch args[1].lowercased() {
case "microphone", "mic":
    let status = AVCaptureDevice.authorizationStatus(for: .audio)
    let name: String
    switch status {
    case .notDetermined: name = "notDetermined"
    case .restricted: name = "restricted"
    case .denied: name = "denied"
    case .authorized: name = "authorized"
    @unknown default: name = "unknown"
    }
    print(name)
default:
    FileHandle.standardError.write("unknown service: \(args[1]) (supported: microphone)\n".data(using: .utf8)!)
    exit(2)
}
