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
// Two modes:
//
//   claude-tcc-probe microphone
//     Side-effect-free read via `AVCaptureDevice.authorizationStatus(.audio)`.
//     Returns the current state without touching the mic or triggering UI.
//
//   claude-tcc-probe microphone --request
//     Calls `AVCaptureDevice.requestAccess(.audio)`, blocks until macOS resolves
//     the request: when state is `notDetermined`, a system prompt appears and
//     the call blocks until the user clicks Allow / Don't Allow; for any other
//     state the callback fires immediately with the cached result. Used by the
//     orchestrator to auto-trigger the prompt during preflight rather than
//     telling the user to run `sox -d -n` by hand.
//
// Build:
//   swiftc -O claude-tcc-probe.swift -o claude-tcc-probe \
//          -framework AVFoundation -framework Foundation
//
// Install: `scripts/bootstrap.sh` handles compile + install to /usr/local/bin/.
//
// Output (stdout): exactly one of `authorized`, `denied`, `notDetermined`,
// `restricted`, or `unknown` — terminated by a newline.
//
// Exit codes: 0 when the probe ran (status is in stdout); 2 on usage error.

import AVFoundation
import Foundation

let args = CommandLine.arguments
guard args.count >= 2 else {
    FileHandle.standardError.write("usage: claude-tcc-probe microphone [--request]\n".data(using: .utf8)!)
    exit(2)
}

switch args[1].lowercased() {
case "microphone", "mic":
    let request = args.count >= 3 && args[2] == "--request"
    if request {
        // `requestAccess` callback fires on a background queue. Block the main
        // thread on a semaphore so the process stays alive until macOS resolves
        // the request — which on `notDetermined` means "until the user clicks
        // Allow or Don't Allow." On any other state the callback fires
        // immediately and we proceed straight to the post-request status read.
        let semaphore = DispatchSemaphore(value: 0)
        AVCaptureDevice.requestAccess(for: .audio) { _ in
            semaphore.signal()
        }
        semaphore.wait()
    }
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
