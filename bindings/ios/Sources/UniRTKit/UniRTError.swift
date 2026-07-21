// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

import CUniRT

public struct UniRTError: Error, CustomStringConvertible {
    public let code: Int32
    public let detail: String

    public var description: String { "UniRT error \(code): \(detail)" }

    static func check(_ status: Int32) throws {
        guard status < 0 else { return }
        let summary = String(cString: unirt_get_error_message(unirt_ErrorCode(rawValue: status)))
        let detail = String(cString: unirt_last_error_message())
        throw UniRTError(code: status, detail: detail.isEmpty ? summary : "\(summary): \(detail)")
    }
}
