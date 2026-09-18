// Local macOS Vision worker: one base64 JPEG request / one pose response per line.
// Frames are already oriented by OpenCV. No images leave this process/device.
import Foundation
import Vision
import ImageIO

let joints: [VNHumanBodyPoseObservation.JointName] = [
    .nose, .leftEye, .rightEye, .leftEar, .rightEar,
    .leftShoulder, .rightShoulder, .leftElbow, .rightElbow,
    .leftWrist, .rightWrist, .leftHip, .rightHip,
    .leftKnee, .rightKnee, .leftAnkle, .rightAnkle
]

func reply(_ value: [String: Any]) {
    if let data = try? JSONSerialization.data(withJSONObject: value),
       let text = String(data: data, encoding: .utf8) {
        print(text)
        fflush(stdout)
    }
}

reply(["ready": true, "backend": "macOS Vision human body pose"])
while let line = readLine() {
    autoreleasepool {
        do {
            guard let requestData = line.data(using: .utf8),
                  let payload = try JSONSerialization.jsonObject(with: requestData) as? [String: Any],
                  let encoded = payload["image"] as? String,
                  let data = Data(base64Encoded: encoded),
                  let source = CGImageSourceCreateWithData(data as CFData, nil),
                  let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
            else { reply(["error": "Invalid JPEG frame"]); return }
            let request = VNDetectHumanBodyPoseRequest()
            let handler = VNImageRequestHandler(cgImage: image, orientation: .up, options: [:])
            try handler.perform([request])
            let people: [[[Double]]] = try (request.results ?? []).map { observation in
                let points = try observation.recognizedPoints(.all)
                return joints.map { name in
                    guard let point = points[name] else { return [0, 0, 0] }
                    return [Double(point.location.x), 1 - Double(point.location.y), Double(point.confidence)]
                }
            }
            reply(["people": people])
        } catch {
            reply(["error": error.localizedDescription])
        }
    }
}
