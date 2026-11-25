import AppKit
import CoreImage
import FoundationModels
import ImageIO
import SwiftUI
import UniformTypeIdentifiers
import Vision

enum Effect: String, CaseIterable, Identifiable {
    case blur = "Blur", pixelate = "Pixelate", solid = "Solid"
    var id: Self { self }
}

struct MaskEdit: Identifiable {
    let id: Int
    let mask: CIImage
    var effect: Effect = .blur
    var strength: Double = 24
    var color: Color = .black
}

struct Subjects {
    let width: Int
    let height: Int
    let labels: [UInt8]
    let masks: [Int: CIImage]

    static func detect(_ original: CGImage) throws -> Subjects {
        let maximumDimension: CGFloat = 1024
        let scale = min(1, maximumDimension / CGFloat(max(original.width, original.height)))
        let image: CGImage
        if scale < 1 {
            let input = CIImage(cgImage: original).transformed(
                by: CGAffineTransform(scaleX: scale, y: scale)
            )
            let context = CIContext(options: [.cacheIntermediates: false])
            guard let resized = context.createCGImage(input, from: input.extent.integral) else {
                throw CocoaError(.fileReadUnknown)
            }
            image = resized
        } else {
            image = original
        }

        let maskTransform = CGAffineTransform(
            scaleX: CGFloat(original.width) / CGFloat(image.width),
            y: CGFloat(original.height) / CGFloat(image.height)
        )
        let request = VNGenerateForegroundInstanceMaskRequest()
        let handler = VNImageRequestHandler(cgImage: image)
        try handler.perform([request])
        guard let result = request.results?.first else {
            return Subjects(width: 0, height: 0, labels: [], masks: [:])
        }
        let buffer = result.instanceMask
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let labels = try readLabels(buffer, width: width, height: height)
        var masks = [Int: CIImage]()
        for id in result.allInstances {
            let mask = try result.generateScaledMaskForImage(forInstances: IndexSet(integer: id), from: handler)
            masks[id] = CIImage(cvPixelBuffer: mask, options: [.colorSpace: NSNull()])
                .transformed(by: maskTransform)
        }
        return Subjects(width: width, height: height, labels: labels, masks: masks)
    }

    private static func readLabels(_ buffer: CVPixelBuffer, width: Int, height: Int) throws -> [UInt8] {
        guard CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_OneComponent8,
              CVPixelBufferLockBaseAddress(buffer, .readOnly) == kCVReturnSuccess else {
            throw CocoaError(.fileReadUnknown)
        }
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buffer)?.assumingMemoryBound(to: UInt8.self) else {
            throw CocoaError(.fileReadUnknown)
        }
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        var labels = [UInt8]()
        labels.reserveCapacity(width * height)
        for row in 0..<height {
            labels.append(contentsOf: UnsafeBufferPointer(start: base + row * stride, count: width))
        }
        return labels
    }

    func hit(_ point: CGPoint, in size: CGSize) -> Int? {
        guard width > 0, height > 0, point.x >= 0, point.y >= 0,
              point.x < size.width, point.y < size.height else { return nil }
        let x = Int(point.x / size.width * CGFloat(width))
        let y = Int(point.y / size.height * CGFloat(height))
        let id = Int(labels[y * width + x])
        return masks[id] == nil ? nil : id
    }
}

@MainActor
final class Editor: ObservableObject {
    @Published var original: CGImage?
    @Published var preview: CGImage?
    @Published var edits: [MaskEdit] = []
    @Published var selection: Int?
    @Published var caption = ""
    @Published var status = ""
    @Published var processingTime = ""
    @Published var message: String?
    @Published var updateURL: URL?
    private var subjects: Subjects?
    private var generation = UUID()
    private var captionTask: Task<Void, Never>?
    private var detectionTask: Task<Void, Never>?
    private let context = CIContext()

    func open() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        load(url)
    }

    func load(_ url: URL) {
        let started = DispatchTime.now().uptimeNanoseconds
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let width = properties[kCGImagePropertyPixelWidth] as? Int,
              let height = properties[kCGImagePropertyPixelHeight] as? Int,
              let image = CGImageSourceCreateThumbnailAtIndex(source, 0, [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceCreateThumbnailWithTransform: true,
                kCGImageSourceThumbnailMaxPixelSize: max(width, height)
              ] as CFDictionary) else {
            message = "This image could not be opened."
            return
        }
        captionTask?.cancel()
        detectionTask?.cancel()
        generation = UUID()
        let current = generation
        original = image
        preview = image
        subjects = nil
        edits = []
        selection = nil
        status = "Finding subjects…"
        processingTime = ""
        caption = "Describing image…"

        detectionTask = Task {
            let result = await Task.detached(priority: .userInitiated) {
                Result { try Subjects.detect(image) }
            }.value
            guard !Task.isCancelled, current == generation else { return }
            switch result {
            case .success(let detected):
                subjects = detected
                status = "Checking subjects for automatic blur…"
                let result = await automaticBlur(image, subjects: detected)
                guard !Task.isCancelled, current == generation else { return }
                for edit in result.edits where !edits.contains(where: { $0.id == edit.id }) {
                    edits.append(edit)
                }
                if !result.edits.isEmpty { refresh() }
                status = result.status
                let elapsed = Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000
                processingTime = String(format: "%.3f s (%.0f ms)", elapsed / 1000, elapsed)
            case .failure:
                status = "Subject detection failed for this image."
            }
        }
        captionTask = Task {
            await detectionTask?.value
            guard !Task.isCancelled, current == generation else { return }
            let model = SystemLanguageModel.default
            if #available(macOS 27.0, *), model.isAvailable {
                let input = scaled(CIImage(cgImage: image), maximum: 768)
                if let thumbnail = context.createCGImage(input, from: input.extent.integral) {
                    let session = LanguageModelSession(model: model)
                    let response = try? await session.respond {
                        "Describe the visible scene in one short sentence."
                        Attachment(thumbnail)
                    }
                    if let response {
                        guard !Task.isCancelled, current == generation else { return }
                        caption = response.content
                        return
                    }
                }
            }
            let tags = await Task.detached(priority: .utility) {
                let request = VNClassifyImageRequest()
                try? VNImageRequestHandler(cgImage: image).perform([request])
                return (request.results ?? []).filter { $0.confidence >= 0.15 }
                    .prefix(3).map { $0.identifier.replacingOccurrences(of: "_", with: " ") }
            }.value
            guard !Task.isCancelled, current == generation else { return }
            caption = tags.isEmpty ? "No description available." : "Detected: " + tags.joined(separator: ", ")
        }
    }

    private func automaticBlur(_ image: CGImage, subjects: Subjects) async -> (edits: [MaskEdit], status: String) {
        guard !subjects.masks.isEmpty else { return ([], "No foreground subjects found.") }
        let model = SystemLanguageModel.default
        guard #available(macOS 27.0, *), model.isAvailable else {
            return ([], "Automatic blur needs Apple Intelligence. Click a subject to mask it manually.")
        }

        let source = CIImage(cgImage: image)
        var bounds: [Int: CGRect] = [:]
        for (index, label) in subjects.labels.enumerated() where label != 0 {
            let pixel = CGRect(x: index % subjects.width, y: index / subjects.width, width: 1, height: 1)
            let id = Int(label)
            bounds[id] = bounds[id]?.union(pixel) ?? pixel
        }
        let sx = source.extent.width / CGFloat(subjects.width)
        let sy = source.extent.height / CGFloat(subjects.height)
        var matches: [MaskEdit] = []
        var failures = 0

        for id in bounds.keys.sorted() {
            guard !Task.isCancelled else { break }
            guard let box = bounds[id], let mask = subjects.masks[id] else { continue }
            let crop = CGRect(
                x: box.minX * sx,
                y: (CGFloat(subjects.height) - box.maxY) * sy,
                width: box.width * sx,
                height: box.height * sy
            ).insetBy(dx: -2 * sx, dy: -2 * sy).intersection(source.extent).integral

            let isolated = source.cropped(to: crop).applyingFilter("CIBlendWithMask", parameters: [
                kCIInputBackgroundImageKey: CIImage(color: CIColor(red: 0.5, green: 0.5, blue: 0.5)).cropped(to: crop),
                kCIInputMaskImageKey: mask.cropped(to: crop)
            ]).cropped(to: crop)
            let input = scaled(isolated, maximum: 384)
            guard let thumbnail = context.createCGImage(input, from: input.extent.integral) else {
                failures += 1
                continue
            }

            do {
                let response = try await LanguageModelSession(model: model).respond {
                    """
                    Classify the isolated subject for photo masking based on visible presentation.
                    Reply exactly BLUR for a female-presenting person.
                    Reply exactly KEEP for a male-presenting person, an object, or an unclear subject.
                    """
                    Attachment(thumbnail)
                }
                let decision = response.content.trimmingCharacters(in: .whitespacesAndNewlines).uppercased()
                if decision == "BLUR" {
                    matches.append(MaskEdit(id: id, mask: mask, strength: max(36, Double(source.extent.width) * 0.035)))
                } else if decision != "KEEP" {
                    failures += 1
                }
            } catch {
                failures += 1
            }
        }

        let summary = "Automatically blurred \(matches.count) " + (matches.count == 1 ? "subject." : "subjects.")
        let detail = failures == 0 ? "Click any detected subject to edit its mask." : "Some subjects could not be classified. Click to mask them manually."
        return (matches, summary + " " + detail)
    }

    func click(_ point: CGPoint, size: CGSize) {
        guard let id = subjects?.hit(point, in: size), let mask = subjects?.masks[id] else {
            selection = nil
            return
        }
        selection = id
        if !edits.contains(where: { $0.id == id }) {
            edits.append(MaskEdit(id: id, mask: mask))
        }
    }

    func removeSelected() {
        edits.removeAll { $0.id == selection }
        selection = nil
    }

    private func scaled(_ image: CIImage, maximum: CGFloat) -> CIImage {
        let scale = min(1, maximum / max(image.extent.width, image.extent.height))
        return image.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
    }

    private func render(maximum: CGFloat) -> CGImage? {
        guard let original else { return nil }
        let source = scaled(CIImage(cgImage: original), maximum: maximum)
        let scale = source.extent.width / CGFloat(original.width)
        let transform = CGAffineTransform(scaleX: scale, y: scale)
        var output = source
        for edit in edits {
            let filtered: CIImage
            switch edit.effect {
            case .blur:
                filtered = source.clampedToExtent().applyingFilter("CIGaussianBlur", parameters: [
                    kCIInputRadiusKey: edit.strength * scale
                ])
            case .pixelate:
                filtered = source.applyingFilter("CIPixellate", parameters: [
                    kCIInputScaleKey: max(1, edit.strength * scale)
                ])
            case .solid:
                filtered = CIImage(color: CIColor(color: NSColor(edit.color)) ?? .black)
            }
            output = filtered.cropped(to: source.extent).applyingFilter("CIBlendWithMask", parameters: [
                kCIInputBackgroundImageKey: output,
                kCIInputMaskImageKey: edit.mask.transformed(by: transform)
            ]).cropped(to: source.extent)
        }
        return context.createCGImage(output, from: source.extent.integral)
    }

    func refresh() { preview = render(maximum: 1600) }

    func export() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.png]
        panel.nameFieldStringValue = "Filtered.png"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        guard let image = render(maximum: .greatestFiniteMagnitude),
              let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            message = "Could not create the export."
            return
        }
        CGImageDestinationAddImage(destination, image, nil)
        if !CGImageDestinationFinalize(destination) { message = "Could not write the PNG." }
    }

    func checkUpdates() async {
        struct Release: Decodable { let tag_name: String; let html_url: URL }
        updateURL = nil
        do {
            let endpoint = URL(string: "https://api.github.com/repos/pmeir99/Image-Filtration-V1/releases/latest")!
            let (data, response) = try await URLSession.shared.data(from: endpoint)
            let code = (response as? HTTPURLResponse)?.statusCode
            if code == 404 {
                message = "No stable release has been published yet."
                return
            }
            guard code == 200 else { throw URLError(.badServerResponse) }
            let release = try JSONDecoder().decode(Release.self, from: data)
            let installed = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "0.1.0"
            let latest = release.tag_name.hasPrefix("v") ? String(release.tag_name.dropFirst()) : release.tag_name
            if latest.compare(installed, options: .numeric) == .orderedDescending {
                updateURL = release.html_url
                message = "Version \(latest) is available."
            } else {
                message = "You’re running the latest version."
            }
        } catch {
            message = "Could not check GitHub for updates."
        }
    }
}

struct EditorView: View {
    @ObservedObject var editor: Editor

    private func fittedSize(_ image: CGImage, in bounds: CGSize) -> CGSize {
        let width = CGFloat(image.width)
        let height = CGFloat(image.height)
        let ratio = Swift.min(bounds.width / width, bounds.height / height)
        return CGSize(width: width * ratio, height: height * ratio)
    }

    var body: some View {
        HStack(spacing: 0) {
            GeometryReader { geometry in
                if let image = editor.preview {
                    let size = fittedSize(image, in: geometry.size)
                    Image(decorative: image, scale: 1)
                        .resizable()
                        .frame(width: size.width, height: size.height)
                        .contentShape(Rectangle())
                        .onTapGesture(coordinateSpace: .local) { point in editor.click(point, size: size) }
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    ContentUnavailableView {
                        Label("Image Filtration", systemImage: "photo")
                    } description: {
                        Text("Open or drop a photo to describe it and mask its subjects.")
                    } actions: {
                        Button("Open Image…", action: editor.open)
                    }
                }
            }
            .background(.black.opacity(0.08))
            .dropDestination(for: URL.self) { urls, _ in
                guard let url = urls.first, url.isFileURL else { return false }
                editor.load(url)
                return true
            }
            Divider()
            VStack(alignment: .leading, spacing: 14) {
                Text("Description").font(.headline)
                Text(editor.caption.isEmpty ? "Open an image to begin." : editor.caption)
                    .textSelection(.enabled)
                if !editor.status.isEmpty {
                    Text(editor.status).font(.callout).foregroundStyle(.secondary)
                }
                if !editor.processingTime.isEmpty {
                    LabeledContent("Detection + blur", value: editor.processingTime)
                        .font(.callout.monospacedDigit())
                }
                Divider()
                ForEach(editor.edits) { edit in
                    Button("Subject \(edit.id)", systemImage: editor.selection == edit.id ? "checkmark.circle.fill" : "circle") {
                        editor.selection = edit.id
                    }.buttonStyle(.plain)
                }
                if let index = editor.edits.firstIndex(where: { $0.id == editor.selection }) {
                    Picker("Effect", selection: $editor.edits[index].effect) {
                        ForEach(Effect.allCases) { Text($0.rawValue).tag($0) }
                    }
                    if editor.edits[index].effect == .solid {
                        ColorPicker("Fill", selection: $editor.edits[index].color, supportsOpacity: false)
                    } else {
                        Slider(value: $editor.edits[index].strength, in: 4...100) { Text("Strength") }
                    }
                    Button("Remove Mask", role: .destructive, action: editor.removeSelected)
                }
                Spacer()
            }
            .padding(18)
            .frame(width: 260)
        }
        .frame(minWidth: 760, minHeight: 480)
        .onChange(of: editor.edits.map { "\($0.id)-\($0.effect)-\($0.strength)-\($0.color)" }) {
            editor.refresh()
        }
        .toolbar {
            Button("Open Image", systemImage: "folder", action: editor.open)
            Button("Export PNG", systemImage: "square.and.arrow.up", action: editor.export)
                .disabled(editor.original == nil)
        }
        .alert("Image Filtration", isPresented: Binding(
            get: { editor.message != nil }, set: { if !$0 { editor.message = nil } }
        )) {
            if let url = editor.updateURL {
                Button("Open Release") { NSWorkspace.shared.open(url) }
            }
            Button("OK", role: .cancel) {}
        } message: { Text(editor.message ?? "") }
    }
}

@main
struct ImageFiltrationApp: App {
    @StateObject private var editor = Editor()

    var body: some Scene {
        Window("Image Filtration", id: "editor") {
            EditorView(editor: editor)
        }
        .defaultSize(width: 1080, height: 700)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("Open Image…", action: editor.open).keyboardShortcut("o")
            }
            CommandGroup(after: .appInfo) {
                Button("Check for Updates…") { Task { await editor.checkUpdates() } }
            }
        }
    }
}
