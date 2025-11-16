// swift-tools-version: 6.1
import ProjectDescription

func coreMLTarget(
    name: String,
    bundleIdSuffix: String,
    patterns: [String],
    excluding: [String] = []
) -> Target {
    .target(
        name: name,
        destinations: [.iPhone, .mac],
        product: .framework,
        bundleId: "com.skyline23.\(bundleIdSuffix)",
        infoPlist: .default,
        sources: [],
        resources: .resources(
            patterns.map { pattern in
                .glob(
                    pattern: .relativeToManifest(pattern),
                    excluding: excluding.map { .relativeToManifest($0) }
                )
            }
        ),
        dependencies: []
    )
}

let project = Project(
    name: "ZenzCoreML",
    targets: [
        coreMLTarget(
            name: "ZenzCoreMLStateless",
            bundleIdSuffix: "ZenzCoreML.Stateless",
            patterns: ["Stateless/*.mlpackage"],
            excluding: ["Stateless/*-8bit.mlpackage"]
        ),
        coreMLTarget(
            name: "ZenzCoreMLStateless8bit",
            bundleIdSuffix: "ZenzCoreML.Stateless8bit",
            patterns: ["Stateless/*-8bit.mlpackage"]
        ),
        coreMLTarget(
            name: "ZenzCoreMLStateful",
            bundleIdSuffix: "ZenzCoreML.Stateful",
            patterns: ["Stateful/*.mlpackage"],
            excluding: ["Stateful/*-8bit.mlpackage"]
        ),
        coreMLTarget(
            name: "ZenzCoreMLStateful8bit",
            bundleIdSuffix: "ZenzCoreML.Stateful8bit",
            patterns: ["Stateful/*-8bit.mlpackage"]
        )
    ]
)
