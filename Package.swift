  // swift-tools-version: 6.1
  import PackageDescription

  let package = Package(
      name: "ZenzCoreMLPackage",
      platforms: [
          .iOS(.v16),
          .macOS(.v13)
      ],
      products: [
          .library(
              name: "ZenzCoreMLStateless",
              targets: ["ZenzCoreMLStateless"]
          ),
          .library(
              name: "ZenzCoreMLStateless8bit",
              targets: ["ZenzCoreMLStateless8bit"]
          ),
          .library(
              name: "ZenzCoreMLStateful",
              targets: ["ZenzCoreMLStateful"]
          ),
          .library(
              name: "ZenzCoreMLStateful8bit",
              targets: ["ZenzCoreMLStateful8bit"]
          )
      ],
      targets: [
          .binaryTarget(
              name: "ZenzCoreMLStateless",
              url: "https://github.com/Skyline-23/zenz-CoreML/releases/download/v3.1.1/ZenzCoreMLStateless.xcframework.zip",
              checksum: "c3bb1f092e839fd02a4ad05e63363bff85e5715cd8b00195522f08e970bb52b1"
          ),
          .binaryTarget(
              name: "ZenzCoreMLStateless8bit",
              url: "https://github.com/Skyline-23/zenz-CoreML/releases/download/v3.1.1/ZenzCoreMLStateless8bit.xcframework.zip",
              checksum: "0fa0ccacf436d94ac12b1260a07a52a2a627d7c95266030e1bb314ecdadace4c"
          ),
          .binaryTarget(
              name: "ZenzCoreMLStateful",
              url: "https://github.com/Skyline-23/zenz-CoreML/releases/download/v3.1.1/ZenzCoreMLStateful.xcframework.zip",
              checksum: "36d23f63e0e51dda2b5d20241a32b88dc8410f86f91c05c094d0d7554b8a1765"
          ),
          .binaryTarget(
              name: "ZenzCoreMLStateful8bit",
              url: "https://github.com/Skyline-23/zenz-CoreML/releases/download/v3.1.1/ZenzCoreMLStateful8bit.xcframework.zip",
              checksum: "0b70982ad90a07510deaaf8ae56ea4c4e7492082c0156c3cc572e406e56eb385"
          )
      ]
  )
