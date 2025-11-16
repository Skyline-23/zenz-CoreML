  // swift-tools-version: 6.1
  import PackageDescription

  let package = Package(
      name: "ZenzCoreMLPackage",
      platforms: [
          .iOS(.v18),
          .macOS(.v15)
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
              url: "https://github.com/Skyline-23/zenz-CoreML/releases/download/v3.1.0/ZenzCoreMLStateless.xcframework.zip",
              checksum: "c94960b1b1fb0529e4ffaa2a90defb8d5076a0f61d5ea207b07d8788326f1bdf"
          ),
          .binaryTarget(
              name: "ZenzCoreMLStateless8bit",
              url: "https://github.com/Skyline-23/zenz-CoreML/releases/download/v3.1.0/ZenzCoreMLStateless8bit.xcframework.zip",
              checksum: "d2ff4097fe1aa371ced80d0a54cee33729f7404791203c3aa6584ee7e8953e6d"
          ),
          .binaryTarget(
              name: "ZenzCoreMLStateful",
              url: "https://github.com/Skyline-23/zenz-CoreML/releases/download/v3.1.0/ZenzCoreMLStateful.xcframework.zip",
              checksum: "0d4fcf1a29ec0d09a1e1adbdc6f587bb889275371c29076ade30053a8a1be5a0"
          ),
          .binaryTarget(
              name: "ZenzCoreMLStateful8bit",
              url: "https://github.com/Skyline-23/zenz-CoreML/releases/download/v3.1.0/ZenzCoreMLStateful8bit.xcframework.zip",
              checksum: "7e36c19ee66be16042361f0feebe7dd227a85097c7672a08570c5dd76988d546"
          )
      ]
  )
