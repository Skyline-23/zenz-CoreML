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
              checksum: "6706583a40fb46b1b5324861fc14496286f453bd163db6f0d0470775bca6bff7"
          ),
          .binaryTarget(
              name: "ZenzCoreMLStateless8bit",
              url: "https://github.com/Skyline-23/zenz-CoreML/releases/download/v3.1.0/ZenzCoreMLStateless8bit.xcframework.zip",
              checksum: "9ccfab5ec41b5dae2813d0f897654c41416e31e8167d3c79e21e85822679aa96"
          ),
          .binaryTarget(
              name: "ZenzCoreMLStateful",
              url: "https://github.com/Skyline-23/zenz-CoreML/releases/download/v3.1.0/ZenzCoreMLStateful.xcframework.zip",
              checksum: "c101db04ba873f121cf2e36053b7766f998f892afc97c14858f9b3778e0e7919"
          ),
          .binaryTarget(
              name: "ZenzCoreMLStateful8bit",
              url: "https://github.com/Skyline-23/zenz-CoreML/releases/download/v3.1.0/ZenzCoreMLStateful8bit.xcframework.zip",
              checksum: "58240fa965678bed92d9acb9278880a8971358ff5dfe04b739a2629229a4507b"
          )
      ]
  )
