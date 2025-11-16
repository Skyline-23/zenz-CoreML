#!/usr/bin/env bash
set -euo pipefail

PROJECT="ZenzCoreML.xcodeproj"
OUT_DIR="BuildArtifacts"

TARGETS=(
  "ZenzCoreMLStateless"
  "ZenzCoreMLStateless8bit"
  "ZenzCoreMLStateful"
  "ZenzCoreMLStateful8bit"
)

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

for TARGET in "${TARGETS[@]}"; do
  echo "=== Building $TARGET ==="

  IOS_ARCHIVE="$OUT_DIR/${TARGET}-iOS.xcarchive"
  MACOS_ARCHIVE="$OUT_DIR/${TARGET}-macOS.xcarchive"
  XCFRAMEWORK="$OUT_DIR/${TARGET}.xcframework"
  ZIP="$OUT_DIR/${TARGET}.xcframework.zip"

  xcodebuild archive \
    -project "$PROJECT" \
    -scheme "$TARGET" \
    -destination "generic/platform=iOS" \
    -archivePath "$IOS_ARCHIVE" \
    SKIP_INSTALL=NO \
    BUILD_LIBRARY_FOR_DISTRIBUTION=YES

  xcodebuild archive \
    -project "$PROJECT" \
    -scheme "$TARGET" \
    -destination "generic/platform=macOS" \
    -archivePath "$MACOS_ARCHIVE" \
    SKIP_INSTALL=NO \
    BUILD_LIBRARY_FOR_DISTRIBUTION=YES

  xcodebuild -create-xcframework \
    -framework "$IOS_ARCHIVE/Products/Library/Frameworks/${TARGET}.framework" \
    -framework "$MACOS_ARCHIVE/Products/Library/Frameworks/${TARGET}.framework" \
    -output "$XCFRAMEWORK"

  (cd "$OUT_DIR" && zip -r "$(basename "$ZIP")" "$(basename "$XCFRAMEWORK")" >/dev/null)

  echo "=== swift package checksum for $TARGET ==="
  swift package compute-checksum "$ZIP"
  echo
done

echo "✅ All xcframeworks are under $OUT_DIR"
