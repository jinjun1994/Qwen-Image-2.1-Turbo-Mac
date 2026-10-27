#!/bin/zsh
set -euo pipefail
TASK_ROOT="${0:A:h:h}"
cd "$TASK_ROOT"
APP="$TASK_ROOT/dist/MQwen.app"
VERSION=$("$TASK_ROOT/.venv/bin/python" -c 'from mqwen import __version__;print(__version__)')
[[ -x "$APP/Contents/MacOS/MQwen" ]] || { print -u2 'Build MQwen.app first'; exit 1; }
cp README.md README.en.md LICENSE THIRD_PARTY.md CHANGELOG.md "$APP/Contents/Resources/"
mkdir -p "$APP/Contents/Resources/docs"
cp -R docs/. "$APP/Contents/Resources/docs/"
codesign --force --deep --sign - "$APP"
codesign --verify --deep --strict "$APP"
STAGE="$TASK_ROOT/build/dmg-stage"
rm -rf "$STAGE"
mkdir -p "$STAGE"
cp -R "$APP" "$STAGE/MQwen.app"
ln -s /Applications "$STAGE/Applications"
cp README.md README.en.md LICENSE THIRD_PARTY.md CHANGELOG.md "$STAGE/"
cp -R docs "$STAGE/docs"
DMG="$TASK_ROOT/dist/MQwen-${VERSION}-arm64.dmg"
rm -f "$DMG"
hdiutil create -volname "MQwen $VERSION" -srcfolder "$STAGE" -ov -format UDZO "$DMG"
(cd "$TASK_ROOT/dist" && shasum -a 256 "${DMG:t}" > "${DMG:t}.sha256")
print "Created $DMG"
