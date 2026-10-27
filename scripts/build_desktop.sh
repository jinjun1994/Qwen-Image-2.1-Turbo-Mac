#!/bin/zsh
set -euo pipefail
TASK_ROOT="${0:A:h:h}"
cd "$TASK_ROOT"
[[ "$(uname -m)" == arm64 ]] || { print -u2 'Build requires Apple Silicon'; exit 1; }
PY="$TASK_ROOT/.venv/bin/python"
"$PY" -c 'import PyInstaller' || { print -u2 'Install packaging tools: uv pip install --python .venv/bin/python pyinstaller==6.22.3'; exit 1; }
mkdir -p reports build dist
"$TASK_ROOT/.venv/bin/pyinstaller" --noconfirm --distpath build/desktop-worker --workpath build/pyinstaller desktop/mqwen-worker.spec > reports/desktop-freeze.log 2>&1
"build/desktop-worker/mqwen-worker/mqwen-worker" --runtime-check > reports/desktop-runtime-check.log 2>&1
APP="$TASK_ROOT/dist/MQwen.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
swiftc -O desktop/MQwen.swift -o "$APP/Contents/MacOS/MQwen" -framework Cocoa -framework WebKit
cp -R build/desktop-worker/mqwen-worker "$APP/Contents/Resources/worker"
cp README.md LICENSE THIRD_PARTY.md "$APP/Contents/Resources/"
"$PY" - <<'PY'
import importlib.metadata as meta
from pathlib import Path
import plistlib,shutil,sysconfig,tomllib
version=tomllib.loads(Path('pyproject.toml').read_text())['project']['version']
app=Path('dist/MQwen.app/Contents')
with (app/'Info.plist').open('wb') as file:
 plistlib.dump(dict(CFBundleName='MQwen',CFBundleDisplayName='MQwen',CFBundleIdentifier='com.mqwen.studio',CFBundleExecutable='MQwen',CFBundlePackageType='APPL',CFBundleShortVersionString=version,CFBundleVersion=version,CFBundleIconFile='AppIcon',LSMinimumSystemVersion='26.0',LSApplicationCategoryType='public.app-category.graphics-design',NSHighResolutionCapable=True,NSAppTransportSecurity=dict(NSAllowsLocalNetworking=True),NSHumanReadableCopyright='MQwen contributors. Source MIT; see THIRD_PARTY.md for dependencies and model terms.'),file)
# Include dependency license files available in the build environment; this is
# deliberately a superset, and does not imply every distribution is executed.
base=app/'Resources/Licenses';base.mkdir()
shutil.copyfile(Path(sysconfig.get_path('stdlib'))/'LICENSE.txt',base/'CPython-LICENSE.txt')
for dist in meta.distributions():
 name=dist.metadata.get('Name','unknown')
 for file in dist.files or []:
  if any(part.lower() in ('licenses','license','copying','notice') or part.lower().startswith(('license.','copying.','notice.')) for part in file.parts):
   src=Path(dist.locate_file(file))
   if src.is_file():
    target=base/name/str(file).replace('../','_parent_/');target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,target)
PY
swift desktop/icon.swift build/icon-png
mkdir -p build/AppIcon.iconset
for pair in '16 16x16' '32 16x16@2x' '32 32x32' '64 32x32@2x' '128 128x128' '256 128x128@2x' '256 256x256' '512 256x256@2x' '512 512x512' '1024 512x512@2x'; do
 read -r size label <<< "$pair"
 cp "build/icon-png/$size.png" "build/AppIcon.iconset/icon_$label.png"
done
iconutil -c icns build/AppIcon.iconset -o "$APP/Contents/Resources/AppIcon.icns"
"$TASK_ROOT/scripts/package_desktop.sh"
