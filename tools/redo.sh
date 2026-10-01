#!/usr/bin/env bash
# Fill in the 7 redone photos and zip them for review.
# Doesn't delete anything: it only generates photos that are missing, so it never pays twice.
cd "$(dirname "$0")"

echo "━━ 1/3 busty on Nano Banana Pro (fills missing ones)"
python3 body_presets.py busty

echo "━━ 2/3 busty leftovers on Seedream (only if Nano Banana gave up)"
python3 body_presets.py busty --model seedream

echo "━━ 3/3 petite (Seedream, fills missing ones)"
python3 body_presets.py petite

FILES="busty/image_17.png busty/image_38.png busty/image_46.png petite/image_01.png petite/image_04.png petite/image_06.png petite/image_44.png"
OK=""
for f in $FILES; do
  if [ -f "$f" ]; then OK="$OK $f"; else echo "⚠ missing: $f (run bash redo.sh again)"; fi
done
rm -f redo.zip
zip -q redo.zip $OK && echo "📦 redo.zip ready ($(echo $OK | wc -w)/7 photos) — send it to Claude"
