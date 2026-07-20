#!/bin/sh
set -eu

out="$(dirname "$0")/in_jpeg"
mkdir -p "$out"

# A small non-synthetic source; gradients and noise both matter, since a flat
# image produces almost no AC coefficients and exercises very little.
src="$(mktemp -t jixel-seed).ppm"
trap 'rm -f "$src"' EXIT
magick "$(dirname "$0")/../assets/abstract.jpg" -resize 32x24! "$src"

cjpeg -quality 80 -sample 1x1 -outfile "$out/base444.jpg" "$src"
cjpeg -quality 80 -sample 2x2 -outfile "$out/base420.jpg" "$src"
cjpeg -quality 80 -sample 2x1 -outfile "$out/base422.jpg" "$src"
cjpeg -quality 80 -sample 2x2 -progressive -outfile "$out/prog420.jpg" "$src"
cjpeg -quality 80 -sample 1x1 -progressive -outfile "$out/prog444.jpg" "$src"
cjpeg -quality 80 -sample 2x2 -restart 1 -outfile "$out/rst420.jpg" "$src"
cjpeg -quality 80 -grayscale -outfile "$out/gray.jpg" "$src"
cjpeg -quality 95 -sample 1x1 -optimize -outfile "$out/opt444.jpg" "$src"

echo "wrote $(ls -1 "$out" | wc -l | tr -d ' ') seeds to $out"
