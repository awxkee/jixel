#!/usr/bin/env bash
#
# // Copyright (c) Radzivon Bartoshyk 7/2026. All rights reserved.
# //
# // Redistribution and use in source and binary forms, with or without modification,
# // are permitted provided that the following conditions are met:
# //
# // 1.  Redistributions of source code must retain the above copyright notice, this
# // list of conditions and the following disclaimer.
# //
# // 2.  Redistributions in binary form must reproduce the above copyright notice,
# // this list of conditions and the following disclaimer in the documentation
# // and/or other materials provided with the distribution.
# //
# // 3.  Neither the name of the copyright holder nor the names of its
# // contributors may be used to endorse or promote products derived from
# // this software without specific prior written permission.
# //
# // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

#
# to_jxl.sh
#
# Converts all images in a source folder to JPEG XL (.jxl) using `cjxl`
# (the reference libjxl encoder), writing the results into a separate
# destination folder. The folder structure (with --recursive) is mirrored
# into the destination, so subfolders line up 1:1 between source and dest.
#
# Both the source and destination folders are REQUIRED, explicit, positional
# arguments — there is no default/current-directory fallback for either.
#
# Usage:
#   ./to_jxl.sh <source_folder> <dest_folder> [--recursive] [--dry-run]
#               [--quality N] [--lossless] [--ext jpg,png,...]
#               [--delete-original] [--overwrite] [--effort N]
#
# Examples:
#   ./to_jxl.sh ./photos ./photos_jxl
#   ./to_jxl.sh ./photos ./photos_jxl --recursive
#   ./to_jxl.sh ./photos ./photos_jxl --recursive --quality 95
#   ./to_jxl.sh ./photos ./photos_jxl --recursive --lossless
#   ./to_jxl.sh ./photos ./photos_jxl --recursive --dry-run
#   ./to_jxl.sh ./photos ./photos_jxl --recursive --delete-original
#
# Notes:
#   - Requires `cjxl` (part of libjxl). Install via: brew install jpeg-xl
#   - The destination folder is created automatically if it doesn't exist,
#     including any mirrored subfolders (with --recursive).
#   - Source and destination must be different paths — the script refuses
#     to run if they resolve to the same directory, to avoid accidentally
#     falling back to in-place conversion.
#   - By default source files are KEPT; pass --delete-original to remove
#     them after a successful conversion.
#   - If an output .jxl already exists at the destination path, it is
#     skipped (won't overwrite) unless --overwrite is set.

set -euo pipefail

# ---- defaults ----
RECURSIVE=false
DRY_RUN=false
DELETE_ORIGINAL=false
OVERWRITE=false
LOSSLESS=false
QUALITY=90
EFFORT=7
EXTENSIONS="jpg,jpeg,png,bmp,gif,ppm,pgm,pfm,pgx,apng"
SRC_DIR=""
DEST_DIR=""

usage() {
    echo "Usage: $0 <source_folder> <dest_folder> [--recursive] [--dry-run]"
    echo "          [--quality N] [--lossless] [--ext ext1,ext2,...]"
    echo "          [--delete-original] [--overwrite] [--effort N]"
    echo ""
    echo "  <source_folder>    REQUIRED. Folder containing images to convert."
    echo "  <dest_folder>      REQUIRED. Folder to write .jxl output into"
    echo "                     (created automatically if it doesn't exist)."
    echo "  --recursive        Also process images in subfolders (mirrors"
    echo "                     the subfolder structure into dest_folder)"
    echo "  --dry-run          Show which files would be converted, without converting"
    echo "  --quality N        JPEG XL quality, 0-100 (default: $QUALITY). Ignored if --lossless."
    echo "  --lossless         Encode losslessly instead of using --quality"
    echo "  --effort N         Encoder effort/speed, 1 (fastest) - 9 (slowest/best) (default: $EFFORT)"
    echo "  --ext              Comma-separated source extensions (default: $EXTENSIONS)"
    echo "  --delete-original  Delete the source file after a successful conversion"
    echo "  --overwrite        Overwrite existing .jxl output files (default: skip if exists)"
    exit 1
}

if [ $# -eq 0 ]; then
    echo "Error: no source/destination folder specified. Both are required." >&2
    echo "" >&2
    usage
fi

while [ $# -gt 0 ]; do
    case "$1" in
        --recursive)
            RECURSIVE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --lossless)
            LOSSLESS=true
            shift
            ;;
        --delete-original)
            DELETE_ORIGINAL=true
            shift
            ;;
        --overwrite)
            OVERWRITE=true
            shift
            ;;
        --quality)
            [ -z "${2:-}" ] && { echo "Error: --quality requires a value" >&2; usage; }
            QUALITY="$2"
            shift 2
            ;;
        --effort)
            [ -z "${2:-}" ] && { echo "Error: --effort requires a value" >&2; usage; }
            EFFORT="$2"
            shift 2
            ;;
        --ext)
            [ -z "${2:-}" ] && { echo "Error: --ext requires a value" >&2; usage; }
            EXTENSIONS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            if [ -z "$SRC_DIR" ]; then
                SRC_DIR="$1"
            elif [ -z "$DEST_DIR" ]; then
                DEST_DIR="$1"
            else
                echo "Error: unexpected argument '$1' (source='$SRC_DIR', dest='$DEST_DIR' already set)" >&2
                usage
            fi
            shift
            ;;
    esac
done

if [ -z "$SRC_DIR" ] || [ -z "$DEST_DIR" ]; then
    echo "Error: both <source_folder> and <dest_folder> are required." >&2
    echo "" >&2
    usage
fi

if [ ! -d "$SRC_DIR" ]; then
    echo "Error: source folder '$SRC_DIR' is not a directory" >&2
    exit 1
fi

# ---- resolve to absolute paths so the same-directory check and the
#      relative-path mirroring below are both reliable regardless of
#      how the paths were typed (relative, trailing slash, etc.) ----
SRC_DIR_ABS=$(cd "$SRC_DIR" && pwd)

if [ ! -d "$DEST_DIR" ]; then
    if [ "$DRY_RUN" = false ]; then
        mkdir -p "$DEST_DIR"
    fi
fi
# If --dry-run and dest doesn't exist yet, resolve it against its parent
# instead of requiring it to already exist.
if [ -d "$DEST_DIR" ]; then
    DEST_DIR_ABS=$(cd "$DEST_DIR" && pwd)
else
    DEST_PARENT=$(dirname "$DEST_DIR")
    if [ ! -d "$DEST_PARENT" ]; then
        echo "Error: destination's parent folder '$DEST_PARENT' does not exist" >&2
        exit 1
    fi
    DEST_DIR_ABS="$(cd "$DEST_PARENT" && pwd)/$(basename "$DEST_DIR")"
fi

if [ "$SRC_DIR_ABS" = "$DEST_DIR_ABS" ]; then
    echo "Error: source and destination resolve to the same folder ('$SRC_DIR_ABS')." >&2
    echo "Refusing to run in-place — pass a different destination folder." >&2
    exit 1
fi

# ---- check cjxl is available ----
if ! command -v cjxl >/dev/null 2>&1; then
    echo "Error: 'cjxl' not found. Install libjxl first:" >&2
    echo "  brew install jpeg-xl" >&2
    exit 1
fi

# ---- build find's extension filter ----
IFS=',' read -ra EXT_ARRAY <<< "$EXTENSIONS"

FIND_ARGS=()
if [ "$RECURSIVE" = false ]; then
    FIND_ARGS+=(-maxdepth 1)
fi
FIND_ARGS+=(-type f "(")
for i in "${!EXT_ARRAY[@]}"; do
    ext="${EXT_ARRAY[$i]}"
    if [ "$i" -gt 0 ]; then
        FIND_ARGS+=(-o)
    fi
    FIND_ARGS+=(-iname "*.${ext}")
done
FIND_ARGS+=(")")

# ---- collect matching files ----
# Bash-3.2-compatible (no `mapfile`/`readarray`), since macOS ships an old
# Bash by default. NUL-delimited to safely handle spaces/special characters.
FILES=()
while IFS= read -r -d '' f; do
    FILES+=("$f")
done < <(find "$SRC_DIR_ABS" "${FIND_ARGS[@]}" -print0)

TOTAL="${#FILES[@]}"
if [ "$TOTAL" -eq 0 ]; then
    echo "No matching image files found in '$SRC_DIR' (extensions: $EXTENSIONS)."
    exit 0
fi

echo "Found $TOTAL image file(s) to convert."
echo "Source:      $SRC_DIR_ABS"
echo "Destination: $DEST_DIR_ABS"
echo "Recursive: $RECURSIVE"
echo "Dry run: $DRY_RUN"
if [ "$LOSSLESS" = true ]; then
    echo "Mode: lossless"
else
    echo "Mode: quality $QUALITY"
fi
echo "Effort: $EFFORT"
echo "Delete original after conversion: $DELETE_ORIGINAL"
echo "Overwrite existing .jxl: $OVERWRITE"
echo ""

COUNT=0
CONVERTED=0
SKIPPED=0
FAILED=0

for f in "${FILES[@]}"; do
    COUNT=$((COUNT + 1))

    # Path of the file relative to the source root, so we can mirror the
    # same relative subfolder structure under the destination root.
    rel_path="${f#"$SRC_DIR_ABS"/}"
    rel_dir=$(dirname "$rel_path")
    base=$(basename "$f")
    name="${base%.*}"

    if [ "$rel_dir" = "." ]; then
        out_dir="$DEST_DIR_ABS"
    else
        out_dir="$DEST_DIR_ABS/$rel_dir"
    fi
    out="$out_dir/$name.jxl"

    if [ -f "$out" ] && [ "$OVERWRITE" = false ]; then
        echo "[$COUNT/$TOTAL] skip (exists): $out"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    if [ "$DRY_RUN" = true ]; then
        echo "[dry-run] would convert: $f -> $out"
        continue
    fi

    mkdir -p "$out_dir"

    echo "[$COUNT/$TOTAL] converting: $f -> $out"

    CJXL_ARGS=(-e "$EFFORT")
    if [ "$LOSSLESS" = true ]; then
        CJXL_ARGS+=(--lossless_jpeg=0 -d 0)
    else
        CJXL_ARGS+=(-q "$QUALITY")
    fi

    if cjxl "${CJXL_ARGS[@]}" "$f" "$out" >/tmp/to_jxl_err.log 2>&1; then
        CONVERTED=$((CONVERTED + 1))
        if [ "$DELETE_ORIGINAL" = true ]; then
            rm -f "$f"
        fi
    else
        echo "  ! failed: $(cat /tmp/to_jxl_err.log)" >&2
        FAILED=$((FAILED + 1))
    fi
done

rm -f /tmp/to_jxl_err.log

echo ""
if [ "$DRY_RUN" = true ]; then
    echo "Dry run complete. $TOTAL file(s) would have been processed."
else
    echo "Done. Converted: $CONVERTED, Skipped (already existed): $SKIPPED, Failed: $FAILED (out of $TOTAL total)."
    if [ "$FAILED" -gt 0 ]; then
        exit 1
    fi
fi