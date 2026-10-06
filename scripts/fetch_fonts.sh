#!/usr/bin/env bash
# Download the fonts the renderer uses into custom_components/eink_dashboard/fonts/.
#
# The fonts are not in git (see .gitignore). The build needs them in the
# package, and the tests need them too: several of them measure text, which
# comes out different with PIL's fallback font. A font that is already there
# is not downloaded again.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${0}")/.." && pwd)"
FONTS_DIR="${REPO_ROOT}/custom_components/eink_dashboard/fonts"

ROBOTO_URL="https://github.com/googlefonts/roboto/raw/main/src/hinted/Roboto-Regular.ttf"
ROBOTO_MEDIUM_URL="https://github.com/googlefonts/roboto/raw/main/src/hinted/Roboto-Medium.ttf"
IBM_PLEX_MONO_URL="https://github.com/IBM/plex/raw/master/packages/plex-mono/fonts/complete/ttf/IBMPlexMono-Regular.ttf"
NOTO_SANS_URL="https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf"
# Monochrome emoji, used as a fallback for characters the text fonts lack.
NOTO_EMOJI_URL="https://github.com/google/fonts/raw/main/ofl/notoemoji/NotoEmoji%5Bwght%5D.ttf"
# Chess pieces for the chess_board widget.
NOTO_SYMBOLS_URL="https://github.com/google/fonts/raw/main/ofl/notosanssymbols2/NotoSansSymbols2-Regular.ttf"

mkdir -p "${FONTS_DIR}"
if [ -f "${FONTS_DIR}/Roboto-Regular.ttf" ]; then
    echo "==> Roboto-Regular.ttf already exists, skipping download"
else
    echo "==> Downloading Roboto-Regular.ttf (Apache 2.0)..."
    curl -fsSL "${ROBOTO_URL}" -o "${FONTS_DIR}/Roboto-Regular.ttf"
fi

if [ -f "${FONTS_DIR}/Roboto-Medium.ttf" ]; then
    echo "==> Roboto-Medium.ttf already exists, skipping download"
else
    echo "==> Downloading Roboto-Medium.ttf (Apache 2.0)..."
    curl -fsSL "${ROBOTO_MEDIUM_URL}" -o "${FONTS_DIR}/Roboto-Medium.ttf"
fi

if [ -f "${FONTS_DIR}/IBMPlexMono-Regular.ttf" ]; then
    echo "==> IBMPlexMono-Regular.ttf already exists, skipping download"
else
    echo "==> Downloading IBMPlexMono-Regular.ttf (SIL Open Font License)..."
    curl -fsSL "${IBM_PLEX_MONO_URL}" -o "${FONTS_DIR}/IBMPlexMono-Regular.ttf"
fi

if [ -f "${FONTS_DIR}/NotoSans-Regular.ttf" ]; then
    echo "==> NotoSans-Regular.ttf already exists, skipping download"
else
    echo "==> Downloading NotoSans-Regular.ttf (SIL Open Font License)..."
    curl -fsSL "${NOTO_SANS_URL}" -o "${FONTS_DIR}/NotoSans-Regular.ttf"
fi

if [ -f "${FONTS_DIR}/NotoEmoji-Regular.ttf" ]; then
    echo "==> NotoEmoji-Regular.ttf already exists, skipping download"
else
    echo "==> Downloading NotoEmoji-Regular.ttf (SIL Open Font License)..."
    curl -fsSL "${NOTO_EMOJI_URL}" -o "${FONTS_DIR}/NotoEmoji-Regular.ttf"
fi

if [ -f "${FONTS_DIR}/NotoSansSymbols2-Regular.ttf" ]; then
    echo "==> NotoSansSymbols2-Regular.ttf already exists, skipping download"
else
    echo "==> Downloading NotoSansSymbols2-Regular.ttf (SIL Open Font License)..."
    curl -fsSL "${NOTO_SYMBOLS_URL}" -o "${FONTS_DIR}/NotoSansSymbols2-Regular.ttf"
fi
