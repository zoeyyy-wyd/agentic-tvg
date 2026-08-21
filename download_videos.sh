#!/usr/bin/env bash
# Download + unpack the video working set (DATA.md §2/§4/§5): ~18 GiB across
# tvg (SFT traces' Charades frames), selfqa (RL train), rl_val (RL val).
# Each zip is deleted right after extraction to keep peak disk down (DATA.md §3).
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p data/archives data/videos/{tvg,selfqa,rl_val}

download () {  # $1 = --include pattern
    hf download longvideotool/LongVT-Source --repo-type dataset \
        --include "$1" --local-dir data/archives
}

extract () {   # $1 = zip glob, $2 = target dir
    for z in data/archives/$1; do
        [ -e "$z" ] || continue
        echo "== extracting $z -> $2"
        unzip -q -o "$z" -d "$2"
        rm -f "$z"
    done
}

# Smallest first: rl_val unblocks extract_rl.py + probe fastest.
download "rl_val_1.zip";  extract "rl_val_1.zip"  data/videos/rl_val
download "selfqa_1.zip";  extract "selfqa_1.zip"  data/videos/selfqa
download "tvg_*.zip";     extract "tvg_*.zip"     data/videos/tvg

echo "== done; layout:"
du -sh data/videos/* 2>/dev/null
find data/videos -type f | head -5
