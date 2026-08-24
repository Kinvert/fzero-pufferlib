#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 WORKSPACE" >&2
    echo "WORKSPACE must contain G-Diffuser/ and PufferLib-5.0/ at the pinned commits." >&2
    exit 2
fi

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
workspace=$(cd "$1" && pwd)
gdiffuser="$workspace/G-Diffuser"
pufferlib="$workspace/PufferLib-5.0"
fzerox="$gdiffuser/decomp"
libultraship="$gdiffuser/libultraship"

gdiffuser_base=c62bf97d490370b05e14cb04d149f36516937270
fzerox_base=6474cfd4fcc7002bb0763ba6346b71d1715025ca
libultraship_base=a31fbffa86658b89dd04083f9bca615706665d45
pufferlib_base=0aa034def1b3c91af9971798288f8bddd1817277

require_head() {
    local repo=$1
    local expected=$2
    local actual
    actual=$(git -C "$repo" rev-parse HEAD)
    if [[ "$actual" != "$expected" ]]; then
        echo "error: $repo is at $actual; expected $expected" >&2
        exit 2
    fi
}

require_head "$gdiffuser" "$gdiffuser_base"
require_head "$fzerox" "$fzerox_base"
require_head "$libultraship" "$libultraship_base"
require_head "$pufferlib" "$pufferlib_base"

git -C "$fzerox" apply --check "$project_root/patches/fzerox.patch"
git -C "$libultraship" apply --check "$project_root/patches/libultraship.patch"
git -C "$gdiffuser" apply --check "$project_root/patches/g-diffuser.patch"
git -C "$pufferlib" apply --check "$project_root/patches/pufferlib-5.0.patch"

git -C "$fzerox" apply "$project_root/patches/fzerox.patch"
git -C "$libultraship" apply "$project_root/patches/libultraship.patch"
git -C "$gdiffuser" apply "$project_root/patches/g-diffuser.patch"
git -C "$pufferlib" apply "$project_root/patches/pufferlib-5.0.patch"

echo "Applied F-Zero PufferLib patches successfully."
