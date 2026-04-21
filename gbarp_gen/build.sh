#!/bin/bash
set -e
cd "$(dirname "$0")"

case "$1" in
    clean) rm -rf build; exit 0 ;;
esac

command -v gcc &>/dev/null || { echo "gcc not found"; exit 1; }
mkdir -p build

gcc -O3 -march=native -DSTANDALONE -o build/gbarp_gen src/gbarp_generator.c -lm
gcc -O3 -march=native -fPIC -shared -o build/libgbarp.so src/gbarp_generator.c -lm

echo "Built: build/gbarp_gen, build/libgbarp.so"
