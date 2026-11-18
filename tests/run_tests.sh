#!/bin/sh
# Builds and runs the roundtrip tests under several configurations:
#   * strict C99 with warnings, debug asserts enabled
#   * NDEBUG build (asserts compiled out) - runtime guards must still work
#   * ASan + UBSan build - no out-of-bounds reads on odd/cropped dimensions
#   * C++ compilation of the single header
set -e

cd "$(dirname "$0")"

CC="${CC:-/home/liiing/tc/gcc}"
SANCC="${SANCC:-/home/liiing/toolchain/bin/zig-cc}"
CXX="${CXX:-/home/liiing/toolchain/bin/zig-cxx}"
OUT=build
mkdir -p "$OUT"

echo "== C99 debug =="
"$CC" -std=c99 -Wall -Wextra -I.. test_roundtrip.c -lm -o "$OUT/test_c99"
"$OUT/test_c99"

echo
echo "== C99 release (NDEBUG, asserts off) =="
"$CC" -std=c99 -Wall -Wextra -DNDEBUG -I.. test_roundtrip.c -lm -o "$OUT/test_ndebug"
"$OUT/test_ndebug"

echo
echo "== ASan + UBSan =="
"$SANCC" -std=c99 -g -O1 -fsanitize=address,undefined -fno-sanitize-recover=all \
    -Wall -Wextra -I.. test_roundtrip.c -lm -o "$OUT/test_asan"
"$OUT/test_asan"

echo
echo "== C++17 compile =="
# -x c++: the test source has a .c extension but must be parsed as C++
"$CXX" -x c++ -std=c++17 -Wall -Wextra -I.. test_roundtrip.c -lm -o "$OUT/test_cpp"
"$OUT/test_cpp"

echo
echo "All test configurations passed."
