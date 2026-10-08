#!/usr/bin/env bash
# One-time build of DARTH (Chatzakis, Papakonstantinou, Palpanas, SIGMOD 2026) for the comparison in
# the PercEF paper. DARTH is its authors' own FAISS fork; we change two things and nothing else:
#   1. faiss/CMakeLists.txt points at "~/lightgbm-install", which CMake does not expand: we write $HOME.
#   2. hnsw-test/VectorDataLoader.cpp only knows the authors' dataset paths: we add a generic layout
#      (the one benchmark_unified.py --export-darth writes) for any dataset name it does not know.
# The search, the features and the predictor are untouched.
#
# Needs (once, with sudo):  sudo apt-get install -y cmake g++ libopenblas-dev libomp-dev git
# and Python packages:     pip install --user lightgbm pandas
set -euo pipefail
ROOT=${DARTH_ROOT:-$HOME/darth-build}
COMMIT=0d9bafcf31d1d79668bc71139fe93fa5e70b5185          # the version inspected for this comparison
mkdir -p "$ROOT"
cd "$ROOT"

# LightGBM needs CMake >= 3.28; Ubuntu 22.04 ships 3.22. A newer CMake from pip needs no sudo.
CMAKE=cmake
have=$(cmake --version 2>/dev/null | head -1 | awk '{print $3}')
if [ -z "$have" ] || [ "$(printf '%s\n' 3.28 "$have" | sort -V | head -1)" != "3.28" ]; then
  python3 -m pip install --user --upgrade "cmake>=3.28"
  CMAKE="$HOME/.local/bin/cmake"
fi
echo "using $($CMAKE --version | head -1)"

# LightGBM C library (DARTH links against it at $HOME/lightgbm-install)
if [ ! -f "$HOME/lightgbm-install/lib/lib_lightgbm.so" ]; then
  [ -d LightGBM ] || git clone --recursive --depth 1 https://github.com/microsoft/LightGBM.git
  rm -rf LightGBM/build
  "$CMAKE" -S LightGBM -B LightGBM/build -DCMAKE_INSTALL_PREFIX="$HOME/lightgbm-install" -DCMAKE_BUILD_TYPE=Release
  "$CMAKE" --build LightGBM/build -j"$(nproc)"
  "$CMAKE" --install LightGBM/build
fi

# DARTH at the pinned commit
if [ ! -d DARTH ]; then
  git clone https://github.com/MChatzakis/DARTH.git
fi
cd DARTH
git checkout -q "$COMMIT"
git checkout -q -- faiss/CMakeLists.txt hnsw-test/VectorDataLoader.cpp     # re-apply patches cleanly

sed -i "s|~/lightgbm-install|$HOME/lightgbm-install|g" faiss/CMakeLists.txt

python3 - <<'EOF'
p = "hnsw-test/VectorDataLoader.cpp"
s = open(p).read()
anchor = 'queryTypeToGroundtruthDistancesMap[NOISY_TESTING]["T2I100M"]'
i = s.index(anchor)
j = s.index("\n}", i)                       # end of initializeDataMaps()
block = r'''

    // Added for the PercEF comparison (loader only): the layout written by
    // benchmark_unified.py --export-darth, for any dataset name not listed above:
    // <dir>/<NAME>/base.fvecs, {train,validation,test}.fvecs, {split}.gt.ivecs, {split}.gtd.fvecs
    if (baseVectorsMap.find(dataset_name) == baseVectorsMap.end()) {
        std::string p = directory_path + dataset_name + "/";
        baseVectorsMap[dataset_name] = p + "base.fvecs";
        const char* split[3] = {"train", "validation", "test"};
        query_type_t qt[3] = {TRAINING, VALIDATION, TESTING};
        for (int i = 0; i < 3; i++) {
            queryTypeToVectorsMap[qt[i]][dataset_name] = p + split[i] + ".fvecs";
            queryTypeToGroundtruthsMap[qt[i]][dataset_name] = p + split[i] + ".gt.ivecs";
            queryTypeToGroundtruthDistancesMap[qt[i]][dataset_name] = p + split[i] + ".gtd.fvecs";
        }
    }'''
open(p, "w").write(s[:j] + block + s[j:])
print("patched", p)
EOF

rm -rf build
"$CMAKE" -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_BUILD_TYPE=Release -B build -S .
make -C build -j"$(nproc)" faiss
make -C build -j"$(nproc)" hnsw_test
echo "built: $ROOT/DARTH/build/hnsw-test/hnsw_test"
