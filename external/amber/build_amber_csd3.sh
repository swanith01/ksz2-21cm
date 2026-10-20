#!/bin/bash
# external/amber/build_amber_csd3.sh [install_dir]          (ksz2-21cm, 2026-10-20)
#
# CSD3 version of the vendored build_amber_tifr_reference.sh. Same recipe, same pinned AMBER commit,
# same field-dump patch, same Makefile.ksz. Differences:
#   * no /apps/intel/oneapi/setvars.sh (that is TIFR). The compiler comes from environment modules.
#   * HEALPix is never used (stub build, Map = no). Nothing here needs full-sky maps.
#   * the compiler is checked and reported, and the script stops with a clear message if it is missing.
#
# Run on a LOGIN node (needs internet for the git clone; the compile itself is light, a minute or two):
#   bash external/amber/build_amber_csd3.sh ~/rds/rds-dirac-dp145-uYnnMXSp128/dc-upad3/amber
# Optional environment:
#   AMBER_MODULES="rhel8/default-icl"    modules to load (default: that one). Space separated.
#   FC=ifort                             compiler command (default: ifort). 'ifx' is untested here.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEST=${1:-$HOME/amber}
AMBER_COMMIT=9703f515a70b17e9bac38457d4891dad9abf7f84    # the commit the dump patch was validated against
AMBER_MODULES=${AMBER_MODULES:-rhel8/default-icl}
FC=${FC:-ifort}

if [ -f /etc/profile.d/modules.sh ]; then . /etc/profile.d/modules.sh; fi
if command -v module >/dev/null 2>&1; then
  module purge
  for m in $AMBER_MODULES; do module load "$m" || { echo "ERROR: cannot load module $m"; exit 1; }; done
  module list 2>&1
else
  echo "NOTE: no 'module' command here; using whatever compiler is already on PATH"
fi

set -euo pipefail
command -v "$FC" >/dev/null || {
  echo "ERROR: compiler '$FC' not found after loading: $AMBER_MODULES"
  echo "Look for Intel compilers/MKL with:  module avail intel 2>&1 | head -40 ;  module avail mkl 2>&1 | head"
  echo "then rerun with  AMBER_MODULES=\"<module names>\"  (and FC=ifx if only ifx exists)"; exit 1; }
[ -n "${MKLROOT:-}" ] || {
  echo "ERROR: MKLROOT is not set, so MKL is not loaded. Find it with:  module avail mkl 2>&1 | head"
  echo "then add it, e.g.  AMBER_MODULES=\"rhel8/default-icl <mkl module>\""; exit 1; }
echo "compiler: $($FC --version 2>&1 | head -1)"
echo "MKLROOT:  $MKLROOT"

# --- source at pinned commit + patch ---------------------------------------------------------
if [ ! -d "$DEST/.git" ]; then
  git clone -q https://github.com/hytrac/amber.git "$DEST"
fi
cd "$DEST"
git fetch -q origin || true
if grep -q cmb_dumpfields src/cmbreion.f90 2>/dev/null && [ "$(git rev-parse HEAD)" = "$AMBER_COMMIT" ]; then
  echo "AMBER already at $AMBER_COMMIT with the patch applied"
else
  git checkout -q -- . && git checkout -q "$AMBER_COMMIT"
  git apply "$HERE/0001-dump-fields-for-ksz-pipeline.patch"
  echo "AMBER checked out at $AMBER_COMMIT, dump patch applied"
fi
cp "$HERE/healpix_stub.f90" "$HERE/Makefile.ksz" src/
cd src

# --- MKL flag: ifort only WARNS on unknown options, so look at the message -------------------
tmp=$(mktemp -d); printf 'program t\nend program t\n' > "$tmp/t.f90"
if $FC -qmkl=parallel "$tmp/t.f90" -o "$tmp/t" 2>&1 | grep -qi "unknown option\|ignoring"; then
  MKLFLAG=-mkl=parallel
else
  MKLFLAG=-qmkl=parallel
fi
rm -rf "$tmp"; echo "MKL flag: $MKLFLAG"

# --- build (stub HEALPix; serial make, the module dependencies are not declared) -------------
make -f Makefile.ksz clean >/dev/null
set +e
make -f Makefile.ksz FC="$FC" MKLFLAG="$MKLFLAG" > build.log 2>&1
rc=$?
set -e
if [ $rc -ne 0 ] || [ ! -x amber.x ]; then
  echo "BUILD FAILED (exit $rc). Last lines of $DEST/src/build.log:"; tail -25 build.log; exit 1
fi
echo
echo "OK: $DEST/src/amber.x"
echo "Runtime: load the same modules ($AMBER_MODULES) in the job script; check  ldd $DEST/src/amber.x | grep -i mkl"
