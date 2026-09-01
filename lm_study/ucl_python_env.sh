#!/bin/bash

# Source this file before invoking the AMN virtual environment from a login node.
export PYROOT=${PYROOT:-/share/apps/python-3.11.9-shared}
export PYTHON_SOURCE=${PYTHON_SOURCE:-/share/apps/source_files/python/python-3.11.9.source}
export OPENSSL_ROOT=${OPENSSL_ROOT:-/share/apps/openssl-3.0.13}
export LIBFFI_ROOT=${LIBFFI_ROOT:-/share/apps/libffi-3.4.6}

if [ -f "$PYTHON_SOURCE" ]; then
  set +u
  # shellcheck disable=SC1090
  source "$PYTHON_SOURCE"
  set -u
fi
if [ -x "$PYROOT/bin/python3" ]; then
  export PATH="$PYROOT/bin:$PATH"
  export LD_LIBRARY_PATH="$PYROOT/lib:${LD_LIBRARY_PATH:-}"
  for runtime_lib in \
    "$OPENSSL_ROOT/lib64" "$OPENSSL_ROOT/lib" \
    "$LIBFFI_ROOT/lib64" "$LIBFFI_ROOT/lib"; do
    test ! -d "$runtime_lib" || export LD_LIBRARY_PATH="$runtime_lib:$LD_LIBRARY_PATH"
  done
else
  module load python/3.8.5 || true
fi
