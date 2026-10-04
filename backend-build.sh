#!/usr/bin/env bash
set -euo pipefail
backend_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$backend_root/tools/build_release.py" "$@"
