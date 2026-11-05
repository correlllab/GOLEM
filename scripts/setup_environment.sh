#!/bin/bash
set -euo pipefail
HERE="$(/usr/bin/dirname -- "$(/usr/bin/readlink -f -- "${BASH_SOURCE[0]}")")"
if [[ $# == 0 ]]; then set -- --help; fi
for arg in "$@"; do
    if [[ $arg == --help || $arg == -h ]]; then
        exec /usr/bin/env -i HOME="$HOME" PATH=/usr/bin:/bin PYTHONNOUSERSITE=1 \
            PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 "$HERE/environment/setup_environment.py" "$@"
    fi
done
source "$HERE/environment/clean_shell.sh"
golem_clean_python setup "$HERE/environment/setup_environment.py" "$@"
