#!/usr/bin/env bash
# Usage: run-test.sh <llc> <plugin.so> <input.mir> <expected.ll> <FileCheck>
set -euo pipefail
LLC=$1
PLUGIN=$2
MIR=$3
LL=$4
FILECHECK=$5
"$LLC" -load "$PLUGIN" -run-pass=ptx-ai "$MIR" -o /dev/null 2>&1 | "$FILECHECK" "$LL"
