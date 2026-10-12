#!/usr/bin/env bash
# Regression checks for neuralforge_eval_accuracy dataset validation.
# Invoked as a CTest script; expects BUILD_DIR and optionally MODEL path.
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build}"
EVAL_BIN="${BUILD_DIR}/bin/neuralforge_eval_accuracy"
MODEL="${MODEL:-models/toy_classifier.nfmodel}"
TMP="${TMPDIR:-/tmp}/nf_eval_dataset_$$"
mkdir -p "$TMP"
trap 'rm -rf "$TMP"' EXIT

if [[ ! -x "$EVAL_BIN" ]]; then
  echo "Missing $EVAL_BIN" >&2
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "Missing $MODEL" >&2
  exit 1
fi

pass=0
fail=0

expect_fail() {
  local name="$1" file="$2"
  if "$EVAL_BIN" "$MODEL" "$file" >/dev/null 2>&1; then
    echo "FAIL: $name (expected non-zero exit)"
    fail=$((fail + 1))
  else
    echo "PASS: $name"
    pass=$((pass + 1))
  fi
}

# Valid tiny dataset matching toy model (2 features, labels in {0,1,2})
cat >"$TMP/ok.txt" <<EOF
2 2
-2.0 -1.5 0
0.0 2.0 2
EOF
if "$EVAL_BIN" "$MODEL" "$TMP/ok.txt" | grep -q 'predictions='; then
  echo "PASS: valid dataset"
  pass=$((pass + 1))
else
  echo "FAIL: valid dataset"
  fail=$((fail + 1))
fi

# Label out of range
cat >"$TMP/bad_label.txt" <<EOF
1 2
0.0 0.0 99
EOF
expect_fail "label out of range" "$TMP/bad_label.txt"

# Feature width mismatch
cat >"$TMP/bad_width.txt" <<EOF
1 3
0.0 0.0 0.0 0
EOF
expect_fail "feature width mismatch" "$TMP/bad_width.txt"

# Non-finite feature
cat >"$TMP/nan.txt" <<EOF
1 2
nan 0.0 0
EOF
expect_fail "non-finite feature" "$TMP/nan.txt"

# Zero samples rejected
cat >"$TMP/zero.txt" <<EOF
0 2
EOF
expect_fail "zero samples" "$TMP/zero.txt"

echo "eval_dataset_tests pass=$pass fail=$fail"
[[ "$fail" -eq 0 ]]
