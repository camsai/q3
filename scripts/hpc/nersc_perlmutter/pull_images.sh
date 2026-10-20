#!/usr/bin/env bash
# Pull the Q3 container images on a Perlmutter login node and check they will be
# visible from the compute nodes.
#
#   GHCR_USER=<github-user> ./scripts/hpc/nersc_perlmutter/pull_images.sh
#   TAGS="latest-cuda" GHCR_USER=<github-user> ./scripts/hpc/nersc_perlmutter/pull_images.sh
#
# While the package is private, this needs a credential. Put the token alone in
# a 600-mode file and point TOKEN_FILE at it, or export GHCR_TOKEN. It is piped
# via --password-stdin so it never lands in argv -- login nodes are shared and
# arguments are visible in `ps`.
#
# Use a CLASSIC personal access token with the `read:packages` scope. A
# fine-grained token authenticates and is then refused for package reads, which
# looks like a bad password but is not: you get "Login Succeeded!" followed by
# "denied" on the pull. See docs/podman_hpc_and_shifter.md.

set -euo pipefail

# Dated tags are reproducible; `latest*` move. Override TAGS to pin a build.
TAGS="${TAGS:-latest latest-mpi latest-cuda}"
REGISTRY="${REGISTRY:-ghcr.io}"
REPOSITORY="${REPOSITORY:-$REGISTRY/camsai/q3}"
GHCR_USER="${GHCR_USER:-}"
TOKEN_FILE="${TOKEN_FILE:-$HOME/.ghcr-token}"

# `podman-hpc images` prints REPOSITORY and TAG as separate columns, so grepping
# for "repo:tag" never matches. Compare the fields instead.
image_rows() {
  local repo="${1%:*}" tag="${1##*:}"
  podman-hpc images 2>/dev/null | awk -v r="$repo" -v t="$tag" '$1==r && $2==t'
}

echo "==> host: $(hostname)  (must be a login node: not a DTN, not a compute node)"
echo "==> tags: $TAGS"

if [ -n "${GHCR_TOKEN:-}" ]; then
  source_description="GHCR_TOKEN environment variable"
  read_token() { printf '%s' "$GHCR_TOKEN"; }
elif [ -r "$TOKEN_FILE" ]; then
  source_description="$TOKEN_FILE"
  read_token() { tr -d '\r\n' < "$TOKEN_FILE"; }
else
  read_token() { :; }
  source_description=""
fi

if [ -n "$source_description" ]; then
  [ -n "$GHCR_USER" ] || { echo "ERROR: set GHCR_USER alongside the token." >&2; exit 1; }
  echo "==> logging in to $REGISTRY as $GHCR_USER (credential from $source_description)"
  read_token | podman-hpc login "$REGISTRY" --username "$GHCR_USER" --password-stdin
else
  echo "==> no credential given; assuming the repository is public"
fi

for tag in $TAGS; do
  image="$REPOSITORY:$tag"
  echo
  echo "======================== $tag ========================"
  # `pull` writes both stores: the login node's read-write one and the squashed
  # copy on the parallel filesystem that compute nodes read.
  podman-hpc pull "$image" || { echo "    PULL FAILED for $tag"; continue; }
  podman-hpc migrate "$image" 2>/dev/null || echo "    (already migrated by pull)"

  count=$(image_rows "$image" | wc -l)
  if [ "$count" -lt 2 ]; then
    echo "    WARNING: $count store row(s), expected 2. Without the squashed copy"
    echo "             compute nodes will not see this image."
  else
    echo "    $count store rows: node-local and squashed copies both present."
  fi

  podman-hpc run --rm "$image" sh -lc \
    'printf "    version: %s\n    binaries: %s\n" "$(q3 --version 2>&1 | head -1)" "$(ls /usr/local/bin | tr "\n" " ")"' \
    2>/dev/null || echo "    (could not query the image)"
done

cat <<'NOTE'

Reminder: every image also carries the serial `q3`, and it comes first on PATH.
Name the parallel executable explicitly or you get a silent serial CPU run:

    latest-mpi   -> q3_mpi        (srun -n N, podman-hpc run --mpi)
    latest-cuda  -> q3_gpu_cuda   (podman-hpc run --gpu)

The read-write store is local to the login node you pulled on, so `podman-hpc
images` can look empty from another one even though batch jobs still work. Jobs
read the squashed copy on scratch. Re-run this if a login-node command cannot
see the image; note that scratch purges eventually take the squashed copy too.
NOTE
