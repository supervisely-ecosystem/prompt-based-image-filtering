#!/bin/bash
set -e


# Determine script and project root directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/..")"

# Download Dockerfile template to script directory
curl -sSL "https://raw.githubusercontent.com/supervisely-ecosystem/workflows/self-contained/docker/hardened/Dockerfile.tmpl" -o "$SCRIPT_DIR/Dockerfile.template"

# Inject model weights download before WORKDIR /app
sed -i 's|^WORKDIR /app$|COPY docker/download_weights.py /tmp/download_weights.py\nRUN /opt/venv/bin/python /tmp/download_weights.py \&\& rm /tmp/download_weights.py\n\nWORKDIR /app|' "$SCRIPT_DIR/Dockerfile.template"


# Set SDK version
SDK_VER=6.73.556
APP_NAME="prompt-based-image-filtering"

# Derive ENTRYPOINT_CMD from config.json
CONFIG_FILE="$PROJECT_ROOT/config.json"
ENTRYPOINT_CMD="$(python3 -c "
import json, sys
with open('$CONFIG_FILE') as f:
    cfg = json.load(f)
ep = cfg.get('entrypoint')
if ep:
    print(ep)
else:
    ms = cfg.get('main_script')
    if ms:
        print('python ' + ms)
    else:
        print('ERROR: neither entrypoint nor main_script found in config.json', file=sys.stderr)
        sys.exit(1)
")"

# Always use project root as build context, and correct relative paths
docker build \
	--no-cache \
	-f "$SCRIPT_DIR/Dockerfile.template" \
	--build-arg tag_ref_name=$SDK_VER \
	--build-arg RUNTIME_BASE_IMAGE=base-py-cuda-hardened \
	--build-arg REQUIREMENTS_FILE=dev_requirements.txt \
	--build-arg ENTRYPOINT_CMD="$ENTRYPOINT_CMD" \
	--label python_sdk_version=$SDK_VER \
	-t supervisely/$APP_NAME:test \
	"$PROJECT_ROOT"

# Ask for confirmation before pushing
read -p "Push image supervisely/$APP_NAME:test to registry? [y/N]: " confirm
if [[ "$confirm" =~ ^[Yy]$ ]]; then
	docker push supervisely/$APP_NAME:test
else
	echo "Push cancelled."
fi
