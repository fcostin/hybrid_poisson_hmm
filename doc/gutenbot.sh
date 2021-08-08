#! /bin/bash

# PURPOSE:
#
# run pdflatex inside a docker container to avoid
# caring about tex package management
#
# USAGE:
#
# Ensure the docker daemon is running.
#
# Ensure your working directory contains this script
# and a directory named src/
# The src directory will be bind-mounted into the
# container for build input and build output
# respectively).
#
# ./gutenbot.sh pdflatex \
#	-interaction nonstopmode \
#	somefile.tex
#
# where somefile.tex lives inside src/

set -euo pipefail

docker build -t gutenbot:dev -f gutenbot.Dockerfile .

ENTRYPOINT=$1

shift 1 # pop $1 from $@; we defer tail of $@ to entrypoint

docker run --rm \
  --name gutenbot \
  --workdir /work/src \
  --mount type=bind,source="$(pwd)"/src,target=/work/src \
  --entrypoint "$ENTRYPOINT" \
  gutenbot:dev \
  $@

