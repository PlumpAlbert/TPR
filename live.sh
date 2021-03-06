#!/bin/sh

xelatex $1
(ls "$1" | entr -pc xelatex $1) &
CONVERTER="$!"
BASE_FILE_NAME=${1%.*}
mupdf "$BASE_FILE_NAME.pdf" &
MUPDF="$!"
ls "$BASE_FILE_NAME.pdf" | entr -c -p kill -SIGHUP "$MUPDF"

kill -SIGTERM "$CONVERTER"
kill -SIGTERM "$MUPDF"
