#!/usr/bin/env bash
cd "$(dirname "$0")"
[[ ! -f h-manifest.conf ]] && echo "h-manifest.conf missing" && exit 1
source h-manifest.conf
[[ ! -f $CUSTOM_CONFIG_FILENAME ]] && echo "config $CUSTOM_CONFIG_FILENAME missing (run h-config.sh)" && exit 1

exec ./supr-meow-tsc $(< "$CUSTOM_CONFIG_FILENAME") 2>&1 | tee "${CUSTOM_LOG_BASENAME}.log"
