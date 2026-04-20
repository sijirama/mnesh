#!/bin/bash
if [[ "$VIRTUAL_ENV" != "" ]]; then
    deactivate
    echo "env ended"
else
    echo "no active env"
fi
