#!/bin/bash
cd ~/code/harvesters_pg || exit 1
export GENICAM_GENTL64_PATH=/usr/local/activesilicon/lib64/
export QT_QPA_PLATFORM=offscreen
exec uv run python stress_ui.py
