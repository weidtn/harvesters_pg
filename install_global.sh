#!/bin/bash
set -e
echo "Installing harvesters_pg system-wide..."
    
#    Create directory with proper permissions
sudo mkdir -p /opt/uv/venv
sudo chown root:root /opt/uv/venv
sudo chmod 755 /opt/uv/venv
    
# Install as root
sudo UV_TOOL_BIN_DIR=/usr/local/bin UV_TOOL_DIR=/opt/uv/venv uv tool install .
    
echo "✓ Installation complete!"
echo "Run 'uv tool run harvesters_pg' from terminal or use desktop shortcut"

