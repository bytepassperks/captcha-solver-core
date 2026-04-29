#!/bin/bash
# Start Xvfb in background for Playwright headed mode
Xvfb :99 -screen 0 1280x800x24 -nolisten tcp &
export DISPLAY=:99

# Wait for Xvfb to be ready
sleep 2

# Start dbus if available (suppresses Chromium warnings)
if command -v dbus-daemon &> /dev/null; then
    mkdir -p /run/dbus
    dbus-daemon --system --nofork &
    sleep 1
fi

# Start uvicorn
exec uvicorn api.server:app --host 0.0.0.0 --port 8000
