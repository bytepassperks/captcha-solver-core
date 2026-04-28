#!/bin/bash
# Start Xvfb in background for Playwright headed mode
Xvfb :99 -screen 0 1280x800x24 -nolisten tcp &
export DISPLAY=:99

# Start uvicorn
exec uvicorn api.server:app --host 0.0.0.0 --port 8000
