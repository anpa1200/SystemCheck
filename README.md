## Usage
# Terminal (color)
python3 systemcheck.py

# Faster (skip deep scan)
python3 systemcheck.py --no-deep

# HTML (pretty report)
python3 systemcheck.py --html --html-file report.html

# JSON
python3 systemcheck.py --json > report.json


--html [--html-file PATH]  # HTML output
--json                      # JSON output
--deep/--no-deep            # deep scan toggle
--top N --max-mounts N --per-mount-timeout S
--no-color                  # TTY colors off
