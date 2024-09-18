# Make it possible to enable test coverage reporting for Python
# code run in children processes.
# http://coverage.readthedocs.io/en/latest/subprocess.html

import os.path as op
from site import getsitepackages

FILE_CONTENT = """\
import coverage; coverage.process_startup()
"""

filename = op.join(getsitepackages()[0], "coverage_subprocess.pth")
with open(filename, mode="w") as f:
    f.write(FILE_CONTENT)

print(f"Installed subprocess coverage support: {filename}")
