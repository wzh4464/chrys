# Copyright (c) 2026 Chrys. All rights reserved.

"""Make ``python -m chrys`` the same program as the ``chrys`` console script.

A built-in profile's ``command: chrys`` resolves, from a source checkout, to
this interpreter running the package -- and without this module that launch
died with "No module named chrys.__main__" before it could write a single
protocol frame. That is how the PACT campaign failed on the first live run
that ever reached it.
"""

from chrys.app.cli.app import main

if __name__ == "__main__":
    raise SystemExit(main())
