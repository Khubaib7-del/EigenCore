"""PyInstaller entry point for the Windows app (see EigenCore.spec)."""

import multiprocessing

from eigencore.desktop.launcher import cli

if __name__ == "__main__":
    multiprocessing.freeze_support()
    cli()
