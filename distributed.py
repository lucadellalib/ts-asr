"""Hotfix for SpeechBrain distributed execution.

Authors
* Luca Della Libera 2023
"""

# Adapted from:
# https://github.com/speechbrain/speechbrain/blob/v0.5.15/speechbrain/utils/distributed.py

import os


__all__ = []


def if_main_process():
    """Checks if the current process is the main process and authorized to run
    I/O commands. In DDP mode, the main process is the one with RANK == 0.
    In standard mode, the process will not have `RANK` Unix var and will be
    authorized to run the I/O commands.

    """
    if "LOCAL_RANK" in os.environ:
        if os.environ["LOCAL_RANK"] == "":
            return False
        else:
            if int(os.environ["LOCAL_RANK"]) == 0:
                return True
            return False
    return True
