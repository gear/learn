"""Quick tutorial for logging in Python.
"""

# Coding: utf-8
# Filename: how_to_log.py
# Created: 2016-07-12 v0.0
# Description: Tutorial for logging in Python.
## v0.0: File created.

import logging

# Create a log file with log level of 'INFO'
# which means it only saves logging with level
# higher or equal 'INFO'.
# 5 levels are: DEBUG, INFO, WARNING, ERROR, CRITICAL
#
# Option: filemode="w" - Overwrite the whole file,
# else, it will append the new log to the file.
logging.basicConfig(filename="learn.log", level=logging.INFO)

logging.debug("This is a message that won't appear because of"
              " its low level")
logging.info("I inform you about this genius code of yours.")
logging.error("Oh shit, deadline in 10 min and you have"
              " me. Good luck bro.")

# At this point, content of file 'learn.log' is:
"""
INFO:root:I inform you about this genius code of yours.
ERROR:root:Oh shit, deadline in 10 min and you have me. Good luck bro.
"""




