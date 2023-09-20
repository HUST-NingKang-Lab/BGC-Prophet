#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import logging
import os
import argparse
import sys


def main(argv=None):
    print(""" ____   ____  ____      ____                  _          _
| __ ) / ___|/ ___|    |  _ \ _ __ ___  _ __ | |__   ___| |_
|  _ \| |  _| |   _____| |_) | '__/ _ \| '_ \| '_ \ / _ \ __|
| |_) | |_| | |__|_____|  __/| | | (_) | |_) | | | |  __/ |_
|____/ \____|\____|    |_|   |_|  \___/| .__/|_| |_|\___|\__|
                                       |_|
          """)
    parser = argparse.ArgumentParser(prog="BGC-Prophet", description="BGC-Prophet: A tool for BGC mining")
    subparsers = parser.add_subparsers(title="subcommands", 
                                       metavar="COMMAND",
                                       dest='cmd', 
                                       help="Use: bgc_prophet COMMAND -h for more information"
                                       )
    subcommnds = [
        'extract',
        'organize',
        'split',
        'predict',
        'output',
        'classify',
        'pipline',
    ]

    for subcommand in subcommnds:
        cmd_instance = subcommand()
        subparser = subparsers.add_parser(cmd_instance.name, description=cmd_instance.description)
        subparser.set_defaults(handle=cmd_instance.handle)
        cmd_instance.add_arguments(subparser)
    
    args = parser.parse_args()
    if hasattr(args, 'handle'):
        args.handle(args)
    else:
        args.print_help()


if __name__=='__main__':
    main()
    