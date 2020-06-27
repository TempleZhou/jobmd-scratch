# -*- coding: utf-8 -*-
import sys

from scrapy.cmdline import execute

if __name__ == '__main__':
    sys.argv = [sys.argv[0], "crawl", "jobmd"]
    sys.exit(execute())
