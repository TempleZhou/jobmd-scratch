# -*- coding: utf-8 -*-
import os
import re
import sys
from scrapy.cmdline import execute
if __name__ == '__main__':
    os.environ['DATA_DIR'] = os.path.abspath(os.curdir) + "/JsonData"
    try:
        os.mkdir(os.environ.get("DATA_DIR"))
    except FileExistsError:
        pass
    print(f'数据目录为：{os.environ.get("DATA_DIR")}')
    sys.argv = [sys.argv[0], "crawl", "jobmd"]
    sys.exit(execute())
