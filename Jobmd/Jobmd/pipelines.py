# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html

import json
import os


class JobmdPipeline(object):
    def process_item(self, item, spider):
        with open(spider.file_url + "/job_desc.json", "a+") as f:
            print(json.dumps(dict(item)))
            f.write(json.dumps(dict(item), ensure_ascii=False))
            f.write("\n")
        return item
