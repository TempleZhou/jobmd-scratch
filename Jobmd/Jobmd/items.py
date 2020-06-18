# -*- coding: utf-8 -*-
# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy
from scrapy.loader import ItemLoader
from w3lib.html import remove_tags
from scrapy.loader.processors import MapCompose


class JobDescItem(scrapy.Item):
    # define the fields for your item here like:
    title = scrapy.Field()
    location = scrapy.Field()
    salary = scrapy.Field()
    workingExp = scrapy.Field()
    education = scrapy.Field()
    workingType = scrapy.Field()
    highLight = scrapy.Field()
    jobDesc = scrapy.Field()
    detailAddress = scrapy.Field()
    techTitle = scrapy.Field()
    major = scrapy.Field()
    englishLevel = scrapy.Field()
    pass


def salary_out_process(data2):
    return data2[0].encode('utf-8')


def workingExp_out_process(data2):
    return data2[2].encode('utf-8')


def education_out_process(data2):
    return data2[3].encode('utf-8')


def workingType_out_process(data2):
    return data2[4].encode('utf-8')


def location_out_process(data2):
    return data2[1].encode('utf-8')


def major_out_process(data2):
    return data2[0].encode('utf-8')


def techTitle_out_process(data2):
    print("techTitle = ==============")
    return data2[3].encode('utf-8')


def englishLevel_out_process(data2):
    return data2[4].encode('utf-8')


def highLight_out_process(data2):
    return " ".join(data2).encode("utf-8")


def detailAddress_out_process(data2):
    return data2[0].replace("\n", "").replace(" ", "").replace("    ", "").encode('utf-8')


def jobDesc_out_process(data2):
    return data2[0].replace(" ", "").replace("\n", "").encode('utf-8')


def title_out_process(data2):
    return data2[0].encode('utf-8')


class JobDescItemLoader(ItemLoader):
    salary_out = salary_out_process
    workingExp_out = workingExp_out_process
    education_out = education_out_process
    workingType_out = workingType_out_process
    location_out = location_out_process
    major_out = major_out_process
    englishLevel_out = englishLevel_out_process
    highLight_in = MapCompose(remove_tags)
    highLight_out = highLight_out_process
    detailAddress_out = detailAddress_out_process
    jobDesc_out = jobDesc_out_process
    title_out = title_out_process
    techTitle_out = techTitle_out_process
