# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html
import scrapy
from scrapy.pipelines.images import ImagesPipeline
import os
from douyu.settings import IMAGES_STORE as images_store

class DouyuPipeline(ImagesPipeline):

    def get_media_requests(self, item, info):
        imag_link = item['vertical_src']
        yield scrapy.Request(imag_link)

    def item_completed(self, results, item, info):
        image_path = [x["path"] for ok, x in results if ok]
        # print(image_path)
        os.rename(images_store+image_path[0], images_store+item["nickname"]+".jpg")
        return item
