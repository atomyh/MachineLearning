# -*- coding: utf-8 -*-
import scrapy
import json
from douyu.items import DouyuItem

class DouyuSpider(scrapy.Spider):
    name = "Douyu"
    allowed_domains = ["douyucdn.cn"]
    baseUrl = "http://capi.douyucdn.cn/api/v1/getVerticalRoom?limit=20&offset="
    offset = 0
    start_urls = [baseUrl + str(offset)]

    def parse(self, response):
       data_list = json.loads(response.body.decode('utf-8'))['data']
       # print(len(data_list))
       if len(data_list) == 0:
           return
       for data in data_list:
           # print(data)
           item = DouyuItem()
           item['nickname'] = data['nickname']
           item['vertical_src'] = data['vertical_src']
           yield item

       self.offset += 20
       yield scrapy.Request(self.baseUrl+str(self.offset), callback=self.parse)



