# -*- coding: utf-8 -*-
import scrapy
from ITcast.items import ItcastItem

class ItcastSpider(scrapy.Spider):
    name = "itcast"
    allowed_domains = ["itcast.cn"]
    start_urls = ["http://www.itcast.cn/channel/teacher.shtml"]

    def parse(self, response):
        node_list = response.xpath("//div[@class='li_txt']")
        # 用来存储所有的item字段
        for node in node_list:
            #创建item字段对象，用来存储信息
            item = ItcastItem()
            #.extract（）将xpath对象转换为unicode字符串
            name = node.xpath("./h3/text()").extract()
            title = node.xpath("./h4/text()").extract()
            info = node.xpath("./p/text()").extract()

            item['name'] = name[0]
            item['title'] = title[0]
            item['info'] = info[0]
            #返回提取到的每个item数据给管道pipeline文件处理，同时还会继续回来执行后面的代码即for循环
            yield item
        print(response.body)
        # pass
