# -*- coding: utf-8 -*-
import scrapy
from agencyCollectIps.items import AgencycollectipsItem
from scrapy.spiders import crawl

class AgencySpider(crawl.CrawlSpider):
    name = "agency"
    allowed_domains = ["www.kuaidaili.com"]
    base_url = "http://www.kuaidaili.com/free/inha/"
    offset = 1
    end = "/"
    start_urls = [base_url + str(offset) + end]

    def parse_start_url(self, response):
        trs = response.xpath("//tr")
        #print(trs)
        for tr in trs[1:]:
            item = AgencycollectipsItem()
            #print(tr)
            item['ip'] = tr.xpath("./td[1]/text()").extract()[0]
            print(item['ip'])
            item['port'] = tr.xpath("./td[2]/text()").extract()[0]
            print(item['port'])
            item['degrees'] = tr.xpath("./td[3]/text()").extract()[0]
            print(item['degrees'])
            item['type'] = tr.xpath("./td[4]/text()").extract()[0]
            print(item['type'])
            item['address'] = tr.xpath("./td[5]/text()").extract()[0]
            print(item['address'])
            item['speed'] = tr.xpath("./td[6]/text()").extract()[0]
            print(item['speed'])
            item['time'] = tr.xpath("./td[7]/text()").extract()[0]
            print(item['time'])
            yield item
        if self.offset < 100:
            self.offset += 1
            url = self.base_url + str(self.offset) + self.end
            yield scrapy.Request(url, callback=self.parse)

