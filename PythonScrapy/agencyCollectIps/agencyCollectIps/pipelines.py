# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html
from agencyCollectIps import settings
import MySQLdb
from twisted.enterprise import adbapi
import MySQLdb.cursors
class AgencycollectipsPipeline(object):
    def __init__(self):
        self.dbpool = adbapi.ConnectionPool('MySQLdb',db='ippool',user='root',passwd='1112juan521*',cursorclass=MySQLdb.cursors.DictCursor,charset='utf8',use_unicode=True)

    def process_item(self, item, spider):
        query = self.dbpool.runInteraction(self._conditional_insert, item)
        query.addErrback(self._handle_error, item, spider)  # 调用异常处理方法
        return item

    def _conditional_insert(self, tx, item):
        sql = ("insert into proxy3(IP,PORT,DEGREES,TYPE,ADDRESS,SPEED,TIME) values(%s,%s,%s,%s,%s,%s,%s)")
        lis = (item['ip'], item['port'], item['degrees'], item['type'], item['address'], item['speed'], item['time'])
            # try:
            #     cur.execute(sql,lis)
            # except Exception as e:
            #     print("Insert error:",e)
            #     con.rollback()
            # else:
            #     con.commit()
            # cur.close()
            # con.close()
        tx.execute(sql,lis)
    def _handle_error(self, failue, item, spider):
        print(failue)
    # def process_item(self, item, spider):
    #     DBKWARGS = settings.DBKWARGS
    #     print(DBKWARGS)
    #     con = MySQLdb.connect(**DBKWARGS)
    #     cur = con.cursor()
    #     print("连接成功")
    #     sql = ("insert into proxy3(IP,PORT,DEGREES,TYPE,ADDRESS,SPEED,TIME) values(%s,%s,%s,%s,%s,%s,%s)")
    #     lis = (item['ip'], item['port'], item['degrees'], item['type'], item['address'], item['speed'], item['time'])
    #     # sql = ("insert into proxy1(IP,PORT) "
    #     #        "values(%s,%s)")
    #     # lis = (item['ip'], item['port'])
    #     try:
    #         cur.execute(sql, lis)
    #     except Exception as e:
    #         print("Insert error:", e)
    #         con.rollback()
    #     else:
    #         con.commit()
    #         print("提交成功")
    #     cur.close()
    #     con.close()
    #     yield item
