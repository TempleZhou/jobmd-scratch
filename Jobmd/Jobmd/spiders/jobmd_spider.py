# -*- coding: utf-8 -*-//
import scrapy
import os

from scrapy.loader import ItemLoader

from Jobmd.items import JobDescItemLoader, JobDescItem


class JobmdSpider(scrapy.Spider):
    name = 'jobmd'
    allowed_domains = ['jobmd.cn']
    start_urls = ['http://jobmd.cn/']

    def parse(self, response):
        # 获取类别列表
        menu_list_items = response.css("div.menu-list_item")
        menu_list_names = menu_list_items.css("h2.menu-list_item-sort").xpath(".//a/text()").getall()

        self.menu_process(menu_list_names)

        name_idx = 0
        content_idx = 0
        self.content_list = []

        vboxes = menu_list_items.css("div.menu-list_item-vbox")
        for vbox in vboxes:
            menu_content_idx = 0
            menu_contents = vbox.css("dl.menu-content_list").xpath(".//dt/strong").css("*::text").getall();

            self.menu_process(menu_contents)

            for menu_content in menu_contents:
                menu_sub_content_selectors = vbox.css("dl.menu-content_list")[menu_content_idx].xpath(".//dd/a")
                menu_content_idx = menu_content_idx + 1
                self.logger.info("menu_sub_content_selectors len is %d" % (len(menu_sub_content_selectors)))
                for sub_content_selector in menu_sub_content_selectors:
                    menu_sub_content = sub_content_selector.css("*::text").get()
                    menu_sub_content_link = sub_content_selector.css("a::attr(href)").get()
                    if menu_sub_content is None or menu_sub_content == "":
                        continue

                    try:
                        dir_name = os.environ.get("DATA_DIR") + "/" +  menu_list_names[name_idx] + "/" + menu_content + "/" + menu_sub_content;
                        self.logger.info(menu_sub_content_link);
                        self.logger.info(dir_name);

                        self.content_list.append(dir_name);
                        os.makedirs(dir_name)
                    except:
                        pass

                    yield response.follow(menu_sub_content_link, cb_kwargs=dict(content_idx=content_idx),
                                          callback=self.parse_job_list)
                    content_idx = content_idx + 1
            name_idx = name_idx + 1
        pass

    # 去除斜杠和空格等无用字符
    def menu_process(self, menu_list):
        for i in range(len(menu_list)):
            menu_list[i] = menu_list[i].replace("/", "").strip().replace("  ", "")

    def parse_job_list(self, response, content_idx):
        job_links = response.css("a.rm-name::attr(href)").getall()

        for job_link in job_links:
            yield response.follow(job_link, cb_kwargs=dict(content_idx=content_idx), callback=self.parse_job_desc)

        next_page = response.css("li.pager-next a::attr(href)").get()
        if next_page is None or len(next_page) == 0:
            return

        yield response.follow(next_page, cb_kwargs=dict(content_idx=content_idx), callback=self.parse_job_list)

    def parse_job_desc(self, response, content_idx):
        self.file_url = self.content_list[content_idx]
        l = JobDescItemLoader(item=JobDescItem(), response=response)
        l.add_css('title', 'div.work-title_fixed-title h2::text')
        l.add_css('location', 'div.box-info_base span::text')
        l.add_css('salary', 'div.box-info_base span::text')
        l.add_css('workingExp', 'div.box-info_base span::text')
        l.add_css('education', 'div.box-info_base span::text')
        l.add_css('workingType', 'div.box-info_base span::text')
        l.add_css('highLight', 'dl.work-welfare span')
        try:
            jobDescSelector = response.css("div.work-content_box-detail dl.work-group")[2]
            l.add_value('jobDesc', "".join(jobDescSelector.css("dd::text").getall()))
        except IndexError:
            self.logger.warn("职位描述缺失！")
            l.add_value('jobDesc', "")

        l.add_css('detailAddress', 'div.work-location_content p::text')
        l.add_css('techTitle', 'ul.work-require li p::text')
        l.add_css('major', 'ul.work-require li p::text')
        l.add_css('englishLevel', 'ul.work-require li p::text')

        self.logger.info(f'正在抓取 {response.request.url}')
        return l.load_item()
