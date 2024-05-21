import scrapy
from scrapy_playwright.page import PageCoroutine
from urllib.parse import urljoin

# Import the list of URLs from the scripts/urls.py file
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '/home/moraa/Documents/10_academy/week4/scalable_data_warehouse/scripts/urls.py', 'scripts'))
from scripts.urls import urls

class RecursiveSpider(scrapy.Spider):
    name = 'recursivespider'

    def start_requests(self):
        for url in urls:
            yield scrapy.Request(
                url,
                meta={
                    "playwright": True,
                    "playwright_page_coroutines": [
                        PageCoroutine("wait_for_selector", "a")  # Adjust this selector as needed
                    ],
                },
                callback=self.parse
            )

    async def parse(self, response):
        # Extract data from the current page
        for item in response.css('div.item'):
            yield {
                'title': item.css('h2.title::text').get(),
                'description': item.css('div.description::text').get(),
            }

        # Follow links to next pages
        for link in response.css('a::attr(href)').getall():
            next_page = urljoin(response.url, link)
            yield scrapy.Request(
                next_page,
                meta={
                    "playwright": True,
                    "playwright_page_coroutines": [
                        PageCoroutine("wait_for_selector", "a")  # Adjust this selector as needed
                    ],
                },
                callback=self.parse
            )

