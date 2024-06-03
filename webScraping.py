import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from bs4 import BeautifulSoup
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import time

class UASpider(CrawlSpider):
    name = "ua_spider"
    allowed_domains = ["u.ae"]
    start_urls = ["https://u.ae/en/information-and-services"]
    rules = (
        Rule(LinkExtractor(allow=r'/en/information-and-services/.*'), callback='parse_item', follow=True, process_links="filter_links"),
    )

    def filter_links(self, links):
        # Limit the depth of the crawl to 1
        return links[:5]

    def parse_item(self, response):
        soup = BeautifulSoup(response.body, 'html.parser')
        data = {
            "url": response.url,
            "title": soup.title.string if soup.title else 'No Title',
            "content": soup.get_text(separator="\n"),
        }
        yield data

class PdfPipeline:
    def open_spider(self, spider):
        self.data_list = []
        self.start_time = time.time()

    def close_spider(self, spider):
        pdf_filename = "docsscraped_data.pdf"
        self.save_to_pdf(self.data_list, pdf_filename)
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        print(f"Scraping completed in {elapsed_time:.2f} seconds")

    def process_item(self, item, spider):
        self.data_list.append(item)
        return item

    def save_to_pdf(self, data_list, filename):
        c = canvas.Canvas(filename, pagesize=letter)
        width, height = letter
        textobject = c.beginText(40, height - 40)
        textobject.setFont("Helvetica", 10)
        textobject.setLeading(12)

        for data in data_list:
            textobject.textLine(f"URL: {data['url']}")
            textobject.textLine(f"Title: {data['title']}")
            textobject.textLine("Content:")

            content_lines = filter(None, data['content'].splitlines())  
            for line in content_lines:
                if textobject.getY() < 50:
                    c.drawText(textobject)
                    c.showPage()
                    textobject = c.beginText(40, height - 40)
                    textobject.setFont("Helvetica", 10)
                    textobject.setLeading(12)

                textobject.textLine(line)

            textobject.textLine("-" * 80)

            if textobject.getY() < 50:
                c.drawText(textobject)
                c.showPage()
                textobject = c.beginText(40, height - 40)
                textobject.setFont("Helvetica", 10)
                textobject.setLeading(12)

        c.drawText(textobject)
        c.save()

# Configure the pipeline
ITEM_PIPELINES = {
    '__main__.PdfPipeline': 300,
}

# Configure settings
settings = {
    'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'ROBOTSTXT_OBEY': False,
    'ITEM_PIPELINES': ITEM_PIPELINES,
}

# Run the spider
process = CrawlerProcess(settings)
process.crawl(UASpider)
start_time = time.time()
process.start()
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total time taken: {elapsed_time:.2f} seconds")
