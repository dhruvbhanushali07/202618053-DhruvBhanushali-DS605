import scrapy


class BooksSpiderSpider(scrapy.Spider):
    name = "books"
    allowed_domains = ["books.toscrape.com"]
    start_urls = ["https://books.toscrape.com/catalogue/page-1.html"]

    page_count = 1
    max_pages = 5

    def parse(self, response):

        books = response.css("article.product_pod")

        for book in books:
            book_url = book.css("h3 a::attr(href)").get()

            yield response.follow(
                book_url,
                callback=self.parse_book
            )

        if self.page_count < self.max_pages:
            next_page = response.css("li.next a::attr(href)").get()

            if next_page:
                self.page_count += 1
                yield response.follow(next_page, callback=self.parse)

    def parse_book(self, response):

        yield {
            "title": response.css("div.product_main h1::text").get(),
            "category": response.css("ul.breadcrumb li:nth-child(3) a::text").get(),
            "price": response.css("p.price_color::text").get(),
            "rating": response.css("p.star-rating::attr(class)").get().split()[-1],
            "availability": " ".join(
                [i.strip() for i in response.css("p.instock.availability::text").getall() if i.strip()]
            ),
            "description": response.css("#product_description + p::text").get(default=""),
            "upc": response.xpath('//th[text()="UPC"]/following-sibling::td/text()').get(),
            "number_of_reviews": response.xpath('//th[text()="Number of reviews"]/following-sibling::td/text()').get(),
            "product_url": response.url,
        }