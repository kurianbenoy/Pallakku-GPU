import re
import urllib.request
import modal


stub = modal.Stub(name="link-scraper")

playwright_image = modal.Image.debian_slim(python_version="3.10").run_commands(
    "apt-get install -y software-properties-common",
    "apt-add-repository non-free",
    "apt-add-repository contrib",
    "apt-get update",
    "pip install playwright==1.30.0",
    "playwright install-deps chromium",
    "playwright install chromium",
)

@stub.function(image=playwright_image)
async def get_links(url):
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(url)
        links = await page.eval_on_selector_all("a[href]", "elements => elements.map(element => element.href)")
        await browser.close()

    print("Links", links)
    return links


@stub.function(schedule=modal.Period(days=1))
def main():
    urls = ["http://modal.com", "http://github.com", "http://kurianbenoy.com"]
    for links in get_links.map(urls):
        for link in links:
            print(link)




