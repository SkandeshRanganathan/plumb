import asyncio
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as pw:
        browser = await pw.webkit.launch(headless=True)
        context = await browser.new_context(user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15')
        page = await context.new_page()
        print("Navigating...")
        await page.goto('https://www.espncricinfo.com/series/1543999/game/1544002', wait_until='domcontentloaded')
        content = await page.content()
        print('__NEXT_DATA__ in content:', '__NEXT_DATA__' in content)
        if '__NEXT_DATA__' not in content:
            with open('dump.html', 'w', encoding='utf-8') as f:
                f.write(content)
            print("Dumped to dump.html")
        await browser.close()

asyncio.run(run())
