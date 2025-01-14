import asyncio
from crawl4ai import AsyncWebCrawler

class Crawler:
    """
    A class that provides two static blocking methods:
      1) get_links(websiteURL) -> set of internal URLs
      2) crawl_links(websiteURL) -> dict of link_name -> crawled markdown

    Implementation details:
    - Each static method uses `asyncio.run()` to execute the underlying
      async calls, so the caller does not need to worry about async/await.
    - Since these are "private" methods (by naming convention), you might
      rename them without underscores if they're meant to be part of the
      public API. 
    """

    @staticmethod
    def get_links(websiteURL: str) -> set[str]:
        """
        Synchronously retrieve all internal links from the given websiteURL.
        
        :param websiteURL: The initial website URL to crawl.
        :return: A set of internal links (strings).
        
        This method blocks until the crawling is complete.
        """
        return asyncio.run(Crawler._async_get_links(websiteURL))

    @staticmethod
    async def _async_get_links(websiteURL: str) -> set[str]:
        """
        The internal async version of link retrieval. Called by _get_links.
        """
        async with AsyncWebCrawler(verbose=True) as crawler:
            result = await crawler.arun(
                url=websiteURL,
                exclude_external_links=True,
                remove_overlay_elements=True
            )
        # Build a set of unique internal links (remove trailing slash)
        return {item['href'].rstrip("/") for item in result.links["internal"]}

    @staticmethod
    def crawl_links(websiteURL: str) -> dict:
        """
        Synchronously crawl all internal links for the given URL and retrieve
        their Markdown content.
        
        :param websiteURL: The initial website URL to crawl.
        :return: A dictionary: { short_link: markdown_content }
        
        This method also blocks, because it calls asyncio.run().
        """
        return asyncio.run(Crawler._async_crawl_links(websiteURL))

    @staticmethod
    async def _async_crawl_links(websiteURL: str) -> dict:
        """
        The internal async version of the crawling workflow:
          1) Gather all links from websiteURL (async).
          2) For each link, fetch the markdown content.
          3) Return a dict of short_link -> markdown content.
        """
        # First, gather the link set (reuse the same logic from _async_get_links)
        link_set = await Crawler._async_get_links(websiteURL)
        
        clean_data_dict = {}
        async with AsyncWebCrawler() as crawler:
            for link in link_set:
                result = await crawler.arun(
                    url=link,
                    exclude_external_links=True,
                    remove_overlay_elements=True,
                    process_iframes=False,
                    excluded_tags=['form', 'header', 'footer', 'nav'],
                    magic=True,
                    exclude_external_images=True
                )
                # Create a readable key name from the link
                short_link = (link
                              .replace("https://www.", "")
                              .replace(".com", "")
                              .replace("/", "_"))
                
                clean_data_dict[short_link] = result.markdown
        return clean_data_dict
