import asyncio
import os
from crawl4ai import AsyncWebCrawler
from openai import OpenAI

class CrawledResult:
    """
    Holds crawled data and provides chainable post-processing.
    """

    def __init__(self, data_dict: dict):
        """
        :param data_dict: { short_link: markdown_content }
        """
        self.data_dict = data_dict

    def check_with_ai(self, model: str, api_key: str) -> "CrawledResult":
        """
        Filters out “useless” entries in-place via OpenAI.
        
        :param model: e.g. "gpt-3.5"
        :param api_key: OpenAI API key
        :return: self
        """
        self.data_dict = Crawler.check_with_ai(self.data_dict, model, api_key)
        return self

    def to_folder(self, folder_path: str) -> "CrawledResult":
        """
        Writes each entry in data_dict to a .txt file in folder_path.
        
        :param folder_path: Destination folder
        :return: self
        """
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        for key, value in self.data_dict.items():
            with open(os.path.join(folder_path, f"{key}.txt"), "w", encoding="utf-8") as f:
                f.write(value)
        return self


class Crawler:
    """
    Provides crawling and AI-based filtering.
    """

    @staticmethod
    def get_links(websiteURL: str) -> set[str]:
        """
        Returns internal links for websiteURL.
        
        :param websiteURL: e.g. "https://example.com"
        :return: set of URLs
        """
        return asyncio.run(Crawler._async_get_links(websiteURL))

    @staticmethod
    async def _async_get_links(websiteURL: str) -> set[str]:
        """
        Internal async link retrieval.
        """
        async with AsyncWebCrawler(verbose=True) as crawler:
            result = await crawler.arun(
                url=websiteURL,
                exclude_external_links=True,
                remove_overlay_elements=True
            )
        return {item['href'].rstrip("/") for item in result.links["internal"]}

    @staticmethod
    def crawl_links(websiteURL: str) -> CrawledResult:
        """
        Gathers internal links, fetches Markdown, returns CrawledResult.
        
        :param websiteURL: e.g. "https://example.com"
        :return: CrawledResult
        """
        data_dict = asyncio.run(Crawler._async_crawl_links(websiteURL))
        return CrawledResult(data_dict)

    @staticmethod
    async def _async_crawl_links(websiteURL: str) -> dict:
        """
        Internal async workflow to fetch Markdown for each link.
        """
        link_set = await Crawler._async_get_links(websiteURL)
        data_dict = {}
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
                short_link = (
                    link
                    .replace("https://www.", "")
                    .replace(".com", "")
                    .replace("https:", "")
                    .replace("/", "_")
                )
                short_link = short_link[:50]
                data_dict[short_link] = result.markdown
        return data_dict

    @staticmethod
    def check_with_ai(clean_data_dict: dict, model: str, api_key: str) -> dict:
        """
        Calls OpenAI to classify each entry, removing 'useless' content.
        
        :param clean_data_dict: { short_link: markdown_content }
        :param model: e.g. "gpt-3.5"
        :param api_key: OpenAI API key
        :return: Filtered dict
        """
        keys_to_remove = []
        client = OpenAI(api_key=api_key)

        for key, value in clean_data_dict.items():
            system_message = """
                System Prompt
                You are an assistant that classifies text as either "useless" or "useful" for a knowledge base.

                A file is "useless" if the majority or essence of its content falls into one or more of the following categories:
                1. It’s a placeholder or error page, such as:
                  - “# This page isn’t available,” with phrases like “Please check that the URL entered is correct,” 
                    “try loading the page again,” or “please contact the website owner.”
                  - A JSON error, for example:
                      {
                        "status":"error",
                        "message":"Cannot parse path tel: 18005462070",
                        "correlationId":"..."
                      }
                  - A page saying only “Loading,” “Close dialog,” or similarly trivial placeholders.
                  - Essentially empty, containing only whitespace, \\n, or a few filler words.
                2. It’s solely composed of error or placeholder text with no substantial content.

                A file is “useful” if there is any significant content beyond trivial or placeholder text, 
                even if it includes some placeholder or error fragments. Do not classify it as “useless” 
                if there is meaningful text that might be relevant to a knowledge base or a RAG system.

                Your output must be a single word: "useless" or "useful".
            """

            user_message = (
                "Decide whether the following text is 'useless' or 'useful' based on the above instructions:\n\n"
                + value
            )
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=20,
                temperature=0.0,
            )
            if "useless" in response.choices[0].message.content.lower():
                keys_to_remove.append(key)

        for useless_key in keys_to_remove:
            clean_data_dict.pop(useless_key, None)

        return clean_data_dict
