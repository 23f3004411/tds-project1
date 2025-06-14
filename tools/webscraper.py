import requests
from bs4 import BeautifulSoup
import pandas as pd
from playwright.sync_api import sync_playwright
import time

# Links collection to be used as sources
links = [
    "https://tds.s-anand.net/#/2025-01/",
    "https://tds.s-anand.net/#/../development-tools",
    "https://tds.s-anand.net/#/../deployment-tools",
    "https://tds.s-anand.net/#/../large-language-models",
    "https://tds.s-anand.net/#/../project-1",
    "https://tds.s-anand.net/#/../data-sourcing",
    "https://tds.s-anand.net/#/../data-preparation",
    "https://tds.s-anand.net/#/../data-analysis",
    "https://tds.s-anand.net/#/../project-2",
    "https://tds.s-anand.net/#/../data-visualization",
    "https://tds.s-anand.net/#/../live-sessions",
    "https://tds.s-anand.net/#/README",
    "https://tds.s-anand.net/#/development-tools",
    "https://tds.s-anand.net/#/deployment-tools",
    "https://tds.s-anand.net/#/large-language-models",
    "https://tds.s-anand.net/#/project-tds-virtual-ta",
    "https://tds.s-anand.net/#/data-sourcing",
    "https://tds.s-anand.net/#/data-preparation",
    "https://tds.s-anand.net/#/data-analysis",
    "https://tds.s-anand.net/#/data-visualization",
    "https://tds.s-anand.net/#/vscode",
    "https://tds.s-anand.net/#/github-copilot",
    "https://tds.s-anand.net/#/uv",
    "https://tds.s-anand.net/#/npx",
    "https://tds.s-anand.net/#/unicode",
    "https://tds.s-anand.net/#/devtools",
    "https://tds.s-anand.net/#/css-selectors",
    "https://tds.s-anand.net/#/json",
    "https://tds.s-anand.net/#/bash",
    "https://tds.s-anand.net/#/llm",
    "https://tds.s-anand.net/#/spreadsheets",
    "https://tds.s-anand.net/#/sqlite",
    "https://tds.s-anand.net/#/git",
    "https://tds.s-anand.net/#/markdown",
    "https://tds.s-anand.net/#/image-compression",
    "https://tds.s-anand.net/#/github-pages",
    "https://tds.s-anand.net/#/colab",
    "https://tds.s-anand.net/#/vercel",
    "https://tds.s-anand.net/#/github-actions",
    "https://tds.s-anand.net/#/docker",
    "https://tds.s-anand.net/#/github-codespaces",
    "https://tds.s-anand.net/#/ngrok",
    "https://tds.s-anand.net/#/cors",
    "https://tds.s-anand.net/#/rest-apis",
    "https://tds.s-anand.net/#/fastapi",
    "https://tds.s-anand.net/#/google-auth",
    "https://tds.s-anand.net/#/ollama",
    "https://tds.s-anand.net/#/prompt-engineering ",
    "https://tds.s-anand.net/#/tds-ta-instructions",
    "https://tds.s-anand.net/#/tds-gpt-reviewer",
    "https://tds.s-anand.net/#/llm-sentiment-analysis",
    "https://tds.s-anand.net/#/llm-text-extraction",
    "https://tds.s-anand.net/#/base64-encoding",
    "https://tds.s-anand.net/#/vision-models",
    "https://tds.s-anand.net/#/embeddings",
    "https://tds.s-anand.net/#/multimodal-embeddings",
    "https://tds.s-anand.net/#/topic-modeling",
    "https://tds.s-anand.net/#/vector-databases",
    "https://tds.s-anand.net/#/rag-cli",
    "https://tds.s-anand.net/#/hybrid-rag-typesense",
    "https://tds.s-anand.net/#/function-calling",
    "https://tds.s-anand.net/#/llm-agents",
    "https://tds.s-anand.net/#/llm-image-generation",
    "https://tds.s-anand.net/#/llm-speech",
    "https://tds.s-anand.net/#/llm-evals",
    "https://tds.s-anand.net/#/scraping-with-excel",
    "https://tds.s-anand.net/#/scraping-with-google-sheets",
    "https://tds.s-anand.net/#/crawling-cli",
    "https://tds.s-anand.net/#/bbc-weather-api-with-python",
    "https://tds.s-anand.net/#/scraping-imdb-with-javascript",
    "https://tds.s-anand.net/#/nominatim-api-with-python",
    "https://tds.s-anand.net/#/wikipedia-data-with-python",
    "https://tds.s-anand.net/#/scraping-pdfs-with-tabula",
    "https://tds.s-anand.net/#/convert-pdfs-to-markdown",
    "https://tds.s-anand.net/#/convert-html-to-markdown",
    "https://tds.s-anand.net/#/llm-website-scraping",
    "https://tds.s-anand.net/#/llm-video-screen-scraping",
    "https://tds.s-anand.net/#/web-automation-with-playwright",
    "https://tds.s-anand.net/#/scheduled-scraping-with-github-actions",
    "https://tds.s-anand.net/#/scraping-emarketer",
    "https://tds.s-anand.net/#/scraping-live-sessions",
    "https://tds.s-anand.net/#/data-cleansing-in-excel",
    "https://tds.s-anand.net/#/data-transformation-in-excel",
    "https://tds.s-anand.net/#/splitting-text-in-excel",
    "https://tds.s-anand.net/#/data-aggregation-in-excel",
    "https://tds.s-anand.net/#/data-preparation-in-the-shell",
    "https://tds.s-anand.net/#/data-preparation-in-the-editor",
    "https://tds.s-anand.net/#/data-preparation-in-duckdb",
    "https://tds.s-anand.net/#/cleaning-data-with-openrefine",
    "https://tds.s-anand.net/#/parsing-json",
    "https://tds.s-anand.net/#/dbt",
    "https://tds.s-anand.net/#/transforming-images",
    "https://tds.s-anand.net/#/extracting-audio-and-transcripts",
    "https://tds.s-anand.net/#/correlation-with-excel",
    "https://tds.s-anand.net/#/regression-with-excel",
    "https://tds.s-anand.net/#/forecasting-with-excel",
    "https://tds.s-anand.net/#/outlier-detection-with-excel",
    "https://tds.s-anand.net/#/data-analysis-with-python",
    "https://tds.s-anand.net/#/data-analysis-with-sql",
    "https://tds.s-anand.net/#/data-analysis-with-datasette",
    "https://tds.s-anand.net/#/data-analysis-with-duckdb",
    "https://tds.s-anand.net/#/data-analysis-with-chatgpt",
    "https://tds.s-anand.net/#/geospatial-analysis-with-excel",
    "https://tds.s-anand.net/#/geospatial-analysis-with-python",
    "https://tds.s-anand.net/#/geospatial-analysis-with-qgis",
    "https://tds.s-anand.net/#/network-analysis-in-python",
    "https://tds.s-anand.net/#/visualizing-forecasts-with-excel",
    "https://tds.s-anand.net/#/visualizing-animated-data-with-powerpoint",
    "https://tds.s-anand.net/#/visualizing-animated-data-with-flourish",
    "https://tds.s-anand.net/#/visualizing-network-data-with-kumu",
    "https://tds.s-anand.net/#/visualizing-charts-with-excel",
    "https://tds.s-anand.net/#/data-visualization-with-seaborn",
    "https://tds.s-anand.net/#/data-visualization-with-chatgpt",
    "https://tds.s-anand.net/#/actor-network-visualization",
    "https://tds.s-anand.net/#/rawgraphs",
    "https://tds.s-anand.net/#/data-storytelling",
    "https://tds.s-anand.net/#/narratives-with-llms",
    "https://tds.s-anand.net/#/marimo",
    "https://tds.s-anand.net/#/revealjs",
    "https://tds.s-anand.net/#/marp",
]

def scrape_dynamic_content_no_table(url):
    main_text = ""

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        try:
            page.goto(url, wait_until='networkidle')

            html_content = page.content()
            soup = BeautifulSoup(html_content, 'html.parser')

            article_content = soup.find('article', class_='markdown-section', id='main')

            if article_content:
                # Remove script and style tags which are not part of the main text
                for element in article_content(['script', 'style']):
                    element.decompose()

                title_element = article_content.find('h1')
                if not title_element:
                    title_element = article_content.find('h2')

                if title_element:
                    article_title = title_element.get_text(strip=True)
                
                main_text = article_content.get_text(separator='\n', strip=True)
                main_text = '\n'.join(line for line in main_text.splitlines() if line.strip())
            else:
                main_text = "Article content not found after JavaScript rendering."
                app_div = soup.find('div', id='app')
                if app_div:
                    main_text = app_div.get_text(separator='\n', strip=True)
                    main_text = '\n'.join(line for line in main_text.splitlines() if line.strip())
                    print("Fallback: Extracted text from #app div.")

        except Exception as e:
            print(f"An error occurred during Playwright operation for {url}: {e}")
        finally:
            browser.close()

    return article_title, main_text

if __name__ == "__main__":
    links_to_scrape = links

    all_scraped_data_for_excel = []

    for i, link in enumerate(links_to_scrape):
        print(f"Scraping {link}...")
        article_title, article_text = scrape_dynamic_content_no_table(link)

        all_scraped_data_for_excel.append({
            'URL': link,
            'Title': article_title,  
            'Main Article Content': article_text,
        })

    if all_scraped_data_for_excel:
        df = pd.DataFrame(all_scraped_data_for_excel)

        column_order = ['URL', 'Main Article Content'] + [col for col in df.columns if col not in ['URL', 'Main Article Content']]
        df = df[column_order]

        excel_file_name = 'scraped_data/scraped_course_data.xlsx'
        df.to_excel(excel_file_name, index=False)
        print(f"\nData successfully saved to {excel_file_name}")
    else:
        print("\nNo data was scraped.")