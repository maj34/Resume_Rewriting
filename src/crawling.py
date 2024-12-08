import json
import os
import argparse
from typing import Set, Dict, Any, List

from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def url_crawl(driver: webdriver.Chrome, start_page: int, end_page: int) -> Set[str]:
    """
    Crawl URLs of cover letters from the specified page range.

    Args:
        driver (webdriver.Chrome): The Selenium WebDriver instance.
        start_page (int): The starting page number.
        end_page (int): The ending page number.

    Returns:
        Set[str]: A set of unique URLs collected from the specified page range.
    """
    url_list = []
    for page in tqdm(range(start_page, end_page), total=end_page - start_page, desc='URL 크롤링 진행도'):
        url = f"https://linkareer.com/cover-letter/search?page={page}&tab=all"
        driver.get(url)
        driver.implicitly_wait(3)

        url_tags = driver.find_elements(
            By.CSS_SELECTOR, 
            'a[href*="cover-letter"][href]:not([href*="search"])'
        )

        for tag in url_tags:
            url_name = tag.get_attribute('href')
            if url_name:
                url_list.append(url_name)

    unique_urls = set(url_list)
    print("URL 수집 완료:", len(unique_urls))
    return unique_urls


def self_introduction(driver: webdriver.Chrome, url: str) -> Dict[str, Any]:
    """
    Extract personal information and cover letter content from a given URL.

    Args:
        driver (webdriver.Chrome): The Selenium WebDriver instance.
        url (str): The URL of the cover letter page.

    Returns:
        Dict[str, Any]: A dictionary containing applicant information and their cover letter.
    """
    person = {}
    driver.get(url)

    # Extract basic information: 지원회사 / 지원부서 / 모집시기
    try:
        info_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (By.XPATH, '//*[@id="__next"]/div[1]/div[4]/div/div[2]/div[1]/div[1]/div/div/div[2]/h1')
            )
        )
        info_parts = [part.strip() for part in info_element.text.split(' / ')]
        person['지원회사'] = info_parts[0]
        person['지원부서'] = info_parts[1]
        person['모집시기'] = info_parts[2]
    except Exception as e:
        print("info 요소를 찾을 수 없습니다:", e)
        person['지원회사'] = "Not Available"
        person['지원부서'] = "Not Available"
        person['모집시기'] = "Not Available"

    # Extract specification: 출신학교 / 학과 / 학점 / 자격증 등
    try:
        specification = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (By.XPATH, '//*[@id="__next"]/div[1]/div[4]/div/div[2]/div[1]/div[1]/div/div/div[3]/h3')
            )
        )
        spec_text = specification.text
        # 출신학교 / 학과 / 학점 / 기타 등
        # Parsing logic:
        # Example: "서울대학교 / 컴퓨터공학과 학점 4.0 / 토익 900"
        # Split by ' / ' first for school and department/other
        spec_parts = [part.strip() for part in spec_text.split(' / ')]
        
        # spec_parts[0]: 출신학교
        # spec_parts[1] 부분에 학과와 학점 정보가 있을 가능성 -> "컴퓨터공학과 학점 4.0"
        # 이후 추가적인 항목이 있을 경우 ' / '로 나뉘어 있음
        
        # Extract 학과, 학점, 기타
        # Find '학점' keyword
        if '학점' in spec_text:
            department_part = spec_text.split(' / ')[1]  # "컴퓨터공학과 학점 4.0"
            department = department_part.split('학점')[0].strip()
            score_part = spec_text.split('학점 ')[1]      # "4.0 / 토익 900"
            score_and_etc = score_part.split(' / ')
            score = score_and_etc[0].strip()
            etc = score_and_etc[1].strip() if len(score_and_etc) > 1 else "없음"
        else:
            # If '학점' keyword not found, fallback to defaults
            department = "Not Available"
            score = "Not Available"
            etc = "없음"

        person['출신학교'] = spec_parts[0] if spec_parts else "Not Available"
        person['학과'] = department
        person['학점'] = score
        person['기타'] = etc
    except Exception as e:
        print("specification 요소를 찾을 수 없습니다:", e)
        person['출신학교'] = "Not Available"
        person['학과'] = "Not Available"
        person['학점'] = "Not Available"
        person['기타'] = "Not Available"

    # Extract cover letter content
    try:
        content = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "coverLetterContent"))
        )
        person['자기소개서'] = content.text
    except Exception as e:
        print("자기소개서 요소를 찾을 수 없습니다:", e)
        person['자기소개서'] = "Not Available"

    return person


def main(start_page: int, end_page: int) -> None:
    """
    Main function to perform web crawling and extract applicant data.

    Args:
        start_page (int): The starting page number.
        end_page (int): The ending page number.

    Returns:
        None
    """
    options = Options()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")

    service = ChromeService(executable_path="/usr/local/bin/chromedriver")
    driver = webdriver.Chrome(service=service, options=options)

    urls = url_crawl(driver, start_page, end_page)

    people = {}
    os.makedirs("result", exist_ok=True)
    output_path = f"./result/linkcareer_data_{start_page}_{end_page}.json"

    for idx, url in tqdm(enumerate(urls), total=len(urls), desc="크롤링 진행도"):
        person = self_introduction(driver, url)
        people[idx] = person
        # Save progress after each extraction
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(people, f, ensure_ascii=False, indent=4)

    driver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linkareer Cover Letter Crawler")
    parser.add_argument("--start_page", type=int, default=1, help="Starting page number for crawling")
    parser.add_argument("--end_page", type=int, default=60, help="Ending page number for crawling")
    args = parser.parse_args()
    main(start_page=args.start_page, end_page=args.end_page)