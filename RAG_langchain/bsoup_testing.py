#pip install webdriver-manager
#pip install selenium
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import time


# Setup Chrome with Selenium WebDriver Manager
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)

try:
	driver.get('https://nutun.com/services/transact/payment-solutions')
	time.sleep(5)  # Adjust the sleep time based on your internet speed and page complexity
	
	driver.execute_script("window.scrollTo(0, 4000)")
	time.sleep(2) 
	driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
	body_text = driver.find_element(By.TAG_NAME, 'body').text
	time.sleep(5) 
	# soup = BeautifulSoup(response.text, 'html.parser')
	# all_text = soup.get_text(separator=' ', strip=True)
	# print(all_text)
	print(body_text)
	
	with open("mytext.txt","w") as f:
		f.write(body_text)
		
finally:
	driver.quit()