import requests
import pandas as pd
from bs4 import BeautifulSoup


def parse_html(soup,data_info):
    li_list = soup.select('.listCentent li')
    # print(len(li_list))
    for li in li_list:
        data_info['网站排名'].append(li.select('.RtCRateWrap .RtCRateCent strong')[0].text)
        data_info['网站名称'].append(li.select('.CentTxt .rightTxtHead a')[0].text)
        data_info['网站网址'].append(li.select('.CentTxt .rightTxtHead .col-gray ')[0].text)
        data_info['网站简介'].append(li.select('.CentTxt .RtCInfo')[0].text)
        data_info['网站得分'].append(li.select('.RtCRateWrap .RtCRateCent span ')[0].text)
    return data_info

data_info ={'网站排名':[],'网站名称':[],'网站网址':[],'网站简介':[],'网站得分':[]}
headers = {'User-Agent':'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Mobile Safari/537.36 Edg/90.0.818.51'}
for i in range(1,18):
    if i ==1:
        url = 'https://top.chinaz.com/all/index_br.html'
    else:
        url = f'https://top.chinaz.com/all/index_br_{i}.html'
    response = requests.get(url=url,headers=headers,timeout=10)
    html_content = response.text
    soup = BeautifulSoup(html_content,'lxml')
    data_info = parse_html(soup,data_info)
    print(f"第{i}页done")

book_info = pd.DataFrame(data_info)
print(book_info.isnull())
print(book_info.duplicated())

# book_info['图书价格'][book_info['图书价格']>100]=None
book_info = book_info.dropna()
book_info.to_excel('网站排名.xlsx')

