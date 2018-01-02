import xlrd
import xlwt
import requests
import re
import time

s = requests.session()
s.keep_alive = False

data = xlrd.open_workbook(r'try.xlsx')
rtable = data.sheets()[0]
wbook = xlwt.Workbook(encoding = 'utf-8',style_compression = 0)
wtable = wbook.add_sheet('sheet1',cell_overwrite_ok = True)

proxies = { 'http': 'http://61.155.164.108:3128',
            'http': 'http://116.199.115.79:80',
            'http': 'http://27.227.70.212:8088',
            'http': 'http://112.114.79.176:8118',
            'http': 'http://220.172.133.118:80',
            'http': 'http://1.48.111.216:8998',
            'http': 'http://119.5.89.114:8000',
            'http': 'http://117.173.179.94:8090',
            'http': 'http://112.67.229.203:8998',
            'http': 'http://140.240.185.220:8998',
            'http': 'http://120.237.14.198:53121',
            'http': 'http://116.199.115.78:80',

}
k=[]
count = 0
for i in range(0,10):
    k.append(rtable.cell(i,0).value)

for item in k:
    r=requests.get(url='https://www.drugbank.ca/drugs/'+str(item),proxies=proxies)
    patterns = re.compile('SMILES</dt><dd class="col-md-10 col-sm-8"><div class="wrap">(.*?)</div></dd></dl>')
    smile = re.findall(patterns,r.text)
    wtable.write(count,0,smile)
    print(smile)
    count += 1
    #time.sleep(0.5)

wbook.save(r'result-1.xls')
