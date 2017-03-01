# -*- coding: utf-8 _*_
import sys
import datetime
import urllib2
from ghost import Ghost, Session
from bs4 import BeautifulSoup
from urllib import quote

ghost = Ghost()
name = "江南果道"
with ghost.start() as session:
    url = "http://weixin.sogou.com/weixin?type=1&query=" + quote(name) + "&ie=utf8&_sug_=n&_sug_type_=2&page=1&ie=utf8"
    page, sources = session.open(url)
    result, resources = session.wait_for_selector(name )

c = 0
while True:
    if c >= 30:
        break
    soup = BeautifulSoup(session.content)
    data = soup.find_all('div', attrs={'class':'wx-rb bg-blue wx-rb_v1 _item'})
    for div in data:
        links = div.find_all('a')
        for a in links:
            print a['href']