# -*- coding:utf-8 -*-
from bs4 import BeautifulSoup
import requests
import urllib2
import re
import os
import socket

def getURLs(url):
    response = requests.get(url)
    page = BeautifulSoup(response.content, "html.parser")
    page.prettify()
    tableURLs = page.find('table', 'table')
    for link in tableURLs.findAll('a'):
        yield str(link.get('href'))

def requestURLs(url, main_url, outdir):
    if not url.startswith('http'):
        url = main_url + url
    print "Current URL is %s" %(url)
    outname = outdir + "/" + url.split('/')[-1]
    # if file exists, exit
    if os.path.exists(outname) or os.path.exists(outname + '.pdf') or os.path.exists(outname + '.html'):
        return None
    if url.endswith('zip'):
        rq = requests.get(url, stream=True)
        # check if url exists
        if rq.status_code == requests.codes.ok:
            with open(outname, 'wb') as fd:
                for chunk in rq.iter_content(chunk_size=1024):
                    if chunk:
                        fd.write(chunk)
            fd.close()
    else:
        rq = urllib2.Request(url)
        try:
            res = urllib2.urlopen(rq, timeout=10)
        except urllib2.URLError, e:
            print "Timed out to connect to this URL"
            return None
        except socket.timeout:
            print "Time out!"
            return None
        content = res.read()
        if url.endswith('pdf'):
            with open(outname, 'wb') as fd:
                fd.write(content)
            fd.close()
        else:
            # check if html format file
            if not re.search(r'.', url.split('/')[-1]):
                if 'text/html' in res.headers.getheader('Content-Type'):
                    print "HTML file!"
                    if not url.endswith('html'):
                        outname += '.html'
                else:
                    print "PDF file!"
                    outname += '.pdf'
            with open(outname, 'wb') as fd:
                fd.write(content)
            fd.close()

def main():
    main_url = "http://cs231n.stanford.edu/"
    target_url = main_url + "syllabus.html"
    outdir  = "/home/fanzong/Desktop/cs231n"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for url in getURLs(target_url):
        requestURLs(url, main_url, outdir)

main()
