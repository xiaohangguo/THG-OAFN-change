"""Search real references for the thesis lit review (Crossref EN + Baidu Xueshu ZH)."""

import json
import re
import time
import urllib.parse
import urllib.request


def get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research)"})
    return urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")


def crossref(query: str, rows: int = 8) -> None:
    q = urllib.parse.quote(query)
    url = (
        "https://api.crossref.org/works?query=" + q
        + f"&rows={rows}&select=title,author,container-title,issued,DOI,type"
    )
    d = json.loads(get(url))
    for it in d["message"]["items"]:
        title = (it.get("title") or [""])[0][:90]
        au = it.get("author") or []
        first = ""
        if au:
            first = (au[0].get("family", "") + ", " + au[0].get("given", ""))[:28]
        venue = (it.get("container-title") or ["?"])[0][:48]
        year = (it.get("issued", {}).get("date-parts", [[None]])[0][0])
        doi = it.get("DOI", "")
        print(f"  {year} | {first} | {title} | {venue} | {doi}")


def baidu_xueshu(query: str) -> None:
    q = urllib.parse.quote(query)
    html = get("https://xueshu.baidu.com/s?wd=" + q + "&rn=20&sc_hit=1")
    titles = re.findall(r'<h3[^>]*>\s*<a[^>]*>(.*?)</a>', html, re.S)
    clean = []
    for t in titles[:20]:
        t = re.sub(r"<[^>]+>", "", t).strip()
        t = re.sub(r"\s+", " ", t)
        if len(t) > 8:
            clean.append(t)
    for t in clean:
        print("  " + t[:90])


def dblp(query: str) -> None:
    q = urllib.parse.quote(query)
    d = json.loads(get("https://dblp.org/search/publ/api?format=json&h=8&q=" + q))
    hits = d["result"]["hits"].get("hit", [])
    for h in hits:
        i = h["info"]
        auth = i.get("authors", {}).get("author", [])
        if isinstance(auth, dict):
            auth = [auth]
        a0 = auth[0]["text"] if auth else "?"
        print(f"  {i.get('year')} | {a0[:24]} | {str(i.get('title'))[:85]} | {i.get('venue','')}")


def cnki_space(query: str) -> None:
    q = urllib.parse.quote(query)
    try:
        html = get("https://search.cnki.com.cn/Search/Result?content=" + q)
        titles = re.findall(r'<a[^>]*class="[^"]*title[^"]*"[^>]*>(.*?)</a>', html, re.S)
        if not titles:
            titles = re.findall(r"<dt[^>]*>.*?<a[^>]*>(.*?)</a>", html, re.S)
        for t in titles[:15]:
            t = re.sub(r"<[^>]+>", "", t).strip()
            t = re.sub(r"\s+", " ", t)
            if len(t) > 8:
                print("  " + t[:92])
        if not titles:
            print("  [no match / blocked]")
    except Exception as e:  # noqa: BLE001
        print(f"  [error] {e}")


if __name__ == "__main__":
    print("=== EN: AML + ML survey ===")
    crossref("anti-money laundering machine learning survey", 8)
    time.sleep(2)
    print("=== EN: GNN + AML ===")
    crossref("graph neural network anti-money laundering transaction", 8)
    time.sleep(2)
    print("=== DBLP: AML ===")
    dblp("anti-money laundering")
    time.sleep(2)
    print("=== DBLP: illicit accounts graph ===")
    dblp("illicit accounts graph")
    time.sleep(2)
    print("=== ZH: CNKI space ===")
    cnki_space("反洗钱 图神经网络")
    time.sleep(2)
    print("=== ZH: CNKI space 2 ===")
    cnki_space("反洗钱 机器学习")
