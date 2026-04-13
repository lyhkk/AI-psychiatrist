import urllib.request, json, http.cookiejar

cj = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))

# 1. start
req = urllib.request.Request(
    'http://127.0.0.1:5000/api/chat/start',
    data=json.dumps({"opening": ""}).encode(),
    headers={'Content-Type': 'application/json'},
    method='POST'
)
r = opener.open(req)
data = json.loads(r.read())
print('start:', data)

# 2. message
req2 = urllib.request.Request(
    'http://127.0.0.1:5000/api/chat/message',
    data=json.dumps({"message": "\u6211\u6700\u8fd1\u5f88\u6cae\u4e27"}).encode(),
    headers={'Content-Type': 'application/json'},
    method='POST'
)
r2 = opener.open(req2, timeout=180)
data2 = json.loads(r2.read())
print('status:', data2.get('status'))
print('reply:', data2.get('reply', '')[:80])
print('cbt_form:', data2.get('cbt_form'))

