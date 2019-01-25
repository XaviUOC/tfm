import datetime
import zlib
import json


def timestamp():
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S")


def log(msg, *args):
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S.%f ") + msg.format(*args))


def uncompress(data, wbits=31):
    res = str(zlib.decompressobj(wbits).decompress(data), encoding="utf-8")
    return json.loads(res)
