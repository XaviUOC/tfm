import json
from lomond import websocket as ws
import tools


def _load_common_symbols():
    with open("common_symbols.json") as f:
        return json.load(f)


class WebSocket:
    def __init__(self, url):
        self._url = url
        fname = "{}_{}.json".format(type(self).__name__, tools.timestamp())
        self._fout = open(fname, "wt")
        self._symbols = _load_common_symbols()
        self._ws = None

    @property
    def symbols(self):
        return self._symbols

    def _send_message(self, **kwargs):
        self._ws.send_json(**kwargs)

    def _process(self, event):
        tools.log("{}: {}".format(type(event), event))

    def start(self):
        self._ws = ws.WebSocket(self._url)
        for event in self._ws:
            self._process(event)

    def stop(self):
        self._ws.close()

    def _dump(self, text):
        self._fout.write(str(text))
        self._fout.write("\n")
        self._fout.flush()

    def __del__(self):
        try:
            self._fout.close()
            self.stop()
        except Exception:
            pass
