from lomond import events
import zlib
import tools
from websocket import WebSocket

API_URL = "https://www.okex.com/v2"
SYMBOLS_URL = "spot/markets/products"
WS_URL = "wss://real.okex.com:10440/ws/v1"
KLINE_CHANNEL = "ok_sub_spot_{}_kline_1min"
DEPTH_CHANNEL = "ok_sub_spot_{}_depth_5"


class OkexWS(WebSocket):
    def __init__(self):
        super().__init__(WS_URL)

    def _send_message(self, **kwargs):
        self._ws.send_json(**kwargs)

    def _process(self, event):
        if isinstance(event, events.Ready):
            self._on_ready()
        elif isinstance(event, events.Binary):
            self._on_binary(event.data)
        else:
            super()._process(event)

    def _on_ready(self):
        # Subscription
        for symbol in self._symbols:
            fsymbol = symbol.lower()
            self._send_message(event="addChannel",
                               channel=KLINE_CHANNEL.format(fsymbol))
            self._send_message(event="addChannel",
                               channel=DEPTH_CHANNEL.format(fsymbol))

    def _on_binary(self, data):
        data_uncompressed = tools.uncompress(data, -zlib.MAX_WBITS)
        for element in data_uncompressed:
            self._dump(element)


if __name__ == '__main__':
    ws = OkexWS()
    ws.start()
