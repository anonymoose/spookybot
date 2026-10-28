"""Live camera view with the detector's boxes drawn on, for watching the vision
work from another machine.

The camera can only be opened by one process, so the bot itself serves the
picture: a tiny MJPEG server that renders a frame only while someone is
watching (no viewers = no cost). It listens on localhost only, because this is a
live feed from a party; reach it through an SSH tunnel (`make view`).
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

log = logging.getLogger(__name__)

FPS = 6.0
PAGE = b"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width">
<title>spookybot</title>
<style>
 :root{color-scheme:dark}
 *{box-sizing:border-box}
 body{margin:0;height:100vh;display:flex;gap:10px;padding:10px;background:#0d0e11;color:#d7dae0;font:13px/1.35 ui-monospace,Menlo,Consolas,monospace}
 #cam{flex:3;min-width:0;display:flex;align-items:center;justify-content:center;background:#000;border-radius:6px}
 #cam img{max-width:100%;max-height:100%}
 #side{flex:2;min-width:0;display:flex;flex-direction:column}
 #bar{padding:4px 2px 8px;color:#8b93a1}
 #bar label{margin-right:14px;cursor:pointer}
 #log{flex:1;overflow-y:auto;background:#07080a;border-radius:6px;padding:8px;white-space:pre-wrap;word-break:break-word}
 .VOICE{color:#7dd3fc}.VISION{color:#86efac}.CHAT{color:#fcd34d}.MOTION{color:#8b93a1}.bad{color:#f87171}
 @media (max-width:900px){body{flex-direction:column;height:auto}#cam{height:50vh}#log{height:45vh}}
</style>
<div id=cam><img src=/stream alt="camera"></div>
<div id=side>
 <div id=bar><label><input type=checkbox id=motion> servo motion</label>
 <label><input type=checkbox id=follow checked> follow</label></div>
 <div id=log></div>
</div>
<script>
const log=document.getElementById('log'),motion=document.getElementById('motion'),
      follow=document.getElementById('follow');
let last=0;
function cls(t){
  if(/ (ERROR|WARNING|CRITICAL) /.test(t)||/Traceback/.test(t))return 'bad';
  const m=t.match(/ (VOICE|VISION|CHAT|MOTION) /);return m?m[1]:'';
}
async function poll(){
  try{
    const r=await fetch('/logs?after='+last),j=await r.json();
    if(j.last<last)log.textContent='';   // bot restarted
    last=j.last;
    for(const l of j.lines){
      const c=cls(l.text);
      if(c==='MOTION'&&!motion.checked)continue;
      const d=document.createElement('div');d.className=c;d.textContent=l.text;log.appendChild(d);
    }
    while(log.childElementCount>600)log.removeChild(log.firstChild);
    if(follow.checked)log.scrollTop=log.scrollHeight;
  }catch(e){}
  setTimeout(poll,700);
}
poll();
</script>"""

GREEN, ORANGE = (60, 220, 90), (255, 160, 40)


def draw_overlay(image, dets, header: str):
    """Draw detections (normalized y0,x0,y1,x1,score,reason) and a header on a
    PIL image. A non-empty reason means the box was seen but rejected."""
    from PIL import ImageDraw, ImageFont

    try:
        font = ImageFont.load_default(size=max(14, image.width // 55))
    except TypeError:  # older Pillow: fixed-size bitmap font
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(image)
    w, h = image.size
    for y0, x0, y1, x1, score, reason in dets:
        color = ORANGE if reason else GREEN
        box = (x0 * w, y0 * h, x1 * w, y1 * h)
        draw.rectangle(box, outline=color, width=3)
        label = f"person {score:.2f}" + (f" ({reason})" if reason else "")
        tb = draw.textbbox((box[0], box[1]), label, font=font)
        draw.rectangle((tb[0] - 2, tb[1] - 2, tb[2] + 2, tb[3] + 2), fill=color)
        draw.text((box[0], box[1]), label, fill=(0, 0, 0), font=font)
    tb = draw.textbbox((6, 6), header, font=font)
    draw.rectangle((tb[0] - 4, tb[1] - 3, tb[2] + 4, tb[3] + 3), fill=(0, 0, 0))
    draw.text((6, 6), header, fill=(255, 255, 255), font=font)
    return image


class LogBuffer(logging.Handler):
    """The bot's recent log lines, for the page's scrolling log panel."""

    def __init__(self, keep: int = 600) -> None:
        super().__init__(logging.INFO)
        self.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"))
        self._lines: deque = deque(maxlen=keep)
        self._next = 1
        self._mutex = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            text = self.format(record)
        except Exception:
            return
        with self._mutex:
            self._lines.append((self._next, text))
            self._next += 1

    def since(self, after: int) -> dict:
        with self._mutex:
            lines = [{"id": i, "text": t} for i, t in self._lines if i > after]
            return {"last": self._next - 1, "lines": lines}


def _handler(render, logs):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the journal for the bot's own lines
            pass

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, "text/html", PAGE)
            elif self.path.startswith("/snapshot"):
                jpeg = render()
                if jpeg is None:
                    self._send(503, "text/plain", b"no camera")
                else:
                    self._send(200, "image/jpeg", jpeg)
            elif self.path.startswith("/stream"):
                self._stream()
            elif self.path.startswith("/logs"):
                after = 0
                if "after=" in self.path:
                    try:
                        after = int(self.path.split("after=")[1].split("&")[0])
                    except ValueError:
                        pass
                self._send(200, "application/json",
                           json.dumps(logs.since(after)).encode())
            else:
                self._send(404, "text/plain", b"not found")

        def _send(self, code, ctype, body):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while True:
                    t0 = time.monotonic()
                    jpeg = render()
                    if jpeg is not None:
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\n"
                            b"Content-Length: " + str(len(jpeg)).encode()
                            + b"\r\n\r\n" + jpeg + b"\r\n")
                    time.sleep(max(0.0, 1.0 / FPS - (time.monotonic() - t0)))
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass  # viewer closed the tab

    return Handler


def start(render, port: int, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    """Serve `render() -> jpeg bytes | None`. Port 0 picks a free one."""
    logs = LogBuffer()
    logging.getLogger().addHandler(logs)
    server = ThreadingHTTPServer((host, port), _handler(render, logs))
    server.daemon_threads = True
    server.log_buffer = logs
    threading.Thread(target=server.serve_forever, daemon=True,
                     name="liveview").start()
    log.info("VISION live view on http://%s:%d (ssh -L %d:localhost:%d, or make view)",
             host, server.server_address[1], server.server_address[1],
             server.server_address[1])
    return server
