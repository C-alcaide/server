"""Build a single sectioned HTML manual from `docs/features/`.

WHY A BUILD RATHER THAN AUTHORING HTML. The markdown is the source of truth: it diffs, it reviews
in a pull request, and it changes in the same commit as the code it describes -- which is the rule
that stops documentation lagging. HTML authored by hand would be a second copy of every claim, and
this tree has already paid for those: an audit on 2026-08-26 found four documented claims that had
outlived their code.

So this is a build step, and the output is disposable. Delete `docs/features.html`, run this, get
it back.

WHAT IT DELIBERATELY DOES NOT DO:

  * No external assets. One self-contained file, images referenced by relative path, so it opens
    from the filesystem with no server and no network.
  * No search, no navigation framework, no JavaScript beyond a theme toggle. A manual that needs a
    build toolchain to read is a manual nobody reads.
  * No editing of the markdown. If a document renders badly the document is wrong, not this script.

Run:  python docs/build_html.py
Out:  docs/features.html
"""
import os
import re
import sys

try:
    import markdown
except ImportError:                                     # pragma: no cover
    sys.exit("needs `python -m pip install markdown` (pure Python, no build step)")

HERE = os.path.dirname(os.path.abspath(__file__))
FEATURES = os.path.join(HERE, "features")
OUT = os.path.join(HERE, "features.html")

#: Reading order, not alphabetical. Grouped the way someone new to the fork would want it, and
#: within a group by how much of the fork the reader has to understand first.
ORDER = [
    ("Start here", ["README.md"]),
    ("Colour", ["colour-grading-and-ocio.md"]),
    ("Projection and virtual production",
     ["projection-and-icvfx.md", "previz.md", "camera-tracking.md"]),
    ("GPU pipeline",
     ["vulkan-mixer.md", "vulkan-output.md", "cuda-prores.md", "cuda-notchlc.md", "hap.md",
      "gstreamer.md", "isf-and-openfx.md", "spout.md"]),
    ("Rewritten upstream modules",
     ["ffmpeg-producer-and-consumer.md", "decklink-output.md", "screen-consumer.md"]),
    ("Signal, sync and control",
     ["ltc-timecode.md", "dmx-sacn-artnet.md", "keyframes.md", "cluster-sync.md", "replay.md",
      "remotewall-and-portaudio.md"]),
    ("Template", ["_TEMPLATE.md"]),
]

CSS = """
:root{
  --bg:#f7f7f8; --panel:#ffffff; --border:#d8d8dc; --text:#1c1c1e; --muted:#6a6a70;
  --title:#0b5aa8; --accent:#0b5aa8; --code-bg:#f0f0f2; --warn-bg:#fff6e5; --warn-br:#e0a83a;
  --nav-bg:#ffffff;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#1a1a1c; --panel:#232326; --border:#3a3a3f; --text:#e4e4e7; --muted:#9a9aa2;
    --title:#7fbcf5; --accent:#7fbcf5; --code-bg:#17171a; --warn-bg:#332a15; --warn-br:#c08a20;
    --nav-bg:#202024;
  }
}
:root[data-theme="dark"]{
  --bg:#1a1a1c; --panel:#232326; --border:#3a3a3f; --text:#e4e4e7; --muted:#9a9aa2;
  --title:#7fbcf5; --accent:#7fbcf5; --code-bg:#17171a; --warn-bg:#332a15; --warn-br:#c08a20;
  --nav-bg:#202024;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
     font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
.wrap{display:flex;align-items:flex-start;max-width:1500px;margin:0 auto;}
nav{position:sticky;top:0;flex:0 0 290px;max-height:100vh;overflow-y:auto;padding:22px 14px 40px;
    background:var(--nav-bg);border-right:1px solid var(--border);}
nav h2{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);
       margin:18px 0 6px;}
nav a{display:block;padding:4px 8px;border-radius:5px;color:var(--text);text-decoration:none;
      font-size:13.5px;}
nav a:hover{background:var(--code-bg);}
main{flex:1 1 auto;min-width:0;padding:28px 34px 90px;}
section{background:var(--panel);border:1px solid var(--border);border-radius:10px;
        padding:24px 28px;margin:0 0 26px;}
h1{font-size:26px;margin:.2em 0 .5em;color:var(--title);}
h2{font-size:19px;margin:1.6em 0 .5em;padding-bottom:.25em;border-bottom:1px solid var(--border);}
h3{font-size:16px;margin:1.3em 0 .4em;}
a{color:var(--accent);}
code{background:var(--code-bg);padding:.12em .38em;border-radius:4px;
     font:13px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace;}
pre{background:var(--code-bg);border:1px solid var(--border);border-radius:8px;padding:12px 14px;
    overflow-x:auto;}
pre code{background:none;padding:0;}
table{border-collapse:collapse;width:100%;margin:1em 0;font-size:13.5px;display:block;
      overflow-x:auto;}
th,td{border:1px solid var(--border);padding:7px 10px;text-align:left;vertical-align:top;}
th{background:var(--code-bg);}
blockquote{margin:1em 0;padding:10px 16px;background:var(--warn-bg);
           border-left:3px solid var(--warn-br);border-radius:0 6px 6px 0;}
blockquote p{margin:.3em 0;}
img{max-width:100%;height:auto;border-radius:8px;border:1px solid var(--border);margin:.6em 0;}
hr{border:0;border-top:1px solid var(--border);margin:1.8em 0;}
.toggle{position:fixed;right:16px;top:14px;z-index:9;background:var(--panel);color:var(--text);
        border:1px solid var(--border);border-radius:6px;padding:6px 11px;cursor:pointer;
        font-size:12.5px;}
.meta{color:var(--muted);font-size:12.5px;margin:0 0 18px;}
@media (max-width:900px){
  .wrap{flex-direction:column} nav{position:static;flex:none;width:100%;max-height:none;
  border-right:0;border-bottom:1px solid var(--border)} main{padding:20px 16px 60px}
}
"""

JS = """
(function(){
  var b=document.getElementById('t');
  function cur(){return document.documentElement.getAttribute('data-theme')
    || (window.matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');}
  b.addEventListener('click',function(){
    var n = cur()==='dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme',n);
    b.textContent = n==='dark' ? 'light' : 'dark';
  });
  b.textContent = cur()==='dark' ? 'light' : 'dark';
})();
"""


def slug(name):
    return re.sub(r"[^a-z0-9]+", "-", os.path.splitext(name)[0].lower()).strip("-")


def build():
    md = markdown.Markdown(extensions=["tables", "fenced_code", "attr_list", "sane_lists"])
    nav, body, missing = [], [], []

    for group, files in ORDER:
        nav.append(f"<h2>{group}</h2>")
        for f in files:
            path = os.path.join(FEATURES, f)
            if not os.path.exists(path):
                missing.append(f)
                continue
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
            # An inter-document link becomes an in-page anchor; a link out of features/ keeps its
            # relative path, which still resolves because the output sits in docs/.
            text = re.sub(r"\]\((?!\.\./|https?://)([A-Za-z0-9_.-]+)\.md\)",
                          lambda m: f"](#{slug(m.group(1))})", text)
            text = text.replace("](../images/", "](images/")
            text = re.sub(r"\]\(\.\./([a-z]+)/", r"](\1/", text)

            md.reset()
            html = md.convert(text)
            title = re.search(r"^#\s+(.+)$", text, re.M)
            title = title.group(1).strip() if title else f
            sid = slug(f)
            nav.append(f'<a href="#{sid}">{title}</a>')
            body.append(f'<section id="{sid}">{html}</section>')

    if missing:
        print("WARNING: listed in ORDER but absent:", ", ".join(missing))
    present = {f for _, fs in ORDER for f in fs}
    extra = sorted(set(os.listdir(FEATURES)) - present - {"features.html"})
    extra = [e for e in extra if e.endswith(".md")]
    if extra:
        # Loud, because a new document silently missing from the manual is exactly the kind of
        # quiet omission this folder exists to avoid.
        print("WARNING: in features/ but NOT in the manual — add to ORDER:", ", ".join(extra))

    out = (
        "<title>CasparVP features</title>\n"
        f"<style>{CSS}</style>\n"
        '<button class="toggle" id="t">dark</button>\n'
        '<div class="wrap">\n'
        f'<nav>{"".join(nav)}</nav>\n'
        f'<main>{"".join(body)}</main>\n'
        "</div>\n"
        f"<script>{JS}</script>\n"
    )
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        fh.write(out)
    print(f"wrote {os.path.normpath(OUT)}  ({len(out) // 1024} KB, "
          f"{len(body)} sections)")


if __name__ == "__main__":
    build()
