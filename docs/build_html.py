"""Build the CasparVP HTML manual from `docs/**/*.md`.

WHY A BUILD RATHER THAN AUTHORING HTML. The markdown is the source of truth: it diffs, it reviews
in a pull request, and it changes in the same commit as the code it describes -- which is the rule
that stops documentation lagging. HTML authored by hand would be a second copy of every claim, and
this tree has already paid for those: an audit on 2026-08-26 found four documented claims that had
outlived their code, and the sweep on 2026-08-27 found roughly a hundred more.

OUTPUT: five self-contained pages in `docs/`, one per audience, plus an index.

    index.html         what each section is for, and how to pick one
    features.html      state, decisions and measured numbers  (docs/features)
    guides.html        how to operate the thing               (docs/guides)
    architecture.html  why it is shaped this way              (docs/architecture)
    reference.html     plans, audits and retired docs         (docs/plans, audits, deprecated)

That split is not cosmetic. It is the same separation `docs/README.md` draws, and the reason it
exists is that a plan read as a description of behaviour is the single way this tree has most often
misled a reader. Putting `plans/` behind a page labelled *history* is part of the fix.

WHAT IT DELIBERATELY DOES NOT DO:

  * No read-time JavaScript beyond a theme toggle, and no external assets. Every page opens from the
    filesystem with no server and no network. Mermaid diagrams are rendered to **inline SVG at build
    time**, so a diagram is a picture in the file rather than a script that has to run.
  * No search framework. Ctrl+F over one page per audience is the search, which is why the pages are
    per-folder rather than per-document.
  * No editing of the markdown to suit the renderer. If a document renders badly the document is
    wrong, not this script.

CROSS-FOLDER LINKS ARE REWRITTEN, and that is the part that makes this coherent rather than five
disconnected dumps. `../guides/SPOUT.md` in a features doc becomes `guides.html#spout`. A link this
script cannot resolve is reported at the end rather than silently emitted as a dead `.md` href --
the markdown tree has zero broken links and the HTML must not introduce any.

Run:  python docs/build_html.py
      python docs/build_html.py --watch          (rebuild whenever a .md changes)
      python docs/build_html.py --no-mermaid     (skip rendering; keep the fenced source)
"""
import hashlib
import os
import re
import shutil
import subprocess
import sys

try:
    import markdown
except ImportError:                                     # pragma: no cover
    sys.exit("needs `python -m pip install markdown` (pure Python, no build step)")

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "_mermaid_cache")

# ── The pages, and the order documents appear on each ────────────────────────────────────
#
# Groups are the same headings `docs/README.md` uses, so the nav and the index agree with the
# folder's own README rather than inventing a second taxonomy.
PAGES = [
    (".", "overview.html", "Overview",
     "The doc tree's own README: what each folder is for and the rule that keeps them apart.",
     [("Start here", ["README.md"])]),

    ("features", "features.html", "Features",
     "State, decisions and measured numbers — what exists and how well it is verified.",
     [("Start here", ["README.md"]),
      ("Colour", ["colour-grading-and-ocio.md", "led-calibration.md"]),
      ("Projection and virtual production",
       ["projection-and-icvfx.md", "previz.md", "camera-tracking.md"]),
      ("GPU pipeline",
       ["vulkan-mixer.md", "vulkan-output.md", "cuda-prores.md", "cuda-notchlc.md", "hap.md",
        "gstreamer.md", "isf-and-openfx.md", "spout.md"]),
      ("Rewritten upstream modules",
       ["ffmpeg-producer-and-consumer.md", "decklink-output.md", "screen-consumer.md",
        "image-consumer-and-producer.md", "html-gpu-direct.md"]),
      ("Signal, sync and control",
       ["ltc-timecode.md", "dmx-sacn-artnet.md", "keyframes.md", "cluster-sync.md", "replay.md",
        "remotewall.md", "portaudio.md"]),
      ("Template", ["_TEMPLATE.md"])]),

    ("guides", "guides.html", "Guides",
     "How to operate it — commands, configuration, syntax and the traps that come with them.",
     [("Start here", ["README.md", "OPERATIONS_GUIDE.md"]),
      ("Colour and HDR",
       ["COLOR_GRADING.md", "HDR_GUIDE.md", "OCIO_USER_GUIDE.md", "LED_CALIBRATION.md",
        "IMAGE_EFFECTS.md", "MIXER_SHAPE.md"]),
      ("Virtual production",
       ["VIRTUAL_PRODUCTION_FEATURES.md", "PROJECTION_CALIBRATION.md", "PREVIZ_3D_MODULE.md",
        "CAMERA_TRACKING.md", "KEYFRAMES.md"]),
      ("Playback and recording",
       ["PLAYBACK_AND_RECORDING_GUIDE.md", "PIPELINE_EFFICIENCY_GUIDE.md",
        "CUDA_PRORES_OPERATION_GUIDE.md", "HAP_PLAYBACK.md", "GSTREAMER_GUIDE.md",
        "REPLAY_MODULE_USAGE.md"]),
      ("Outputs",
       ["DECKLINK_OUTPUT.md", "VULKAN_OUTPUT.md", "SPOUT.md", "DMX_LIGHTING.md",
        "LTC_TIMECODE.md", "PORTAUDIO_MODULE.md"]),
      ("Plug-ins and remote sources",
       ["ISF_USER_AND_SHADER_GUIDE.md", "OPENFX_USER_AND_PLUGIN_GUIDE.md",
        "REMOTEWALL_MODULE.md", "CLUSTER_SYNC.md"])]),

    ("architecture", "architecture.html", "Architecture",
     "Why it is shaped this way — read from the source, and the source is the authority.",
     [("Start here", ["README.md"]),
      ("The mixers",
       ["VULKAN_MIXER_IMPLEMENTATION.md", "UPSTREAM_VULKAN_COMPARISON.md",
        "OCIO_INTEGRATION_STUDY.md"]),
      ("GPU interop and output",
       ["GPU_INTEROP_ARCHITECTURE.md", "GPU_CODEC_HANDOFF.md", "VULKAN_OUTPUT.md",
        "DECKLINK_GPU_DIRECT_OUTPUT.md"]),
      ("Codecs and decode",
       ["CUDA_PRORES_IMPLEMENTATION_GUIDE.md", "HAP_DECODE_ROUTES.md", "FFMPEG_8_MIGRATION.md"]),
      ("Everything else",
       ["OPENFX_IMPLEMENTATION.md", "CAMERA_TRACKING_TRANSFORM.md", "CLUSTER_SYNC_DESIGN.md"])]),

    (None, "reference.html", "Reference and history",
     "Plans, audits and retired documents. <strong>None of this describes the current build</strong> "
     "— read it as intent or as a dated record, never as behaviour.",
     [("Plans and studies — intent, not state", [("plans", None)]),
      ("Audits — dated snapshots", [("audits", None)]),
      ("Deprecated — superseded, kept for citation", [("deprecated", None)])]),
]

CSS = """
:root{
  --bg:#f7f7f8; --panel:#ffffff; --border:#d8d8dc; --text:#1c1c1e; --muted:#6a6a70;
  --title:#0b5aa8; --accent:#0b5aa8; --code-bg:#f0f0f2; --warn-bg:#fff6e5; --warn-br:#e0a83a;
  --nav-bg:#ffffff;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#17171a; --panel:#1e1e22; --border:#33333a; --text:#e6e6e8; --muted:#9a9aa2;
    --title:#67aef5; --accent:#67aef5; --code-bg:#26262c; --warn-bg:#2e2718; --warn-br:#8a6a20;
    --nav-bg:#202024;
  }
}
:root[data-theme="dark"]{
  --bg:#17171a; --panel:#1e1e22; --border:#33333a; --text:#e6e6e8; --muted:#9a9aa2;
  --title:#67aef5; --accent:#67aef5; --code-bg:#26262c; --warn-bg:#2e2718; --warn-br:#8a6a20;
  --nav-bg:#202024;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
     font:15px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
.wrap{display:flex;align-items:flex-start;}
nav{position:sticky;top:0;flex:0 0 300px;max-height:100vh;overflow-y:auto;padding:18px 14px 40px;
    background:var(--nav-bg);border-right:1px solid var(--border);}
nav .home{display:block;font-weight:700;color:var(--title);text-decoration:none;margin:0 8px 14px;}
nav h2{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);
       margin:18px 8px 6px;}
nav a{display:block;padding:4px 8px;border-radius:5px;color:var(--text);text-decoration:none;
      font-size:13.5px;}
nav a:hover{background:var(--code-bg);}
main{flex:1 1 auto;min-width:0;padding:28px 40px 90px;max-width:1000px;}
section{border-top:1px solid var(--border);padding-top:26px;margin-top:34px;}
section:first-of-type{border-top:none;margin-top:0;}
h1{font-size:27px;color:var(--title);margin:.2em 0 .5em;}
h2{font-size:21px;margin:1.5em 0 .5em;}
h3{font-size:17px;margin:1.3em 0 .4em;}
h4{font-size:15px;margin:1.2em 0 .3em;color:var(--muted);}
a{color:var(--accent);}
code{background:var(--code-bg);padding:.12em .34em;border-radius:4px;font-size:.9em;
     font-family:ui-monospace,SFMono-Regular,Consolas,monospace;}
pre{background:var(--code-bg);padding:12px 14px;border-radius:7px;overflow-x:auto;}
pre code{background:none;padding:0;}
blockquote{margin:1em 0;padding:.6em 1em;background:var(--warn-bg);
           border-left:4px solid var(--warn-br);border-radius:0 5px 5px 0;}
blockquote p:first-child{margin-top:0} blockquote p:last-child{margin-bottom:0}
table{border-collapse:collapse;margin:1em 0;display:block;overflow-x:auto;max-width:100%;}
th,td{border:1px solid var(--border);padding:6px 10px;text-align:left;vertical-align:top;}
th{background:var(--code-bg);}
img{max-width:100%;height:auto;border-radius:6px;}
.mermaid-svg{margin:1em 0;overflow-x:auto;}
.mermaid-svg svg{max-width:100%;height:auto;}
hr{border:none;border-top:1px solid var(--border);margin:2em 0;}
#theme{position:fixed;right:16px;bottom:16px;z-index:9;background:var(--panel);
       border:1px solid var(--border);color:var(--text);border-radius:20px;padding:7px 14px;
       cursor:pointer;font-size:13px;}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(255px,1fr));gap:16px;margin:1.6em 0;}
.card{border:1px solid var(--border);border-radius:9px;padding:16px 18px;background:var(--panel);}
.card h3{margin:0 0 .35em;}
.card a{text-decoration:none;font-weight:600;}
.card p{margin:.3em 0 0;color:var(--muted);font-size:13.5px;}
.count{color:var(--muted);font-size:12px;}
@media (max-width:900px){
  .wrap{flex-direction:column} nav{position:static;flex:none;width:100%;max-height:none;
        border-right:none;border-bottom:1px solid var(--border);}
  main{padding:20px 18px 70px;}
}
"""

JS = """
(function(){
  var b=document.getElementById('theme');
  if(!b)return;
  var k='casparvp-docs-theme';
  var s=localStorage.getItem(k);
  if(s)document.documentElement.setAttribute('data-theme',s);
  b.addEventListener('click',function(){
    var cur=document.documentElement.getAttribute('data-theme');
    var mq=window.matchMedia('(prefers-color-scheme: dark)').matches;
    var next=(cur? cur : (mq?'dark':'light'))==='dark' ? 'light':'dark';
    document.documentElement.setAttribute('data-theme',next);
    localStorage.setItem(k,next);
  });
})();
"""


def slug(name):
    """A stable anchor for a document, from its filename."""
    return re.sub(r"[^a-z0-9]+", "-", os.path.splitext(name)[0].lower()).strip("-")


def first_title(text, fallback):
    m = re.search(r"^#\s+(.+)$", text, re.M)
    if not m:
        return fallback
    t = m.group(1).strip()
    t = re.sub(r"`([^`]*)`", r"\1", t)                   # code spans read badly in a nav
    return t


# ── Mermaid → inline SVG, at build time ──────────────────────────────────────────────────

def render_mermaid(source, enabled):
    """Return inline SVG for a mermaid block, or None to keep the fenced source.

    Cached by content hash under `_mermaid_cache/`, which is committed: it makes the build
    reproducible offline, and an SVG is text so it reviews like anything else here.
    """
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    cached = os.path.join(CACHE, digest + ".svg")
    if os.path.exists(cached):
        with open(cached, encoding="utf-8") as fh:
            return fh.read()
    if not enabled:
        return None
    os.makedirs(CACHE, exist_ok=True)
    src = os.path.join(CACHE, digest + ".mmd")
    with open(src, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(source)
    exe = shutil.which("npx")
    if not exe:
        return None
    try:
        subprocess.run([exe, "--yes", "@mermaid-js/mermaid-cli@latest",
                        "-i", src, "-o", cached, "-b", "transparent"],
                       capture_output=True, timeout=300, check=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None
    finally:
        if os.path.exists(src):
            os.remove(src)
    if not os.path.exists(cached):
        return None
    with open(cached, encoding="utf-8") as fh:
        return fh.read()


MERMAID_RE = re.compile(r"^```mermaid[ \t]*\n(.*?)^```[ \t]*$", re.S | re.M)


def extract_mermaid(text, enabled, stats):
    """Replace mermaid fences with placeholders; return (text, {placeholder: svg})."""
    blocks = {}

    def sub(m):
        svg = render_mermaid(m.group(1), enabled)
        if svg is None:
            stats["unrendered"] += 1
            return m.group(0)                            # leave the source visible
        stats["rendered"] += 1
        key = "MERMAIDBLOCK%dENDMERMAID" % len(blocks)
        # Strip the XML prologue and any fixed width so it scales in the page.
        svg = re.sub(r"^<\?xml[^>]*\?>\s*", "", svg.strip())
        blocks[key] = '<div class="mermaid-svg">%s</div>' % svg
        return "\n\n" + key + "\n\n"

    return MERMAID_RE.sub(sub, text), blocks


# ── Link and image rewriting ─────────────────────────────────────────────────────────────

def build_link_map():
    """`folder/FILE.md` -> `page.html#slug`, for every document the site includes."""
    out = {}
    for folder, page, _t, _d, groups in PAGES:
        for _group, entries in groups:
            for entry in entries:
                if isinstance(entry, tuple):             # a whole folder
                    sub = entry[0]
                    d = os.path.join(HERE, sub)
                    if not os.path.isdir(d):
                        continue
                    for f in sorted(os.listdir(d)):
                        if f.endswith(".md"):
                            out[f"{sub}/{f}"] = f"{page}#{sub}-{slug(f)}"
                else:
                    # normpath so the docs-root page ("." folder) keys as `README.md`, which is
                    # what `../README.md` from inside a folder resolves to.
                    key = os.path.normpath(os.path.join(folder, entry)).replace("\\", "/")
                    out[key] = f"{page}#{slug(entry)}"
    return out


ANY_LINK_RE = re.compile(r"\]\(([^)\s]+?)((?:#[^)]*)?)\)")


def rewrite_links(text, folder, link_map, unresolved):
    """Point every markdown link at its place in the built site.

    ONE PASS, deliberately. An earlier version rewrote `.md` targets and then re-rooted every
    remaining relative path, which re-rewrote the results of the first pass and pushed
    `../CLAUDE.md` to `CLAUDE.md`. Each target is now classified once and rewritten once.
    """
    def sub(m):
        target, frag = m.group(1), m.group(2)
        if target.startswith(("http://", "https://", "mailto:", "#", "data:")):
            return m.group(0)

        # An image: the pages sit in docs/, and docs/images/ is where they are.
        if target.startswith("images/") or "/images/" in target:
            i = target.index("images/")
            return "](%s%s)" % (target[i:], frag)

        norm = os.path.normpath(os.path.join(folder or "", target)).replace("\\", "/")

        # A document the site includes -> its section anchor.
        if norm in link_map:
            return "](%s)" % link_map[norm]

        # Anything else that is relative -- source files, scripts, a folder, a .md outside
        # the site. Re-root from docs/<folder>/ to docs/ so it still resolves on disk, which
        # is the whole value of an architecture doc pointing at the code.
        if target.startswith("."):
            resolved = os.path.normpath(os.path.join(HERE, folder or "", target))
            return "](%s%s)" % (os.path.relpath(resolved, HERE).replace("\\", "/"), frag)

        if target.endswith(".md"):
            unresolved.append(f"{folder or '.'}/{target}")
            return "](#)"
        return m.group(0)

    return ANY_LINK_RE.sub(sub, text)


def demote_headings(text):
    """The document's own H1 becomes the section heading, so shift the rest down one."""
    lines, out, first = text.split("\n"), [], True
    fenced = False
    for line in lines:
        if line.startswith("```"):
            fenced = not fenced
        if not fenced and line.startswith("#"):
            if line.startswith("# ") and first:
                first = False
                continue                                  # the section header carries it
            line = "#" + line
        out.append(line)
    return "\n".join(out)


# ── Page assembly ────────────────────────────────────────────────────────────────────────

def page_shell(title, nav_html, body_html, home=True):
    return (
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">\n"
        f"<title>{title} — CasparVP</title>\n"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n"
        f"<style>{CSS}</style></head><body>\n"
        f"<div class=\"wrap\">{nav_html}<main>{body_html}</main></div>\n"
        "<button id=\"theme\">theme</button>\n"
        f"<script>{JS}</script>\n</body></html>\n"
    )


def build_page(spec, md, link_map, mermaid_enabled, report):
    folder, page, title, desc, groups = spec
    nav = ['<a class="home" href="index.html">← CasparVP docs</a>']
    body = [f"<h1>{title}</h1><p class=\"count\">{desc}</p>"]

    for group, entries in groups:
        expanded = []
        for entry in entries:
            if isinstance(entry, tuple):
                sub = entry[0]
                d = os.path.join(HERE, sub)
                if os.path.isdir(d):
                    expanded += [(sub, f) for f in sorted(os.listdir(d)) if f.endswith(".md")]
            else:
                expanded.append((folder, entry))
        if not expanded:
            continue
        nav.append(f"<h2>{group}</h2>")
        for sub, name in expanded:
            path = os.path.join(HERE, sub, name)
            if not os.path.exists(path):
                report["missing"].append(f"{sub}/{name}")
                continue
            with open(path, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
            sid = (f"{sub}-{slug(name)}" if folder is None else slug(name))
            doc_title = first_title(text, name)
            text, blocks = extract_mermaid(text, mermaid_enabled, report)
            text = rewrite_links(text, sub, link_map, report["unresolved"])
            text = demote_headings(text)
            md.reset()
            html = md.convert(text)
            for key, svg in blocks.items():
                html = html.replace(f"<p>{key}</p>", svg).replace(key, svg)
            nav.append(f'<a href="#{sid}">{doc_title}</a>')
            body.append(f'<section id="{sid}"><h1>{doc_title}</h1>{html}</section>')
            report["docs"] += 1

    out = os.path.join(HERE, page)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(page_shell(title, f'<nav>{"".join(nav)}</nav>', "".join(body)))
    return out


def build_index(report):
    cards = []
    for folder, page, title, desc, groups in PAGES:
        n = 0
        for _g, entries in groups:
            for e in entries:
                if isinstance(e, tuple):
                    d = os.path.join(HERE, e[0])
                    n += len([f for f in os.listdir(d) if f.endswith(".md")]) if os.path.isdir(d) else 0
                else:
                    n += 1
        cards.append(
            f'<div class="card"><h3><a href="{page}">{title}</a></h3>'
            f'<p>{desc}</p><p class="count">{n} documents</p></div>')

    body = f"""<h1>CasparVP documentation</h1>
<p>A fork of CasparCG Server with virtual-production work: ACES colour management, a full grading
chain, a Vulkan mixer, GPU-direct paths, 360&deg;/curved projection, DMX/Art-Net and extra
consumers.</p>

<div class="cards">{"".join(cards)}</div>

<h2>Which section answers which question</h2>
<p>The split is deliberate, and it is the one thing to understand before reading anything here.</p>
<table>
<tr><th>You want to know</th><th>Read</th></tr>
<tr><td>Does this exist, and how well is it verified?</td><td><a href="features.html">Features</a> &mdash; state, decisions and every measured number</td></tr>
<tr><td>How do I drive it?</td><td><a href="guides.html">Guides</a> &mdash; commands, config elements, syntax, traps</td></tr>
<tr><td>Why is it built like this?</td><td><a href="architecture.html">Architecture</a> &mdash; read from the source, and the source wins</td></tr>
<tr><td>What was planned, or what did an audit find?</td><td><a href="reference.html">Reference and history</a></td></tr>
</table>

<blockquote><p><strong>A plan is not a description of behaviour.</strong> Everything under
<a href="reference.html">Reference and history</a> is intent or a dated snapshot. That distinction is
the single way this documentation has most often misled a reader, which is why those documents sit
behind their own page and carry a status line naming what shipped.</p></blockquote>

<h2>How much of this is checked mechanically</h2>
<p>Sixteen standing checks in the test harness assert what can be asserted, because prose goes stale
and nobody notices: every declared module path exists, all 91 fork-specific AMCP commands appear
somewhere, every battery is named in a document and every battery a document names exists, no
broken links, the required header fields are present, config defaults match the code, and each
plan&rsquo;s status is checked against its own declared falsifier.</p>
<p>What they cannot check is whether a paragraph is <em>true</em>. That still needs a reader, and a
sweep on 2026-08-27 found roughly a hundred claims the code contradicted &mdash; including three
settings documented as working that are compiled out, a command with no documented syntax at all,
and worked examples whose parameters were off by one position.</p>

<p class="count">Built by <code>docs/build_html.py</code> from the markdown in
<code>docs/</code>. The markdown is the source of truth; these pages are generated, and every
diagram is inline SVG rendered at build time, so a page opens from the filesystem with no server
and no network.</p>"""
    out = os.path.join(HERE, "index.html")
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(page_shell("Documentation", "", body))
    return out


def sources():
    """Every file a rebuild depends on: the markdown, and this script."""
    out = [os.path.abspath(__file__)]
    for dp, dn, fns in os.walk(HERE):
        dn[:] = [d for d in dn if d not in ("_mermaid_cache", "images", "diagrams")]
        out += [os.path.join(dp, f) for f in fns if f.endswith(".md")]
    return out


def fingerprint():
    """mtime+size per source. Cheap enough to poll: ~100 stat calls."""
    fp = {}
    for p in sources():
        try:
            st = os.stat(p)
            fp[p] = (st.st_mtime, st.st_size)
        except OSError:
            pass
    return fp


def watch(mermaid_enabled):
    """Rebuild on every change until interrupted.

    A poll rather than a filesystem-watch API: 100 stat calls is nothing next to the 2.4 s
    rebuild, and it needs no third-party package -- which is the same reason this script uses
    `markdown` and nothing else. A full rebuild rather than an incremental one for the same
    reason: at 2.4 s warm, the bookkeeping to work out which page changed would cost more
    than it saves and could get the answer wrong, which an incremental build does silently.
    """
    import time
    # Line-buffer stdout for the whole session. Python block-buffers when piped, so
    # `--watch | tee build.log` showed nothing at all until the process died -- and a watch
    # tool whose progress is invisible is worse than no watch tool.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass
    print("watching docs/**/*.md — Ctrl-C to stop")
    last = None
    while True:
        fp = fingerprint()
        if fp != last:
            if last is not None:
                changed = sorted(
                    os.path.relpath(p, HERE) for p in set(fp) ^ set(last)
                ) or sorted(os.path.relpath(p, HERE) for p in fp
                            if last.get(p) != fp[p])
                print(f"\n[{time.strftime('%H:%M:%S')}] changed: "
                      + ", ".join(changed[:4]) + (" …" if len(changed) > 4 else ""))
            t0 = time.time()
            rc = build_all(mermaid_enabled, quiet=last is not None)
            print(f"[{time.strftime('%H:%M:%S')}] "
                  + ("rebuilt" if rc == 0 else "rebuilt WITH PROBLEMS — see above")
                  + f" in {time.time() - t0:.1f}s")
            last = fp
        time.sleep(1.0)


def build_all(mermaid_enabled, quiet=False):
    md = markdown.Markdown(extensions=["tables", "fenced_code", "attr_list", "sane_lists",
                                       "toc", "md_in_html"])
    link_map = build_link_map()
    report = {"docs": 0, "rendered": 0, "unrendered": 0, "missing": [], "unresolved": []}

    written = [build_index(report)]
    for spec in PAGES:
        written.append(build_page(spec, md, link_map, mermaid_enabled, report))

    if not quiet:
        total = 0
        for p in written:
            kb = os.path.getsize(p) // 1024
            total += kb
            print(f"wrote {os.path.relpath(p, HERE):<20} {kb:>5} KB")
        print(f"\n{report['docs']} documents, {total} KB total")
        print(f"mermaid: {report['rendered']} rendered to inline SVG, "
              f"{report['unrendered']} left as source")
    if report["missing"]:
        print(f"MISSING ({len(report['missing'])}): {report['missing']}")
    if report["unresolved"]:
        uniq = sorted(set(report["unresolved"]))
        print(f"UNRESOLVED LINKS ({len(uniq)}) — these became '#':")
        for u in uniq:
            print(f"  {u}")
        return 1
    return 0


def main():
    mermaid_enabled = "--no-mermaid" not in sys.argv
    if "--watch" in sys.argv:
        try:
            watch(mermaid_enabled)
        except KeyboardInterrupt:
            print("\nstopped")
        return 0
    return build_all(mermaid_enabled)


if __name__ == "__main__":
    sys.exit(main())
