"""Build the executive brief: one page per capability, print-ready A4 landscape.

WHO IT IS FOR. Supervisors and heads of department deciding where effort and budget go --
not operators and not engineers. So every page answers four questions in the same order:
what it does in plain language, where it earns its keep, what it would take to exploit
fully, and how the market solves the same problem.

WHY A GENERATOR rather than hand-written HTML. The content is a table of facts, and a table
of facts hand-carried into markup is a second copy that drifts -- the failure this doc tree
has paid for repeatedly. The readiness column in particular comes straight from what the
harness measured, so it has to be editable as data.

READINESS IS NOT MARKETING. `proven` means a battery gates it and was run; `partial` means
some of it is measured and the doc says which; `untested` means no battery drives it at all.
An untested capability is not a broken one -- most of these work -- but a supervisor asked
to fund something deserves to know which claims rest on a measurement.

Run:   python docs/build_exec_brief.py
Out:   docs/EXECUTIVE_BRIEF.html   (+ .pdf if Chrome is found)
"""
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_HTML = os.path.join(HERE, "EXECUTIVE_BRIEF.html")
OUT_PDF = os.path.join(HERE, "EXECUTIVE_BRIEF.pdf")

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]

# ── Readiness vocabulary ─────────────────────────────────────────────────────────────────
PROVEN = ("proven", "#1a6b1a", "#6fbf6f")
PARTIAL = ("partly proven", "#7a5c00", "#d0a02a")
UNTESTED = ("not yet measured", "#5a3a1a", "#c88a4a")

# ── The pages ────────────────────────────────────────────────────────────────────────────
#
# `gap` is the honest answer to "what else would it take" and comes from the coverage audit,
# not from a wish list. `next` is the one actionable line the summary matrix shows, and it is a
# FIELD rather than the first sentence of `gap` -- deriving it produced rows reading "Two
# specifics." and "Measured and honest." under a column headed "the first thing it needs", which
# looks like information and is not. `market` names how the problem is solved elsewhere, at a level of
# detail worth standing behind -- see the caveat page at the end.
FEATURES = [
    dict(
        n=1, title="Colour management to a published standard",
        lead="ACES and OpenColorIO, applied per layer and per output, instead of a house recipe.",
        img="ocio_stages.png",
        buys=[
            "A camera's log footage, a graphics fill and a stock clip all land in the same "
            "reference space before anything is graded, so they match without eyeballing.",
            "The deliverable is describable: 'ACES 2.0, Rec.2100 PQ at 1000 nits' is a "
            "configuration here rather than a conversation.",
            "The same transforms the post house already uses — the tooling is shared with "
            "Nuke, Resolve, Maya and Blender.",
        ],
        earns="A shoot mixing an ARRI camera feed, pre-rendered plates and live graphics on the "
              "same wall. Without a reference space, matching them is trial and error that has to "
              "be redone whenever a source changes.",
        gap="Nothing outstanding for the built-in path — it is gated at 1 LSB on both renderers "
            "and was re-measured this week. The OCIO half is measured against OCIO's own CPU "
            "processor; no one has yet compared an OCIO HDR view against the built-in PQ path "
            "end to end.",
        market="ACES is the AMPAS standard and OpenColorIO is the ASWF library behind most "
               "post-production colour tooling. Media servers differ widely here: some expose a "
               "fixed pipeline, some none at all. Using the standards rather than a private "
               "pipeline is the point.",
        next='Nothing blocking — compare an OCIO HDR view against the built-in PQ path',
        status=PROVEN,
        evidence="conformance 100/100 conversions within 1 LSB, worst 0.55 · grading 48/48 · "
                 "ocio 18/18 · both renderers",
    ),
    dict(
        n=2, title="A full grading chain, on the frame path",
        lead="Twenty-six operators — CDL, LUTs, curves, secondary qualifier, windowed nodes.",
        img="grading_controls.png",
        buys=[
            "A look can be built and adjusted at the server rather than baked into media, so a "
            "note from the director does not mean a re-render.",
            "ASC CDL files from the grading suite load directly, so the on-set look and the "
            "post look are the same numbers.",
            "Windowed nodes mean a correction can be confined to part of the frame — a hot "
            "practical, one panel of a wall.",
        ],
        earns="Live correction during a shoot: the wall reads too warm on camera, and the fix is "
              "one command against the channel instead of a media turnaround.",
        gap="The operators are measured individually and in stacks. What is not covered is the "
            "tweened form of each command — animating a grade over time is exercised by nothing — "
            "and the windowed-node feature is a prototype with one operation.",
        market="Comparable to the correction layers in disguise and Pixera, and to the CDL "
               "support in Resolume. The distinguishing part here is that it sits in the same "
               "colour-managed pipeline as the output transform rather than beside it.",
        next='Cover the animated form of each grading command',
        status=PROVEN,
        evidence="grading 48/48 both renderers, neutrals exactly 0.00 · cdl-file · grade-window "
                 "byte-identical on both",
    ),
    dict(
        n=3, title="LED wall calibration",
        lead="A measured correction applied once to the whole channel, after compositing.",
        img="led_tiling.png",
        buys=[
            "The wall's own colour error is corrected in the server, so every consumer attached "
            "to the channel gets the corrected picture.",
            "Takes the calibration a camera-based solve produces — the same file the LED "
            "vendor's own tooling would consume.",
            "Applied once to the composite rather than per layer, so it cannot drift between "
            "sources.",
        ],
        earns="A volume whose panels have aged unevenly, or two panel batches that do not match. "
              "The processor can only do so much; this closes the rest without touching content.",
        gap="Verified on two of five consumer types. Panel- or tile-level addressing is "
            "deliberately not attempted — that stays the LED processor's job — and the "
            "colorimetric solve itself is not ours.",
        market="OpenVPCal (Netflix) is the open-source camera-based solve this consumes. "
               "Brompton Tessera and Megapixel Helios do calibration in the processor. The two "
               "are complementary: the processor corrects the panel, this corrects what is sent "
               "to it.",
        next='Verify on the remaining three consumer types',
        status=PROVEN,
        evidence="calibration 32/32 within 1 LSB · proven to be channel-master rather than "
                 "per-layer, 17.9 LSB apart · verified on a second consumer",
    ),
    dict(
        n=4, title="In-camera VFX — the inner frustum",
        lead="The region the camera sees is rendered for the camera; the rest lights the scene.",
        img="icvfx_frustum.png",
        buys=[
            "The camera sees correct perspective and parallax, which is the whole premise of "
            "shooting against an LED volume rather than a green screen.",
            "The outer region can be dimmed or tinted independently, so the wall keeps lighting "
            "the set without polluting the shot.",
            "Per-channel colour trim on inner and outer regions separately.",
        ],
        earns="Any LED-volume shoot with a moving camera. Without it the wall is a backdrop; "
              "with it the wall is the set.",
        gap="This is the largest gap in the brief and the most worth closing. Only the "
            "per-channel gain is covered by a test. The mask geometry, the feather, the "
            "inner-frustum reprojection and every animated form are driven by nothing — and an "
            "audit found a real colour defect in exactly this area precisely because nothing "
            "drove it.",
        market="Unreal Engine's nDisplay and disguise are the reference implementations and both "
               "are mature. This is not a claim to match them; it is a claim that the capability "
               "exists inside a playout server we control, which changes what a small rig costs.",
        next='Drive the mask, feather and reprojection in test — not just the colour gain',
        status=PARTIAL,
        evidence="icvfx-parity: gain exchange, worst 0 LSB, both renderers · everything else in "
                 "the feature: no coverage",
    ),
    dict(
        n=5, title="Projection onto shapes that are not flat",
        lead="Cylinders, domes, fisheye and corner-pinned surfaces, with soft-edge blending.",
        img="projection_geometry.png",
        buys=[
            "Content designed flat can be shown on a curved wall without pre-warping it, so the "
            "same media works on more than one venue.",
            "Edge blending across several outputs, so a wide surface reads as one picture.",
            "Lens distortion correction, so a projector's own optics stop being a content problem.",
        ],
        earns="A curved cyclorama or a dome where the geometry changes between venues. Pre-warping "
              "media per venue is the alternative, and it does not survive a last-minute change.",
        gap="The geometry is measured. What is not: the animated forms of these commands, and the "
            "interaction between a curved warp and a blend mask on the same layer.",
        market="Projection warping is the core of Pixera, disguise and Watchout. This does not "
               "replace a dedicated warping suite for a complex install, but it removes the need "
               "for one on the many jobs that are a single curve.",
        next='Cover the animated commands, and warp combined with a blend mask',
        status=PARTIAL,
        evidence="geometry · blend-mask · mixer-parity across six rasters · the tweened forms are "
                 "not driven",
    ),
    dict(
        n=6, title="Camera tracking",
        lead="Eighteen commands turning a tracking feed into the render camera, with a lens model.",
        img="tracking_coords.png",
        buys=[
            "Consumes what the tracking vendors already output, so the choice of tracker stays "
            "open.",
            "A lens profile maps zoom and focus to field of view and distortion, so a real lens "
            "and the virtual camera agree.",
            "A survey-based world alignment, so the tracker's coordinate system and the stage's "
            "agree without trial and error.",
        ],
        earns="Any moving-camera virtual production. It is the input the inner frustum needs to "
              "be worth anything.",
        gap="No test drives any of the eighteen commands, and none has been run against real "
            "tracking hardware here. The composition order was undocumented until this week — it "
            "is now written down, read from the source, and verified by nothing.",
        market="Mo-Sys, stYpe, Vicon, OptiTrack and Ncam supply the tracking; this is the "
               "consumer of it. The comparable integration in disguise and Unreal is mature and "
               "supported — that is the gap to be honest about.",
        next='A trial against real tracking hardware, then a test for the 18 commands',
        status=UNTESTED,
        evidence="18 commands registered · no battery · no hardware trial recorded",
    ),
    dict(
        n=7, title="3D pre-visualisation of the stage",
        lead="Load the venue, map channel output onto the screens, and look at it before load-in.",
        img="stage_layout.png",
        buys=[
            "A projection or LED design can be checked before anyone is on site, which is where "
            "the expensive surprises live.",
            "Screens, presets and camera positions are addressable by name at runtime, so a "
            "rehearsal can move between looks quickly.",
            "The same ICVFX state the live path uses, so what is previewed is what will run.",
        ],
        earns="Quoting and de-risking a job. A design error found in previz costs an afternoon; "
              "the same error found at load-in costs a day of crew.",
        gap="Thirteen commands and no coverage of any kind — the largest untested surface here. "
            "It also renders through one graphics API on both renderers, and the consequence of "
            "that for colour has not been measured.",
        market="disguise's Designer previz is the benchmark and is a major part of why that "
               "platform is standard. This is a smaller capability in the same shape.",
        next='Any coverage at all — 13 commands are driven by nothing',
        status=UNTESTED,
        evidence="13 commands registered · no battery references PREVIZ at all",
    ),
    dict(
        n=8, title="Playback that never touches the CPU",
        lead="Decode on the GPU and hand the picture to the mixer without a host round trip.",
        img="recording_decode_paths.png",
        buys=[
            "More layers and more channels from the same machine, because the bottleneck was "
            "never the decoder — it was copying frames through main memory.",
            "Four decode routes, chosen per source, so an unusual codec does not force the whole "
            "channel onto the slow path.",
            "Ten-bit sources stay ten-bit through the transfer, so the saving is not paid for in "
            "precision.",
        ],
        earns="Any job whose channel count is set by hardware cost. This is the feature that "
              "changes how many machines a show needs.",
        gap="Measured and honest. What remains is a decision rather than work: the newest route "
            "(video decode through Vulkan) has no dedicated test, and the A/B between it and the "
            "established route has not been run.",
        market="NVDEC and NVIDIA GPUDirect are the underlying vendor technology, available to "
               "everyone. What differs between servers is how much of the path avoids host "
               "memory; this fork's does so end to end for the supported codecs.",
        next='A dedicated test for the newest decode route, and an A/B against the established one',
        status=PROVEN,
        evidence="decode-cost with engagement required, so a silently-declined route cannot be "
                 "reported as a saving · gpu-direct-parity against a software reference",
    ),
    dict(
        n=9, title="Recording without stealing the render",
        lead="ISO recordings encoded on the GPU, from the composited frame already there.",
        img="recording_encode_paths.png",
        buys=[
            "Every channel can be recorded without halving how many channels the machine can "
            "play, because the frame is not copied back for the encoder.",
            "Broadcast-friendly formats — ProRes at several profiles, h.264/HEVC, and a "
            "purpose-built CUDA ProRes recorder.",
            "A capture-card bypass recorder for taking an SDI input straight to disk.",
        ],
        earns="Deliverables and compliance recording on a show that also needs its channels. The "
              "usual answer is a second machine.",
        gap="Well measured for picture and cost. The gap is operational: the quality setting that "
            "takes this from one recording to eight is raster-dependent and has to be chosen per "
            "job rather than left at a default.",
        market="NVENC is the vendor encoder everyone uses. Dedicated recorders (AJA, Blackmagic "
               "HyperDeck) do this in hardware and remain the right answer for guaranteed "
               "compliance recording; this removes the need for one per channel.",
        next='Choose the recording quality setting per raster rather than per default',
        status=PROVEN,
        evidence="encode-matrix and encode-parity across four codecs · iso-scaling for capacity",
    ),
    dict(
        n=10, title="GPU-native intermediate codecs",
        lead="ProRes, NotchLC and HAP decoded on the GPU, in the form the mixer wants.",
        img="feature_codec_handoff.png",
        buys=[
            "The codecs the design and post pipelines already produce, played back at high layer "
            "counts.",
            "HAP in five variants including the newest BC7 form, so content authored for VJ and "
            "installation tooling plays natively.",
            "NotchLC, so material coming out of Notch does not need transcoding.",
        ],
        earns="An installation or broadcast job whose content arrives as ProRes or HAP and must "
              "run many layers deep.",
        gap="Two specifics. Neither HAP nor NotchLC has ever been compared against a reference "
            "decoder — only against our other renderer, which cannot catch a fault both share. "
            "And the newest HAP variant has no test material at all.",
        market="HAP is Vidvox's open codec, standard in Resolume and VDMX. NotchLC is Notch's. "
               "ProRes is Apple's. Supporting all three on the GPU is what a modern media server "
               "is expected to do; the distinguishing part is doing it without a host copy.",
        next='Compare HAP and NotchLC against a reference decoder, and obtain BC7 test material',
        status=PARTIAL,
        evidence="prores-parity against FFmpeg's CPU decoder · HAP measured between renderers "
                 "only · NotchLC has no reference comparison",
    ),
    dict(
        n=11, title="SDI output straight from the GPU",
        lead="The composited frame packed and delivered to the card without a host round trip.",
        # NOT decklink/datapath.png -- that one is the engineering diagram, and names
        # cudaMemcpy, SSBOs and AVX2. Correct, and the wrong register for this reader.
        img="exec_multiport.png",
        buys=[
            "Lower latency and more headroom on the machine, for the same picture.",
            "Several SDI ports driven from one channel as one synchronised group, so a canvas "
            "wider than a single cable is not a special case.",
            "Correct colour and HDR signalling on the wire, including the ancillary-data form "
            "broadcast chains expect.",
        ],
        earns="Broadcast and large-format delivery, where SDI is not optional and the downstream "
              "chain trusts what the wire says about the picture.",
        gap="The single-port path is measured to the decimal over a real loopback. Multi-port is "
            "not driven by any test, and one operational interaction is worth knowing: the "
            "latency mode that sounds most like 'synchronise' switches the driver's own "
            "multi-port sync off.",
        market="This is Blackmagic DeckLink hardware, so the card is the same one other servers "
               "use. The difference is how the frame reaches it — via host memory, or not.",
        next='Drive the multi-port group in test',
        status=PROVEN,
        evidence="sdi-output over the 1→4 loopback, re-verified this week: 62.92 dB with a "
                 "placed sub-region, and every documented figure reproduced exactly",
    ),
    dict(
        n=12, title="Straight to the LED processor",
        lead="Present a channel to a display output — HDMI or DisplayPort — with no SDI in between.",
        img="exec_direct_display.png",
        buys=[
            "Removes a capture card and a converter from the signal path, with the cost and the "
            "failure points that go with them.",
            "Sync groups so several outputs present together, and EDID handling for processors "
            "that report a mode they do not want.",
            "Reports what it actually negotiated — surface format, colour space, HDR state — "
            "rather than what was asked for.",
        ],
        earns="An LED volume fed from display outputs, which is how most modern processors prefer "
              "to be driven.",
        gap="Two concrete blockers, both external. HDR over this path needs Windows 11 — the "
            "direct-scanout extension does not exist on Windows 10, and the machine falls back to "
            "a path with no HDR surface. And the genlock and EDID features are untested because "
            "they need hardware we have not driven.",
        market="disguise, Pixera and Unreal-based systems all output to processors this way; "
               "Brompton and Megapixel processors take HDMI/DP natively. This is table stakes for "
               "LED work rather than a differentiator — which is exactly why it matters that it "
               "exists here.",
        next='A Windows 11 machine with an HDR display; then genlock and EDID hardware',
        status=PARTIAL,
        evidence="vulkan-output-signalling 3/3 consistent · the HDR degradation on Windows 10 is "
                 "named by the test rather than hidden",
    ),
    dict(
        n=13, title="HDR that survives to the wire",
        lead="PQ and HLG rendered, and correctly labelled, on SDI and in files and streams.",
        img="exec_hdr.png",
        buys=[
            "HDR deliverables without a separate finishing step: the mastering-display numbers "
            "travel with the picture.",
            "Correct signalling on SDI, in recorded files and over SRT/UDP streams — the same "
            "four numbers, spelled the same way in each.",
            "Ingested HDR is read and reported, so an operator can see what arrived.",
        ],
        earns="Any HDR deliverable, and any LED volume being shot for an HDR finish.",
        gap="Two halves, differently mature. File, stream and SDI signalling are measured. The "
            "display-output half is blocked on the Windows 11 point on the previous page, and "
            "nothing yet reads back what a display actually received.",
        market="HDR10, PQ and HLG are open standards; the mastering-display metadata is the same "
               "block every finishing tool writes. Broadcast chains and LED processors both "
               "expect it, and getting the label wrong is a delivery failure rather than a "
               "picture one.",
        next='An instrument that reads back what a display actually received',
        status=PARTIAL,
        evidence="signalling on the card · ffprobe read-back per transport · display-side "
                 "read-back has no instrument",
    ),
    dict(
        n=14, title="More machines, one frame number",
        lead="Frame-accurate playback across servers, with commands scheduled to a target frame.",
        img="exec_cluster.png",
        buys=[
            "A wall too wide, or a show too heavy, for one machine stops being an architectural "
            "problem.",
            "Commands are stamped with the frame they should act on, so 'go' means the same "
            "instant on every node.",
            "Content drift between nodes is monitored rather than assumed.",
        ],
        earns="Large volumes and multi-surface venues — the jobs where the alternative is a "
              "bigger, more expensive single machine.",
        gap="It has never been run as a cluster. That needs a second machine and nothing else, "
            "which makes it the cheapest item on this list to move from 'written' to 'proven'. "
            "The parts a single machine can check are checked.",
        market="disguise's director/actor model and Unreal's nDisplay clustering are the "
               "references, both built on the same time-synchronisation standards. Using PTP "
               "rather than something bespoke means it can share a clock with the rest of the "
               "facility.",
        next='A second machine — no development needed',
        status=UNTESTED,
        evidence="the frame-number arithmetic is verified against its own model · no multi-machine "
                 "run exists",
    ),
    dict(
        n=15, title="Automation and lighting",
        lead="Keyframed mixer state, and house lighting driven from the picture itself.",
        img="exec_lighting.png",
        buys=[
            "Any of 184 mixer properties can be animated on a timeline — moves, grades, "
            "projection and ICVFX included — so a cue is data rather than an operator.",
            "Colour sampled from regions of the frame and sent as Art-Net or sACN, so practicals "
            "and cyc lights follow the content.",
            "Both are standard protocols, so nothing bespoke is needed at the lighting desk.",
        ],
        earns="A show where the wall and the room have to agree, and where repeatability between "
              "takes matters more than live operation.",
        gap="Neither is driven by a test. The lighting path has a battery; the keyframe system "
            "has none, and the projection and ICVFX fields it can animate were undocumented "
            "until this week.",
        market="Art-Net and sACN are what every lighting desk speaks; Resolume and similar tools "
               "offer comparable content-to-light features. Timeline automation is standard in "
               "media servers — the unusual part here is how much of the mixer is addressable.",
        next='A test for the keyframe system (8 commands, 184 fields)',
        status=PARTIAL,
        evidence="dmx battery covers the lighting transports · keyframes: 8 commands, 184 fields, "
                 "no coverage",
    ),
    dict(
        n=16, title="Fitting into the room",
        lead="Texture sharing, plug-in hosting, GStreamer pipelines and remote tile-wall input.",
        img="gstreamer_caspar_routes.png",
        buys=[
            "Frames shared with Notch, Resolume, TouchDesigner or Unreal on the same machine "
            "with no encode, no file and no network.",
            "Industry-standard effect plug-ins hosted directly, so a look built in a post tool "
            "can run live.",
            "GStreamer pipelines in and out, which covers the transports a broadcast facility "
            "already runs.",
        ],
        earns="Any job where this is one tool among several. The alternative is a capture card "
              "and a conversion between every pair of applications.",
        gap="Mostly a coverage gap rather than a capability one. Texture sharing has no test in "
            "either direction, and the plug-in hosts are unmeasured. GStreamer is measured.",
        market="Spout is the Windows standard for this and is what Resolume and TouchDesigner "
               "use; OpenFX is the Open Effects Association's plug-in standard, hosted by Resolve "
               "and Nuke; ISF is Vidvox's shader format. Speaking the existing standards is the "
               "whole feature.",
        next='Cover texture sharing in both directions, and the two plug-in hosts',
        status=PARTIAL,
        evidence="gstreamer 14/14 both renderers · spout, ISF and OpenFX: no coverage",
    ),
]


def _pill(status):
    label, bg, fg = status
    return f'<span class="pill" style="border-color:{fg};color:{fg}">{label}</span>'


def _feature_page(f):
    buys = "".join(f"<li>{b}</li>" for b in f["buys"])
    return f"""
<section class="page feature">
  <header class="phead">
    <div class="pnum">{f['n']:02d}</div>
    <div>
      <h2>{f['title']}</h2>
      <p class="lead">{f['lead']}</p>
    </div>
    <div class="pstatus">{_pill(f['status'])}</div>
  </header>
  <div class="pbody">
    <figure class="pfig"><img src="images/{f['img']}" alt=""></figure>
    <div class="pcol">
      <h3>What it buys us</h3>
      <ul>{buys}</ul>
      <h3>Where it earns its keep</h3>
      <p>{f['earns']}</p>
      <h3 class="warn">What it would take to use fully</h3>
      <p>{f['gap']}</p>
      <h3 class="mkt">How the market solves this</h3>
      <p class="mktp">{f['market']}</p>
    </div>
  </div>
  <footer class="pfoot"><span class="ev">Evidence</span> {f['evidence']}</footer>
</section>"""


def _matrix():
    rows = ""
    for f in FEATURES:
        label, _bg, fg = f["status"]
        rows += (f"<tr><td class='mn'>{f['n']:02d}</td><td>{f['title']}</td>"
                 f"<td style='color:{fg}'>{label}</td>"
                 f"<td class='mg'>{f['next']}</td></tr>")
    counts = {}
    for f in FEATURES:
        counts[f["status"][0]] = counts.get(f["status"][0], 0) + 1
    return rows, counts


CSS = """
@page { size: A4 landscape; margin: 0; }
/* --bg is EXACTLY the #1e1e1e the 42 diagrams in docs/images are drawn on, so a figure sits
   on the page rather than reading as a lighter grey card pasted onto it. Matching the deck to
   the existing palette is cheaper and more consistent than redrawing the diagrams. */
:root{
  --bg:#1e1e1e; --panel:#262629; --panel2:#2d2d31; --line:#3a3a40;
  --text:#e4e4e7; --muted:#9a9aa4; --title:#67aef5; --accent:#2f6bd0;
  --warn:#d0a02a; --mkt:#8fb8d8; --ok:#6fbf6f;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{background:var(--bg);color:var(--text);
  font:12px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  -webkit-print-color-adjust:exact;print-color-adjust:exact;}
.page{width:297mm;height:210mm;padding:12mm 13mm;position:relative;overflow:hidden;
  page-break-after:always;break-after:page;display:flex;flex-direction:column;
  background:var(--bg);}
.page:last-child{page-break-after:auto;break-after:auto}

/* ── cover ───────────────────────────────────────────────────────────── */
.cover{justify-content:center;background:
  radial-gradient(1300px 700px at 14% 8%, #223148 0%, var(--bg) 62%);}
.cover .cbody{display:flex;gap:14mm;align-items:flex-start}
.cover .cleft{flex:1 1 auto;min-width:0}
.cover .cright{flex:0 0 84mm;border-left:1px solid var(--line);padding-left:9mm}
.cover .kicker{color:var(--title);letter-spacing:.22em;text-transform:uppercase;
  font-size:10.5px;font-weight:700;margin-bottom:8mm}
.cover h1{font-size:37px;line-height:1.12;font-weight:700}
.cover h1 em{color:var(--title);font-style:normal}
.cover .sub{color:var(--muted);font-size:13.5px;margin-top:6mm;line-height:1.6}
.cover .meta{position:absolute;left:13mm;right:13mm;bottom:11mm;display:flex;
  justify-content:space-between;color:var(--muted);font-size:10px;
  border-top:1px solid var(--line);padding-top:4mm}
.cover .stats{display:flex;gap:11mm;margin-top:11mm;flex-wrap:wrap}
.cover .stat b{display:block;color:var(--title);font-size:25px;line-height:1}
.cover .stat span{color:var(--muted);font-size:10px}
.cover .ctitle{color:var(--muted);text-transform:uppercase;letter-spacing:.14em;
  font-size:9px;font-weight:700;margin-bottom:4.5mm}
.cover .toc{list-style:none}
.cover .toc li{font-size:10.5px;color:#c8c8d0;padding:1.35mm 0;line-height:1.3;
  border-bottom:1px solid #2e2e32;display:flex;gap:3.5mm}
.cover .toc .tn{color:#5c5c68;font-variant-numeric:tabular-nums;flex:0 0 6mm}

/* ── contents / matrix ───────────────────────────────────────────────── */
h1.ph{font-size:23px;color:var(--title);margin-bottom:2mm}
p.psub{color:var(--muted);font-size:12px;margin-bottom:6mm;max-width:230mm}
/* Padding is 1.5mm rather than 2.1mm because at 2.1mm the 16-row matrix ran 30px past the
   page and `overflow:hidden` CUT the last rows and the legend -- caught by check_overflow(),
   not by looking at it, which is the entire argument for that gate. */
table{width:100%;border-collapse:collapse;font-size:10px}
th,td{text-align:left;padding:1.5mm 3mm;border-bottom:1px solid var(--line);
  vertical-align:top}
th{color:var(--muted);text-transform:uppercase;letter-spacing:.09em;font-size:9px;
  border-bottom:1px solid #45454e}
td.mn{color:var(--muted);font-variant-numeric:tabular-nums;width:9mm}
td.mg{color:var(--muted);width:105mm}
.legend{display:flex;gap:9mm;margin-top:5mm;font-size:10px;color:var(--muted)}
.legend b{color:var(--text);font-weight:600}

.scopefig{flex:1 1 auto;display:flex;align-items:center;justify-content:center;min-height:0}
.scopefig img{max-width:100%;max-height:100%}

/* ── feature page ────────────────────────────────────────────────────── */
.phead{display:flex;gap:6mm;align-items:flex-start;border-bottom:1px solid var(--line);
  padding-bottom:4mm;margin-bottom:5mm}
.pnum{font-size:30px;font-weight:700;color:#3a3a44;line-height:.9;
  font-variant-numeric:tabular-nums}
.phead h2{font-size:21px;color:var(--title);line-height:1.15}
.phead .lead{color:var(--muted);font-size:12px;margin-top:1.5mm}
.pstatus{margin-left:auto;flex:0 0 auto;padding-top:1mm}
.pill{border:1px solid;border-radius:11px;padding:1.4mm 3.4mm;font-size:9px;
  font-weight:700;text-transform:uppercase;letter-spacing:.09em;white-space:nowrap}
.pbody{display:flex;gap:8mm;flex:1 1 auto;min-height:0}
.pfig{flex:0 0 170mm;display:flex;align-items:flex-start;justify-content:center}
.pfig img{max-width:100%;max-height:136mm}
.pcol{flex:1 1 auto;min-width:0}
.pcol h3{font-size:10.3px;text-transform:uppercase;letter-spacing:.1em;color:var(--title);
  margin:0 0 2.2mm}
.pcol h3.warn{color:var(--warn)} .pcol h3.mkt{color:var(--mkt)}
.pcol h3:not(:first-child){margin-top:5.6mm}
.pcol ul{list-style:none}
.pcol li{position:relative;padding-left:4.6mm;margin-bottom:2.6mm;font-size:11.8px;
  line-height:1.5}
.pcol li:before{content:"";position:absolute;left:0;top:1.7mm;width:1.7mm;height:1.7mm;
  border-radius:50%;background:var(--accent)}
.pcol p{font-size:11.8px;line-height:1.5;color:#d2d2d8}
.pcol p.mktp{color:var(--muted)}
.pfoot{border-top:1px solid var(--line);margin-top:4mm;padding-top:3mm;
  color:var(--muted);font-size:9.5px}
.pfoot .ev{color:var(--ok);text-transform:uppercase;letter-spacing:.1em;font-size:8.5px;
  font-weight:700;margin-right:2.5mm}

/* ── closing ─────────────────────────────────────────────────────────── */
.two{display:flex;gap:9mm;margin-top:2mm}
.two>div{flex:1 1 0;min-width:0}
.box{background:var(--panel);border:1px solid var(--line);border-radius:5px;
  padding:5mm 5.5mm;margin-bottom:4mm}
.box h4{color:var(--title);font-size:12px;margin-bottom:2.5mm}
.box.warnb{border-color:#5a4a20} .box.warnb h4{color:var(--warn)}
.box ul{list-style:none}
.box li{position:relative;padding-left:4.2mm;margin-bottom:1.8mm;font-size:10.5px;
  line-height:1.45;color:#d2d2d8}
.box li:before{content:"";position:absolute;left:0;top:1.6mm;width:1.6mm;height:1.6mm;
  border-radius:50%;background:var(--accent)}
.box.warnb li:before{background:var(--warn)}
.box p{font-size:10.5px;line-height:1.5;color:#d2d2d8}
.closer{margin-top:2mm;border:1px solid #2c4a72;border-left:3px solid var(--title);
  border-radius:5px;background:linear-gradient(90deg,#20304a 0%,var(--panel) 70%);
  padding:5mm 6mm}
.closer .cq{color:var(--title);text-transform:uppercase;letter-spacing:.13em;font-size:9px;
  font-weight:700;margin-bottom:2.5mm}
.closer p{font-size:11.5px;line-height:1.55;color:#dcdce2}
.closer b{color:#fff;font-weight:600}
.foot{position:absolute;left:13mm;right:13mm;bottom:8mm;color:#6a6a74;font-size:8.5px;
  border-top:1px solid var(--line);padding-top:2.5mm;display:flex;
  justify-content:space-between}
"""


# ── overflow gate ────────────────────────────────────────────────────────────────────────
#
# WHY THIS EXISTS. `.page` is a fixed A4 box with `overflow:hidden`, so a page whose content
# does not fit does not look broken -- the surplus is silently CUT, and the PDF renders a
# tidy page missing its last sentence. That is the same failure the diagram generator's
# `layout_check` was written for, and the same one that clipped the fourth column of
# `exec_scope.png` until someone looked at the PNG.
#
# So the fit is measured rather than trusted: Chrome lays the document out, reports
# scrollHeight against clientHeight for every page and every text column, and the build
# fails on any overflow. A one-page-per-feature promise needs a check that the page holds it.
PROBE_JS = """
<script>
window.addEventListener('load', function () {
  var out = [];
  document.querySelectorAll('.page').forEach(function (p, i) {
    var over = p.scrollHeight - p.clientHeight;
    if (over > 1) out.push('page ' + (i + 1) + ' overflows by ' + over + 'px');
    p.querySelectorAll('.pcol, .box, .scopefig, table').forEach(function (c) {
      var co = c.scrollHeight - c.clientHeight;
      if (co > 1) out.push('page ' + (i + 1) + ' .' + c.className.split(' ')[0] +
                           ' overflows by ' + co + 'px');
    });
  });
  var pre = document.createElement('pre');
  pre.id = 'overflow-report';
  pre.textContent = out.length ? out.join(String.fromCharCode(10)) : 'FIT-OK';
  document.body.appendChild(pre);
});
</script>
"""


def check_overflow():
    """Lay the document out in Chrome and fail on any clipped box. Returns findings."""
    exe = next((p for p in CHROME_CANDIDATES if os.path.exists(p)), None)
    if not exe:
        print("  overflow gate SKIPPED — no Chrome; a clipped page would go unnoticed")
        return []
    probe_html = OUT_HTML.replace(".html", ".__probe.html")
    with open(OUT_HTML, encoding="utf-8") as fh:
        doc = fh.read()
    with open(probe_html, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(doc.replace("</body>", PROBE_JS + "</body>"))
    try:
        r = subprocess.run(
            [exe, "--headless", "--disable-gpu", "--virtual-time-budget=9000", "--dump-dom",
             "file:///" + probe_html.replace(os.sep, "/")],
            capture_output=True, timeout=180, check=False)
        dom = r.stdout.decode("utf-8", "replace")
    except (subprocess.TimeoutExpired, OSError) as e:
        print("  overflow gate FAILED to run:", e)
        return []
    finally:
        if os.path.exists(probe_html):
            os.remove(probe_html)
    m = re.search(r'<pre id="overflow-report">(.*?)</pre>', dom, re.S)
    if not m:
        print("  overflow gate produced no report — treat as unmeasured, not as a pass")
        return []
    body = m.group(1).strip()
    if body == "FIT-OK":
        print("  overflow gate: every page and column fits")
        return []
    findings = [ln.strip() for ln in body.splitlines() if ln.strip()]
    for f in findings:
        print("  OVERFLOW:", f)
    return findings


def build():
    n = len(FEATURES)
    proven = sum(1 for f in FEATURES if f["status"] is PROVEN)
    partial = sum(1 for f in FEATURES if f["status"] is PARTIAL)
    untested = sum(1 for f in FEATURES if f["status"] is UNTESTED)
    rows, _counts = _matrix()

    toc = "".join(
        f"<li><span class='tn'>{f['n']:02d}</span>{f['title']}</li>" for f in FEATURES)
    pages = [f"""
<section class="page cover">
  <div class="cbody">
    <div class="cleft">
      <div class="kicker">Capability brief &middot; internal</div>
      <h1>What we built on top of <em>CasparCG</em>,<br>and what it is worth</h1>
      <p class="sub">A fork of the CasparCG playout server carrying colour management, a grading
      chain, virtual-production geometry, GPU-direct video paths and multi-machine sync. One page
      per capability: what it does, where it pays for itself, what it would still take to exploit
      fully, and how the rest of the market solves the same problem.</p>
      <div class="stats">
        <div class="stat"><b>{n}</b><span>capabilities</span></div>
        <div class="stat"><b>91</b><span>added control commands</span></div>
        <div class="stat"><b>{proven}</b><span>gated by measurement</span></div>
        <div class="stat"><b>{partial + untested}</b><span>with a named next step</span></div>
      </div>
    </div>
    <div class="cright">
      <div class="ctitle">In this brief</div>
      <ol class="toc">{toc}</ol>
    </div>
  </div>
  <div class="meta"><span>Prepared for supervisors and heads of department</span>
    <span>Every figure in this brief is a measurement, not an estimate</span></div>
</section>

<section class="page">
  <h1 class="ph">What we added, and what we left alone</h1>
  <p class="psub">The base server is untouched underneath: we track upstream CasparCG rather than
  diverge from it, so upstream fixes keep arriving and our work keeps applying. Everything above
  the line is ours.</p>
  <figure class="scopefig"><img src="images/exec_scope.png" alt=""></figure>
  <div class="foot"><span>Capability brief</span><span>page 2</span></div>
</section>

<section class="page">
  <h1 class="ph">Where we stand, at a glance</h1>
  <p class="psub">Readiness here means one thing only: whether an automated test gates the
  capability and was run. It is not a judgement of how well something works &mdash; most of these
  run in production &mdash; but a supervisor asked to fund something deserves to know which
  claims rest on a measurement and which rest on someone's word.</p>
  <table>
    <tr><th></th><th>Capability</th><th>Readiness</th><th>The first thing it needs</th></tr>
    {rows}
  </table>
  <div class="legend">
    <span><b style="color:#6fbf6f">proven</b> &mdash; a test gates it and was run ({proven})</span>
    <span><b style="color:#d0a02a">partly proven</b> &mdash; some of it is measured ({partial})</span>
    <span><b style="color:#c88a4a">not yet measured</b> &mdash; works, but nothing tests it ({untested})</span>
  </div>
  <div class="foot"><span>Capability brief</span><span>page 3</span></div>
</section>"""]

    pages += [_feature_page(f) for f in FEATURES]

    pages.append(f"""
<section class="page">
  <h1 class="ph">What it would take, in order of what it returns</h1>
  <p class="psub">Every item below came out of a coverage audit rather than a wish list, so each
  one is a known gap with a known cost.</p>
  <div class="two">
    <div>
      <div class="box"><h4>Cheap, and unlocks the most</h4>
      <ul>
        <li><b>A second machine.</b> Multi-machine sync is written and has never been run as a
        cluster. Nothing else is needed &mdash; no development, no hardware beyond the server.</li>
        <li><b>One SDI cable.</b> A second loopback pair doubles how much of the SDI matrix can be
        measured in parallel, and is currently the limit on that.</li>
        <li><b>A Windows 11 machine with an HDR display.</b> HDR over display outputs is
        unreachable on Windows 10 &mdash; the extension does not exist there. This is an OS
        upgrade, not a code change.</li>
        <li><b>Test material for the newest HAP variant.</b> A complete decode route that nothing
        has ever rendered.</li>
      </ul></div>
      <div class="box"><h4>Development, with a clear return</h4>
      <ul>
        <li><b>Drive ICVFX properly in test.</b> The mask, feather and reprojection are the core
        of the virtual-production offer and only the colour gain is covered. An audit found a real
        defect here precisely because nothing drove it.</li>
        <li><b>Cover camera tracking and pre-visualisation.</b> Thirty-one commands between them,
        zero tests. These are the two capabilities a client is most likely to ask to see.</li>
      </ul></div>
    </div>
    <div>
      <div class="box warnb"><h4>Known and deliberately unresolved</h4>
      <ul>
        <li>One ProRes colour-tag reading disagrees with the published standard. Correcting it
        changes existing pictures, so it needs test material and a measurement first &mdash; it is
        recorded rather than quietly changed.</li>
        <li>Neither HAP nor NotchLC has been compared against a reference decoder. Comparing our
        two renderers to each other cannot catch a fault they share.</li>
        <li>Display-side HDR read-back has no instrument at all &mdash; we can prove what we sent,
        not what a display received.</li>
      </ul></div>
      <div class="box"><h4>How to read the market comparisons</h4>
      <p>The named products are there to place each capability, not to claim parity.
      disguise, Pixera and Unreal-based systems are mature platforms with support
      organisations behind them; this is a playout server we control, which is a different
      trade rather than a better one.</p>
      <p style="margin-top:2.5mm">Product capabilities move quickly. Treat every comparison
      here as a starting point to verify before it is quoted outside this document.</p>
      </div>
    </div>
  </div>
  <div class="closer">
    <div class="cq">The one-sentence version</div>
    <p>The capabilities are built and most of them are measured; what is missing is mostly
    <b>proof rather than function</b> &mdash; and the three cheapest items on this page
    (a second machine, one SDI cable, a Windows 11 host) convert four of the ten unproven
    entries into measured ones without a line of new code.</p>
  </div>
  <div class="foot"><span>Capability brief</span><span>page {n + 4}</span></div>
</section>""")

    html = ("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">\n"
            "<title>CasparVP — Capability Brief</title>\n"
            f"<style>{CSS}</style></head><body>\n{''.join(pages)}\n</body></html>\n")
    with open(OUT_HTML, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(html)
    print(f"wrote {os.path.relpath(OUT_HTML, HERE)}  "
          f"({os.path.getsize(OUT_HTML) // 1024} KB, {n + 4} pages)")
    return n + 4


def to_pdf():
    exe = next((p for p in CHROME_CANDIDATES if os.path.exists(p)), None) or shutil.which("chrome")
    if not exe:
        print("no Chrome/Edge found — open EXECUTIVE_BRIEF.html and print to PDF "
              "(A4 landscape, no margins, background graphics ON)")
        return 1
    url = "file:///" + OUT_HTML.replace("\\", "/")
    cmd = [exe, "--headless", "--disable-gpu", "--no-pdf-header-footer",
           "--run-all-compositor-stages-before-draw", "--virtual-time-budget=12000",
           f"--print-to-pdf={OUT_PDF}", url]
    try:
        subprocess.run(cmd, capture_output=True, timeout=240, check=False)
    except (subprocess.TimeoutExpired, OSError) as e:
        print("PDF export failed:", e)
        return 1
    if not os.path.exists(OUT_PDF):
        print("PDF export produced nothing")
        return 1
    print(f"wrote {os.path.relpath(OUT_PDF, HERE)}  "
          f"({os.path.getsize(OUT_PDF) // 1024} KB)")
    return 0


if __name__ == "__main__":
    build()
    bad = check_overflow()
    rc = to_pdf()
    sys.exit(1 if bad else rc)
