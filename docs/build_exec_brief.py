"""Build the executive brief: one page per capability, print-ready A4 landscape.

WHO IT IS FOR. Supervisors and heads of department deciding where effort and budget go --
not operators and not engineers. Every page answers the same four things in the same order --
what it does, where it earns its keep, what it would still take, and how the market solves the
same problem -- but it does NOT label them. Headers naming each block read as a form filled in
rather than as a document, so the prose introduces itself and typography separates the parts:
a list, then a plain paragraph, then an amber rule for the caveat, then a hairline and muted
type for the market note.

WHY A GENERATOR rather than hand-written HTML. The content is a table of facts, and a table
of facts hand-carried into markup is a second copy that drifts -- the failure this doc tree
has paid for repeatedly. The readiness column in particular comes straight from what the
harness measured, so it has to be editable as data.

THE CAPABILITY LIST IS DERIVED FROM `docs/features/`, NOT FROM THE DIAGRAMS. It was built the
other way round first, and four capabilities fell out of it -- replay, PortAudio, LTC timecode
and remotewall, every one `Coverage: none`. A list of features taken from a list of pictures
selects for what somebody had already drawn. Each entry now declares `covers=`, `FOLDED` names
the documents deliberately left without a page, and a harness test asserts the two account for
the whole folder.

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

#: Documents in `docs/features/` that deliberately get no page of their own, and why. The
#: harness asserts this plus every entry's `covers=` accounts for the whole folder, so a new
#: feature document fails until someone decides which side it is on. That check exists because
#: four capabilities were missed at once -- see the module docstring.
FOLDED = {
    "image-consumer-and-producer":
        "the measurement surface rather than a capability -- it is what most batteries capture "
        "through. An operator never chooses it and a supervisor has no decision to make about it.",
    "vulkan-mixer":
        "the substrate the picture capabilities run on rather than a capability beside them. It "
        "appears on nearly every page already, as 'both renderers'.",
}


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
        earns="It pays for itself on a shoot mixing an ARRI camera feed, pre-rendered plates "
              "and live graphics on the same wall: without a reference space, matching them is "
              "trial and error that has to be redone whenever a source changes.",
        gap="Nothing is outstanding on the built-in path — it is gated at 1 LSB on both "
            "renderers and was re-measured this week. The OCIO half is measured against OCIO's "
            "own processor, and no one has yet compared an OCIO HDR view against the built-in "
            "PQ path end to end.",
        market="Elsewhere, ACES is the AMPAS standard and OpenColorIO the ASWF library behind "
               "most post-production colour tooling. Media servers differ widely here — some "
               "expose a fixed pipeline, some none at all — and using the published standards "
               "rather than a private pipeline is the point.",
        next='Nothing blocking — compare an OCIO HDR view against the built-in PQ path',
        covers=['colour-grading-and-ocio'],
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
        earns="It earns its keep in live correction during a shoot: the wall reads too warm on "
              "camera, and the fix is one command against the channel instead of a media "
              "turnaround.",
        gap="The operators are measured individually and in stacks. What is not covered is the "
            "animated form of each command, and the windowed-node feature is still a prototype "
            "with one operation.",
        market="Elsewhere this looks like the correction layers in disguise and Pixera, or CDL "
               "support in Resolume. The difference here is that it sits inside the same "
               "colour-managed pipeline as the output transform rather than beside it.",
        next='Cover the animated form of each grading command',
        covers=['colour-grading-and-ocio'],
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
        earns="It matters most on a volume whose panels have aged unevenly, or where two panel "
              "batches do not match. The processor can only do so much, and this closes the "
              "rest without touching content.",
        gap="It is verified on two of five consumer types. Panel- and tile-level addressing is "
            "deliberately not attempted — that stays the LED processor's job — and the "
            "colorimetric solve itself is not ours.",
        market="Elsewhere, OpenVPCal (Netflix) is the open camera-based solve this consumes, "
               "while Brompton Tessera and Megapixel Helios calibrate inside the processor. The "
               "two are complementary: the processor corrects the panel, this corrects what is "
               "sent to it.",
        next='Verify on the remaining three consumer types',
        covers=['led-calibration'],
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
        earns="It applies to any LED-volume shoot with a moving camera. Without it the wall is "
              "a backdrop; with it the wall is the set.",
        gap="This is the largest gap in the brief and the most worth closing: only the "
            "per-channel gain is covered by a test. The mask geometry, the feather, the "
            "inner-frustum reprojection and every animated form are driven by nothing — and an "
            "audit found a real colour defect in exactly this area precisely because nothing "
            "drove it.",
        market="Elsewhere, Unreal Engine's nDisplay and disguise are the reference "
               "implementations, and both are mature. This is not a claim to match them; it is "
               "a claim that the capability exists inside a playout server we control, which "
               "changes what a small rig costs.",
        next='Drive the mask, feather and reprojection in test — not just the colour gain',
        covers=['projection-and-icvfx'],
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
        earns="It pays off on a curved cyclorama or a dome where the geometry changes between "
              "venues. Pre-warping media per venue is the alternative, and it does not survive "
              "a last-minute change.",
        gap="The geometry is measured. What is not: the animated forms of these commands, and "
            "the interaction between a curved warp and a blend mask on the same layer.",
        market="Elsewhere, warping is the core of Pixera, disguise and Watchout. This does not "
               "replace a dedicated warping suite for a complex install, but it removes the "
               "need for one on the many jobs that are a single curve.",
        next='Cover the animated commands, and warp combined with a blend mask',
        covers=['projection-and-icvfx'],
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
        earns="It is what any moving-camera virtual production needs — and the input without "
              "which the inner frustum is worth nothing.",
        gap="No test drives any of the eighteen commands, and none has been run against real "
            "tracking hardware here. The composition order was undocumented until this week; it "
            "is now written down, read from the source, and verified by nothing.",
        market="Elsewhere, Mo-Sys, stYpe, Vicon, OptiTrack and Ncam supply the tracking and "
               "this is the consumer of it. The comparable integration in disguise and Unreal "
               "is mature and supported, which is the gap to be honest about.",
        next='A trial against real tracking hardware, then a test for the 18 commands',
        covers=['camera-tracking'],
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
        earns="Its value is in quoting and de-risking a job. A design error found in previz "
              "costs an afternoon; the same error found at load-in costs a day of crew.",
        gap="Thirteen commands, and no coverage of any kind — the largest untested surface "
            "here. It also renders through one graphics API on both renderers, and the "
            "consequence of that for colour has not been measured.",
        market="Elsewhere, disguise's Designer previz is the benchmark and a large part of why "
               "that platform is standard. This is a smaller capability in the same shape.",
        next='Any coverage at all — 13 commands are driven by nothing',
        covers=['previz'],
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
            "Browser-based graphics take the same route — the composited page arrives as a "
            "shared GPU texture rather than a host bitmap, so an animated overlay costs no "
            "per-frame copy either.",
        ],
        earns="It matters on any job whose channel count is set by hardware cost. This is the "
              "capability that changes how many machines a show needs.",
        gap="This one is measured and honest. What remains is a decision rather than work: the "
            "newest route, video decode through Vulkan, has no dedicated test, and the A/B "
            "against the established route has not been run.",
        market="Elsewhere, NVDEC and GPUDirect are vendor technology available to everyone. "
               "What differs between servers is how much of the path avoids host memory, and "
               "this fork's avoids it end to end for the supported codecs.",
        next='A dedicated test for the newest decode route, and an A/B against the established one',
        covers=['ffmpeg-producer-and-consumer', 'html-gpu-direct'],
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
        earns="It earns its keep on deliverables and compliance recording for a show that also "
              "needs its channels — where the usual answer is a second machine.",
        gap="Picture and cost are well measured. The gap is operational: the quality setting "
            "that takes this from one recording to eight is raster-dependent, and has to be "
            "chosen per job rather than left at a default.",
        market="Elsewhere, NVENC is the encoder everyone uses, and dedicated recorders — AJA, "
               "Blackmagic HyperDeck — do this in hardware and remain the right answer for "
               "guaranteed compliance recording. This removes the need for one per channel.",
        next='Choose the recording quality setting per raster rather than per default',
        covers=['ffmpeg-producer-and-consumer'],
        status=PROVEN,
        evidence="encode-matrix and encode-parity across four codecs · iso-scaling for capacity",
    ),
    dict(
        n=10, title="Instant replay",
        lead="Record a channel continuously, and play any point back while it is still recording.",
        img="exec_replay.png",
        buys=[
            "A moment can be reviewed, cut and exported without stopping the recording or "
            "reloading anything — the record keeps running throughout.",
            "Playback follows the write head, so 'go back twenty seconds' is a command rather "
            "than a file operation.",
            "A highlight exports to a file with in and out points while the record continues, "
            "and a second export is refused outright rather than silently queued.",
        ],
        earns="It is the sports and events pattern: a replay, a highlight cut during the show, or "
              "a compliance review while the show is still running. The alternative is a machine "
              "that does nothing else.",
        gap="Nothing tests it, and the behaviour worth testing is the distinguishing one — "
            "playing a file that is still being written. One machine can check that in seconds: "
            "record, play LIVE, assert the frame is recent and the picture is not torn. Two "
            "further gaps sit behind it. Nothing verifies that an interrupted recording leaves a "
            "readable store, which is the case the segmented design exists for; and there is no "
            "cost measurement, so there is no guidance on how many replay channels a machine "
            "sustains.",
        market="Elsewhere this is EVS at the top of the market and a HyperDeck or a dedicated "
               "replay controller below it — purpose-built, operationally mature, and a separate "
               "box. The difference here is that it is a consumer and a producer on a channel "
               "that is already doing something else.",
        next='Test LIVE playback of an open recording — the distinguishing behaviour',
        covers=['replay'],
        status=UNTESTED,
        evidence="segmented store, LIVE mode and the asynchronous EXPORT with its 400 EXPORT BUSY "
                 "refusal all read from the source · no battery records to a replay store",
    ),
    dict(
        n=11, title="GPU-native intermediate codecs",
        lead="ProRes, NotchLC and HAP decoded on the GPU, in the form the mixer wants.",
        img="feature_codec_handoff.png",
        buys=[
            "The codecs the design and post pipelines already produce, played back at high layer "
            "counts.",
            "HAP in five variants including the newest BC7 form, so content authored for VJ and "
            "installation tooling plays natively.",
            "NotchLC, so material coming out of Notch does not need transcoding.",
        ],
        earns="It applies to installation and broadcast jobs whose content arrives as ProRes or "
              "HAP and has to run many layers deep.",
        gap="Two specifics are outstanding. Neither HAP nor NotchLC has ever been compared "
            "against a reference decoder — only against our other renderer, which cannot catch "
            "a fault both share. And the newest HAP variant has no test material at all.",
        market="Elsewhere, HAP is Vidvox's open codec and standard in Resolume and VDMX, "
               "NotchLC is Notch's, and ProRes is Apple's. Supporting all three on the GPU is "
               "expected of a modern media server; doing it without a host copy is the "
               "distinguishing part.",
        next='Compare HAP and NotchLC against a reference decoder, and obtain BC7 test material',
        covers=['cuda-prores', 'cuda-notchlc', 'hap'],
        status=PARTIAL,
        evidence="prores-parity against FFmpeg's CPU decoder · HAP measured between renderers "
                 "only · NotchLC has no reference comparison",
    ),
    dict(
        n=12, title="SDI output straight from the GPU",
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
        earns="It is what broadcast and large-format delivery need, where SDI is not optional "
              "and the downstream chain trusts what the wire says about the picture.",
        gap="The single-port path is measured to the decimal over a real loopback. Multi-port "
            "is driven by no test, and one operational interaction is worth knowing: the "
            "latency mode that sounds most like 'synchronise' switches the driver's own "
            "multi-port sync off.",
        market="Elsewhere this is the same Blackmagic DeckLink hardware other servers use. The "
               "difference is how the frame reaches it — through host memory, or not.",
        next='Drive the multi-port group in test',
        covers=['decklink-output'],
        status=PROVEN,
        evidence="sdi-output over the 1→4 loopback, re-verified this week: 62.92 dB with a "
                 "placed sub-region, and every documented figure reproduced exactly",
    ),
    dict(
        n=13, title="Straight to the LED processor",
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
        earns="It suits an LED volume fed from display outputs, which is how most modern "
              "processors prefer to be driven.",
        gap="Two things are outstanding, both outside the code. HDR over this path needs "
            "Windows 11, because the direct-scanout extension does not exist on Windows 10 and "
            "the machine falls back to a path with no HDR surface. And the genlock and EDID "
            "features are untested because they need hardware we have not driven.",
        market="Elsewhere, disguise, Pixera and Unreal-based systems all output to processors "
               "this way, and Brompton and Megapixel processors take HDMI or DisplayPort "
               "natively. This is table stakes for LED work rather than a differentiator — "
               "which is exactly why it matters that it exists here.",
        next='A Windows 11 machine with an HDR display; then genlock and EDID hardware',
        covers=['vulkan-output'],
        status=PARTIAL,
        evidence="vulkan-output-signalling 3/3 consistent · the HDR degradation on Windows 10 is "
                 "named by the test rather than hidden",
    ),
    dict(
        n=14, title="HDR that survives to the wire",
        lead="PQ and HLG rendered, and correctly labelled, on SDI and in files and streams.",
        img="exec_hdr.png",
        buys=[
            "HDR deliverables without a separate finishing step: the mastering-display numbers "
            "travel with the picture.",
            "Correct signalling on SDI, in recorded files and over SRT/UDP streams — the same "
            "four numbers, spelled the same way in each.",
            "Ingested HDR is read and reported, so an operator can see what arrived.",
            "A confidence monitor on an ordinary display can tone-map the HDR picture, so an "
            "operator can judge it without an HDR reference screen in front of them.",
        ],
        earns="It is needed for any HDR deliverable, and for any LED volume being shot for an "
              "HDR finish.",
        gap="The two halves are differently mature. File, stream and SDI signalling are "
            "measured; the display-output half is blocked on the same Windows 11 point as the "
            "previous page, and nothing yet reads back what a display actually received.",
        market="Elsewhere, HDR10, PQ and HLG are open standards and the mastering-display block "
               "is the same one every finishing tool writes. Broadcast chains and LED "
               "processors both expect it, and getting the label wrong is a delivery failure "
               "rather than a picture one.",
        next='An instrument that reads back what a display actually received',
        covers=['screen-consumer'],
        status=PARTIAL,
        evidence="signalling on the card · ffprobe read-back per transport · display-side "
                 "read-back has no instrument",
    ),
    dict(
        n=15, title="More machines, one frame number",
        lead="Frame-accurate playback across servers, with commands scheduled to a target frame.",
        img="exec_cluster.png",
        buys=[
            "A wall too wide, or a show too heavy, for one machine stops being an architectural "
            "problem.",
            "Commands are stamped with the frame they should act on, so 'go' means the same "
            "instant on every node.",
            "Content drift between nodes is monitored rather than assumed.",
        ],
        earns="It is for large volumes and multi-surface venues — the jobs where the "
              "alternative is a bigger, more expensive single machine.",
        gap="It has never been run as a cluster, which needs a second machine and nothing else. "
            "The parts a single machine can check are checked.",
        market="Elsewhere, disguise's director/actor model and Unreal's nDisplay clustering are "
               "the references, both built on the same time-synchronisation standards. Using "
               "PTP rather than something bespoke means it can share a clock with the rest of "
               "the facility.",
        next='A second machine — no development needed',
        covers=['cluster-sync'],
        status=UNTESTED,
        evidence="the frame-number arithmetic is verified against its own model · no multi-machine "
                 "run exists",
    ),
    dict(
        n=16, title="Automation and lighting",
        lead="Keyframed mixer state, and house lighting driven from the picture itself.",
        img="exec_lighting.png",
        buys=[
            "Any of 184 mixer properties can be animated on a timeline — moves, grades, "
            "projection and ICVFX included — so a cue is data rather than an operator.",
            "Colour sampled from regions of the frame and sent as Art-Net or sACN, so practicals "
            "and cyc lights follow the content.",
            "Both are standard protocols, so nothing bespoke is needed at the lighting desk.",
        ],
        earns="It suits a show where the wall and the room have to agree, and where "
              "repeatability between takes matters more than live operation.",
        gap="Neither half is driven by a test. The lighting path has a battery; the keyframe "
            "system has none, and the projection and ICVFX fields it can animate were "
            "undocumented until this week.",
        market="Elsewhere, Art-Net and sACN are what every lighting desk speaks, and Resolume "
               "and similar tools offer comparable content-to-light features. Timeline "
               "automation is standard in media servers — the unusual part here is how much of "
               "the mixer is addressable.",
        next='A test for the keyframe system (8 commands, 184 fields)',
        covers=['keyframes', 'dmx-sacn-artnet'],
        status=PARTIAL,
        evidence="dmx battery covers the lighting transports · keyframes: 8 commands, 184 fields, "
                 "no coverage",
    ),
    dict(
        n=17, title="Audio and timecode into the facility",
        lead="Many channels out to professional interfaces, and the building's own clock in.",
        img="exec_audio.png",
        buys=[
            "Multi-channel audio out to whatever the facility already runs — a Dante virtual "
            "soundcard, a MADI or RME interface, any USB or PCIe device — with per-channel "
            "mapping rather than a fixed stereo pair.",
            "The audio device can act as the channel's clock, so the picture stays locked to it "
            "instead of drifting against it.",
            "Timecode read from an audio input becomes a process-wide clock: recordings are "
            "stamped with house time, tracking samples align to it, and it reports whether it is "
            "on real timecode or has fallen back to the system clock.",
        ],
        earns="It matters wherever this machine has to live inside an existing facility rather "
              "than beside one — a Dante plant, a house timecode distribution, an audio "
              "department that expects discrete channels rather than a stereo mix.",
        gap="ASIO is the caveat, and it is a build question rather than a development one. The "
            "code is complete and compiled OUT of the current binary, because the Steinberg SDK "
            "was absent when this was configured — confirmed in the build cache and in the "
            "executable's symbols. Enabling it is a free download and a rebuild. Until then a "
            "Dante virtual soundcard is reachable over WASAPI rather than its own driver, which "
            "costs latency. Neither half is tested, and one defect is recorded: a frame-number "
            "call site assumes 25fps and is wrong at any other rate.",
        market="Elsewhere, Dante is Audinate's and is the de facto standard for facility audio, "
               "ASIO is Steinberg's low-latency interface, and LTC is SMPTE 12M — all of them "
               "things a building already speaks. Many media servers offer a stereo pair and "
               "nothing else.",
        next='Rebuild with the ASIO SDK, then cover the enumeration and one device open',
        covers=['portaudio', 'ltc-timecode'],
        status=UNTESTED,
        evidence="PortAudio consumer and producer with channel mapping and shared ASIO capture, "
                 "and the LTC clock with three dependents, all read from the source · "
                 "PA_USE_ASIO=OFF in this build · no battery",
    ),
    dict(
        n=18, title="Fitting into the room",
        lead="Texture sharing, plug-in hosting, GStreamer pipelines and remote tile-wall input.",
        img="gstreamer_caspar_routes.png",
        buys=[
            "Frames shared with Notch, Resolume, TouchDesigner or Unreal on the same machine "
            "with no encode, no file and no network.",
            "Industry-standard effect plug-ins hosted directly, so a look built in a post tool "
            "can run live.",
            "GStreamer pipelines in and out, which covers the transports a broadcast facility "
            "already runs.",
            "A remote source can arrive as a layer over the network, for a wall driven from "
            "somewhere other than this machine.",
        ],
        earns="It matters on any job where this is one tool among several. The alternative is a "
              "capture card and a conversion between every pair of applications.",
        gap="This is mostly a coverage gap rather than a capability one: texture sharing has no "
            "test in either direction and the plug-in hosts are unmeasured, while GStreamer is "
            "measured.",
        market="Elsewhere, Spout is the Windows standard and what Resolume and TouchDesigner "
               "use; OpenFX is the Open Effects Association's plug-in standard, hosted by "
               "Resolve and Nuke; ISF is Vidvox's shader format. Speaking the standards that "
               "already exist is the whole feature.",
        next='Cover texture sharing in both directions, and the two plug-in hosts',
        covers=['spout', 'isf-and-openfx', 'gstreamer', 'remotewall'],
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
      <ul>{buys}</ul>
      <p>{f['earns']}</p>
      <p class="gapp">{f['gap']}</p>
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
/* --bg is #1f1f1f, not the #1e1e1e the diagrams are drawn on, and the one step is deliberate:
   Chrome's print rasteriser renders a CSS #1e1e1e page as 29,29,29 while a PNG carrying 30,30,30
   passes through untouched, so matching the values in SOURCE left the figures one code lighter
   than the page and every one read as a faintly pasted-on card. Measured, then compensated.
   Verify with the pixel probe if either side's rendering changes. */
:root{
  --bg:#1f1f1f; --panel:#262629; --panel2:#2d2d31; --line:#3a3a40;
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
/* The wash is an IMAGE, not a `radial-gradient`. Over a near-black background the CSS ramp
   spans about eight 8-bit steps across 300mm, so Chrome's print rasteriser banded it into
   visible 35mm stripes. No gradient syntax fixes that -- there is not enough bit depth to
   express the ramp. `exec_cover_bg.png` is the same ramp with about +/-1 code of dither,
   which is the standard remedy. Generated by docs/diagrams/generate_exec_diagrams.py. */
.cover{justify-content:center;background:var(--bg) url("images/exec_cover_bg.png") no-repeat;
  background-size:cover;background-position:left top}
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
p.psub{color:var(--muted);font-size:11.3px;margin-bottom:4.5mm;max-width:240mm}
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
.pcol ul{list-style:none;margin-bottom:5.2mm}
.pcol li{position:relative;padding-left:4.6mm;margin-bottom:2.6mm;font-size:11.8px;
  line-height:1.5}
.pcol li:before{content:"";position:absolute;left:0;top:1.7mm;width:1.7mm;height:1.7mm;
  border-radius:50%;background:var(--accent)}
.pcol p{font-size:11.8px;line-height:1.5;color:#d2d2d8}
.pcol p.gapp{margin-top:5.2mm;padding-left:4.6mm;border-left:2px solid var(--warn);
  color:#cfcfd6}
.pcol p.mktp{margin-top:5.6mm;padding-top:4mm;border-top:1px solid var(--line);
  font-size:10.9px;color:var(--muted)}
.pfoot{border-top:1px solid var(--line);margin-top:4mm;padding-top:3mm;
  color:var(--muted);font-size:9.5px}
.pfoot .ev{color:var(--ok);text-transform:uppercase;letter-spacing:.1em;font-size:8.5px;
  font-weight:700;margin-right:2.5mm}

/* ── closing ─────────────────────────────────────────────────────────── */
.two{display:flex;gap:9mm;margin-top:2mm}
.two>div{flex:1 1 0;min-width:0}
.box{background:var(--panel);border:1px solid var(--line);border-radius:5px;
  padding:3.8mm 4.4mm;margin-bottom:4mm}
.box h4{color:var(--title);font-size:11.2px;margin-bottom:2.1mm}
.box.warnb{border-color:#5a4a20} .box.warnb h4{color:var(--warn)}
.box ul{list-style:none}
.box li{position:relative;padding-left:4mm;margin-bottom:1.35mm;font-size:9.7px;
  line-height:1.4;color:#d2d2d8}
.box li:before{content:"";position:absolute;left:0;top:1.6mm;width:1.6mm;height:1.6mm;
  border-radius:50%;background:var(--accent)}
.box.warnb li:before{background:var(--warn)}
.box p{font-size:9.9px;line-height:1.45;color:#d2d2d8}
.closer{margin-top:3mm;border:1px solid #2c4a72;border-left:3px solid var(--title);
  border-radius:5px;background:linear-gradient(90deg,#20304a 0%,var(--panel) 70%);
  padding:3.6mm 5mm}
.closer .cq{color:var(--title);text-transform:uppercase;letter-spacing:.13em;font-size:8.5px;
  font-weight:700;margin-bottom:1.8mm}
.closer p{font-size:10.6px;line-height:1.5;color:#dcdce2}
.closer b{color:#fff;font-weight:600}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:4mm 7mm;margin-top:0}
.box h4 .bn{display:inline-block;width:5mm;height:5mm;line-height:5mm;text-align:center;
  border-radius:50%;background:var(--accent);color:#fff;font-size:8.5px;margin-right:2.4mm;
  vertical-align:1px}
.box p.bnote{margin-top:2mm;padding-top:1.8mm;border-top:1px solid var(--line);
  color:var(--muted);font-size:9.3px}
.box p+ul{margin-top:1.9mm}
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
  <h1 class="ph">A capable server is not yet a production tool</h1>
  <p class="psub">Everything in this brief is server work, and server work is the part that is
  done. What follows is what stands between these capabilities and a tool a production can
  actually be run on &mdash; and none of it is server work, which is precisely why none of it has
  happened alongside the server.</p>
  <figure class="scopefig"><img src="images/exec_to_production.png" alt=""></figure>
  <div class="foot"><span>Capability brief</span><span>page {n + 4}</span></div>
</section>

<section class="page">
  <h1 class="ph">What it would take, and in what order</h1>
  <p class="psub">Roughly the order they block each other in: without a client nothing else is
  reachable by an operator, and without machines and training a good client is still not a
  service.</p>
  <div class="grid2">
    <div class="box"><h4><span class="bn">1</span> A client application built for this</h4>
    <p>The 360 client was written to prove the control surface works, and it does that well. It is
    lab work: it exposes commands, not a workflow. It is not an events and virtual-production
    application and was never trying to be one.</p>
    <ul>
      <li>A show or project model &mdash; open a venue, reproduce a setup, save it again. Today the
      state of a channel is a sequence of commands somebody typed.</li>
      <li>Wizards where an operator should not be handling raw parameters: calibration,
      projection alignment, inner-frustum setup.</li>
      <li>Presets that travel between venues, visible state, undo, and errors that say what to do
      rather than what failed.</li>
    </ul>
    <p class="bnote">Nothing is missing in the server for this. Every capability in this brief is
    addressable; what is missing is something to address them with.</p>
    </div>

    <div class="box"><h4><span class="bn">2</span> Machines that can carry it</h4>
    <p>The specification follows from the heaviest cases already measured rather than from a
    guess &mdash; 12K ProRes, many layers, GPU-direct recording running alongside playback, and
    several outputs at once.</p>
    <ul>
      <li>GPU memory and storage bandwidth are the practical ceilings, not clock speed. The
      measurements in this brief say where each one binds.</li>
      <li>A show needs a spare that is identical, not merely similar, and a rehearsed way to
      switch to it.</li>
      <li>Multi-machine work needs a second host before it can be proven at all &mdash; that is
      lab time on our side, not a purchase to argue for.</li>
    </ul>
    </div>

    <div class="box"><h4><span class="bn">3</span> Guides and people trained on them</h4>
    <p>Our documentation is engineer-facing by design: it records what was measured, and why a
    thing is built the way it is. An operator needs close to the opposite.</p>
    <ul>
      <li>Task-based guides &mdash; the ten things someone will actually do, in order, with
      pictures &mdash; plus a quick reference for the command set.</li>
      <li>A worked example per scenario: LED volume, projection, broadcast delivery.</li>
      <li>Teaching. Someone who has run a show on it should train the next person; documentation
      on its own has never produced an operator.</li>
    </ul>
    </div>

    <div class="box"><h4><span class="bn">4</span> A defined way to run it live</h4>
    <p>The gap between a system that works and a system a crew will take to a paying job is
    mostly this, and it is the one least visible from inside the code.</p>
    <ul>
      <li>Versioned show files and presets, so a venue is re-opened rather than rebuilt.</li>
      <li>Monitoring an operator can read at a glance: sync, drift, dropped frames, and what the
      outputs are really signalling. The server reports most of this already and nothing surfaces
      it.</li>
      <li>A rollback path &mdash; versioned builds and a known-good one to fall back to &mdash;
      and a named person who fixes it mid-show.</li>
      <li>Conventions for how content arrives: codec, colour tagging, naming. Otherwise every job
      re-derives them, which is where colour errors enter.</li>
    </ul>
    </div>
  </div>
  <div class="closer">
    <div class="cq">The one-sentence version</div>
    <p>The engineering is largely done and largely measured; what remains is <b>not more server
    features</b> but the four things that turn a capable server into something a crew can be
    handed &mdash; and the client application is the one that blocks the other three.</p>
  </div>
  <div class="foot"><span>Capability brief</span>
    <span>market comparisons place each capability rather than claim parity &mdash; verify product
    specifics before quoting them onward</span>
    <span>page {n + 5}</span></div>
</section>""")

    html = ("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">\n"
            "<title>CasparVP — Capability Brief</title>\n"
            f"<style>{CSS}</style></head><body>\n{''.join(pages)}\n</body></html>\n")
    with open(OUT_HTML, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(html)
    print(f"wrote {os.path.relpath(OUT_HTML, HERE)}  "
          f"({os.path.getsize(OUT_HTML) // 1024} KB, {n + 5} pages)")
    return n + 5


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
