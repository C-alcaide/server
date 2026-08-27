# Guides - operator-facing how-to

How to drive a feature. Written for someone running a show, not someone changing the code.

**These describe intended behaviour and are not verified on every commit.** For what is actually
measured - which battery, which numbers, which date - read the matching document in
[../features/](../features/). Where a guide and a feature document disagree, the feature document
is the one derived from the code.

**Three guides were added on 2026-08-27 to close coverage gaps**, found by asking which fork
features an operator can drive and which of those had no guide at all:

* `DECKLINK_OUTPUT.md` — the primary SDI path, 2,382 lines divergent from upstream, and previously
  described only in fragments across five other guides. Read §3 before using a subregion: on the
  GPU readback paths, four of its six numbers are silently dropped.
* `SPOUT.md` and `LTC_TIMECODE.md` — both had **zero mentions in any guide**, and both are small,
  self-contained, entirely operator-driven features. Those were the clearest gaps in the set.

Two guides carry explicit limits worth knowing before quoting them:

* `LED_CALIBRATION.md` reports 32/32 at 1 LSB **and says in the next sentence** that only the
  IMAGE consumer was captured and only one layer composited. That is "the LUT is applied
  correctly", not "`CALIBRATION` works".
* `PLAYBACK_AND_RECORDING_GUIDE.md` carries channel-count ceilings measured on **one** rig, and
  the ladder that produces them resolves no finer than about +/-1 channel.
