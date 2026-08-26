# Deprecated - superseded, kept for provenance

**Do not read these for current behaviour.** They are here so that a commit message or a review
comment referring to them still resolves.

| document | superseded by | why it is here |
| :--- | :--- | :--- |
| `VULKAN_MIXER_PLAN.md` | `../architecture/VULKAN_MIXER_IMPLEMENTATION.md` | the mixer shipped |
| `GSTREAMER_INTEGRATION_PLAN.md` | `../guides/GSTREAMER_GUIDE.md` | the module shipped, with batteries |
| `OPENFX_INTEGRATION_PLAN.md` | `../architecture/OPENFX_IMPLEMENTATION.md`, `../guides/OPENFX_USER_AND_PLUGIN_GUIDE.md` | the host shipped |
| `VULKAN_SCREEN_CONSUMER_draft.md` | `../architecture/VULKAN_OUTPUT.md` | a draft, never finished |

**Each was checked before being moved here**: no other document and no source comment cited any of
them *at the time*. That is no longer true of `GSTREAMER_INTEGRATION_PLAN.md`, which four live
documents now cite deliberately as the historical record — `FFMPEG_8_MIGRATION.md` (twice),
`features/gstreamer.md` and `GSTREAMER_GUIDE.md`. **That is the folder working, not a mistake:**
these files exist so such a reference resolves, which is also why nothing here should be deleted
merely for being uncited. An archive whose contents get pruned for lack of inbound links is not an
archive. `OCIO_INTEGRATION_STUDY.md` was initially filed here and then moved to `../architecture/`
when a check found eight source files citing it by section number - which is the test for this
folder. Nothing depended on the four above; something depended on that one.
