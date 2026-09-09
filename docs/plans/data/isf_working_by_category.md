# ISF inventory — the 314 shaders that render, by category

> Generated 2026-09-09 from Vidvox's **MIT** collection after the vertex-shader fix
> (`1a4121267`). Every shader here was played into a real channel and produced a picture;
> see `EFFECT_SOURCES_CATALOGUE.md` §6.1 for the method and its limits — chiefly that
> **"renders" is not "is correct"**, since nothing was compared against a reference.
>
> `gen` draws itself · `flt` wraps a source · `trn` blends two sources.
> **params** is how many `CALL <ch>-<l> ISF SET <name> …` handles it exposes.
> A shader may appear under more than one category; that is the collection's own tagging.

```
PLAY 1-10 [ISF] "<name>"                     generator
PLAY 1-10 [ISF] "<name>" <source>             filter
PLAY 1-10 [ISF] "<name>" TRANSITION <a> <b> 50   transition
CALL 1-10 ISF LIST                           exact parameter spelling
```

## Stylize  (39)

| shader | | params | first parameters |
| :--- | :--- | ---: | :--- |
| `ASCII Art` | flt | 5 | size, gamma, tint, tintColor, alphaMode |
| `Boxinator` | flt | 6 | rate, edge, blend, randomize, gamma, grid |
| `Broken LCD` | flt | 21 | flickerLevel, patternStyle, glitchBrightness, glitchBrightnessCurve, glitchRadius, glitchSmoothness, … |
| `Chroma Zoom` | flt | 5 | master_zoom, red_zoom, green_zoom, blue_zoom, center |
| `City Lights` | flt | 6 | blurAmount, intensity, edgeIntensity, edgeThreshold, erodeIntensity, erodeRadius |
| `Diagonalize` | flt | 2 | width, angle |
| `Double Vision` | flt | 4 | hShift, vShift, mixAmount1, mixAmount2 |
| `Edge Blowout` | flt | 8 | leftEdge, rightEdge, bottomEdge, topEdge, doHorizontal, doVertical, … |
| `Edge Trace` | flt | 3 | intensity, spread, invert_lines |
| `Edges` | flt | 4 | intensity, threshold, sobel, opaque |
| `Emboss` | flt | 3 | intensity, colorize, brightness |
| `Flipbook` | flt | 4 | flipRate, stretchMode, flipDirection, holdTime |
| `Frosted Glass` | flt | 2 | magnitude, seed |
| `Ghosting` | flt | 7 | uBias, uScale, uGhosts, uGhostDispersal, uAdditive, uDirection, … |
| `Glow` | flt | 1 | intensity |
| `Glow-Fast` | flt | 2 | intensity, blurAmount |
| `God Rays` | flt | 3 | lightDirDOTviewDir, lightPositionOnScreen, quality |
| `JPEG Block Corruption` | flt | 6 | blockSize, quantize, chromaSubsample, dcCorrupt, blockDisplace, seed |
| `Kaleidoscope` | flt | 5 | sides, angle, slidex, slidey, center |
| `Meta Image` | flt | 4 | cell_size, zoom_tile, mixAmt, mode |
| `Multi-Pixellate` | flt | 5 | cell_size, min_cell_size, rSeed, shape, round_to_divisions |
| `MultiFrame 2x2` | flt | 2 | lag, hueShift |
| `MultiFrame 3x3` | flt | 2 | lag, hueShift |
| `Neon` | flt | 3 | intensity, gain, neonColor |
| `Night Vision` | flt | 4 | luminanceThreshold, colorAmplification, noiseLevel, visionColor |
| `Noise Pixellate` | flt | 4 | cell_size, sigGain, mode, shape |
| `Optical Flow Distort` | flt | 6 | amt, maskHold, inputScale, inputOffset, inputLambda, resetNow |
| `Pixellate` | flt | 2 | cell_size, shape |
| `Poly Glitch` | flt | 4 | randomSeed, sizeSpread, sizeGain, sampleMode |
| `Posterize` | flt | 2 | gamma, numColors |
| `Sketch` | flt | 1 | intensity |
| `Smoke Screen` | flt | 3 | smokeColor, smokeIntensity, smokeDirection |
| `Thermal Camera` | flt | 0 | — |
| `Toon` | flt | 2 | normalEdgeThreshold, qLevel |
| `Triangle Mosaic` | flt | 5 | cellSize, jitter, seed, outlineWidth, outlineColor |
| `Triangles` | flt | 2 | cell_size, style |
| `Video Snake` | flt | 9 | corner, cols, rows, direction, edgeMode, fpsThrottle, … |
| `v002 Light Leak` | flt | 3 | amount, length, angle |
| `v002 Technicolor` | flt | 2 | amount, style |

## Color Effect  (38)

| shader | | params | first parameters |
| :--- | :--- | ---: | :--- |
| `Auto Color Tone` | flt | 5 | sampleMode, colorModeOverride, colorCount, baseColor, pixelFollowLocation |
| `Auto Levels` | flt | 4 | min_threshold, mid_point, max_threshold, adapt_rate |
| `Channel Slide` | flt | 2 | slideAmt, reflection |
| `Chroma Desaturation Mask` | flt | 13 | threshold, mask_color, hueTol, hueMinBracket, hueMaxBracket, satTol, … |
| `Chroma Mask` | flt | 17 | mask_color, showAlpha, applyAlpha, applyCutOff, alphaMode, cutoffThresh, … |
| `Chroma Zoom` | flt | 5 | master_zoom, red_zoom, green_zoom, blue_zoom, center |
| `Chromatic Aberration` | flt | 5 | amount, falloff, balance, center, direction |
| `Color Blowout` | flt | 1 | intensity |
| `Color Invert` | flt | 0 | — |
| `Color Monochrome` | flt | 2 | intensity, color |
| `Color Posterize` | flt | 1 | levels |
| `Color Replacement` | flt | 14 | threshold, mask_color, target_color, hueTol, hueMinBracket, hueMaxBracket, … |
| `Convergence` | flt | 4 | horizontal_magnitude, vertical_magnitude, color_magnitude, mode |
| `Corner Color Tint` | flt | 5 | color1, color2, color3, color4, rotationAngle |
| `Duotone` | flt | 4 | threshold, softness, brightColor, darkColor |
| `False Color` | flt | 2 | brightColor, darkColor |
| `Ghosting` | flt | 7 | uBias, uScale, uGhosts, uGhostDispersal, uAdditive, uDirection, … |
| `HSVtoRGB` | flt | 0 | — |
| `Layer Mask` | flt | 5 | maskSizingMode, bright, contrast, alphaMode, applyAlpha |
| `Long Exposure` | flt | 2 | absorptionRate, dischargeRate |
| `Luminance Posterize` | flt | 1 | levels |
| `Maximum Component` | flt | 0 | — |
| `Minimum Component` | flt | 0 | — |
| `Motion Heat Map` | flt | 4 | motionThreshold, motionGain, motionColor, displayMode |
| `Multi Hue Shift` | flt | 3 | shiftLow, shiftMid, shiftHigh |
| `Posterize` | flt | 2 | gamma, numColors |
| `RGB Invert` | flt | 4 | r, g, b, a |
| `RGB Strobe` | flt | 5 | r, g, b, a, strobeRates |
| `RGB Trails 3.0` | flt | 4 | rWeight, gWeight, bWeight, aWeight |
| `RGBA Swap` | flt | 4 | redInput, greenInput, blueInput, alphaInput |
| `RGBtoHSV` | flt | 0 | — |
| `Saturation Bleed` | flt | 3 | bleedLevel, depth, gainLevel |
| `Sepia Tone` | flt | 1 | contrast |
| `Solarize` | flt | 4 | centerBrightness, powerCurve, colorize, inverse |
| `Strobe` | flt | 3 | strobeRate, strobeMode, strobeColor |
| `Thermal Camera` | flt | 0 | — |
| `Trio Tone` | flt | 3 | darkColor, midColor, brightColor |
| `v002 Technicolor` | flt | 2 | amount, style |

## Wipe  (35)

| shader | | params | first parameters |
| :--- | :--- | ---: | :--- |
| `Angular` | trn | 2 | progress, startingAngle |
| `Bounce` | trn | 4 | progress, bounces, shadow_height, shadow_colour |
| `Bow Tie Horizontal` | trn | 1 | progress |
| `Bow Tie Vertical` | trn | 1 | progress |
| `Circle` | trn | 3 | progress, backColor, center |
| `Circle Crop` | trn | 2 | progress, bgcolor |
| `Circle Open` | trn | 3 | progress, smoothness, opening |
| `Colour Distance` | trn | 2 | progress, power |
| `Directional` | trn | 2 | progress, direction |
| `Directional Wipe` | trn | 3 | progress, direction, smoothness |
| `Doom Screen Transition` | trn | 6 | progress, frequency, noise, bars, dripScale, amplitude |
| `Doorway` | trn | 4 | progress, reflection, perspective, depth |
| `Grid Flip` | trn | 6 | progress, pause, dividerWidth, randomness, bgcolor, size |
| `Heart Transition` | trn | 1 | progress |
| `Inverted Page Curl` | trn | 1 | progress |
| `Mosaic` | trn | 3 | progress, endy, endx |
| `Perlin Transition` | trn | 4 | progress, seed, smoothness, scale |
| `Pinwheel` | trn | 2 | progress, speed |
| `Polar Function` | trn | 2 | progress, segments |
| `Polka Dots Curtain` | trn | 3 | progress, dots, center |
| `Radial` | trn | 2 | progress, smoothness |
| `Squares Wire` | trn | 4 | progress, smoothness, direction, squares |
| `Squeeze` | trn | 2 | progress, colorSeparation |
| `Stereo Viewer` | trn | 3 | progress, zoom, corner_radius |
| `Swap Transition` | trn | 4 | progress, perspective, depth, reflection |
| `TV Static` | trn | 2 | progress, offset |
| `Undulating Burn Out` | trn | 4 | progress, center, color, smoothness |
| `Wind` | trn | 2 | progress, size |
| `Window Slice` | trn | 3 | progress, count, smoothness |
| `Wipe Down` | trn | 1 | progress |
| `Wipe Left` | trn | 1 | progress |
| `Wipe Right` | trn | 1 | progress |
| `Wipe Up` | trn | 1 | progress |
| `Zoom In Circles` | trn | 1 | progress |
| `cube` | trn | 5 | progress, reflection, persp, unzoom, floating |

## Glitch  (26)

| shader | | params | first parameters |
| :--- | :--- | ---: | :--- |
| `Bad TV` | flt | 8 | noiseLevel, distortion1, distortion2, speed, scroll, scanLineThickness, … |
| `Broken LCD` | flt | 21 | flickerLevel, patternStyle, glitchBrightness, glitchBrightnessCurve, glitchRadius, glitchSmoothness, … |
| `Comet Tails` | flt | 2 | absorptionRate, dischargeRate |
| `Convergence` | flt | 4 | horizontal_magnitude, vertical_magnitude, color_magnitude, mode |
| `FastMosh` | flt | 6 | update_keyframe, update_rate, sharpen, blur, posterize, mode |
| `Glitch Memories` | trn | 1 | progress |
| `Glitch Shifter` | flt | 7 | glitch_size, glitch_horizontal, glitch_vertical, randomize_size, randomize_zoom, use_alt_image, … |
| `Interlace` | flt | 1 | lineSize |
| `Interlace Mirror` | flt | 2 | horizontal, vertical |
| `JPEG Block Corruption` | flt | 6 | blockSize, quantize, chromaSubsample, dcCorrupt, blockDisplace, seed |
| `Key Frame Artifacts` | flt | 4 | updateKeyFrame, adaptRate, numColors, buffQuality |
| `Micro Buffer` | flt | 5 | inputDelay, inputDelay2, inputDelay3, inputRate, mode |
| `Micro Buffer RGB` | flt | 3 | inputRate, delayMode, inputDelay |
| `Pattern Glitch` | flt | 3 | maxGlitchSize, glitchRate, patternMode |
| `Poly Glitch` | flt | 4 | randomSeed, sizeSpread, sizeGain, sampleMode |
| `Random Freeze` | flt | 3 | maxUpdateSize, maxBlendAmount, resetImage |
| `Resize Glitch` | flt | 8 | randomFrequency, glitchNow, levelX, levelY, center, randomizeWidth, … |
| `Slit Scan` | flt | 4 | spacing, line_width, angle, shift |
| `Sorting Smear` | flt | 5 | resetInput, adaptLevel, sortRate, horizontalSort, verticalSort |
| `Stylize Glitch` | flt | 3 | maxGlitchSize, glitchRate, stylizeMode |
| `Time Glitch RGB` | flt | 8 | inputDelay, inputRate, glitch_size, glitch_horizontal, glitch_vertical, randomize_size, … |
| `Trail Mask` | flt | 4 | maskSizingMode, bright, contrast, RGB_mode |
| `VHS Glitch` | flt | 16 | autoScan, xScanline, xScanline2, yScanline, xScanlineSize, xScanlineSize2, … |
| `Vertical Tearing` | flt | 1 | tearPosition |
| `Y-C Time Blur` | flt | 2 | yFeedbackLevel, cFeedbackLevel |
| `v002 Glitch Analog` | flt | 6 | inputDistortion, inputBarsAmount, inputVSYNC, inputHSYNC, inputResolution, inputResolutionMix |

## Utility  (26)

| shader | | params | first parameters |
| :--- | :--- | ---: | :--- |
| `3d Rotate` | flt | 4 | xrot, yrot, zrot, zoom |
| `Apply Alpha` | flt | 0 | — |
| `Color Bars` | gen | 1 | colorShift |
| `Color Controls` | flt | 4 | bright, contrast, hue, saturation |
| `Color Invert` | flt | 0 | — |
| `Color Test Grid` | gen | 4 | gridCols, gridRows, colorShift, colorRange |
| `ConstantAlpha` | flt | 1 | alpha |
| `Cursor Overlay` | flt | 2 | cursor_scale, cursor_position |
| `Deinterlace` | flt | 0 | — |
| `Digital Clock` | gen | 5 | colorInput, clockMode, yOffset, blinkingColons, twentyFourHourStyle |
| `Freeze Frame` | flt | 1 | freeze |
| `Gamma Correction` | flt | 1 | gamma |
| `HSVtoRGB` | flt | 0 | — |
| `Highlighter Overlay` | flt | 6 | box_width, box_height, border_thickness, box_position, box_color, border_color |
| `Layer Mask` | flt | 5 | maskSizingMode, bright, contrast, alphaMode, applyAlpha |
| `Layer Position` | flt | 2 | offset, repeatImage |
| `Maximum Component` | flt | 0 | — |
| `Minimum Component` | flt | 0 | — |
| `Motion Heat Map` | flt | 4 | motionThreshold, motionGain, motionColor, displayMode |
| `RE RGB Gradient Generator` | flt | 27 | frequency1, phase1, amplitude1, offset1, angle1, curve1, … |
| `RGB Invert` | flt | 4 | r, g, b, a |
| `RGBtoHSV` | flt | 0 | — |
| `Rotate` | flt | 1 | angle |
| `Set Alpha` | flt | 1 | newAlpha |
| `Solid Color` | gen | 1 | Color |
| `Test Pattern Generator` | gen | 8 | pattern, brightness, contrast, saturation, hue, rotation, … |

## Distortion Effect  (24)

| shader | | params | first parameters |
| :--- | :--- | ---: | :--- |
| `Bump Distortion` | flt | 3 | level, radius, center |
| `Circle Splash Distortion` | flt | 3 | radius, streaks, center |
| `Circle Warp` | flt | 3 | radius, width, resultRotation |
| `Circle Wrap Distortion` | flt | 5 | inputAngle, inputRadius, inputCenter, mirror, correctAspect |
| `Cubic Warp` | flt | 2 | level, center |
| `Diagonalize` | flt | 2 | width, angle |
| `Displace` | flt | 4 | uDisplaceAmt, xComponent, yComponent, relativeShift |
| `Edge Distort` | flt | 2 | intensity, invert_map |
| `Hyperspace` | flt | 5 | centerX, scrollAmount, rightScrollOffset, midHeight, seamless |
| `Noise Displace` | flt | 5 | displaceX, displaceY, detailX, detailY, updateTime |
| `Optical Flow Distort` | flt | 6 | amt, maskHold, inputScale, inputOffset, inputLambda, resetNow |
| `Pixel Shifter` | flt | 8 | hPhase, hFrequency, hRandom, vPhase, vFrequency, vRandom, … |
| `Power Warp` | flt | 6 | power_x, power_y, shift_x, shift_y, mode_x, mode_y |
| `Ripples` | flt | 6 | level, offset, x_smear, y_smear, center, mode |
| `Shape Morph Wrap` | flt | 14 | mixPoint, shape1, shape2, shapeWobble, preRotateAngle, angleShift, … |
| `Shockwave` | flt | 5 | positionVal, distortion, magnitude, center, background |
| `Shockwave Pulse` | flt | 5 | pulse, rate, magnitude, distortion, center |
| `Sphere Map` | flt | 3 | imageScale, radiusScale, pointInput |
| `Trapezoid Distortion` | flt | 3 | topWidth, bottomWidth, heightScale |
| `Triangle Warp` | flt | 2 | peakPoint, distortX |
| `Twirl` | flt | 3 | radius, amount, center |
| `Waveform Displace` | flt | 5 | audio, displaceX, displaceY, detailX, detailY |
| `v002 Glitch Analog` | flt | 6 | inputDistortion, inputBarsAmount, inputVSYNC, inputHSYNC, inputResolution, inputResolutionMix |
| `v002-CRT-Displacement` | flt | 1 | Amount |

## Retro  (23)

| shader | | params | first parameters |
| :--- | :--- | ---: | :--- |
| `ASCII Art` | flt | 5 | size, gamma, tint, tintColor, alphaMode |
| `Bad TV` | flt | 8 | noiseLevel, distortion1, distortion2, speed, scroll, scanLineThickness, … |
| `CMYK Halftone` | flt | 2 | gridSize, smoothing |
| `CMYK Halftone-Lookaround` | flt | 2 | gridSize, smoothing |
| `Color Monochrome` | flt | 2 | intensity, color |
| `Color Posterize` | flt | 1 | levels |
| `Dither-Bayer` | flt | 2 | colorize, matrixMode |
| `Dot Screen` | flt | 5 | sharpness, angle, scale, colorize, center |
| `Heart Transition` | trn | 1 | progress |
| `Interlace` | flt | 1 | lineSize |
| `Interlace Mirror` | flt | 2 | horizontal, vertical |
| `Inverted Page Curl` | trn | 1 | progress |
| `Luminance Posterize` | flt | 1 | levels |
| `Posterize` | flt | 2 | gamma, numColors |
| `RGB Halftone` | flt | 2 | gridSize, smoothing |
| `RGB Halftone-lookaround` | flt | 2 | gridSize, smoothing |
| `Sepia Tone` | flt | 1 | contrast |
| `Stereo Viewer` | trn | 3 | progress, zoom, corner_radius |
| `Swap Transition` | trn | 4 | progress, perspective, depth, reflection |
| `VHS Glitch` | flt | 16 | autoScan, xScanline, xScanline2, yScanline, xScanlineSize, xScanlineSize2, … |
| `Zooming Feedback` | flt | 11 | preShift, feedbackLevel, rotateAngle, zoomLevel, zoomCenter, feedbackShift, … |
| `v002-CRT-Displacement` | flt | 1 | Amount |
| `v002-CRT-Mask` | flt | 2 | Amount, style |

## Distortion  (21)

| shader | | params | first parameters |
| :--- | :--- | ---: | :--- |
| `Butterfly Wave Scrawler` | trn | 4 | progress, amplitude, waves, colorSeparation |
| `Chromatic Aberration` | flt | 5 | amount, falloff, balance, center, direction |
| `Crazy Parametric Fun` | trn | 5 | progress, b, smoothness, amplitude, a |
| `Crosswarp` | trn | 1 | progress |
| `Directional Warp` | trn | 2 | progress, direction |
| `Displacement` | trn | 2 | progress, strength |
| `Dreamy` | trn | 1 | progress |
| `Dreamy Zoom` | trn | 3 | progress, rotation, scale |
| `Fly Eye` | trn | 4 | progress, size, colorSeparation, zoom |
| `Glitch Displace` | trn | 1 | progress |
| `Glitch Memories` | trn | 1 | progress |
| `Hexagonalize` | trn | 3 | progress, steps, horizontalHexagons |
| `Kaleidoscope Transition` | trn | 4 | progress, power, speed, angle |
| `Morph` | trn | 2 | progress, strength |
| `Pixelize` | trn | 3 | progress, steps, squaresMin |
| `Ripple Transition` | trn | 3 | progress, amplitude, speed |
| `Rotate Scale Fade` | trn | 5 | progress, scale, backColor, rotations, center |
| `Simple Zoom Transition` | trn | 2 | progress, zoom_quickness |
| `Swirl` | trn | 1 | progress |
| `Water Drop` | trn | 3 | progress, speed, amplitude |
| `cube` | trn | 5 | progress, reflection, persp, unzoom, floating |

## Blur  (20)

| shader | | params | first parameters |
| :--- | :--- | ---: | :--- |
| `Bloom` | flt | 2 | blurAmount, intensity |
| `Diagonal Blur` | flt | 3 | width, angle, quality |
| `Dilate` | flt | 2 | intensity, radius |
| `Dilate-Fast` | flt | 2 | intensity, radius |
| `Edge Blur` | flt | 3 | intensity, blurAmount, invert_map |
| `Erode` | flt | 2 | intensity, bleedRadius |
| `Erode-Fast` | flt | 2 | intensity, radius |
| `Fast Blur` | flt | 1 | blurAmount |
| `Frosted Glass` | flt | 2 | magnitude, seed |
| `Gloom` | flt | 2 | blurAmount, intensity |
| `Hatch Blur` | flt | 3 | width, angle, quality |
| `Median` | flt | 1 | radius |
| `Multi Pass Gaussian Blur` | flt | 1 | blurAmount |
| `RGB Trails 3.0` | flt | 4 | rWeight, gWeight, bWeight, aWeight |
| `Smudged Lens` | flt | 7 | scale, brightness, brightnessCurve, radius, noiseSeed, blurLevel, … |
| `Soft Blur` | flt | 2 | softness, depth |
| `VVMotionBlur 3.0` | flt | 1 | blurAmount |
| `Y-C Time Blur` | flt | 2 | yFeedbackLevel, cFeedbackLevel |
| `v002 Dilate` | flt | 1 | amount |
| `v002 Erode` | flt | 1 | amount |

## Geometry Adjustment  (20)

| shader | | params | first parameters |
| :--- | :--- | ---: | :--- |
| `3d Rotate` | flt | 4 | xrot, yrot, zrot, zoom |
| `Collage` | flt | 4 | seed, cell_size, allow_flips_h, allow_flips_v |
| `Crop and Feather` | flt | 8 | Top, Bottom, Left, Right, TopFeather, BottomFeather, … |
| `Dual Side Scroller And Flip` | flt | 8 | slidetop, shifttop, mirrorHorizontaltop, mirrorVerticaltop, slidebot, shiftbot, … |
| `Flip H` | flt | 0 | — |
| `Flip V` | flt | 0 | — |
| `HorizVertHold` | flt | 3 | hHold, vHold, flashEvent |
| `Layer Position` | flt | 2 | offset, repeatImage |
| `Mirror` | flt | 2 | horizontal, vertical |
| `Quad Mask` | flt | 6 | pt1, pt2, pt3, pt4, invertMask, maskApplyMode |
| `Resize Glitch` | flt | 8 | randomFrequency, glitchNow, levelX, levelY, center, randomizeWidth, … |
| `Rotate` | flt | 1 | angle |
| `Shake` | flt | 2 | magnitude, intensity |
| `Side Scroller And Flip` | flt | 4 | slide, shift, mirrorHorizontal, mirrorVertical |
| `Sliding Strips` | flt | 4 | xShiftAmount, yShiftAmount, xTileSize, yTileSize |
| `Soft Flip` | flt | 5 | angle, centerPt, lineWidth, flipH, flipV |
| `Triple Rotate` | flt | 7 | angle1, angle2, angle3, angle4, radius1, radius2, … |
| `Vertex Manipulator` | flt | 4 | topleft, bottomleft, topright, bottomright |
| `XYZoom` | flt | 3 | levelX, levelY, center |
| `Zoom` | flt | 2 | level, center |

## Dissolve  (17)

| shader | | params | first parameters |
| :--- | :--- | ---: | :--- |
| `Burn` | trn | 2 | progress, color |
| `Color Phase` | trn | 3 | progress, fromStep, toStep |
| `Colour Distance` | trn | 2 | progress, power |
| `CrossZoom` | trn | 2 | progress, strength |
| `Crosshatch` | trn | 4 | progress, fadeEdge, center, threshold |
| `Fade` | trn | 1 | progress |
| `Fade Color` | trn | 3 | progress, colorPhase, color |
| `Fade Gray Scale` | trn | 2 | progress, intensity |
| `Film Burn` | trn | 2 | progress, Seed |
| `Linear Blur` | trn | 2 | progress, intensity |
| `Luma Transition` | trn | 1 | progress |
| `Luminance Melt` | trn | 4 | progress, direction, l_threshold, above |
| `Multiply Blend` | trn | 1 | progress |
| `Random Squares` | trn | 3 | progress, smoothness, size |
| `Ripple Transition` | trn | 3 | progress, amplitude, speed |
| `Simple Zoom Transition` | trn | 2 | progress, zoom_quickness |
| `Window Blinds` | trn | 1 | progress |

## Color  (16)

| shader | | params | first parameters |
| :--- | :--- | ---: | :--- |
| `Brick Pattern` | gen | 5 | brickSize, fillSize, brickOffset, fillColor, brickColor |
| `Checkerboard` | gen | 5 | width, offset, color1, color2, splitPos |
| `Color History` | gen | 2 | displayMode, data |
| `Color Scales` | gen | 3 | iAuthor, iNote, iPreviewMode |
| `Color Schemes` | gen | 3 | baseColor, colorModeOverride, colorCount |
| `Color Test Grid` | gen | 4 | gridCols, gridRows, colorShift, colorRange |
| `Corner Colors` | gen | 5 | color1, color2, color3, color4, rotationAngle |
| `Graph Paper` | gen | 7 | bgColor, lineColor, majorDivisions, minorHDivisions, minorVDivisions, majorDivisionLineWidth, … |
| `Linear Gradient` | gen | 6 | offset, frequency, curve, vertical, startColor, endColor |
| `Multi Gradient` | flt | 27 | frequency1, phase1, amplitude1, offset1, angle1, curve1, … |
| `RE RGB Gradient Generator` | flt | 27 | frequency1, phase1, amplitude1, offset1, angle1, curve1, … |
| `Radial Gradient` | gen | 5 | radius1, radius2, startColor, endColor, location |
| `Random Checkerboard` | gen | 10 | width, offset, hue, saturation, brightness, randHue, … |
| `Random Stripes` | gen | 11 | width, offset, hue, saturation, brightness, vertical, … |
| `Sine Warp Gradient` | gen | 7 | size, rotation, angle, shift, xcolor, ycolor, … |
| `Solid Color` | gen | 1 | Color |

## Geometry  (16)

| shader | | params | first parameters |
| :--- | :--- | ---: | :--- |
| `Basic Shape` | gen | 8 | color, maskShapeMode, shapeWidth, shapeHeight, center, invertMask, … |
| `Bordered Box` | gen | 6 | box_width, box_height, border_thickness, box_position, box_color, border_color |
| `Grid Warp` | gen | 10 | level, radius, center, bgColor, lineColor, majorDivisions, … |
| `Heart` | gen | 2 | size, color |
| `Line Group` | gen | 11 | startLineLength, endLineLength, startAngle, endAngle, lineThickness, lineCount, … |
| `Lines` | gen | 6 | spacing, line_width, angle, shift, color1, color2 |
| `Poly Star` | gen | 6 | pointCount, buldge, pointRadiusInside, pointRadiusOutside, pointRotation, starColor |
| `Random Lines` | gen | 10 | lineCount, lineWidth, randomSeed, wobbleAmount, hueRange, colorSaturation, … |
| `Random Shape Blast` | gen | 6 | saturation, brightness, mixAmount, maskShapeMode, anchorToBottom, resetImage |
| `Spiral` | gen | 6 | rotation, count, width, softness, color1, color2 |
| `Star` | gen | 4 | size, bordersize, color1, color2 |
| `Stripes` | gen | 6 | width, offset, vertical, color1, color2, splitPos |
| `Triangle` | gen | 5 | pt1, pt2, pt3, fillColor, bgColor |
| `Truchet Tile` | gen | 5 | tSize, nSeed, color1, color2, lineMode |
| `VU Meter` | gen | 4 | audioLevel, color1, color2, color3 |
| `Worley Cells` | gen | 10 | density, jitter, speed, metric, colorMode, cellColor1, … |

## Color Adjustment  (13)

| shader | | params | first parameters |
| :--- | :--- | ---: | :--- |
| `Apply Alpha` | flt | 0 | — |
| `Auto Colors Histogram` | flt | 5 | colorMode, colorAdjustment, maxGain, threshold, timeSmoothing |
| `Bright` | flt | 1 | bright |
| `Color Controls` | flt | 4 | bright, contrast, hue, saturation |
| `Color Levels` | flt | 10 | minLevel, midLevel, maxLevel, offset1, offset2, offset3, … |
| `Exposure Adjust` | flt | 1 | inputEV |
| `Gamma Correction` | flt | 1 | gamma |
| `LGG` | flt | 4 | lift, gamma, gain, saturation |
| `RGB EQ` | flt | 4 | red, green, blue, gain |
| `Set Alpha` | flt | 1 | newAlpha |
| `Vibrance` | flt | 1 | vibrance |
| `White Point Adjust` | flt | 1 | newWhite |
| `v002 Bleach Bypass` | flt | 1 | amount |

## Tile Effect  (11)

| shader | | params | first parameters |
| :--- | :--- | ---: | :--- |
| `Kaleidoscope Tile` | flt | 7 | size, sides, rotation, angle, slide1, slide2, … |
| `Meta Image` | flt | 4 | cell_size, zoom_tile, mixAmt, mode |
| `Mirror Edge` | flt | 2 | angle, shift |
| `MultiFrame 2x2` | flt | 2 | lag, hueShift |
| `MultiFrame 3x3` | flt | 2 | lag, hueShift |
| `Quad Tile` | flt | 6 | size, rotation, angle, slide1, slide2, shift |
| `Replicate` | flt | 9 | startSize, startOpacity, startCenter, startPadding, endSize, endOpacity, … |
| `Replicate Random` | flt | 3 | randomSeed, repetitions, randomizeOpacity |
| `Sine Warp Tile` | flt | 4 | size, rotation, angle, shift |
| `Video Bricks` | flt | 5 | brickSize, borderSize, borderColor, crop, brickOffset |
| `Video Snake` | flt | 9 | corner, cols, rows, direction, edgeMode, fpsThrottle, … |

## Film  (10)

| shader | | params | first parameters |
| :--- | :--- | ---: | :--- |
| `Bloom` | flt | 2 | blurAmount, intensity |
| `Dirty Lens` | flt | 6 | scale, brightness, brightnessCurve, radius, noiseSeed, alphaMode |
| `Gloom` | flt | 2 | blurAmount, intensity |
| `Lens Flare` | flt | 10 | uBias, uScale, uSource, uChromatic, uGhosts, uGhostDispersal, … |
| `Long Exposure` | flt | 2 | absorptionRate, dischargeRate |
| `Smudged Lens` | flt | 7 | scale, brightness, brightnessCurve, radius, noiseSeed, blurLevel, … |
| `v002 Bleach Bypass` | flt | 1 | amount |
| `v002 Light Leak` | flt | 3 | amount, length, angle |
| `v002 Technicolor` | flt | 2 | amount, style |
| `v002 Vignette` | flt | 3 | vignette, vignetteEdge, vignetteMix |

## Masking  (10)

| shader | | params | first parameters |
| :--- | :--- | ---: | :--- |
| `Apply Alpha` | flt | 0 | — |
| `Chroma Desaturation Mask` | flt | 13 | threshold, mask_color, hueTol, hueMinBracket, hueMaxBracket, satTol, … |
| `Chroma Mask` | flt | 17 | mask_color, showAlpha, applyAlpha, applyCutOff, alphaMode, cutoffThresh, … |
| `Dirty Lens` | flt | 6 | scale, brightness, brightnessCurve, radius, noiseSeed, alphaMode |
| `Layer Mask` | flt | 5 | maskSizingMode, bright, contrast, alphaMode, applyAlpha |
| `Motion Mask` | flt | 7 | threshold, persistence, updateBackground, dilate, blur, showAlpha, … |
| `Quad Mask` | flt | 6 | pt1, pt2, pt3, pt4, invertMask, maskApplyMode |
| `Random Squares Mask` | flt | 6 | width, offset, alpha1, alpha2, seed1, randomThreshold |
| `Shape Mask` | flt | 8 | maskShapeMode, shapeWidth, shapeHeight, center, invertMask, horizontalRepeat, … |
| `Slit Scan Mask` | flt | 5 | spacing, line_width, angle, shift, edgeSharpness |

## v002  (10)

| shader | | params | first parameters |
| :--- | :--- | ---: | :--- |
| `v002 Bleach Bypass` | flt | 1 | amount |
| `v002 Crosshatch` | flt | 6 | invert, separation, greyscale, thickness, front, back |
| `v002 Dilate` | flt | 1 | amount |
| `v002 Erode` | flt | 1 | amount |
| `v002 Glitch Analog` | flt | 6 | inputDistortion, inputBarsAmount, inputVSYNC, inputHSYNC, inputResolution, inputResolutionMix |
| `v002 Light Leak` | flt | 3 | amount, length, angle |
| `v002 Technicolor` | flt | 2 | amount, style |
| `v002 Vignette` | flt | 3 | vignette, vignetteEdge, vignetteMix |
| `v002-CRT-Displacement` | flt | 1 | Amount |
| `v002-CRT-Mask` | flt | 2 | Amount, style |

## Feedback  (9)

| shader | | params | first parameters |
| :--- | :--- | ---: | :--- |
| `Circular Feedback Mask` | flt | 8 | maskRadius, feedbackRate, twirlAmount, fadeRate, centerFeedback, feedbackCenter, … |
| `Comet Tails` | flt | 2 | absorptionRate, dischargeRate |
| `Echo Trace` | flt | 4 | thresh, gain, hardCutoff, invert |
| `FastMosh` | flt | 6 | update_keyframe, update_rate, sharpen, blur, posterize, mode |
| `Ghosting` | flt | 7 | uBias, uScale, uGhosts, uGhostDispersal, uAdditive, uDirection, … |
| `Long Exposure` | flt | 2 | absorptionRate, dischargeRate |
| `Shape Morph Feedback Mask` | flt | 12 | maskRadius, feedbackRate, mixPoint, shape1, shape2, shapeWobble, … |
| `Sorting Smear` | flt | 5 | resetInput, adaptLevel, sortRate, horizontalSort, verticalSort |
| `Zooming Feedback` | flt | 11 | preShift, feedbackLevel, rotateAngle, zoomLevel, zoomCenter, feedbackShift, … |

## Halftone Effect  (9)

| shader | | params | first parameters |
| :--- | :--- | ---: | :--- |
| `CMYK Halftone` | flt | 2 | gridSize, smoothing |
| `CMYK Halftone-Lookaround` | flt | 2 | gridSize, smoothing |
| `Circular Screen` | flt | 5 | sharpness, offset, scale, colorize, center |
| `Dither-Bayer` | flt | 2 | colorize, matrixMode |
| `Dot Screen` | flt | 5 | sharpness, angle, scale, colorize, center |
| `Line Screen` | flt | 6 | sharpness, offset, angle, scale, colorize, fill |
| `RGB Halftone` | flt | 2 | gridSize, smoothing |
| `RGB Halftone-lookaround` | flt | 2 | gridSize, smoothing |
| `v002 Crosshatch` | flt | 6 | invert, separation, greyscale, thickness, front, back |

## Noise  (9)

| shader | | params | first parameters |
| :--- | :--- | ---: | :--- |
| `Life` | gen | 4 | restartNow, startThresh, randomRegrowth, randomDeath |
| `Noise` | gen | 5 | seed, cell_size, threshold, use_time, color_mode |
| `Noise Adapt` | flt | 3 | adaptRate, threshold, useRGBA |
| `Noise Displace` | flt | 5 | displaceX, displaceY, detailX, detailY, updateTime |
| `Noise Pixellate` | flt | 4 | cell_size, sigGain, mode, shape |
| `Ridgelines` | gen | 11 | scale, octaves, persistence, lacunarity, sharpness, gain, … |
| `Shake` | flt | 2 | magnitude, intensity |
| `Simplex Noise` | gen | 9 | scale, octaves, persistence, lacunarity, speed, contrast, … |
| `Worley Cells` | gen | 10 | density, jitter, speed, metric, colorMode, cellColor1, … |

## Pattern  (5)

| shader | | params | first parameters |
| :--- | :--- | ---: | :--- |
| `Brick Pattern` | gen | 5 | brickSize, fillSize, brickOffset, fillColor, brickColor |
| `Checkerboard` | gen | 5 | width, offset, color1, color2, splitPos |
| `Graph Paper` | gen | 7 | bgColor, lineColor, majorDivisions, minorHDivisions, minorVDivisions, majorDivisionLineWidth, … |
| `Random Checkerboard` | gen | 10 | width, offset, hue, saturation, brightness, randHue, … |
| `Random Stripes` | gen | 11 | width, offset, hue, saturation, brightness, vertical, … |

## Audio Visualizer  (4)

| shader | | params | first parameters |
| :--- | :--- | ---: | :--- |
| `Audio Waveform Shape` | gen | 13 | audioWave, audioGain, lineWidth, shapeRadius, feedbackLevel, zoomLevel, … |
| `FFT Color Lines` | gen | 11 | fftImage, waveImage, gainFFT, rangeFFT, waveSize, vertical, … |
| `FFT Filled Waveform` | gen | 8 | fftImage, strokeSize, gain, minRange, maxRange, topColor, … |
| `Waveform Displace` | flt | 5 | audio, displaceX, displaceY, detailX, detailY |

## Kaleidoscope  (4)

| shader | | params | first parameters |
| :--- | :--- | ---: | :--- |
| `Kaleidoscope` | flt | 5 | sides, angle, slidex, slidey, center |
| `Kaleidoscope Tile` | flt | 7 | size, sides, rotation, angle, slide1, slide2, … |
| `Radial Replicate` | flt | 5 | postRotateAngle, numberOfDivisions, preRotateAngle, centerRadiusStart, centerRadiusEnd |
| `Sine Warp Tile` | flt | 4 | size, rotation, angle, shift |

## Generator  (3)

| shader | | params | first parameters |
| :--- | :--- | ---: | :--- |
| `Ridgelines` | gen | 11 | scale, octaves, persistence, lacunarity, sharpness, gain, … |
| `Simplex Noise` | gen | 9 | scale, octaves, persistence, lacunarity, speed, contrast, … |
| `Worley Cells` | gen | 10 | density, jitter, speed, metric, colorMode, cellColor1, … |

## Overlay  (3)

| shader | | params | first parameters |
| :--- | :--- | ---: | :--- |
| `Cursor Overlay` | flt | 2 | cursor_scale, cursor_position |
| `Doodler Overlay` | flt | 9 | penDown, eraseMode, eraseAndReset, penRate, drawColor, penSize, … |
| `Highlighter Overlay` | flt | 6 | box_width, box_height, border_thickness, box_position, box_color, border_color |

## Sharpen  (3)

| shader | | params | first parameters |
| :--- | :--- | ---: | :--- |
| `Sharpen Luminance` | flt | 1 | intensity |
| `Sharpen RGB` | flt | 3 | intensityR, intensityG, intensityB |
| `Unsharp Mask` | flt | 1 | intensity |

## Test Pattern  (3)

| shader | | params | first parameters |
| :--- | :--- | ---: | :--- |
| `Color Bars` | gen | 1 | colorShift |
| `Color Test Grid` | gen | 4 | gridCols, gridRows, colorShift, colorRange |
| `Test Pattern Generator` | gen | 8 | pattern, brightness, contrast, saturation, hue, rotation, … |

## Drawing  (2)

| shader | | params | first parameters |
| :--- | :--- | ---: | :--- |
| `Doodler` | gen | 9 | penDown, eraseMode, eraseAndReset, penRate, drawColor, penSize, … |
| `Etch-a-Sketch` | gen | 9 | moveUp, moveDown, moveLeft, moveRight, penColor, penSize, … |

## Histogram  (1)

| shader | | params | first parameters |
| :--- | :--- | ---: | :--- |
| `Auto Colors Histogram` | flt | 5 | colorMode, colorAdjustment, maxGain, threshold, timeSmoothing |
