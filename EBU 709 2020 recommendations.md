|     |
| --- |
| **Recommendation ITU-R BT.2087-0**<br><br>**(10/2015)** |
| **Colour conversion from  <br>Recommendation ITU-R BT.709 to  <br>Recommendation ITU-R BT.2020** |
| **BT Series**<br><br>**Broadcasting service**<br><br>**(television)** |

Foreword

The role of the Radiocommunication Sector is to ensure the rational, equitable, efficient and economical use of the radio-frequency spectrum by all radiocommunication services, including satellite services, and carry out studies without limit of frequency range on the basis of which Recommendations are adopted.

The regulatory and policy functions of the Radiocommunication Sector are performed by World and Regional Radiocommunication Conferences and Radiocommunication Assemblies supported by Study Groups.

# Policy on Intellectual Property Right (IPR)

ITU-R policy on IPR is described in the Common Patent Policy for ITU-T/ITU-R/ISO/IEC referenced in Annex 1 of Resolution ITU-R 1. Forms to be used for the submission of patent statements and licensing declarations by patent holders are available from http://www.itu.int/ITU-R/go/patents/en where the Guidelines for Implementation of the Common Patent Policy for ITU‑T/ITU‑R/ISO/IEC and the ITU-R patent information database can also be found.

|     |     |
| --- | --- |
| Series of ITU-R Recommendations<br><br>(Also available online at http://www.itu.int/publ/R-REC/en) |     |
| **Series** | Title |
| **BO** | Satellite delivery |
| **BR** | Recording for production, archival and play-out; film for television |
| **BS** | Broadcasting service (sound) |
| BT  | Broadcasting service (television) |
| **F** | Fixed service |
| **M** | Mobile, radiodetermination, amateur and related satellite services |
| **P** | Radiowave propagation |
| **RA** | Radio astronomy |
| **RS** | Remote sensing systems |
| **S** | Fixed-satellite service |
| **SA** | Space applications and meteorology |
| **SF** | Frequency sharing and coordination between fixed-satellite and fixed service systems |
| **SM** | Spectrum management |
| **SNG** | Satellite news gathering |
| **TF** | Time signals and frequency standards emissions |
| **V** | Vocabulary and related subjects |

**_Note_**: _This ITU-R Recommendation was approved in English under the procedure detailed in Resolution ITU-R 1._

_Electronic Publication_

Geneva, 2015

© ITU 2015

All rights reserved. No part of this publication may be reproduced, by any means whatsoever, without written permission of ITU.

RECOMMENDATION ITU-R BT.2087-0<sup>[\[1\]](#footnote-1)</sup>\*

Colour conversion from Recommendation ITU-R BT.709 to  
Recommendation ITU-R BT.2020

(2015)

Scope

This Recommendation addresses a method of colour conversion from Recommendation ITU-R BT.709 to Recommendation ITU-R BT.2020 for use when HDTV programme content is included within UHDTV programmes. Two sets of conversion equations are specified. One set is based on an opto-electronic transfer function (OETF) and its inverse. The other set is based on an electro-optical transfer function (EOTF) and its inverse.

Keywords

UHDTV, colour conversion

The ITU Radiocommunication Assembly,

considering

_a)_ that Recommendation ITU-R BT.2020 – Parameter values for ultra-high definition television systems for production and international programme exchange, specifies the parameter values for the UHDTV image systems, and one of the features of UHDTV is its colour gamut wider than that of HDTV as specified in Recommendation ITU-R BT.709;

_b)_ that an increasing number of television broadcasters and programme makers around the world are starting to produce UHDTV programmes;

_c)_ that HDTV programmes may well be used for making UHDTV programmes, which necessitates colour conversion from Recommendation ITU-R BT.709 to Recommendation ITU‑R BT.2020;

_d)_ that it is required that colours of Recommendation ITU-R BT.709 content should be unchanged by the colour conversion to Recommendation ITU-R BT.2020 and that the conversion method should be mathematically definable,

recommends

**1** that when colour conversion from Recommendation ITU-R BT.709 to Recommendation ITU-R BT.2020 is required for UHDTV programme production and international exchange, the method described in Annex 1 should be used.

Annex 1  
<br/>Method for colour conversion from Recommendation ITU-R BT.709  
to Recommendation ITU-R BT.2020

Figure 1 shows a block diagram of the colour conversion from Recommendation ITU-R BT.709 (Rec. 709) to the non-constant luminance signal format in Table 4 of Recommendation ITU-R BT.2020 (Rec. 2020). The input and output of this diagram are digitally represented _Y′C′<sub>B</sub>C′<sub>R</sub>_ signals or _R′G′B′_ signals.

Figure 1

Block diagram of colour conversion from Rec. 709 _Y′C′<sub>B</sub>C′<sub>R</sub>_ or _R′G′B′_ to Rec. 2020 _Y′C′<sub>B</sub>C′<sub>R</sub>_ or _R′G′B′_  
for the non-constant luminance signal format in Recommendation ITU-R BT.2020

The functions and equations of each block in Fig. 1 are as follows:

&nbsp;

Q<sub>YC</sub><sup>\-1</sup>

Inverse-quantisation of digitally represented luminance and colour-difference signals _D′<sub>Y</sub>D′<sub>CB</sub>D′<sub>CR</sub>_ (Rec. 709) in the bit-depth of _N_<sub>709</sub> bits to normalized luminance and colour-difference signals _E′<sub>Y</sub>E′<sub>CB</sub>E′<sub>CR</sub>_ (Rec. 709):

Q<sub>RGB</sub><sup>\-1</sup>

Inverse-quantisation of digitally represented colour signals _D′<sub>R</sub>D′<sub>G</sub>D′<sub>B</sub>_ (Rec. 709) in the bit-depth of _N_<sub>709</sub> bits to normalized colour signals _E′<sub>R</sub>E′<sub>G</sub>E′<sub>B</sub>_ (Rec. 709):

M1

Conversion from normalized luminance and colour-difference signals _E′<sub>Y</sub>E′<sub>CB</sub>E′<sub>CR</sub>_ (Rec. 709) to normalized _R′G′B′_ colour signals _E′<sub>R</sub>E′<sub>G</sub>E′<sub>B</sub>_ (Rec. 709):

Non-linear to linear conversion from normalized _R′G′B′_ colour signals _E′<sub>R</sub>E′<sub>G</sub>E′<sub>B</sub>_ (Rec. 709) to linearly represented, normalized _RGB_ colour signals _E<sub>R</sub>E<sub>G</sub>E<sub>B</sub>_ (Rec. 709) is accomplished by one of two equations which produce slightly different colours from each other:

Case #1: In the case where the goal is to preserve colours seen on a Rec. 709 display<sup>[\[2\]](#footnote-2)</sup> when displayed on a Rec. 2020 display<sup>[\[3\]](#footnote-3)</sup>, an approximation of the electro-optical transfer function (EOTF) from Recommendation ITU-R BT.1886 (Rec. 1886) is used:

Case #2: In the case where the source is a direct camera output and the goal is to match the colours of a direct Rec. 2020 camera output, an approximation of the Rec. 709 inverse opto-electronic transfer function (OETF) is used (see Annex 2):

NOTE 1 – Recommendation ITU-R BT.1886 specifies the reference EOTF which is used to display Rec. 709 signals. This transfer function is expressed as _L_ = _a_(max\[(_V+b_),0\])<sup>2.40</sup>, where _a_ =(_L<sub>W</sub>_<sup>1/2.40</sup>–_L<sub>B</sub>_<sup>1/2.40</sup>)<sup>2.40</sup> and _b_ = _L<sub>B</sub>_<sup>1/2.40</sup>/(_L<sub>W</sub>_<sup>1/2.40</sup>–_L<sub>B</sub>_<sup>1/2.40</sup>). The approximated, normalized form of this transfer function is shown in this document, which is found by setting _L<sub>W</sub>_ = 1 and _L<sub>B</sub>_ = 0.

NOTE 2 – _E_ and _E'_ are defined within the range of 0 to 1 in Recommendation ITU‑R BT.709. However, the definition of the video signal quantization allows values above 1 or below 0. The above equation may also be applied to those values above 1 or below 0 with an appropriate treatment of the sign for negative values.

M2

Colour conversion from linearly represented, normalized _RGB_ colour signals _E<sub>R</sub>E<sub>G</sub>E<sub>B</sub>_ (Rec. 709) to linearly represented, normalized _RGB_ colour signals _E<sub>R</sub>E<sub>G</sub>E<sub>B</sub>_ (Rec. 2020):

NOTE 3 – All matrix values above were calculated with high precision and then rounded to four decimal digits.

Linear to non-linear conversion from linearly represented, normalized _RGB_ colour signals _E<sub>R</sub>E<sub>G</sub>E<sub>B</sub>_ (Rec. 2020) to normalized _R′G′B′_ colour signals _E′<sub>R</sub>E′<sub>G</sub>E′<sub>B</sub>_ (Rec. 2020) is accomplished by applying the inverse of the non-linear to linear conversion equation.

Case #1: In the cases where the goal is to preserve colours seen on a Rec. 709 display, an approximation of the inverse of Rec. 1886 EOTF is used:

Case #2: In the case where the source is a direct camera output and the goal is to match the colours of a direct Rec. 2020 camera output, an approximation of the Rec. 2020 OETF is used (see Annex 2):

NOTE 4 – _E_ and _E'_ are defined within the range of 0 to 1 in Recommendation ITU‑R BT.2020. However, the definition of the video signal quantization allows values above 1 or below 0. The above equation may also be applied to those values above 1 or below 0 with an appropriate treatment of the sign for negative values.

M3

Conversion from normalized _R′G′B′_ colour signals _E′<sub>R</sub>E′<sub>G</sub>E′<sub>B</sub>_ (Rec. 2020) to normalized luminance and colour-difference signals _E′<sub>Y</sub>E′<sub>CB</sub>E′<sub>CR</sub>_ (Rec. 2020):

Q<sub>RGB</sub>

Quantisation of normalized colour signals _E′<sub>R</sub>E′<sub>G</sub>E′<sub>B</sub>_ (Rec. 2020) to digitally represented colour signals _D′<sub>R</sub>D′<sub>G</sub>D′<sub>B</sub>_ (Rec. 2020) in the bit-depth of _N_<sub>2020</sub> bits:

Q<sub>YC</sub>

Quantisation of normalized luminance and colour-difference signals _E′<sub>Y</sub>E′<sub>CB</sub>E′<sub>CR</sub>_ (Rec. 2020) to digitally represented luminance and colour-difference signals _D′<sub>Y</sub>D′<sub>CB</sub>D′<sub>CR</sub>_ (Rec. 2020) in the bit‑depth of _N_<sub>2020</sub> bits:

Figure 2 shows a block diagram for the colour conversion from Rec. 709 to the constant luminance signal format in Table 4 of Recommendation BT.2020. The input signals of this diagram are digitally represented _R′G′B′_ and _Y′C′<sub>B</sub>C′<sub>R</sub>_. And the output signals are digitally represented _R′G′B′_ and _Y′<sub>C</sub>C′<sub>BC</sub>C′<sub>RC</sub>_ where the addition of the ‘c’ subscript indicates the constant luminance signal format.

Figure 2

Block diagram of colour conversion from Rec. 709 _Y′C′<sub>B</sub>C′<sub>R</sub>_ or _R′G′B′_ to Rec. 2020  
_Y′<sub>C</sub>C′<sub>BC</sub>C′<sub>RC</sub>_ or _R′G′B′_ for the constant luminance signal format in Recommendation ITU-R BT.2020

The functions and equations of each block in Fig. 2 are as follows:

For the five blocks inside the black broken line, the same equations and input/output signals are applied as in the descriptions for Fig. 1. These blocks correspond to the conversion from the digitally represented luminance and colour-difference _D′<sub>Y</sub>D′<sub>CB</sub>D′<sub>CR</sub>_ and colour _D′<sub>R</sub>D′<sub>G</sub>D′<sub>B</sub>_ signals (Rec. 709) to the linearly represented, normalized RGB colour signals _E<sub>R</sub>E<sub>G</sub>E<sub>B</sub>_ (Rec. 2020).

For the ‘M4’ and ‘C’ blocks in Fig. 2 (for the constant luminance signal format) are different compared with the blocks in Fig. 1 (for the non-constant luminance signal format). The same non‑linear function and quantization equations are applied for , ‘Q<sub>YcCc</sub>’ and ‘Q<sub>RGB</sub>’ blocks. To differentiate between the non-constant and constant signal format, the ‘c’ subscript is added for the constant luminance signal format.

M4

Conversion from linearly represented, normalized _RGB_ colour signals _E<sub>R</sub>E<sub>G</sub>E<sub>B</sub>_ (Rec. 2020) to normalized constant-luminance signal _E<sub>Yc</sub>_ (Rec. 2020):

Linear to non-linear conversion from linearly represented, normalized _RB_ colour signals _E<sub>R</sub>E<sub>B</sub>_ and normalized constant-luminance signal _E<sub>Yc</sub>_ (Rec. 2020) to non-linearly represented, normalized _R′B′_ colour signals _E′<sub>R</sub>E′<sub>B</sub>_ and normalized constant-luminance signal _E′<sub>Yc</sub>_ (Rec. 2020) is accomplished by applying the inverse of the non-linear to linear conversion equation.

Case #1: In the case where the goal is to preserve colours seen on a Rec. 709 display when displayed on a Rec. 2020 display, an approximation of the Rec. 1886 inverse EOTF is used:

Case #2: In the case where the source is a direct camera output and the goal is to match the colours of a direct Rec. 2020 camera output, an approximation of the Rec. 2020 OETF is used (see Annex 2):

NOTE 5 – The range of _E_ or _E'_ is defined within the range of 0 to 1 in Recommendation ITU‑R BT.2020. However, the definition of the video signal quantization allows values above 1 or below 0. The above equation may also be applied to those values above 1 or below 0.

C

Conversion from non-linearly represented, normalized _R′B′_ colour signals _E′<sub>R</sub>E′<sub>B</sub>_ and normalized constant-luminance signal _E′<sub>Yc</sub>_ (Rec. 2020) to normalized colour-difference signals _E′<sub>CBc</sub>E′<sub>CRc</sub>_ (Rec. 2020):

Q<sub>RGB</sub>

Quantisation of normalized colour signals _E′<sub>R</sub>E′<sub>G</sub>E′<sub>B</sub>_ (Rec. 2020) to digitally represented colour signals _D′<sub>R</sub>D′<sub>G</sub>D′<sub>B</sub>_ (Rec. 2020) in the bit-depth of _N_<sub>2020</sub> bits:

Q<sub>YcCc</sub>

Quantisation of normalized constant-luminance and colour-difference signals _E′<sub>Yc</sub>E′<sub>CBc</sub>E′<sub>CRc</sub>_  
(Rec. 2020) to digitally represented constant-luminance and colour-difference signals _D′<sub>Yc</sub>D′<sub>CBc</sub>D′<sub>CRc</sub>_ (Rec. 2020) in the bit-depth of _N_<sub>2020</sub> bits:

Annex 2 (informative)  
<br/>Non-linear transfer functions for colour conversion

A concept of signal flow from scene light to display light in video systems is modelled as shown in Fig. 3, consisting of four functions: camera adjustments for creative rendering, opto-electronic transfer function (OETF), electro-optical transfer function (EOTF), and display adjustments to compensate for viewing environment.

Camera adjustments include linear segment near black, pre-knee, knee point, knee slope, and other adjustments. The Rec. 709 and Rec. 2020 OETFs are similar to a square root function. The deviation of these OETFs from a 1/2.0-power function including the linear segment near black can be decomposed into the camera adjustment function. So the OETF itself can be regarded as a square root function.

On the basis of this concept, the square function and square root function should be used for the conversion between linear and non-linear signal representations for the Case #2 OETF-based conversion.

Figure 3

Block diagram of OETF and EOTF in video systems

Annex 3 (informative)  
<br/>Examples of the two use cases for colour conversion

As described in Annex 1, there are two general use cases where colour conversion from Rec. 709 to Rec. 2020 is desired. In the first use case (Case #1), the goal is to preserve colours originally seen on a Rec. 709 display on a Rec. 2020 display. Note that a Rec. 709 display is a display device with _RGB_ primaries that correspond to those in Recommendation ITU-R BT.709, a D65 white point, and an EOTF which conforms to Recommendation ITU-R BT.1886. Likewise, a Rec. 2020 display is a display device with _RGB_ primaries that correspond to those in Recommendation ITU-R BT.2020, a D65 white point, and an EOTF which conforms to Recommendation ITU-R BT.1886. In the second use case (Case #2), the goal is to match the colours of a direct Rec. 2020 camera output. The following example is intended to illustrate the difference between the two cases, and the need for two different conversion approaches.

For this example, a red object is captured by two different cameras: one of which conforms to the Rec. 709 specification and the other conforms to the Rec. 2020 specification. The Rec. 709 camera is connected to a Rec. 709 display, which is operating in a typical reference setup (Rec. 1886 EOTF with a 100 cd/m<sup>2</sup> white level, 0.005 cd/m<sup>2</sup> black level, in a Rec. 2035 viewing environment). Similarly, the Rec. 2020 camera is connected to a Rec. 2020 display, with the same reference setup (Rec. 1886 EOTF with a 100 cd/m<sup>2</sup> white level, 0.005 cd/m<sup>2</sup> black level, in a Rec. 2035 viewing environment).

The red object is selected to be at 20 cd/m<sup>2</sup> luminance and the same chromaticity as the Rec. 709 red primary. This can be expressed in _Yxy_ coordinates as _Y_ \= 20, _x_ = 0.64, _y_ = 0.33.

If the Rec. 709 camera is assumed to have a sensor which utilizes perfect CIE1931 colour matching functions, and the iris is adjusted so the red object produces a normalized Y output from the sensor of 0.2, the result is a 10-bit Rec. 709 encoded _R’G’B’_ output of _R’_ \= 914, _G’_ \= 64, _B’_ \= 64. After being decoded by the Rec. 709 display, the result is an output of _Y_ \= 19.8, _x_ \= 0.640, _y_ \= 0.330 which is very close to the original scene colour.

If the Rec. 2020 camera sensor is assumed to utilize the same colour matching functions and the same iris setting, the result is a 10-bit Rec. 2020 encoded _R’G’B’_ output of _R’_ = 737, _G’_ = 258, _B’_ = 125. The values are very different from the Rec. 709 camera output because the red colour is not near the red primary of the Rec. 2020 system as it was with the Rec. 709 system. After being decoded by the Rec. 2020 display the result is an output of _Y_ = 16.2, _x_ = 0.677, _y_ = 0.316 which is slightly dimmer and slightly more reddish than the original scene colour. This change is an effect of the system gamma rendering taking place in a larger colour space.

Now if the Rec. 709 output of _R’_ = 914, _G’_ = 64, _B’_ = 64 is converted to Rec. 2020 with the Case #1 EOTF-based conversion, the result is a Rec. 2020 output of _R’_ = 764, _G’_ = 343, _B’_ = 217. After being decoded by the Rec. 2020 display the result is an output of _Y_ = 20.3, _x_ = 0.634, _y_ = 0.331 which is very close to the original Rec. 709 display colour (a DeltaE2000 difference of 0.75). It is very different from the Rec. 2020 capture and display colour (a DeltaE2000 difference of 5.9).

If the Rec. 709 output of _R’_ = 914, _G’_ = 64, _B’_ = 64 is instead converted to Rec. 2020 with the Case #2 OETF-based conversion, the result is a Rec. 2020 output of _R’_ \= 737, _G’_ \= 287, _B’_ \= 173. After being decoded by the Rec. 2020 display the result is an output of _Y_ \= 17.0, _x_ \= 0.660, _y_ \= 0.321 which is a better match than Case #1 to the original Rec. 2020 capture and display colour (a DeltaE2000 difference of 2.3). But it is a worse match to the original Rec. 709 display colour (a DeltaE2000 difference of 3.4).

So it seems clear that for converting pre-produced content, which was originally approved on a Rec. 709 display, the Case #1 EOTF-based conversion can be preferred. But for mixing live outputs of Rec. 709 and Rec. 2020 cameras, the Case #2 OETF-based conversion can be preferred.

1.  \* Radiocommunication Study Group 6 made editorial amendments to this Recommendation in the year 2016 in accordance with Resolution ITU-R 1. [↑](#footnote-ref-1)
    
2.  A Rec. 709 display is a display device with RGB primaries that correspond to those in Recommendation ITU-R BT.709, a D65 white point, and an EOTF which conforms to Recommendation ITU-R BT.1886. [↑](#footnote-ref-2)
    
3.  A Rec. 2020 display is a display device with RGB primaries that correspond to those in Recommendation ITU-R BT.2020, a D65 white point, and an EOTF which conforms to Recommendation ITU-R BT.1886. [↑](#footnote-ref-3)