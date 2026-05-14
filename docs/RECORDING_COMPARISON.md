# Análisis Comparativo: Grabación y Replay — CasparVP vs. Soluciones de Mercado

## Resumen Ejecutivo

CasparVP integra capacidades de **grabación ISO** y **replay instantáneo** dentro de su motor de gráficos y playout, eliminando la necesidad de hardware dedicado para estas funciones. A continuación se compara con las soluciones líderes del mercado.

---

## Comparativa por Funcionalidad

> ✅ = Sí &nbsp;&nbsp; ⚠️ = Parcial/Limitado &nbsp;&nbsp; ❌ = No

| Capacidad | **CasparVP** | **EVS XT-VIA** | **Metus INGEST** | **REC Multichannel** | **Blackmagic HyperDeck** | **AJA Ki Pro** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Grabación ISO continua** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Replay instantáneo** | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Cámara lenta variable** | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Play-while-Record** | ✅ | ✅ | ❌ | ❌ | ⚠️ | ❌ |
| **Buffer circular (time-shift)** | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Exportación de highlights** | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Grabación HDR (PQ/HLG)** | ✅ | ✅ | ❌ | ❌ | ✅ | ⚠️ solo Ki Pro Ultra 12G |
| **Grabación multi-canal** | ✅ multi-ch (límite = HW) | ✅ hasta 6 ch HD | ✅ multi-ch | ✅ multi-ch (UI: 9, límite = HW) | ⚠️ hasta 4 ch (Extreme 8K) | ⚠️ hasta 4 ch (Ki Pro GO) |
| **Codecs profesionales (ProRes, DNx)** | ✅ vía FFmpeg | ⚠️ propietario (ProRes/DNx requiere X-File, coste extra) | ✅ | ✅ DNx, ProRes, XDCAM | ✅ | ✅ |
| **Proxy simultáneo (hi-res + proxy)** | ❌ | ✅ | ✅ | ❌ | ❌ | ⚠️ solo Ki Pro Ultra 12G |
| **Timecode LTC embebido** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Enc. por hardware (NVENC/QSV)** | ❌ | ❌ | ⚠️ | ✅ | N/A | N/A |
| **Captura sincronizada multi-tarjeta** | ✅ sync HW (DeckLink) | ✅ | ⚠️ | ✅ sync HW (DeckLink) | N/A (appliance) | N/A (appliance) |
| **Streaming de red (entrada)** | ✅ NDI, SRT, RTMP, RTP, UDP | ✅ ST 2110 | ✅ NDI, SRT, RTMP | ❌ | ❌ | ❌ |
| **Streaming de red (salida)** | ✅ NDI, SRT, RTMP, UDP/TS | ✅ ST 2110 | ✅ SRT, RTMP, UDP | ❌ | ❌ | ❌ |
| **Control remoto por red** | ✅ AMCP (TCP) + OSC | ✅ protocolo propio + REST | ⚠️ app remota | ❌ solo app local | ✅ HyperDeck Protocol (TCP) | ✅ REST API / web |
| **Controlador hardware dedicado** | ❌ protocolo AMCP | ✅ LSM-VIA | ❌ | ❌ | ❌ | ❌ |
| **Hardware propietario necesario** | ❌ PC estándar | ✅ | ❌ | ❌ PC + Decklink | ✅ | ✅ |

---

## Posicionamiento por Segmento

| | **Coste** | **Mejor para** | **Limitación principal** |
|:---|:---|:---|:---|
| **CasparVP** | Bajo (software open-source + PC estándar) | Producciones que ya usan CasparCG y necesitan replay integrado sin inversión adicional | Sin controlador hardware dedicado; operación por comandos o integración con software de terceros |
| **EVS** | Muy alto (€150k–500k+) | Deportes en directo de primer nivel (Fórmula 1, fútbol, Juegos Olímpicos) | Inversión y coste de mantenimiento elevados; ecosistema cerrado |
| **Metus** | Medio (licencia por dongle USB; canales y codecs extra de pago; precio bajo cotización, estimado ~€2.000–5.000 según configuración) | Ingest ISO multicanal para archivo y post-producción | No tiene replay; solo grabación |
| **REC Multichannel** | Bajo (software propio + PC + Decklink) | Grabación ISO sincronizada de muchas cámaras | Solo graba; sin replay, sin playout, sin HDR |
| **HyperDeck** | Bajo-medio (€455–4.475 por unidad) | Grabación ISO dedicada, fiable y autónoma; hasta 4 ch en Extreme 8K; control por TCP y RS-422 | Solo graba; sin replay ni slow-mo |
| **AJA Ki Pro** | Medio (€4.500–5.500 por unidad) | Grabación ISO en campo; Ki Pro GO graba hasta 4 ch simultáneos (HDMI o SDI) | Solo graba; sin replay ni slow-mo |

---

## Ventajas Diferenciales de CasparVP

1. **Todo-en-uno sin coste de licencia:** Gráficos, playout, grabación ISO y replay en una misma plataforma open-source. No requiere hardware propietario.

2. **Replay integrado en el flujo de producción:** El operador puede grabar un canal, revisar jugadas en cámara lenta, exportar highlights y volver a directo — todo desde el mismo sistema.

3. **Coste de entrada muy bajo:** Un PC con tarjeta Decklink (≈€500) puede sustituir funciones que normalmente requieren un EVS de seis cifras, en producciones de escala media.

4. **Soporte HDR nativo:** Las grabaciones preservan metadatos BT.2020/PQ/HLG tanto en el container como a nivel de frame, algo que muchas soluciones de gama media aún no ofrecen.

---

## Dónde EVS y otros siguen siendo superiores

- **Deportes de máximo nivel:** EVS ofrece controladores físicos (LSM-VIA) con jog/shuttle táctil, integración con bases de datos de clips (IPDirector), y flujos de trabajo probados en miles de producciones mundiales. CasparVP no tiene un controlador hardware equivalente.

- **Multi-cámara masivo:** EVS puede ingestar hasta 6 cámaras simultáneamente en HD con redundancia de disco (más servidores = más canales). REC Multichannel soporta múltiples entradas Decklink sincronizadas con encoding por GPU.

- **Super slow-motion dedicado:** Cámaras a 120/180/360 fps con servidores EVS es un estándar que CasparVP no cubre (trabaja a la velocidad del canal, típicamente 25/50/60 fps).

- **Soporte y garantía 24/7:** Para emisoras con acuerdos de nivel de servicio estrictos, EVS ofrece soporte global. CasparVP depende de la comunidad y del equipo de desarrollo propio.

---

## Conclusión

CasparVP cubre el **80% de las necesidades de grabación y replay** de una producción en directo de escala pequeña a media, a una **fracción del coste** de las soluciones establecidas. Es especialmente atractivo para operaciones que ya utilizan CasparCG como motor de playout/gráficos y quieren añadir replay sin duplicar infraestructura. Para producciones deportivas de primer nivel o entornos con requisitos de redundancia y soporte crítico, EVS sigue siendo el estándar de referencia.
