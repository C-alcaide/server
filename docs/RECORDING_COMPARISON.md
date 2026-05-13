# Análisis Comparativo: Grabación y Replay — CasparVP vs. Soluciones de Mercado

## Resumen Ejecutivo

CasparVP integra capacidades de **grabación ISO** y **replay instantáneo** dentro de su motor de gráficos y playout, eliminando la necesidad de hardware dedicado para estas funciones. A continuación se compara con las soluciones líderes del mercado.

---

## Comparativa por Funcionalidad

> ✅ = Sí &nbsp;&nbsp; ⚠️ = Parcial/Limitado &nbsp;&nbsp; ❌ = No

| Capacidad | **CasparVP** | **EVS XT-VIA** | **Metus INGEST** | **Blackmagic HyperDeck** | **AJA Ki Pro** |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Grabación ISO continua** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Replay instantáneo** | ✅ | ✅ | ❌ | ❌ | ❌ |
| **Cámara lenta variable** | ✅ | ✅ | ❌ | ❌ | ❌ |
| **Reproducción inversa** | ✅ | ✅ | ❌ | ❌ | ❌ |
| **Play-while-Record** | ✅ | ✅ | ❌ | ⚠️ | ❌ |
| **Buffer circular (time-shift)** | ✅ | ✅ | ❌ | ❌ | ❌ |
| **Exportación de highlights** | ✅ | ✅ | ❌ | ❌ | ❌ |
| **Grabación HDR (PQ/HLG)** | ✅ | ✅ | ⚠️ | ✅ | ✅ |
| **Grabación multi-canal** | ✅ multi-ch (límite = HW) | ✅ 12+ ch | ✅ multi-ch | ⚠️ 1 ch/unidad | ⚠️ 1 ch/unidad |
| **Codecs profesionales (ProRes, DNx)** | ✅ vía FFmpeg | ⚠️ propietario | ✅ | ✅ | ✅ |
| **Timecode LTC embebido** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Controlador hardware dedicado** | ❌ protocolo AMCP | ✅ LSM-VIA | ❌ | ❌ | ❌ |
| **Hardware propietario necesario** | ❌ PC estándar | ✅ | ❌ | ✅ | ✅ |

---

## Posicionamiento por Segmento

| | **Coste** | **Mejor para** | **Limitación principal** |
|:---|:---|:---|:---|
| **CasparVP** | Bajo (software open-source + PC estándar) | Producciones que ya usan CasparCG y necesitan replay integrado sin inversión adicional | Sin controlador hardware dedicado; operación por comandos o integración con software de terceros |
| **EVS** | Muy alto (€150k–500k+) | Deportes en directo de primer nivel (Fórmula 1, fútbol, Juegos Olímpicos) | Inversión y coste de mantenimiento elevados; ecosistema cerrado |
| **Metus** | Medio (licencia software) | Ingest ISO multicanal para archivo y post-producción | No tiene replay; solo grabación |
| **HyperDeck** | Bajo-medio (€1.500–5.000 por unidad) | Grabación ISO dedicada, fiable y autónoma | Solo graba; sin replay ni slow-mo |
| **AJA Ki Pro** | Medio (€3.000–8.000 por unidad) | Grabación ISO en campo con codecs broadcast | Solo graba; sin replay ni slow-mo |

---

## Ventajas Diferenciales de CasparVP

1. **Todo-en-uno sin coste de licencia:** Gráficos, playout, grabación ISO y replay en una misma plataforma open-source. No requiere hardware propietario.

2. **Replay integrado en el flujo de producción:** El operador puede grabar un canal, revisar jugadas en cámara lenta, exportar highlights y volver a directo — todo desde el mismo sistema.

3. **Coste de entrada muy bajo:** Un PC con tarjeta Decklink (≈€500) puede sustituir funciones que normalmente requieren un EVS de seis cifras, en producciones de escala media.

4. **Soporte HDR nativo:** Las grabaciones preservan metadatos BT.2020/PQ/HLG tanto en el container como a nivel de frame, algo que muchas soluciones de gama media aún no ofrecen.

---

## Dónde EVS y otros siguen siendo superiores

- **Deportes de máximo nivel:** EVS ofrece controladores físicos (LSM-VIA) con jog/shuttle táctil, integración con bases de datos de clips (IPDirector), y flujos de trabajo probados en miles de producciones mundiales. CasparVP no tiene un controlador hardware equivalente.

- **Multi-cámara masivo:** EVS puede ingestar 6–12+ cámaras simultáneamente con redundancia de disco. CasparVP gestiona un canal por instancia.

- **Super slow-motion dedicado:** Cámaras a 120/180/360 fps con servidores EVS es un estándar que CasparVP no cubre (trabaja a la velocidad del canal, típicamente 25/50/60 fps).

- **Soporte y garantía 24/7:** Para emisoras con acuerdos de nivel de servicio estrictos, EVS ofrece soporte global. CasparVP depende de la comunidad y del equipo de desarrollo propio.

---

## Conclusión

CasparVP cubre el **80% de las necesidades de grabación y replay** de una producción en directo de escala pequeña a media, a una **fracción del coste** de las soluciones establecidas. Es especialmente atractivo para operaciones que ya utilizan CasparCG como motor de playout/gráficos y quieren añadir replay sin duplicar infraestructura. Para producciones deportivas de primer nivel o entornos con requisitos de redundancia y soporte crítico, EVS sigue siendo el estándar de referencia.
