# Analisi delle immagini: cosa rende difficile la segmentazione

Misure su tutte le 3110 immagini del pool (nostro dataset 329, OrgaSegment 231, OrganoID 2550), confrontate con i problemi descritti nella sezione 2.2 della tesi (`paper/thesis.pdf`) e messe in relazione con il Dice per immagine dei modelli addestrati in LODO. Script: `scripts/analyze_dataset_issues.py` (misure per immagine in `outputs/dataset_analysis/features.csv`, riepilogo per dataset in `domain_summary.txt`, relazione con il Dice in `dice_relation.txt`).

## 1. Come è stata fatta l'analisi

Per ogni immagine, sui livelli di grigio e sulla maschera di riferimento:

- contrasto e uso della dinamica (percentili 1-99, deviazione standard, livelli usati, saturazione);
- illuminazione: ampiezza e quota di varianza spiegata da un piano adattato ai pixel di sfondo (rampa), variazione a bassa frequenza;
- rumore (stima di Immerkaer), nitidezza (varianza del laplaciano), blocchi JPEG (salto di grigio sulla griglia 8x8);
- contrasto organoide/sfondo con segno (positivo: organoidi più chiari dello sfondo), rapporto segnale-rumore, nitidezza del bordo;
- morfologia a scala 480x640: frazione di organoide, numero di componenti, quota di componenti sotto 200 px, diametro mediano, solidità, rapporto perimetro/area, componenti a contatto del bordo.

Il Dice per immagine è quello del riferimento (Y-Net grezzo + scala F=2) e di C (con consistenza e immagini di test), medio sui semi disponibili, sulle immagini di test dei tre domini di LODO: 329 (nostro), 231 (OrgaSegment), 32 (OrganoID). Le correlazioni sono di Spearman dentro ogni dataset; la correlazione parziale toglie dal Dice (sui ranghi) l'effetto di sette misure di morfologia.

## 2. I tre dataset sono molto diversi

| misura (mediana) | nostro | OrgaSegment | OrganoID |
|---|---|---|---|
| dimensione immagine | 1280x960 | 512-2048 (mediana 1024) | 512 (test 1024) |
| organoidi per immagine | 116 | 50 | 20 |
| rumore, sigma in livelli di grigio | 2,5 | 0,5 | 1,2 |
| blocchi JPEG (1 = assenti) | 1,0 | **1,64** | 1,0 |
| quota di varianza spiegata dalla rampa di illuminazione | 0,62 | 0,45 | 0,14 |
| contrasto organoide/sfondo, assoluto | 1,13 | 1,58 | 1,72 |
| ampiezza p1-p99 | 0,50 | 0,71 | 0,60 |
| immagini con organoidi più chiari dello sfondo | 0% | 0% | 58% (train), 16% (test) |

Le differenze maggiori (d di Cohen oltre 2) sono: dimensione dell'immagine, numero di organoidi, rumore, blocchi JPEG, rampa di illuminazione e modalità/polarità. Dopo il ridimensionamento a 480x640 il diametro mediano degli organoidi è simile nei tre dataset (17-19 px).

## 3. Cosa spiega il Dice dentro ogni dataset

**La morfologia domina.** Le misure che correlano di più con il Dice sono le stesse nei tre domini: quota di area fatta da oggetti piccoli (rho da -0,50 a -0,57), rapporto perimetro/area (da -0,48 a -0,55), organoidi a contatto del bordo (da +0,43 a +0,47), frazione di organoide (+0,52 nel nostro, +0,55 in OrgaSegment). Sette misure di morfologia spiegano R² = 0,44 (nostro), 0,43 (OrgaSegment) e 0,47 (OrganoID, solo 32 immagini) dei ranghi del Dice.

**La fotometria conta in modo moderato, dopo aver tolto la morfologia.**

- Nostro dataset (329 immagini): nitidezza del laplaciano +0,39, deviazione standard -0,34, ampiezza p1-p99 -0,32, variazione a bassa frequenza -0,29, variabilità dello sfondo -0,28, ampiezza della rampa -0,26, contrasto organoide/sfondo +0,26, nitidezza del bordo (gradiente sul bordo rispetto allo sfondo) +0,26 (tutti p < 0,01).
- OrgaSegment: solo la nitidezza del bordo (+0,36) e la variabilità dello sfondo (-0,17).
- OrganoID (32 immagini): contrasto organoide/sfondo (+0,34) e gradiente sul bordo (+0,32), non significativi a p < 0,01; la misura di nitidezza del bordo normalizzata sul contrasto ha segno opposto (-0,41) e va letta con cautela.
- Il rumore e i blocchi JPEG non peggiorano il Dice dentro un dataset (correlazione del rumore da -0,01 a +0,30, cioè mai dannosa).

**Polarità e modalità in OrganoID.** Il 58% delle immagini di addestramento di OrganoID e il 16% di quelle di test (5 su 32) mostrano organoidi con bordo chiaro su fondo scuro (aspetto da campo scuro/contrasto di fase, con il bordo della piastra visibile), cosa che non compare mai in nostro e OrgaSegment. In LODO con OrganoID come test, il modello non ha quindi mai visto questa modalità: Dice 72,5 sulle 5 immagini chiare contro 84,9 sulle 27 scure (riferimento; con C 73,9 contro 86,0). Con 5 immagini è un indizio, non una prova.

## 4. Confronto con i problemi della tesi (sezione 2.2)

| problema nella tesi | cosa dicono i dati |
|---|---|
| Basso contrasto, semitrasparenza | **Confermato.** Il nostro dataset è il meno contrastato (contrasto 1,13 contro 1,58 e 1,72). Contrasto e nitidezza del bordo predicono il Dice in tutti e tre i domini, anche dopo aver tolto la morfologia. |
| Vignettatura, illuminazione non uniforme | **Confermato, soprattutto nel nostro dataset**, dove la rampa è la più forte (R² mediano 0,62) e predice un Dice più basso (-0,26 dopo la morfologia). In OrgaSegment e OrganoID l'effetto è debole o non significativo. |
| Detriti, polvere, graffi simili agli organoidi | **Non misurabile direttamente**: non ci sono annotazioni di detriti. Gli oggetti piccoli sono il fattore che pesa di più sul Dice in tutti i domini: il 35-44% dei componenti ha meno di 200 px, e le maschere di riferimento marcano molti puntini minuscoli (vedi `outputs/dataset_analysis/organoid_test_bright_vs_dark.png`: immagini di test di OrganoID con le maschere). La precisione sulle istanze sotto 200 px è circa 20% e il richiamo 35-65%: il modello sbaglia sia per mancato rilevamento sia per falsi positivi. Non si distingue un detrito da un piccolo organoide senza annotazioni. |
| Domain shift per illuminazione, rumore, risoluzione | **Confermato come differenza tra dataset** (rumore sigma 2,5 / 1,2 / 0,5; dimensione da 512 a 2048; rampa 0,62 / 0,45 / 0,14). Ma dentro un dataset il rumore non spiega gli errori. |

**Cosa non c'è nella tesi e i dati mostrano:**

1. **Morfologia e dimensione degli oggetti**: è il fattore che spiega più Dice (circa metà dei ranghi), più di tutta la fotometria.
2. **Modalità di acquisizione e polarità** in OrganoID, assente nei dataset di addestramento di LODO.
3. **Compressione JPEG** in OrgaSegment (blocchi 1,64 contro 1,0): grande differenza tra dataset, effetto lieve sul Dice (-0,35 di correlazione dentro il dataset, che dopo la morfologia scompare).
4. **Oggetti a contatto del bordo** e bordo della piastra (menisco) nelle immagini intere di pozzetto.

## 5. Che cosa implica per il modello

- HarmNet è costruito contro illuminazione, contrasto e rumore. I dati dicono che questi fattori spiegano una quota minore del Dice rispetto alla morfologia, e il rumore non spiega nulla dentro un dataset. È coerente con il guadagno piccolo osservato.
- Il guadagno di C (consistenza con immagini di test) sul riferimento è +1,16 su OrganoID, +1,03 su OrgaSegment e **-1,18 sul nostro dataset** (migliorano il 66%, l'88% e il 27% delle immagini). Sul nostro dataset il guadagno è peggiore dove la texture, la variabilità dello sfondo e la rampa di illuminazione sono più forti (correlazione da -0,40 a -0,56). Attenzione: sul nostro dataset il riferimento ha solo il seme 42 e C i semi 42 e 43.
- Idee non provate che i dati suggeriscono: inversione del contrasto e simulazione di campo scuro come augmentation (polarità di OrganoID); peso maggiore agli oggetti piccoli nella perdita o risoluzione maggiore; trattamento esplicito dei componenti a contatto del bordo.

## 6. Limiti

- Sono associazioni, non cause: morfologia e fotometria sono correlate tra loro (la regressione sui ranghi ne tiene conto solo in parte).
- OrganoID ha 32 immagini di test, di cui 5 chiare: le sue correlazioni sono poco stabili.
- Il Dice per immagine viene da pochi semi: OrganoID 3, nostro e OrgaSegment ancora incompleti al momento dell'analisi (riferimento: solo il seme 42).
- Il segno del contrasto (organoide più chiaro o più scuro) è la differenza tra le medie dentro e fuori maschera: non coincide sempre con la polarità visiva. Controllato su 8 immagini di OrganoID.
- I detriti non sono annotati: il loro effetto è inferito dagli oggetti piccoli.
- La sezione della tesi è stata letta solo dal testo estratto dal PDF (figure non viste).
