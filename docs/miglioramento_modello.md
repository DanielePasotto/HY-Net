# Come migliorare il modello: esperimenti, risultati e decisioni

Scopo: rendere HY-Net (HarmNet + Y-Net a due flussi con gate) più generalizzabile e più preciso, mantenendo l'idea architetturale.
Il documento raccoglie tutto quello che è stato provato dopo la riscrittura del codice, con numeri, file e job, in modo da poter riprendere il lavoro.
Ultimo aggiornamento: 2026-11-13. Gli esperimenti della sezione 7 sono in corso: le celle vuote vanno riempite con `scripts/summarize_improvement.py`.

## 1. Protocollo

- **Dati.** Tre dataset nel pool: `ours` (329 immagini, 66 pozzetti), `orgaseg` (OrgaSegment), `organoid` (OrganoID).
- **LODO** (leave-one-dataset-out): si addestra sui due dataset non di test e si testa sul terzo. Folds in `folds/lodo_{ours,orgaseg,organoid}.json`, dati in `data/pool`.
- **Metrica principale.** Dice di primo piano per pozzetto (media sui pozzetti, intervallo di confidenza bootstrap 95%). Per i risultati a più semi si riporta la media dei semi (42, 43, 44).
- **Classi di dimensione** (componenti connesse, 480x640): <200 px, 200–1000 px, >=1000 px. Si riportano recall e precisione di istanza (soglia IoU 0,5) da `summary.json`, campo `by_size`.
- **Modelli.** `hynet` = HarmNet (riaddestrato sul pool, poi end-to-end) + `AsymmetricGatedYNet`. `ynet_raw` = lo stesso Y-Net con l'immagine grezza in entrambi i flussi, senza HarmNet. `unet`, `campeod`: baseline.
- **Cluster.** slurm, nodi geppetto e geminio (RTX 3060), `-w` per fissare il nodo. Un addestramento dura 1–3 ore. Si lancia con `scripts/lodo.sh <dominio> <modello>` tramite `slurm/lodo.sbatch`.
- **Regola di `lodo.sh`:** se esiste `$ROOT/<dominio>/runs/<modello>/fold0/DONE` l'addestramento viene saltato e si valuta il modello già presente. Ogni esperimento deve avere una `ROOT` nuova (collisione trovata il 2026-11-13 con `outputs/lodo_aug_s43/s44`, che sono run del 2 novembre senza scala).

## 2. Punto di partenza (baseline LODO, seme 42)

| modello | ours | orgaseg | organoid |
|---|---|---|---|
| HY-Net | 83,8 | 86,8 | 72,2 |
| Y-Net grezzo | 83,3 | 86,9 | 79,0 |
| U-Net | 80,4 | 87,0 | 78,8 |
| CAMPEOD | 83,35 | 86,97 | 83,75 |

A 3 semi su OrganoID: Y-Net grezzo 78,26 (79,05 / 76,99 / 78,74), HY-Net 72,60 (72,24 / 73,33 / 72,22).
Su `ours` e `orgaseg` HarmNet non cambia il risultato; su OrganoID lo peggiora.

Errori per classe di dimensione su OrganoID (seme 42, recall/precisione %):

| modello | <200 px | 200–1000 px | >=1000 px |
|---|---|---|---|
| Y-Net grezzo | 39/21 | 68/69 | 72/87 |
| HY-Net | 22/10 | 51/53 | 69/83 |
| U-Net | 46/26 | 70/70 | 72/88 |
| CAMPEOD | 41/25 | 69/69 | 77/85 |

Il deficit di HY-Net è concentrato negli organoidi medi; tutti i modelli sono deboli sotto 200 px (troppe predizioni piccole).

Distribuzione delle componenti in OrganoID: mediana 169 px, media 1349, p10 2 px, p90 2831, 39% sotto 100 px. In `ours` e `orgaseg` la mediana è 261 e 230 px, p90 circa 1000 e 832 px. (Una prima stima "mediana 2650 px" era un artefatto di calcolo ed è stata ritirata.)

## 3. Cosa dice la tesi (paper/thesis.pdf) e dove è debole

- HarmNet: loss regionale (primo piano verso il target della pipeline classica, sfondo verso l'ingresso sfocato 5x5), quindi la firma dell'illuminazione è conservata per costruzione.
- Nessun controllo "ingresso grezzo" nelle ablazioni; i baseline sono pre-addestrati e valutati zero-shot; CAMPEOD 83,2 è su un dataset proprio e non è confrontabile con 87,35 su OrganoID.

## 4. Armonizzazione: cosa non ha funzionato

Misure con classificatori di dominio (ridge su istogramma e momenti, spettro radiale, CNN sonda) e con il Dice del segmentatore.

| tentativo | esito |
|---|---|
| HarmNet con target della pipeline classica | la pipeline aumenta la riconoscibilità del dominio |
| HarmNet con loss avversaria di dominio, con penalità sui momenti (`--harm_domain_adv`, `--harm_moment_match`) | nessun miglioramento cross-dataset; con momenti λ=3 la sonda CNN scende al 42,6% ma l'immagine è distrutta (SSIM 0,818, CNR 0,33) |
| "content floor" | le sole maschere di verità identificano il dataset all'83,5%: l'accuratezza del classificatore di dominio non è un obiettivo valido |
| attenuazione spettrale deterministica + normalizzazione robusta (`src/hynet/spectral.py`) | Y-Net su OrganoID 76,46 contro 78,26 con grezzo; solo normalizzazione robusta 71,85 (un seme) |
| GroupNorm (`--norm group`), AdaBN (`src/hynet/adabn.py`) | implementati; i job GroupNorm 2148/2151/2152 non sono mai stati richiesti e restano in hold; nessun risultato riportato qui |

Conclusione: nessuna variante di HarmNet o di preprocessing deterministico migliora i risultati cross-dataset.

## 5. Scale augmentation: il guadagno più grande finora

OrganoID ha organoidi molto più piccoli. Aggiunto zoom casuale (`src/hynet/scale.py`, `--scale_aug F`, `SCALE=F` in `lodo.sh`): con probabilità 1/2, fattore log-uniforme in [1/F, F], F=2 (massimo ammesso). Ingrandimento = ritaglio + resize (bilineare per l'immagine, nearest per la maschera); riduzione = resize con antialias + riempimento a riflessione.

Dice per pozzetto su OrganoID (3 semi):

| modello | base | con scala | delta |
|---|---|---|---|
| Y-Net grezzo | 78,26 (79,05/76,99/78,74) | **82,94** (84,08/82,37/82,37) | +4,7 |
| HY-Net | 72,60 (72,24/73,33/72,22) | 76,20 (75,62/77,42/75,55) | +3,6 |

- Tutti i semi migliorano. Y-Net grezzo + scala eguaglia CAMPEOD (83,75, un solo seme) con una rete molto più piccola e senza pre-addestramento.
- Recall <200 px: Y-Net 39% a 50–56%; HY-Net circa 20% a 27–35%. La precisione resta bassa (~20% Y-Net, ~10% HY-Net) e le predizioni in eccesso aumentano.
- Classe 200–1000 px: Y-Net sale a 69–76% di recall; HY-Net resta a ~51–52%.
- La scala non spiega il ritardo di HY-Net: il divario da Y-Net resta di circa 6,7 punti.
- Non provati: F diverso da 2, le direzioni `ours` e `orgaseg`.
- Radici: `outputs/lodo_scale{,_s43,_s44}`; per HY-Net sono stati riusati gli HarmNet dei LODO base.

## 6. HarmNet residuale (senza target classico)

Idea: `ResidualHarmNet` (`src/hynet/models/harmnet.py`), `out = x + 0,5 * tanh(conv(...))` con ultimo strato a zero, quindi identità all'inizio; nessuna pre-fase su target classico, addestramento solo con la loss di segmentazione (modello `hynet_res`, `needs_harmnet=False`).

Dice su OrganoID, scala F=2 (radici `outputs/lodo_res{,_s43,_s44}`):

| modello | s42 | s43 | s44 | media |
|---|---|---|---|---|
| HY-Net (HarmNet classico) | 75,62 | 77,42 | 75,55 | 76,20 |
| **`hynet_res`** | 80,59 | 83,90 | 81,64 | **82,04** |
| Y-Net grezzo | 84,08 | 82,37 | 82,37 | 82,94 |

- +5,8 punti su HY-Net, ma -0,9 rispetto a Y-Net grezzo (dentro la variabilità tra semi).
- Recall 200–1000 px: 64–69% (HY-Net 51–52%).

### 6.1 Misure su `hynet_res` (3 semi; `scripts/measure_harm_delta.py`, `scripts/analyze_residual_harm.py`)

Non è rimasto vicino all'identità:

| | OrganoID (test) | ours+orgaseg (train) |
|---|---|---|
| `\|x_harm - x\|` medio | 0,22 / 0,27 / 0,24 | 0,24 / 0,29 / 0,26 |
| massimo | 0,49–0,50 (saturato) | 0,48–0,50 |
| pixel con `\|delta\|` > 0,02 | 95–97% | 96–97% |
| RMS(delta) / std(immagine) | 3,9–4,6 | 1,6–1,9 |

- Una mappa affine di `x` spiega solo il 22–31% di delta (34–48% sul train); la correlazione con l'alta frequenza è 0,12–0,20: non è solo luminosità/contrasto né enhancement dei bordi.
- Cambia più lo sfondo (`|delta|` 0,22–0,28) del primo piano (0,17–0,21).
- Dalle immagini (seme 42): bordi e texture degli organoidi conservati e accentuati; sullo sfondo compaiono ampie macchie a bassa frequenza, saturate, assenti nel grezzo. Visualizzazione: `outputs/analysis/res_vis_s42.png` sul cluster (colonne: grezzo, grezzo stirato p1–p99, armonizzato, delta+0,5, maschera).

Il flusso armonizzato è usato (Dice per pozzetto su OrganoID, sostituendo un flusso):

| seme | normale | flusso armonizzato sostituito da `x` | flusso grezzo sostituito dall'armonizzato |
|---|---|---|---|
| 42 | 80,59 | 32,95 | 71,83 |
| 43 | 83,90 | 58,97 | 65,81 |
| 44 | 81,64 | 49,78 | 62,09 |

Il gate α (peso sul flusso armonizzato) è 0,43–0,64 per livello. Il valore "normale" coincide con quello salvato. Le sostituzioni portano il Y-Net fuori distribuzione: provano la dipendenza, non l'utilità (il Dice finale è pari a Y-Net grezzo).

Ipotesi non verificata: le macchie derivano dalla InstanceNorm di HarmNet (normalizzazione per immagine). Controllo non eseguito: HarmNet vincolato (clamp a [0,1] o penalità sulla bassa frequenza di delta); previsto un esito intorno a 82–83.

## 7. Consistenza non supervisionata (in corso)

Obiettivo: dare a HarmNet un segnale che premi l'invarianza, invece della sola loss di segmentazione.

Implementazione (`src/hynet/consistency.py`, integrata in `engine.training`):
- Due viste fotometriche indipendenti (le perturbazioni di `--augment`) della stessa immagine, stessa geometria.
- Loss = `--consist` * MSE tra le probabilità predette + `--consist_harm` * MSE tra le due immagini armonizzate (solo modelli con HarmNet).
- `--consist_pairs 2` coppie per passo; immagini = training, più (con `--consist_target true`) metà delle coppie dalle immagini non annotate del dominio di test. **Le maschere del test non sono mai lette, ma il test diventa transduttivo (non zero-shot): va dichiarato in ogni confronto.**
- Pesi usati: 5 e 5, scelti a occhio, un solo valore provato. 94 test passati; smoke test end-to-end sul cluster riuscito.

Disegno (OrganoID, scala F=2, semi 42/43/44):

| gruppo | modello | consistenza | immagini di test | radice | job |
|---|---|---|---|---|---|
| B | `hynet_res` | pred + harm | sì | `outputs/lodo_cB{,_s43,_s44}` | 2188–2190 |
| A | `hynet_res` | pred + harm | no | `outputs/lodo_cA{,_s43,_s44}` | 2194–2196 |
| C | Y-Net grezzo | solo pred | sì | `outputs/lodo_cC{,_s43,_s44}` | 2191–2193 |
| E | Y-Net grezzo | nessuna, `--augment` supervisionato | no | `outputs/lodo_augsc{,_s43,_s44}` | 2200–2202 |

A contro B separa consistenza (punto 3) da uso di immagini di test (punto 4). C dice se il guadagno viene da HarmNet o dalla sola consistenza. E dice se basta la normale augmentation fotometrica.

Risultati parziali (Dice per pozzetto):

| gruppo | s42 | s43 | s44 | media |
|---|---|---|---|---|
| Y-Net grezzo + scala (rif.) | 84,08 | 82,37 | 82,37 | 82,94 |
| `hynet_res` (rif.) | 80,59 | 83,90 | 81,64 | 82,04 |
| B | 83,87 | 83,49 | in corso | 83,68 (n=2) |
| A | | | | |
| C | | | | |
| E | | | | |

B/42 e B/43: recall/precisione 200–1000 px 67/78 e 61/81 (riferimento `hynet_res` 64–69 / 68–75); <200 px recall 40 e 26. Con 2 semi su 3 e un'incertezza tra semi di circa 1–2 punti non si può dire che B superi Y-Net grezzo (+0,7).

Da fare a fine corsa: tabella completa con `python3 scripts/summarize_improvement.py` (sul cluster); per A/B ripetere `scripts/measure_harm_delta.py` e `scripts/analyze_residual_harm.py` (adattando le radici) per vedere se le macchie spariscono, quanto cambia il gate e cosa succede sostituendo i flussi.

## 8. Decisioni aperte e prossimi passi

1. Chiudere la sezione 7 e decidere in base ad A, B, C, E se HarmNet porta un vantaggio reale oltre Y-Net grezzo + scala + augmentation.
2. Se B e C sono pari: HarmNet è neutro e la strada è un Y-Net grezzo con scala, augmentation fotometrica e (se accettabile) consistenza sulle immagini di test.
3. Se B supera C: HarmNet con consistenza è utile; verificare allora con più semi e sulle altre due direzioni (`ours`, `orgaseg`).
4. Provare scala su `ours` e `orgaseg` e F > 2 (richiede modificare il vincolo `MAX_FACTOR` in `scale.py`).
5. Leve non ancora provate: encoder pre-addestrati (ImageNet) per i flussi, Tversky alpha 0,5 o area minima per ridurre le predizioni piccole, modello finale addestrato sul pool con bilanciamento di dominio.

## 9. Limiti noti di tutto il lavoro

- Molti confronti hanno un solo seme o tre semi senza test di significatività sulla differenza tra medie.
- Una sola impostazione di iperparametri per ogni esperimento (F=2, 0,5 di scala residua, pesi 5/5).
- Lo spazio di ricerca è stato guidato soprattutto da OrganoID: non è dimostrato che le conclusioni valgano per `ours` e `orgaseg`.
- I confronti con CAMPEOD sono a un solo seme.
- Il lavoro è tutto non committato nel repository.

## 10. Codice rimosso (pulizia del 2026-11-13)

Il codice degli esperimenti chiusi è stato tolto dal ramo `wip/model-improvement`; resta recuperabile dal commit 8cf9ff0 (`git show 8cf9ff0:<percorso>`).
- Armonizzazione avversaria e a momenti: `src/hynet/adversarial.py`, `tests/test_adversarial.py`, flag `--harm_domain_adv` e `--harm_moment_match`, variabili `HARM_ADV`/`HARM_MATCH` di `lodo.sh`. La classe `DomainDiscriminator` (usata dalla sonda CNN) è stata spostata in `scripts/analyze_harmonization.py`.
- Armonizzazione spettrale: `src/hynet/spectral.py`, `tests/test_spectral.py`, `scripts/spectral_pool.py`, `slurm/spectral_pool.sbatch`.
- Driver di esperimenti chiusi: `scripts/adabn.sh`, `scripts/summarize_adabn.py`, `scripts/finetune_external.sh`, `scripts/summarize_finetune.py`. Restano `src/hynet/adabn.py` e il flag `--adabn` di `hynet-evaluate`, e i flag `--init_weights`/`--n_train` del fine-tuning.
- Le sezioni 4 e 5 descrivono quei tentativi: per riprodurli serve tornare al commit 8cf9ff0.

## 11. File creati o modificati in questo ciclo

- Codice: `src/hynet/scale.py`, `src/hynet/consistency.py`, `src/hynet/spectral.py`, `src/hynet/adversarial.py`, `src/hynet/models/harmnet.py` (`ResidualHarmNet`), `src/hynet/models/hynet.py` (`forward_proc`, `residual_harm`), `src/hynet/models/factory.py` (`hynet_res`), `src/hynet/engine.py`, `src/hynet/config.py`, `src/hynet/cli/train.py`, `src/hynet/data.py`, `src/hynet/splits.py`.
- Script: `scripts/lodo.sh` (SCALE, CONSIST*, ...), `scripts/summarize_improvement.py`, `scripts/measure_harm_delta.py`, `scripts/analyze_residual_harm.py`, `scripts/analyze_harmonization.py`, `scripts/spectral_pool.py`.
- Test: `tests/test_scale.py`, `tests/test_consistency.py`, `tests/test_spectral.py`, `tests/test_models.py`.
