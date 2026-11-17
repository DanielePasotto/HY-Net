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

## 7. Consistenza non supervisionata

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
| D | Y-Net grezzo | peso 1e-6 (solo effetto BatchNorm) | sì | `outputs/lodo_cD{,_s43,_s44}` | 2235–2237 |

A contro B separa consistenza (punto 3) da uso di immagini di test (punto 4). C dice se il guadagno viene da HarmNet o dalla sola consistenza. E dice se basta la normale augmentation fotometrica.

Risultati parziali (Dice per pozzetto):

Risultati finali (Dice per pozzetto su OrganoID, 3 semi; tabella con `scripts/summarize_improvement.py`):

| gruppo | s42 | s43 | s44 | media |
|---|---|---|---|---|
| Y-Net grezzo + scala (rif.) | 84,08 | 82,37 | 82,37 | 82,94 |
| `hynet_res` (rif.) | 80,59 | 83,90 | 81,64 | 82,04 |
| B (`hynet_res`, consistenza, immagini di test) | 83,87 | 83,49 | 83,44 | 83,60 |
| A (`hynet_res`, consistenza, solo sorgente) | 60,17 | 48,22 | 39,62 | **49,34** |
| C (Y-Net grezzo, consistenza, immagini di test) | 84,67 | 84,53 | 83,11 | **84,10** |
| E (Y-Net grezzo, solo `--augment` + scala) | 82,13 | 81,47 | 82,35 | 81,98 |
| D (come C, ma peso consistenza 1e-6: solo effetto sulle statistiche BatchNorm) | 82,21 | 81,93 | 82,80 | 82,31 |
| F (Y-Net grezzo, consistenza solo su immagini sorgente, `--consist_bn_eval`: BatchNorm in eval nei forward di consistenza; non transduttivo) | 79,01 | 76,56 | 77,44 | **77,67** |

Valutazione con AdaBN (`ADABN=true`: statistiche BatchNorm ristimate sulle immagini di test, pesi invariati), stessi modelli:

| gruppo | senza AdaBN | con AdaBN |
|---|---|---|
| Y-Net grezzo + scala | 82,94 | 77,96 |
| `hynet_res` | 82,04 | 78,52 |
| A | 49,34 | 76,93 |
| B | 83,60 | 83,07 |
| C | 84,10 | 79,81 |

Letture:
- **A è un artefatto delle statistiche BatchNorm, non un fallimento dell'invarianza.** Durante l'addestramento le statistiche correnti della BatchNorm si aggiornano anche sui forward di consistenza, cioè su immagini perturbate; senza immagini di test nel mix, a OrganoID non si adattano e il modello crolla (49,3). Con AdaBN sale a 76,9, quindi i pesi sono validi. In B e C le immagini di test entrano nelle statistiche e il crollo non c'è.
- **Il vantaggio di B e C sul riferimento è piccolo e confondibile.** C batte il riferimento di +1,2 (positivo su 3 semi su 3: +0,6, +2,2, +0,7) e batte E (sola augmentation) di +2,1; ma parte del vantaggio poteva venire dalle statistiche BatchNorm che assorbono le immagini di test e non da una invarianza imparata. Il controllo D separa i due effetti.
- **Controllo D: il guadagno di C viene dalla consistenza, non dalle statistiche BatchNorm.** Con le stesse immagini di test nel mix ma peso di consistenza 1e-6, D ottiene 82,31, cioè 0,6 punti sotto il riferimento (82,94) e 1,8 sotto C (84,10). C è sopra D in 3 semi su 3 (+2,5, +2,6, +0,3); il margine del seme 44 è piccolo. Le immagini di test nelle statistiche BatchNorm da sole non aiutano.
- **AdaBN applicato dopo l'addestramento peggiora** (per esempio 82,94 a 77,96 sul riferimento), quindi la semplice ristima delle statistiche sul test non spiega da sola il guadagno di C.
- **HarmNet non aiuta con la consistenza:** B (83,60) è sotto C (84,10) di 0,5 punti, dentro la variabilità tra semi.
- Il test è transduttivo (usa immagini non annotate del dominio di test): il confronto con i modelli zero-shot non è equo.

Altri domini (lanciati 2026-11-15): riferimento R (Y-Net grezzo + scala) e C (con consistenza e immagini di test) sui domini di test `ours` e `orgaseg`, 3 semi; job 2242–2253, radici `outputs/lodo_Rx*` e `outputs/lodo_Cx*` (sottocartella del dominio). Risultati da riempire qui.

Da fare: analisi delle macchie su B (`scripts/measure_harm_delta.py`, `scripts/analyze_residual_harm.py`, adattando le radici).

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
- I confronti equi con CAMPEOD e U-Net (fine della sezione 12) hanno due semi; i numeri di CAMPEOD della sezione 2 sono a un solo seme e senza scala né augmentation.
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

## 12. Piano del 2026-11-15 e job lanciati

Motivazione: la conclusione "HarmNet non serve" si basa su prove quasi tutte su OrganoID dopo la scala (vedi `docs/analisi_immagini.md`: OrganoID non è intestino e contiene una modalità assente negli altri dati). Sull'intestino, senza augmentation HY-Net batte Y-Net grezzo sul nostro dataset (83,90 contro 81,81) e pareggia su OrgaSegment (86,80 contro 86,91); con l'augmentation fotometrica pareggia ovunque. Nuova valutazione sui soli domini intestinali (nostro e OrgaSegment come domini di test), OrganoID come addestramento aggiuntivo e stress test.

Regola di decisione per HarmNet: resta nel modello finale se (a) batte Y-Net grezzo con la ricetta completa sull'intestino con un margine coerente nei semi, oppure (b) vince in modo netto fuori dai range di augmentation senza perdere in-domain. Altrimenti si documenta come risultato negativo.

Gruppi (radici `outputs/lodo_<tag>[_s43|_s44]/<dominio>`, domini `ours` e `orgaseg`; scala F=2, 3 job per cella al massimo):

| tag | modello | ricetta | semi | scopo |
|---|---|---|---|---|
| Rx | `ynet_raw` | scala | 42-44 | riferimento (già in corso) |
| Cx | `ynet_raw` | scala + consistenza (immagini di test) | 42-44 | C sugli altri domini (già in corso) |
| Ya | `ynet_raw` | scala + `--augment` | 42, 43 | confronto con la ricetta completa |
| Ha | `hynet` (HarmNet pre-addestrato copiato dalle radici `lodo*`) | scala + `--augment` | 42, 43 | idem |
| Ra | `hynet_res` | scala + `--augment` | 42, 43 | idem |
| Pa | `hynet_par` | scala + `--augment` + `--consist_harm 5` (solo sorgente) | 42, 43 | HarmNet vincolato |
| Nx | `ynet_raw` | scala, addestrato senza OrganoID (`folds/lodo_<dominio>_nooid.json`) | 42-44 | ablazione di OrganoID nell'addestramento |
| Sx | `ynet_raw` | scala + `--small_weight 4` | 42, 43 | peso agli oggetti sotto 200 px |

Per ogni modello Ya/Ha/Ra/Pa c'è un job di robustezza (`hynet-robustness`, tutta la sweep, anche i livelli fuori dai range di augmentation) in dipendenza `afterok` dall'addestramento: gruppi `outputs/robustness/lodo_<tag>_<dominio>[_s43]`. I job e gli ID sono in `~/jobs_plan.txt` sul cluster (2255-2296 più i job Rx/Cx 2242-2253).

Codice nuovo di questo piano:
- `ParametricHarmNet` (`src/hynet/models/harmnet.py`) e modello `hynet_par`: l'armonizzatore predice solo un campo di illuminazione (guadagno e offset su una griglia 12x16 interpolata) e una curva di tono monotona, con standardizzazione per immagine; parte da zero (identità a meno della standardizzazione). Con `--consist_harm` senza `--consist` la consistenza usa solo HarmNet (il Y-Net non vede le viste perturbate, quindi le statistiche BatchNorm restano pulite).
- `--small_weight` e `hynet.losses.small_object_weights`; `--consist_bn_eval` (controllo F, risultato negativo: 77,67).
- `folds/lodo_ours_nooid.json`, `folds/lodo_orgaseg_nooid.json`; `scripts/lodo.sh` accetta `FOLDS_FILE` e `SMALL_WEIGHT`.
- Test: 94 passano in locale.

Risultati: vedi sotto (2026-11-16).

### Hardware dei job (da dichiarare nel paper)

Il cluster ha anche un nodo `claudio` (RTX 5070 Ti), usato dal 2026-11-15 oltre ai due nodi con RTX 3060 (`geppetto`, `geminio`). Assegnazione: tutti i job del seme 43 dei gruppi Ya, Ha, Ra, Pa (con robustezza), Nx e Sx (ID 2275-2294) sono su `claudio`; il resto, compresi Rx e Cx, sulle 3060. Le celle di un seme sono sullo stesso tipo di GPU, quindi i confronti tra modelli dello stesso seme restano appaiati. Il job 2297 (`outputs/lodo_Ya_claudio`) è lo stesso addestramento di `Ya_ours_42` (job 2298, 3060) eseguito su `claudio`: misura l'effetto della GPU a parità di seme. Nel paper vanno dichiarate entrambe le GPU (RTX 3060 e RTX 5070 Ti). L'elenco dei job con il nodo è in `~/jobs_plan.txt` sul cluster.

Aggiornamento della notte del 2026-11-15: `claudio` risulta circa 2,5 volte più veloce di una 3060, ed era libero. Ho spostato su `claudio` anche i job `Nx_orgaseg_42`, `Nx_orgaseg_44`, `Sx_orgaseg_42` e le robustezze di `Ya_ours_42`, `Ha_orgaseg_42`, `Pa_orgaseg_42`. Ho aggiunto il gruppo Rc (la ricetta di Rx ripetuta su `claudio`: `ours` seme 43, `orgaseg` semi 42-44), riferimento appaiato per Nx e Sx su quella GPU; la differenza Rc meno Rx misura l'effetto della sola GPU. Il riepilogo è `scripts/summarize_plan.py`. Valutazione di `Rx_ours_43` (job 2300) rimasta sulle 3060.

### Risultati del piano (2026-11-16)

Dice per pozzetto, test su `ours` (66 pozzetti) e `orgaseg` (231 immagini). Tabella generata da `scripts/summarize_plan.py`. Semi 42/43/44; i gruppi con la ricetta completa hanno solo i semi 42 e 43. Il seme 43 di Ya, Ha, Ra, Pa, Nx, Sx gira su `claudio` (vedi sopra).

| gruppo | `ours` s42 / s43 / s44 | media | `orgaseg` s42 / s43 / s44 | media |
|---|---|---|---|---|
| Rx: Y-Net grezzo + scala | 85,12 / 83,65 / 83,28 | 84,02 | 87,39 / 87,61 / 87,61 | 87,54 |
| Rc: Rx ripetuto su `claudio` | - / 83,86 / - | 83,86 | 87,24 / 87,48 / 87,49 | 87,40 |
| Cx: scala + consistenza (immagini di test) | 83,45 / 84,40 / 83,54 | 83,80 | 88,42 / - / 88,57 | 88,50 |
| Nx: scala, senza OrganoID | 82,35 / 86,03 / 83,48 | 83,96 | 86,59 / 86,43 / 86,85 | 86,62 |
| Sx: scala + peso 4 sugli oggetti piccoli | 85,42 / 86,74 / - | 86,08 | 85,00 / 85,40 / - | 85,20 |
| Ya: Y-Net grezzo + scala + augment | 87,03 / 88,88 / - | 87,95 | 87,22 / 87,36 / - | 87,29 |
| Ha: `hynet` + scala + augment | 85,44 / 85,23 / - | 85,33 | 86,88 / 87,15 / - | 87,01 |
| Ra: `hynet_res` + scala + augment | 86,71 / 86,38 / - | 86,55 | 87,20 / 87,11 / - | 87,16 |
| Pa: `hynet_par` + scala + augment + consistenza su HarmNet | 84,47 / 84,28 / - | 84,37 | 86,93 / 86,66 / - | 86,80 |

Differenze appaiate per seme (media): Ha-Ya -2,62 (`ours`) e -0,27 (`orgaseg`); Ra-Ya -1,41 e -0,13; Pa-Ya -3,58 e -0,49. Cx-Rx -0,22 e +1,00; Nx-Rx -0,06 e -0,92 (3 semi su 3 negativi su `orgaseg`); Sx-Rx +1,70 (richiamo degli oggetti piccoli da 41% a 52%) e -2,30 (la precisione sotto 200 px scende da 45% a 34%). Effetto della GPU (Rc-Rx): +0,21 su `ours` (1 seme), -0,13 su `orgaseg` (3 semi); Ya su `ours` seme 42: 86,35 su `claudio`, 87,03 sulla 3060.

Robustezza (`hynet-robustness`, differenza di Dice appaiata sui pozzetti rispetto a Ya, due semi insieme; livelli fuori dai range di default dell'augmentation: gamma 0,5, blur 4, noise 0,1):

| | `ours` pulito | `ours` gamma 0,5 | `ours` blur 4 | `orgaseg` pulito | `orgaseg` noise 0,1 | `orgaseg` blur 4 | media perturbate (`ours` / `orgaseg`) |
|---|---|---|---|---|---|---|---|
| Ha (`hynet`) | -2,62 | +3,19 | +4,75 | -0,27 | +2,12 | +3,14 | -1,54 / +0,07 |
| Ra (`hynet_res`) | -1,40 | +0,57 | +1,74 | -0,13 | +1,69 | +2,71 | -0,96 / +0,16 |
| Pa (`hynet_par`) | -3,58 | -4,98 | -13,00 | -0,49 | -4,03 | -3,07 | -4,36 / -0,92 |

Letture:
- **Regola (a) non soddisfatta.** Con la ricetta completa nessuna variante con HarmNet batte Y-Net grezzo: sul nostro dataset stanno 1,4-3,6 punti sotto (sempre negative in entrambi i semi), su OrgaSegment pareggiano (da -0,1 a -0,5). Con 2 semi e un'oscillazione tra semi di circa 2,5 punti su `ours` (Ya 87,03 e 88,88), le differenze piccole sono dentro il rumore; le differenze di 2-3,6 punti su `ours` sono coerenti nei due semi.
- **Regola (b) soddisfatta solo in parte.** `hynet` e `hynet_res` reggono meglio i livelli estremi fuori range (gamma 0,5, blur 4, noise 0,1: da +1,7 a +4,8 punti), senza perdite in-domain su `orgaseg` (da -0,1 a -0,3); sul nostro dataset perdono 1,4-2,6 punti a immagine pulita. `hynet_res` su `orgaseg` è l'unico caso con media delle perturbazioni migliore con margine (+0,16, IC [+0,04, +0,28]).
- **`hynet_par` (armonizzatore vincolato) è peggiore su tutto:** -3,6 su `ours`, e crolla con blur 4 (-13,0). Va scartato nella forma attuale.
- **Verifica mancante per la regola (b):** il confronto vale solo contro Y-Net addestrato con i range di default. Bisogna ripeterlo con `--augment_range wide` (range che contengono tutta la sweep) per vedere se il vantaggio fuori range sopravvive a un'augmentation che lo copre. Eseguita il 2026-11-16: vedi la sezione successiva.
- **Ablazione OrganoID:** toglierlo dall'addestramento peggiora `orgaseg` di 0,9 punti (3 semi su 3); su `ours` non c'è effetto netto (+2,4, -2,8, +0,2).
- **Peso sugli oggetti piccoli (Sx):** aumenta il richiamo ma funziona su `ours` e peggiora `orgaseg`: va tarato per dominio o usato solo in combinazione con altro.
- **Consistenza (C):** utile su `orgaseg` (+1,0), neutra su `ours` (-0,2).
- **Augmentation fotometrica:** Ya contro Rx vale +3,9 su `ours` e -0,25 su `orgaseg`.

### Verifica con augmentation ampia (lanciata e in parte conclusa il 2026-11-16)

Per decidere la regola (b): `ynet_raw` (tag Wy) e `hynet_res` (tag Wr), scala + `--augment` con `--augment_range wide` (range che contengono tutta la sweep di robustezza), `ours` e `orgaseg`, semi 42 e 43, solo sulle 3060 (geppetto e geminio); radici `outputs/lodo_Wy*` e `outputs/lodo_Wr*`, job 2307-2322 (addestramenti e robustezza in dipendenza). Confronto: Wr contro Wy sulla sweep completa e a immagine pulita, e Wy contro Ya (stessi range di default). Se il vantaggio di `hynet_res` fuori range sparisce con Wy, HarmNet non ha un ruolo di robustezza dimostrabile.

**Risultati (tutti i job conclusi alle 21:03 del 2026-11-16).**

Dice per pozzetto a immagine pulita (seme 42 / seme 43 / media):

| gruppo | `ours` | `orgaseg` |
|---|---|---|
| Ya: Y-Net grezzo, augment di default | 87,03 / 88,88 / 87,95 | 87,22 / 87,36 / 87,29 |
| Wy: Y-Net grezzo, augment ampia | 86,34 / 86,89 / 86,62 | 86,64 / 86,97 / 86,81 |
| Wr: `hynet_res`, augment ampia | 87,49 / 87,26 / 87,38 | 87,12 / 87,79 / 87,45 |

Differenze appaiate per seme: Wy-Ya -0,68 e -1,98 su `ours` (media -1,33), -0,58 e -0,39 su `orgaseg` (media -0,48); Wr-Wy +1,15 e +0,37 su `ours` (media +0,76), +0,47 e +0,82 su `orgaseg` (media +0,65): positive in tutti e 4 i confronti.

Robustezza (`hynet-robustness`, differenza appaiata sui pozzetti con intervallo di confidenza bootstrap al 95% sui pozzetti; semi 42 e 43 insieme):

| Wr meno Wy | `ours` | `orgaseg` |
|---|---|---|
| pulito | +0,76 [+0,35, +1,19] | +0,65 [+0,50, +0,80] |
| gamma 0,5 | +2,49 | +0,33 |
| gamma 2,0 | +0,83 | +1,03 |
| noise 0,1 | +0,57 | +0,27 |
| blur 4 | +3,51 | +1,70 |
| media perturbate | +1,16 [+0,75, +1,62] | +0,64 [+0,50, +0,78] |
| perdita media rispetto al pulito (positivo = più robusto) | +0,40 [+0,34, +0,46] | -0,01 [-0,05, +0,04] |

Wr contro Ya (Dice assoluto, stesse condizioni di valutazione): su `ours` il pulito è 87,38 contro 87,95 (-0,57), la media delle perturbate 86,32 contro 85,98 (+0,34), con gamma 0,5 +4,6 e con blur 4 +4,9. Su `orgaseg` il pulito è 87,45 contro 87,29 (+0,16), la media delle perturbate 87,16 contro 86,35 (+0,81), con noise 0,1 +2,9 e con blur 4 +7,1. Attenzione: il seme 43 di Ya gira su `claudio` e quello di Wy e Wr sulle 3060, quindi per `ours` questo confronto non è appaiato sulla GPU (l'effetto della GPU misurato è piccolo, vedi sopra, ma non nullo).

Letture:
- **A parità di augmentation ampia, HarmNet residuo batte Y-Net grezzo** sia a immagine pulita sia su tutta la sweep, su entrambi i domini (positivo in tutti e 4 i confronti per seme). Il vantaggio non sparisce quando l'augmentation copre i livelli di test, quindi non dipende solo dal fatto che Y-Net non li ha visti.
- **L'augmentation ampia costa in-domain a Y-Net** (Wy-Ya -1,33 su `ours`, -0,48 su `orgaseg`) e protegge dagli estremi. Ya è migliore sul pulito e dentro i range, ma crolla fuori range (`orgaseg`: noise 0,1 e blur 4 sono 2-5 punti sotto Wy).
- **Wr non supera Ya sul pulito su `ours`** (-0,57) e lo pareggia su `orgaseg` (+0,16, dentro il rumore): la regola (a) resta non soddisfatta. La regola (b) è soddisfatta nella forma "vince in modo netto fuori dai range": Wr guadagna da 2,9 a 7,1 punti agli estremi (noise 0,1, gamma 0,5, blur 4) e la media delle perturbate è migliore su entrambi i domini (+0,34 su `ours`, +0,81 su `orgaseg`), con una perdita sul pulito solo su `ours`. Si legge come un compromesso a favore della robustezza, non come un guadagno in accuratezza.
- **Cautele.** `ours` ha un'oscillazione tra semi di circa 2,5 punti con due semi: le differenze di +0,4/+1,2 di Wr-Wy sono nella stessa direzione nei due semi ma piccole rispetto al rumore, mentre gli intervalli di confidenza sono sui pozzetti e non sui semi. Gli estremi (blur 4, noise 0,1, gamma 0,5) sono sintetici e non sono immagini reali con altri microscopi.

### Confronto equo con i baseline (2026-11-17)

Finora U-Net e CAMPEOD erano valutati senza scala e senza augmentation (CAMPEOD: un solo seme, `outputs/lodo_cam_micro4`), mentre i nostri modelli avevano la ricetta completa, quindi il confronto non era equo. Ho addestrato i due baseline con la stessa ricetta di Ya (`SCALE=2 AUGMENT=true`, augmentation di default), tre domini di test, semi 42 e 43. Ho aggiunto anche `hynet_res` con la stessa ricetta su OrganoID (tag Ra), dove mancava. Radici `outputs/lodo_{Ua,Ka,Ra}[_s43]` (Ua = U-Net, Ka = CAMPEOD, Ra su OrganoID), job 2326-2344, elenco in `~/jobs_plan.txt`. Il riepilogo è `scripts/summarize_plan.py --domains ours orgaseg organoid` (gruppi Ua, Ka, e `augsc` per il Y-Net grezzo su OrganoID, cioè il gruppo E della sezione 7).

Dettagli di esecuzione: il seme 42 è su geppetto e geminio (RTX 3060), il seme 43 su claudio (RTX 5070 Ti); CAMPEOD con `MICRO=4` (stesso batch accumulato su 4 passi, per stare in 12 GB). I primi lanci di CAMPEOD (job 2325-2336) sono falliti con `KeyError: 'campeod'`, perché la voce era sparita da `src/hynet/models/factory.py` (la pulizia della sezione 10); l'ho reinserita (import locale, perché `campeod.py` e la cartella `CAMPEOD/` non sono nel repository) e ho rilanciato i 6 job (2339-2344).

Dice per pozzetto, media dei semi 42 e 43 (valori per seme tra parentesi):

| modello (ricetta: scala 2 + augment standard) | `ours` | `orgaseg` | `organoid` |
|---|---|---|---|
| Y-Net grezzo (Ya; su OrganoID: E, 3 semi) | 87,95 (87,03 / 88,88) | 87,29 (87,22 / 87,36) | 81,98 (82,13 / 81,47 / 82,35) |
| `hynet_res` (Ra) | 86,55 (86,71 / 86,38) | 87,16 (87,20 / 87,11) | 82,85 (81,87 / 83,83) |
| `hynet_res`, augment ampia (Wr) | 87,38 (87,49 / 87,26) | 87,45 (87,12 / 87,79) | non provato |
| CAMPEOD (Ka) | 87,74 (86,87 / 88,61) | 87,13 (87,41 / 86,85) | 83,85 (83,73 / 83,96) |
| U-Net (Ua) | 85,85 (86,54 / 85,16) | 87,14 (87,10 / 87,19) | 80,52 (80,39 / 80,64) |

Differenze appaiate per seme (media di 42 e 43): Ka-Ya -0,21 su `ours` e -0,16 su `orgaseg`; Ka-E +2,04 su `organoid` (+1,60 e +2,49). Ua-Ya -2,10, -0,15; Ua-E -1,29. Ra-Ya -1,41, -0,13; Ra-E +1,05 (-0,26 e +2,35). Richiamo/precisione degli oggetti sotto 200 px (media dei semi): su `organoid` Ka 51/26, Ra 48/21, Ua 60/18.

Letture:
- **Non battiamo CAMPEOD.** Sui due domini intestinali i modelli con HarmNet (Ra, Wr) e CAMPEOD sono a pari entro 0,3 punti su `orgaseg` (Wr 87,45 contro 87,13) e a 0,4-1,2 punti su `ours` (Wr 87,38, Ra 86,55 contro 87,74), cioè dentro l'oscillazione tra semi (circa 2,5 punti su `ours`). Su OrganoID CAMPEOD è sopra di circa 1 punto a Ra e di circa 2 al Y-Net grezzo, e per il Y-Net grezzo la differenza è coerente nei due semi.
- **U-Net è battuta** sul nostro dataset (Ra +0,70, Wr +1,53, Y-Net grezzo +2,10, positivi in entrambi i semi) e su OrganoID (+2,3 per Ra), ed è pari su `orgaseg`.
- **Il numero di CAMPEOD in tabella 2 non è più il riferimento**: con la ricetta completa CAMPEOD sale sul nostro dataset a 87,74 (dai 83,35 senza scala e augmentation), mentre su `orgaseg` resta a 87,13 (86,97 prima).
- **Cosa resta in vantaggio a HY-Net:** nessun guadagno di accuratezza misurabile. Il punto di forza dimostrato è la robustezza (sezione precedente) e il costo, ma il costo non è stato misurato in questo ciclo (parametri, tempo per epoca: da rilevare per il paper).

Cautele: due semi (tre solo per la riga E); il seme 43 di Ka, Ua e Ra gira su una GPU diversa dal seme 42 (l'effetto della GPU misurato è piccolo, vedi sopra); il riferimento E di OrganoID ha anche il seme 44, ma le differenze appaiate usano solo 42 e 43; Wr non è stato provato su OrganoID; la robustezza di Ka e Ua non è stata misurata; nessun intervallo di confidenza sulle differenze tra modelli; scelte di ricetta fatte guardando i domini di test.

### Robustezza di CAMPEOD e U-Net, e costo dei modelli (2026-11-17)

Robustezza (`hynet-robustness`, tutta la sweep, semi 42 e 43 insieme, differenza appaiata sui pozzetti, intervallo bootstrap al 95% sui pozzetti). Job 2346-2353; gruppi `outputs/robustness/lodo_{Ka,Ua}_<dominio>[_s43]`. Le valutazioni del seme 43 sono state eseguite sulle 3060 (i modelli del seme 43 sono stati addestrati su `claudio`). Tutti i modelli sono addestrati con scala 2 e augmentation fotometrica di default, tranne Wr che usa l'augmentation ampia.

Dice assoluto (media su pozzetti e semi):

| impostazione | `ours`: CAMPEOD | Wr | Y-Net Ya | U-Net | `orgaseg`: CAMPEOD | Wr | Y-Net Ya | U-Net |
|---|---|---|---|---|---|---|---|---|
| pulito | 87,74 | 87,38 | 87,95 | 85,85 | 87,13 | 87,45 | 87,29 | 87,14 |
| gamma 0,5 | 80,58 | 81,39 | 76,76 | 77,48 | 88,49 | 88,43 | 88,68 | 88,78 |
| noise 0,1 | 86,36 | 85,93 | 86,09 | 84,10 | 83,98 | 86,90 | 84,05 | 86,09 |
| blur 2 | 82,90 | 86,04 | 85,91 | 83,92 | 86,57 | 87,20 | 86,72 | 86,60 |
| blur 4 | 50,16 | 82,78 | 77,87 | 76,91 | 74,43 | 85,90 | 78,81 | 79,73 |
| media perturbate | 84,10 | 86,32 | 85,98 | 84,24 | 85,88 | 87,16 | 86,35 | 86,40 |

Wr (`hynet_res` con augmentation ampia) meno CAMPEOD: pulito -0,36 [-0,65, -0,07] su `ours` e +0,32 [+0,15, +0,49] su `orgaseg`; media delle perturbate +2,22 [+1,85, +2,62] e +1,27 [+1,02, +1,53]; blur 4 +32,6 e +11,5; noise 0,1 -0,43 e +2,92; gamma 0,5 +0,81 e -0,06. Su `ours` CAMPEOD è leggermente meglio di Wr su rumore, ombreggiatura (-0,5 e -0,8) e JPEG; su `orgaseg` Wr è sopra su quasi tutte le perturbazioni.

Il Y-Net grezzo Ya (con augmentation di default, come CAMPEOD) supera CAMPEOD sulla media delle perturbate di +1,87 su `ours` e +0,47 su `orgaseg` (blur 4: +27,7 e +4,4), ma resta peggiore di CAMPEOD con gamma 0,5 su `ours` (-3,82). U-Net: perde 1,9 punti a immagine pulita su `ours`, è pari su `orgaseg`, e dà +26,8 e +5,3 con blur 4 rispetto a CAMPEOD.

Letture:
- **CAMPEOD è fragile alla sfocatura** (blur 4: 50,2 su `ours`, 74,4 su `orgaseg`) e al rumore forte su `orgaseg` (noise 0,1: 84,0). Questo spiega quasi tutto il vantaggio di robustezza di HY-Net e anche degli altri modelli senza CAMPEOD; non è specifico di HarmNet. Probabilmente dipende anche dal ridimensionamento a 512x512 dell'ingresso di CAMPEOD (non verificato).
- **Il confronto non è del tutto equo a favore di HY-Net**: Wr è addestrato con l'augmentation ampia, CAMPEOD con quella di default; CAMPEOD con augmentation ampia non è stato provato. Il confronto equo sullo stesso range è Ya contro Ka (sopra), dove il vantaggio esiste ma è più piccolo su `orgaseg`.
- **Cosa si può dire nel paper:** a Dice pulito pari (entro 0,4 sui domini intestinali), HY-Net con augmentation ampia è molto più robusto di CAMPEOD alla sfocatura e al rumore forte, perde poco o nulla su gamma, ombreggiatura e JPEG, e ha un quinto dei parametri. Il merito dell'armonizzatore residuo isolato resta quello della sezione precedente (Wr meno Wy: da +0,6 a +1,2 sulla media delle perturbate).

Costo (RTX 3060, job `bench_cost` 2354 e `bench_cam` 2355, `hynet-benchmark`, 5 iterazioni cronometrate con AMP; la durata per epoca è un'estrapolazione a 195 immagini):

| modello | parametri | inferenza fp32 (480x640, batch 1) | memoria di picco in addestramento | min/epoca (batch) |
|---|---|---|---|---|
| U-Net | 31,0 M | 58 ms | 4,25 GB | 0,31 (4) |
| Y-Net grezzo | 36,1 M | 88 ms | 4,10 GB | 0,49 (2) |
| `hynet_res` e `hynet` | 37,5 M | 133 ms | 6,17 GB | 0,73 (2) |
| CAMPEOD | 177,1 M | 117 ms | 5,33 GB | 0,77 (1); a batch 4 esaurisce la memoria |

CAMPEOD ha 4,7 volte i parametri di HY-Net ma l'inferenza è più veloce (117 contro 133 ms): il vantaggio di HY-Net è sui parametri, non sulla velocità. I tempi per epoca sono a batch diversi (quello usato nel LODO è 4 con accumulo: 2 passi da 2 per i modelli con HarmNet, 4 da 1 per CAMPEOD), quindi vanno letti come ordine di grandezza.
