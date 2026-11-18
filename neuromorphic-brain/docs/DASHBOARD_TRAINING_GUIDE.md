# Vezérlőpult és tanítási útmutató

Ez az útmutató egy helyen foglalja össze a Neuromorphic Brain webes vezérlőpultját, a helyben futtatható tanítási feladatokat, az adatok eredetét, a metrikák értelmezését és az ellenőrzés módját. A komponensek matematikai és belső architekturális részleteit az [ARCHITECTURE.md](ARCHITECTURE.md), az API-adatszerződéseket az [API_SPECIFICATION.md](API_SPECIFICATION.md) írja le.

## 1. Indítás és leállítás

A parancsokat a `neuromorphic-brain` könyvtárból futtasd:

```powershell
python dashboard_app.py
```

Alapértelmezésben elindul a webes vezérlőpult, a terminálos nézet és egy háromepochos demonstráció. A demonstráció befejezése után a webkiszolgáló tovább fut, amíg `Ctrl+C`-vel le nem állítod.

| Parancs | Működés |
|---|---|
| `python dashboard_app.py` | Webes felület + terminálos nézet + demonstráció |
| `python dashboard_app.py --web` | Webes felület; nem indul automatikus tanítás |
| `python dashboard_app.py --web --simulate --no-tui` | Webes felület + demonstráció, terminálos nézet nélkül |
| `python dashboard_app.py --tui` | Terminálos nézet és demonstráció webes felület nélkül |
| `python dashboard_app.py --web --port 8013` | Webes felület a megadott porton |

A webes felület alapértelmezett címe `http://localhost:8000`. Leállításkor a folyamat bezárja a dashboard szervert is.

## 2. A felület részei

### Control Center

- **Metrikacsík:** aktuális epoch, batchenkénti helyi veszteség, batch-pontosság, meglepetés, viselkedési mód, döntési útvonal és futási állapot.
- **Homeostatic State:** a kíváncsiság, hibaarány, CPU-terhelésből simított erőforrás-stressz és bizonytalanság értéke.
- **Neural Activity:** a tényleges bemeneti, rejtett rétegbeli és kimeneti aktivitásból összesített ábra. A rajzolt csomópontok és azok elrendezése szemléltető jellegű, nem a teljes hálózat egyedi neuronjait mutatja.
- **Memória:** az aktuális futás lokális epizodikus eseményei, szemantikus fogalmai és kapcsolatai, valamint az aktuális perceptuális állapot aktív/inaktív jelzése.
- **Charts és System Logs:** a futás közben mért batch-értékek és a friss eseménynapló.
- **Tanítási vezérlők:** prompt, tanítási ráta (LR), batchméret, tervkérés, indítás és leállítás.

Mérés előtt a mérőmezők `—` jelet mutatnak. Ez azt jelenti, hogy még nem érkezett teljes metrikaminta; nem nulla eredményt vagy üres memóriát jelent.

### Tesztközpont

A Control Center melletti nézetváltóval érhető el. Három csoportot tartalmaz:

1. helyben futtatható feladatok és azok promptmintái;
2. külső benchmarkok, amelyekhez jelenleg nincs futtatóadapter csatlakoztatva;
3. tanítási paradigmák és a tényleges implementáltságuk.

A prompt-súgó ugyanebből a katalógusból készül, ezért a felületen látható leírások a `/api/catalog` válaszát követik.

## 3. Tanítási munkafolyamat

1. Adj meg egy feladatot a promptban.
2. Állítsd be az epochok számát, a tanítási rátát és a batchméretet.
3. Kérd le a tervet. A terv a felismerhető feladattípust, adatforrást és beállításokat írja le; önmagában nem futtat tanítást.
4. Indítsd el a futást. A felület batchenként közli a mért adatokat.
5. A **Stop** gomb leállítási jelzést küld; az aktuális batch után a folyamat szabályosan befejezi a leállást.

A paramétereknek pozitív értékűeknek kell lenniük. Üres prompt, hibás paraméter vagy nem támogatott benchmark esetén az API hibát jelez, és nem indít helyette másik, félrevezető feladatot. Egy időben csak egy tanítási folyamat futhat.

## 4. Helyben támogatott feladatok és adatforrások

| Feladat | Promptfelismerés | Adat és osztályok |
|---|---|---|
| XOR | `xor` | Szintetikus, kétbites XOR tüskeminta; 2 kimeneti osztály |
| Bináris paritás | `parity` vagy `paritás` | Szintetikus, nyolcbites paritásfeladat; 2 kimeneti osztály |
| Általános osztályozás | `classification` vagy más általános osztályozási prompt | 10 osztály; `data/n_mnist/processed_nmnist.pt` betöltése, ha létezik, különben szintetikus DVS |
| Szekvencia-osztályozás | `sequence` vagy `szekvencia` | Kísérleti alias az általános osztályozási útvonalhoz; nem külön szekvencia-benchmark |

A feldolgozott N-MNIST fájlnak a `neuromorphic-brain/data/n_mnist/processed_nmnist.pt` útvonalon kell lennie, és `samples`, `labels` mezőket kell tartalmaznia. Ha a fájl nincs jelen, az általános osztályozás **Synthetic DVS** adatot használ; ez valós pipeline-futtatás szintetikus bemeneten, nem valódi N-MNIST mérés. Az XOR és a paritás mindig szintetikus feladat.

A ténylegesen kiválasztott forrás az első batch feldolgozása után a **DATA** mezőben és az API `data_source` értékében látható.

## 5. A kijelzett mérőszámok jelentése

| Mező | Mit mutat? | Fontos korlát |
|---|---|---|
| Epoch | Az éppen feldolgozott epoch sorszáma | Nem az összes befejezett epoch kumulált száma |
| Batch accuracy | A legutóbb feldolgozott batch helyesen osztályozott mintáinak aránya | Tanítóbatch-érték, nem külön teszt- vagy validációs pontosság |
| Local loss / batch | A tanító batch DECOLLE-rétegein mért lokális veszteségek átlaga | Nem teszthalmaz- vagy epoch-átlag |
| Surprise | A prediktív modul batchből származó meglepetési értéke | A modell prediktív kimenete, nem külső minősítés |
| Curiosity, error rate, uncertainty | Simított belső értékek: a kíváncsiság a prediktív meglepetésből, a hibaarány az aktuális batch hibájából, a bizonytalanság pedig a router komplexitási jeléből származik | Nem külső vagy kalibrált minősítések |
| Process CPU load (`/api/state`) | Két egymást követő batch-minta között mért folyamat-CPU-idő-különbség / eltelt falióra-idő, `[0,1]` tartományra korlátozva; az intervallum a batch-ek közötti várakozást is tartalmazza | API-mező; a felületen a belőle simított stressz látható, nem külön nyers CPU-gauge. Egy logikai maghoz viszonyított folyamatmérés, nem teljes gépszintű CPU- vagy RAM-mérés |
| Process CPU stress | A fenti CPU-terhelésből simított homeosztatikus érték | Nem memóriatelítettség vagy operációsrendszer-szintű terhelés |
| Route / Mode | A pipeline aktuális döntési útvonala és viselkedési módja | Belső router- és homeosztázis-eredmény |
| Spikes / batch | A feldolgozott batch rejtett rétegeiben található nemnulla tüskék száma | Batchenként változik |
| Connections | A hálózati rétegek súlymátrixainak és fix readout-mátrixainak elemszáma | Modellstruktúrából számított elemszám, nem élő aktivitás |
| Layer activity | Bemeneti átlagok, rejtett rétegek átlagos tüskearánya és kimeneti átlagos valószínűségek | Az ábra ezeket kevesebb, szemléltető csomópontra sűríti |
| Output | A tényleges kimeneti aktivitásvektor mérete | XOR/paritás esetén 2, általános osztályozásnál 10 |
| Current percept | Megvan-e az aktív pipeline `current_perceptual_state` állapota | Aktív/inaktív jelző, nem általános munkamemória-darabszám |
| Local episodic events | Az aktív pipeline folyamaton belüli eseménymásolatának száma | Nem az összes korábbi Qdrant-folyamat globális elemszáma |
| Session semantic concepts / relations | Az aktív pipeline szemantikus gráfjának csomópont- és élmennyisége | Az aktuális munkamenet memóriája |
| Training progress | `step / total_steps`, a tényleges dataloader batchjeiből számolva | Az epoch-jelző ettől külön az aktuális epochot mutatja |

A chartok batchenként frissülnek. A neurális ábra nem használ véletlen tüskéket vagy mesterséges pulzálást; aktivitás hiányában „Waiting for telemetry” felirat jelenik meg.

## 6. Benchmarkok, tanítási paradigmák és rangsor

- **Felügyelt tanulás:** a helyi XOR, paritás és osztályozási útvonal címkézett adatokkal működik.
- **Felügyelet nélküli / prediktív tanulás:** prediktív kiegészítő veszteség létezik, de önálló, címkézetlen kezelőfelületi tanítási mód nincs.
- **Megerősítéses tanulás / RLHF:** nincs csatlakoztatott jutalommodell vagy interaktív tanulási környezet.
- **MMLU, HumanEval, GSM8K/MATH és LMSYS Arena:** katalógusbejegyzések, nem a helyi SNN-futtatóval jelenleg végrehajtható benchmarkok. Ezek promptként történő indítását a rendszer visszautasítja.
- **AI rangsor:** LMSYS Chatbot Arena, BenchLM és Epoch AI választható típusok. Jelenleg egyikhez sincs hivatalos forrás-URL konfigurálva, nincs forrásadapter és nincsenek összehasonlítható Digitalis-Agy eredmények. Emiatt az oldal nem gyárt helyezést vagy értéket; frissítéskor a hiányzó konfigurációt jelzi.

Különböző rangsorok eltérő mérőszámai nem hasonlíthatók közvetlenül össze. A helyi batch-pontosság nem helyettesíti például az Arena Elo-pontszámát vagy az MMLU eredményét.

## 7. HTTP API röviden

| Metódus és útvonal | Funkció |
|---|---|
| `GET /` vagy `/index.html` | Dashboard HTML |
| `GET /api/state` | Aktuális telemetriai pillanatkép |
| `GET /api/stream` | SSE élő állapotfolyam |
| `GET /api/catalog` | Helyi workflow-, benchmark-, paradigma- és rangsorkatalógus |
| `POST /api/plan` | Tanítási terv; JSON: `prompt`, opcionálisan `epochs`, `lr`, `batch_size` |
| `POST /api/train/start` | Helyi tanítás indítása ugyanilyen paraméterekkel |
| `POST /api/train/stop` | Egy futó tanítás kooperatív leállításának kérése |
| `POST /api/update` | Külső telemetriafrissítés; `has_metrics: true` csak teljes, valós mérési minta esetén |
| `GET /api/leaderboard?type=<id>` | Rangsor metaadatainak lekérése |
| `POST /api/leaderboard/refresh` | Rangsor frissítése; jelenleg nincs konfigurált forrásadapter |

Hibás terv- vagy indítási paraméter, illetve nem támogatott benchmark esetén az API `400` hibát ad. Ismeretlen rangsortípus szintén `400`; konfigurálatlan rangsor frissítése `409`; adapter nélküli konfigurált forrás `501` választ adna.
A `has_metrics: true` jelzéshez teljes telemetriaminta szükséges; hiányos külső frissítés `400` hibát kap. Későbbi részleges frissítés megtartja a már érvényes minta mérési állapotát.

## 8. Ellenőrzés és hibakeresés

A projekt tesztjei a `neuromorphic-brain` könyvtárból futtathatók:

```powershell
python -m pytest tests -q
```

A repository gyökeréből, a projekt virtuális környezetével:

```powershell
& '.\neuromorphic-brain\.venv\Scripts\python.exe' -m pytest '.\neuromorphic-brain\tests' -q
git diff --check
```

Gyakori, helyes állapotok:

- **Synthetic DVS** jelenik meg: a feldolgozott N-MNIST fájl nem található; a pipeline ettől még valós, csak a bemeneti adat generált.
- **`—` a metrikák helyén:** még nem érkezett teljes batchmérés, vagy még nem kezdődött el a tanítás.
- **A rangsor frissítése nem ad sorokat:** nincs konfigurált hivatalos forrás és adapter; ez szándékosan nem helyettesül kitalált adatokkal.
- **Külső benchmark promptja hibát ad:** a benchmark katalógusban szerepel, de nincs helyi futtatója.
- **A Stop nem azonnal állítja le a batch-et:** a leállítás kooperatív; a folyamat a folyamatban lévő batch után áll meg.
