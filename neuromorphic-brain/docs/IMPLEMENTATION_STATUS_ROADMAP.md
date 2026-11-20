# Megvalósítási állapot és fejlesztési ütemterv

## 1. Cél

Az ötlet célja egy moduláris, futás közben alkalmazkodó neuromorf kognitív agent:

- helyi, szinaptikus szintű tanulás;
- hierarchikus munkamemória, epizodikus memória és szemantikus tudásgráf;
- belső homeosztatikus állapotkezelés;
- prediktív hibák alapján történő útvonalválasztás;
- Rendszer 1 / Rendszer 2 döntési kapuzás;
- absztrakt látens térben működő világmodell és többlépéses szimuláció;
- alvási fázisban végzett konszolidáció és tartós, biztonságos tudásfrissítés.

Ez a dokumentum a dokumentált célállapotot, a jelenlegi implementációt és a következő megvalósítási lépéseket különíti el egymástól.

## 2. Pillérszintű állapot

| Pillér | Jelenlegi állapot | Dokumentált célhoz viszonyítva |
|---|---|---|
| DECOLLE és LIF hálózat | Használható prototípus | Jelentős részben megvalósult |
| AdaMax helyi optimizálás | Megvalósítva rétegenként | Megvalósult |
| Szinaptikus gráf és Hebb-pruning | DECOLLE együttaktivitás, JSON checkpoint, 100 lépésenkénti pruning | A fő funkciók megvannak; a hosszabb benchmark és nagy hálózatos skálázás további validációt igényel |
| Munkamemória | Redis + szálbiztos in-memory fallback | Megvalósult |
| Epizodikus memória | Qdrant + in-memory fallback | Alapfunkciók megvalósultak |
| Szemantikus memória | NetworkX irányított gráf | Folyamatbeli prototípus; tartós tárolás hiányzik |
| Alvási konszolidáció | Egyszerű epizód-összegzés és driftvédő | Klaszterezés és háttérworker hiányzik |
| Globális homeosztázis | Négy belső állapot és viselkedési mód | Megvalósult prototípus |
| Prediktív kódolás | PyTorch autoenkóder és meglepetési küszöb | Alapmodell megvan; Mamba SSM hiányzik |
| CDR útvonalválasztás | Súlyozott, validált négydimenziós router | A fő bemenetek jelentős része még kiegészítésre szorul |
| Rendszer 1 / 2 architektúra | Elkülönült rövid és lassú router osztály | A Rendszer 2 még demonstratív |
| JEPA világmodell | Tanítható látens átmenet- és értékfej | Prototípus; valódi akció- és környezetmodell hiányzik |
| Webdashboard és API | Futtatható dashboard, telemetria és tanítási vezérlés | Helyi prototípusokra jól illeszkedik |
| Konténeres infrastruktúra | Docker Compose brain-core, Redis és Qdrant | Megvalósult alaptelepítés |

## 3. Ami már működőképes vagy prototípus szinten létezik

### 3.1 Neuromorf tanulási mag

- LIF szintű membránpotenciál és tüsketermelés.
- Surrogate gradiens alapú tüskefelévezetés.
- Fix, véletlen rétegbeli readout.
- Rétegenkénti DECOLLE veszteség és AdaMax optimizer.
- Membrán- és tüskepályák detatcholása a rétegek között.
- Tüskearány-regularizáció.
- CPU és GPU eszköz kiválasztás.

Kapcsolódó implementáció:

- [decolle_layer.py](/d:/Digitalis-Agy/neuromorphic-brain/src/core/decolle_layer.py)
- [pipeline.py](/d:/Digitalis-Agy/neuromorphic-brain/src/core/pipeline.py)

### 3.2 Memóriahierarchia

- Redis-alapú vagy lokális in-memory munkamemória TTL-lel.
- Qdrant-alapú vagy lokális vektortár epizodikus eseményekhez.
- Pydantic esemény- és vektorséma ellenőrzés.
- NetworkX szemantikus irányított gráf.
- Épülő epizodikus driftellenőrzés.

Kapcsolódó implementáció:

- [working_memory.py](/d:/Digitalis-Agy/neuromorphic-brain/src/memory/working_memory.py)
- [episodic_memory.py](/d:/Digitalis-Agy/neuromorphic-brain/src/memory/episodic_memory.py)
- [semantic_memory.py](/d:/Digitalis-Agy/neuromorphic-brain/src/memory/semantic_memory.py)
- [sleep_consolidation.py](/d:/Digitalis-Agy/neuromorphic-brain/src/memory/sleep_consolidation.py)

### 3.3 Prediktív és homeosztatikus szabályozás

- Batchátlaggal számított prediktív residual hiba.
- Rejtett veszteség feláramolása és meglepetési küszöb.
- Kíváncsiság, hibaarány, erőforrás-stressz és bizonytalanság.
- Cautious, Conservation, Exploration és Balanced viselkedési mód.
- Magas meglepetés vagy biztonsági mód esetén Rendszer 2 kényszeres útválasztás.

Kapcsolódó implementáció:

- [residual_error.py](/d:/Digitalis-Agy/neuromorphic-brain/src/predictive/residual_error.py)
- [state_vector.py](/d:/Digitalis-Agy/neuromorphic-brain/src/homeostasis/state_vector.py)
- [cdr_router.py](/d:/Digitalis-Agy/neuromorphic-brain/src/routing/cdr_router.py)

### 3.4 Tanítási és telemetriai felület

- Szintetikus DVS, XOR és bináris paritás adatkészlet.
- Feldolgozott N-MNIST betöltés helyi fájlból.
- Batchenkénti telemetria.
- Tanítás indítása, leállítása és állapotstreamelése.
- Helyi webes dashboard és terminálos nézet.

Kapcsolódó dokumentáció:

- [DASHBOARD_TRAINING_GUIDE.md](/d:/Digitalis-Agy/neuromorphic-brain/docs/DASHBOARD_TRAINING_GUIDE.md)
- [API_SPECIFICATION.md](/d:/Digitalis-Agy/neuromorphic-brain/docs/API_SPECIFICATION.md)

## 4. Ami hiányzik vagy demonstrációs szinten maradt

### 4.1 Kritikus hiányosságok

1. **A szinaptikus gráf integrációja részleges.**
   - A pipeline mostantól az egymás melletti DECOLLE-rétegek azonos időlépésben együtt tüzelő neuronjait irányított gráfélekként rögzíti.
   - A gráf kapcsolatszáma megjelenik a pipeline eredményeiben.
   - A verziózott JSON checkpoint mentése és betöltése megvalósult.
   - Inaktív élek ritka, 100 pipeline-lépésenkénti pruningja megvalósult; az időbélyeg megőrzi a tényleges utolsó aktivációt.
   - Az 50 epochos kis CPU stabilitásteszt lefedi a súlyok, élek és aktivitáselőzmények alapvető korlátait; nagy hálózatos benchmark még szükséges.

2. **Az alvási konszolidáció nem elosztott és nem teljes.**
   - Hiányzik az epizodikus hasonlósági klaszterezés.
   - Hiányzik a koszinusz küszöbön alapuló redundanciaszűrés.
   - Hiányzik a felesleges epizódok végleges törlése.
   - Nincs időzített vagy ütemezett NREM/REM háttérfolyamat.

3. **A Rendszer 2 nem valódi tervezőmotor.**
   - A DualProcessRouter fix szöveges választ ad.
   - Nincs MCTS vagy szimbolikus feladatlezárás.
   - A JEPA nem cselekvésekből tanul meg környezeti dinamikát, hanem neutral vagy egyhot osztályjelöltekből.
   - Az egyik különálló JEPA sandbox nincs integrálva a fő döntési folyamatba.

4. **A CDR még nem valós feladatdimenziókat mér.**
   - A `stakeholder_count` a pipeline-ban fixen egy.
   - A domain-crossing becslése csak batch-pontosságon alapul.
   - A router nem olvas valós szemantikus részgráfot vagy entitásfüggőséget.

5. **A szemantikus tudás nem tartós.**
   - A NetworkX gráf processzciklusonként újratelepülhet.
   - Hiányzik a verziózott vagy különálló Graph Database / szerveres tároló.

### 4.2 Jelentős hiányosságok

- Nincs Mamba vagy más dokumentált állapottérmodell a prediktív modulban.
- Qdrant kapcsolaton elérhető események esetén a `retrieve_recent()` nem olvas direkt a szerverből.
- Hiányzik a reprezentációk időbeli változását mérő, kalibrált bizonytalanság.
- A memória-, router- és JEPA-módosítások nem rendelkeznek felülírási és visszaállítási szabályzattal.
- Az épített benchmarkkatalógus nem tartalmaz valódi MMLU, HumanEval, GSM8K vagy MATH adaptert.
- Nincs CPU-/memória-/latencia mérés szervezeti vagy többprocesszes környezetben.
- Nincs éles, osztott elosztott futtatás és feladatütemezés.

## 5. Célállapot mérhető kritériumokkal

A projekt akkor tekinthető az eredeti elképzeléshez közelebbinek, ha a következő kritériumok teljesülnek.

### Fázis 1 – Stabil integrált szinaptikus rendszer

- [x] A SynapticGraphManager csomópontjai és élei a DECOLLE neurális rétegek aktivitásához kapcsolódnak.
- [x] Új edge csak ténylegesen, azonos batch-elemben és időlépésben együtt aktivált neuronpárból jöhet létre.
- [x] A pruning idősorozatos karbantartási műveletként, 100 pipeline-lépésenként lefut.
- [x] Gráfstruktúra checkpointból helyreállítható.
- [x] Legalább 50 epochos ismételt kis CPU-futásban a súlyok és gráfértékek végesek, az aktivitáselőzmények korlátosak.

### Fázis 2 – Valódi memória konszolidáció

- [ ] Epizodikus események cosine vagy beágyazás-alapú klaszterezése.
- [ ] 0.85 körüli hasonlósági küszöbön egyetlen absztrakt fogalommá vonás.
- [ ] Konfigurálható redundancia-, életkor- és relevanciaszűrés.
- [ ] Szemantikus gráf tartós tárolása és betöltése.
- [ ] Környezetfüggetlen, időzített háttérworker.
- [ ] Drift-rejekció mérhető identitás- vagy referenciavektor ellen.

### Fázis 3 – Valódi Rendszer 2 és világmodell

- [ ] Minimum 5–10 valóban eltérő akciójelölt.
- [ ] Observált környezeti átmenetekkel betanított JEPA transition head.
- [ ] Soklépéses rollout és stabilitás vagy kockázat alapú értékelés.
- [ ] MCTS vagy dokumentált alternatív keresőmotor integrálása.
- [ ] Szemantikus részgráf és epizodikus hasonlóság a tervező bemenetében.
- [ ] A Rendszer 2 csak skálázott, mérhető bonyolultság és bizonytalanság esetén indul.

### Fázis 4 – Mérhető tanulási eredmény

- [ ] XOR és paritás mellett külön validációs és tesztelosztás.
- [ ] Held-out pontosság és loss jelentése.
- [ ] Valódi N-MNIST vagy más dokumentált adatforrás futtatása.
- [ ] Benchmarkadapterek és ismételhető baseline eredmények.
- [ ] Online tanulás során új feladatra emlékezés és régi tudás visszaesésének mérése.
- [ ] Minden jelentős mérőszám definíciója és reprodukálható futtatási parancsa.

### Fázis 5 – Termelési szintű üzemeltetés

- [ ] Redis és Qdrant tartós kötetekkel.
- [ ] Authentikáció és API hozzáférés-szabályozás.
- [ ] Strukturált naplózás és perzisztens metrikák.
- [ ] Egyszerű skálázás és egészségi ellenőrzés.
- [ ] Graceful shutdown, state checkpoint és helyreállítás.
- [ ] Docker és natív futtatás azonos telemetriai sémával.

## 6. Ajánlott sorrend

1. **Pipeline alapú integráció**
   - Szinaptikus gráf aktivitásjel összekötése a DECOLLE pipeline-lal.

2. **Memóriatartóság és konszolidáció**
   - Qdrant és szemantikus gráf helyes perzisztálása.
   - Aszinkron konszolidáció és epizód-klaszterezés.

3. **Valós CDR bemenet**
   - Entitásszám, tartományváltás és bizonytalanság mérésének implementálása.

4. **JEPA környezet és tervező**
   - Observált átmenetekkel tanuló akciómodell.
   - Akciójelöltekből álló rollout- és értékelőmotor.

5. **MCTS/Rendszer 2**
   - Szimbolikus vagy keresőalapú tervező, amely tényleges cselekvési eredményeket ad.

6. **Benchmark és reprodukálhatóság**
   - Adapterek, validáció, baseline és hosszútávú felejtési teszt.

## 7. Jelenlegi érvényességi korlát

Az implementáció jelenleg demonstrációs és kutatási prototípus. A dashboard, XOR, paritás és szintetikus DVS futtatás helyi pipeline-ként működhet, de a projekt még nem tekinthető kész neuromorf agyemulátornak.

Az eredeti dokumentum célállapota és a tényleges kód között a legnagyobb különbség a szinaptikus gráf, az alvási háttérmunkás, a valódi Rendszer 2 tervezés, az akcióalapú JEPA, a tartós szemantikus memória és a külső benchmarkok hiánya.
