# Agent cél és fejlesztési terv

## 1. Végső cél

Olyan általános célú agent létrehozása, amely természetes nyelvű feladatokat értelmez, és fokozatosan megtanul eszközöket használni. A kezdeti képességek közé tartozzon a fájlok olvasása és írása; a hosszabb távú cél a helyi gépen, asztali alkalmazásokban, weboldalakon és API-kon végzett, több lépésből álló feladatmegoldás.

Az agent ne kapjon rögtön korlátlan hozzáférést egy valódi géphez. Először elkülönített sandboxban gyakoroljon, és csak előre rögzített, ismételhető teszteken és biztonsági ellenőrzéseken megfelelt, verziózott változatot lehessen korlátozott valódi környezetben kipróbálni.

Az „eszközök megtanulása” a dokumentált eszközleírások, példák és a műveletek eredményeiből való tanulást jelenti. Nem jelenti azt, hogy az agent automatikusan, korlátozás nélkül hozzáférhet bármilyen programhoz vagy művelethez.

## 2. Jelenlegi projektállapot

A projekt neuromorf tanulási és memória-prototípus. Nem általános, szöveges feladatokat megoldó vagy eszközöket végrehajtó agent.

| Terület | Jelenlegi állapot |
|---|---|
| DECOLLE/LIF hálózat | PyTorch-alapú prototípus, helyi osztályozási feladatokra |
| Memória | Munkamemória, epizodikus vektormemória és szemantikus gráf; nem agent általános fájlrendszer-hozzáférése |
| Döntési útválasztás | Komplexitás- és bizonytalanság-alapú prototípus |
| JEPA és tervezés | Akciófeltételes látens modell és beam planner; megfigyelt átmeneteket igényel, valós környezetben még nincs validálva, és nem hajt végre fizikai vagy digitális műveleteket |
| N-MNIST | Importáló, előfeldolgozás és elkülönített train/test adatok rendelkezésre állnak. A teljes adatos tanítás dokumentált kísérlete a tesztkiértékelés előtt leállt, ezért a dokumentált train eredmény nem teszteredmény |
| Természetes nyelv | Nincs nyelvi modell vagy általános szövegértési/-generálási réteg |
| Eszközhasználat | Nincs eszköz-regiszter, eszközválasztó, végrehajtó-hurok, jogosultsági modell vagy sandboxos tanítás |
| Külső benchmarkok | Helyi XOR/paritás/osztályozási benchmarkok vannak; MMLU, HumanEval és hasonló feladatok adapterei nincsenek bekötve. Egyes külső rangsoradatok lekérése nem azonos benchmarkfuttatóval |

A jelenlegi teljes automatikus tesztcsomag ellenőrzéskor 159 sikeres teszttel futott le. Ez a meglévő kód tesztjeire vonatkozik; nem igazolja az általános agentcél teljesülését.

Részletek:

- [Megvalósítási állapot és fejlesztési ütemterv](IMPLEMENTATION_STATUS_ROADMAP.md)
- [Architektúra-specifikáció](ARCHITECTURE.md)
- [API-specifikáció](API_SPECIFICATION.md)
- [N-MNIST és tanítási útmutató](DASHBOARD_TRAINING_GUIDE.md)

## 3. Ajánlott architekturális irány

Ne próbáljuk a meglévő osztályozó SNN-t közvetlenül általános nyelvi agentté alakítani. A nyelvértés, a tervezés és az eszközvezérlés külön agentréteget igényel. Első megvalósításként célszerű egy meglévő nyelvi modellt használni, amelyhez az agent biztonságos eszköz-API-t ad. A neuromorf komponensek maradhatnak kísérleti memória-, predikciós vagy útválasztási modulok; szerepüket külön méréssel kell igazolni.

Az agent alapvető végrehajtási ciklusa:

1. A kérés és a rendelkezésre álló eszközök leírásának értelmezése.
2. Terv és egy konkrét, sémának megfelelő eszközhívás kiválasztása.
3. Jogosultságellenőrzés és a hívás futtatása az engedélyezett környezetben.
4. Az eszköz eredményének ellenőrzése és visszaadása az agentnek.
5. Következő lépés, befejezés vagy emberi jóváhagyás kérése.
6. A teljes lépéssor, eszközkimenet és értékelési eredmény naplózása.

Minden eszköznek géppel ellenőrizhető leírása legyen: neve, célja, paraméterei, eredménytípusa, szükséges jogosultságai, kockázati szintje és hibaválaszai. Az agent ne hajthasson végre tetszőleges műveletet pusztán azért, mert azt szöveges válaszként előállította.

## 4. Fokozatos fejlesztési szakaszok

### 0. szakasz – Követelmények és alapmérés

- Rögzíteni kell az első támogatott feladatokat és az elfogadási mérőszámokat.
- Külön kell választani a modellt, az agent vezérlőhurkát, az eszközöket, a memóriát és az értékelőt.
- Meg kell határozni, mely feladatok igényelnek olvasást, írást, böngészést, API-hívást vagy asztali alkalmazáskezelést.
- Készüljön rögzített, nem tanításra használt tesztkészlet az összehasonlításhoz.

### 1. szakasz – Olvasás és írás egy fájlsandboxban

- Az agent csak ideiglenes, tesztadatokkal feltöltött munkakönyvtárban olvashasson és írhasson.
- Kezdetben legyen fájllista, fájlolvasás és fájlírás eszköz; törlés, tetszőleges shell-parancs és hálózat ne legyen engedélyezve.
- Ellenőrizni kell a célútvonalat, a fájlméretet, a megengedett kiterjesztéseket és azt, hogy az írás ne érjen a sandboxon kívüli fájlhoz.
- A feladatok például strukturált szöveg kinyerése, egy szövegfájl szerkesztése és a módosítás ellenőrzése lehetnek.

**Elfogadási feltétel:** a feladatok külön, korábban nem látott teszteken is teljesülnek; a kimenet megfelel a specifikációnak; tiltott útvonalhoz vagy művelethez nem fér hozzá; minden fájlváltozás visszaállítható és naplózott.

### 2. szakasz – További eszközök és több lépéses feladatok

- Új eszközök egyenként, dokumentált sémával és minimális jogosultsággal kerüljenek be.
- Webes/API-eszközöknél legyen host-engedélylista, időkorlát, válaszméret-korlát és titokkezelés.
- Asztali alkalmazáskezelés először szimulált vagy eldobható környezetben történjen.
- A tanító példák és a tesztesetek legyenek elkülönítve; a tesztadat ne szivárogjon a tanulásba.

**Elfogadási feltétel:** az agent helyesen választ eszközt, kezeli a hibás vagy hiányos eredményeket, és egy feladat több eszközhívásán át is ellenőrizhető állapotot tart fenn.

### 3. szakasz – Sandboxos eszköztanulás és biztonsági kapuk

- Az új eszköz használatát leírásokkal, példákkal és sandboxos visszajelzésekkel kell gyakoroltatni.
- A pontozás mérje a feladat sikerét, az eszközválasztás helyességét, a kimenet érvényességét, a tiltott műveletek próbálkozásait és a feladat erőforrásigényét.
- A tesztek között legyenek félreérthető kérések, hibás eszközválaszok, részleges fájlok és olyan esetek, amelyeknél helyesen meg kell állni vagy jóváhagyást kell kérni.
- Minden tanult változat külön verziót, tesztjelentést és visszaállítható állapotot kapjon.

**Előléptetési feltétel:** előre meghatározott, reprezentatív és rejtett teszteken eléri a célértékeket, nincs súlyos tiltott mellékhatása, az eredmények megismételhetők, és egy korábbi stabil verzióra vissza lehet állni. A pontos pontszámküszöböket a feladatkészlet és a kockázat alapján még meg kell határozni.

### 4. szakasz – Korlátozott próba valódi környezetben

- Az első valós próbák kizárólag engedélyezett, alacsony kockázatú eszközöket használjanak.
- A hozzáférés legyen fokozatos: először olvasás, utána jóváhagyással írás, majd csak külön indoklással más művelet.
- A mentés, visszaállítás, naplózás, időkorlát és leállítási lehetőség a próba előtt legyen ellenőrizve.
- A pénzügyi, fiók-, biztonsági és visszafordíthatatlan változtatásokhoz emberi jóváhagyás szükséges.

## 5. Sandbox és biztonsági követelmények

- A sandbox legyen elkülönítve a felhasználó valódi fájljaitól, hitelesítő adataitól és futó alkalmazásaitól.
- Alapértelmezésben ne legyen külső hálózati hozzáférés; a szükséges webes/API-hozzáférés később célzott engedélylistával adható hozzá.
- A fájlrendszerhez és eszközökhöz csak explicit, feladathoz kötött jogosultság járjon.
- Legyen CPU-, memória-, idő- és kimenetméret-korlát, valamint naplózás és azonnali leállítás.
- A tanult eszközpolitika ne módosíthassa saját jogosultságait vagy biztonsági szabályait.
- A teszteredmény soha ne adjon automatikusan korlátlan hozzáférést a teljes géphez.

## 6. Mi maradjon a neuromorf projekt szerepe?

A jelenlegi projektet érdemes külön kutatási prototípusként továbbvinni: valós N-MNIST-teszteredmény, reprodukálható benchmarkok és a JEPA valós környezetes validálása továbbra is fontos. Az általános agent megvalósítását azonban külön rétegként kell kezelni, hogy a kísérleti SNN vagy világmodell ne kapjon olyan képességet, amelyet még nem igazoltak.

Különösen fontos az alábbiak szétválasztása:

- **Agent által kért System 2 útvonal** nem jelenti azt, hogy a tervező ténylegesen lefutott.
- **Modell által javasolt művelet** nem jelenti azt, hogy az eszköz azt végrehajtotta.
- **Tanítóeredmény** nem helyettesíti a különálló teszteredményt.
- **Memória vagy checkpoint mentése** nem egyenértékű az általános fájlkezelési képességgel.

## 7. Nyitott döntések

Az első agentverzió megkezdése előtt rögzíteni kell:

1. Melyik modellt használja a rendszer: meglévő helyi/felhős nyelvi modellt, vagy saját modell betanítása a cél?
2. Melyek legyenek az első sandboxos feladatok és engedélyezett eszközök?
3. Milyen tesztpontszám, ismételhetőség és biztonsági feltétel szükséges a következő szakaszba lépéshez?
4. A valódi gépes próba alatt mely műveletek maradjanak mindig emberi jóváhagyáshoz kötve?

Ezek a döntések jelentősen befolyásolják a rendszer architektúráját, költségét és biztonsági modelljét; nem szabad őket pusztán technológiai alapértelmezésből eldönteni.
