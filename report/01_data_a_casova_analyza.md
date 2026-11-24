# 1 Dáta a časová analýza

*Autor kapitoly: osoba A. Kód: `src/data/`, `src/analysis/temporal/`, `src/baselines/`. Všetky čísla
sú reprodukovateľné príkazmi v časti 1.8; tabuľky sú v `results/tables/A/`, obrázky v `results/figures/A/`.*

Cieľom kapitoly je opísať, aké dáta máme, ako sme ich spracovali do spoločného formátu pre všetky tri
modely, aké časové vlastnosti má meškanie odletov a ako sa v čase prejavuje počasie. Každé zistenie je
uvedené číslom, odkazom na graf alebo tabuľku a vetou o tom, čo z neho plynie pre modelovanie.

---

## 1.1 Zdroje dát a rozsah

| Dataset | Rozsah | Po filtri na náš problém |
|---|---|---|
| `WeatherEvents_Jan2016-Dec2022.csv` | 8 627 181 udalostí počasia, celé USA, 2016–2022, čas v UTC | 16 761 udalostí na 6 letiskách (2 143 v roku 2018); 803 177 udalostí z 227 staníc v mriežke regiónu |
| `2018_flights_on_time.csv` | 7 206 195 letov (BTS On-Time), 1. 1.–31. 12. 2018, 109 stĺpcov, čas miestny | 1 323 693 letov so štartom alebo cieľom na 6 letiskách; 745 226 odletov; 166 789 letov priamo medzi nimi |
| `airports_lookup.csv` | súradnice letísk (IATA) | 6 letísk SEA, PDX, SFO, SJC, LAX, SAN |

Jediný rok, ktorý pokrývajú oba datasety, je **2018**. Na ňom beží celá analýza meškaní, cieľ aj
modely. Počasie z rokov 2016–2022 slúži iba ako kontext na posúdenie, či bol rok 2018 typický (1.6).

Uzly sú v pevnom poradí `[SEA, PDX, SFO, SJC, LAX, SAN]`, teda zo severu na juh. Každé letisko má
vlastnú meteostanicu (KSEA … KSAN) vzdialenú od súradníc letiska najviac 2,0 km
(`A4_letisko_stanica.csv`), takže spojenie letisko ↔ stanica je jednoznačné.

## 1.2 Spracovanie dát (dátový kontrakt)

**Lety** (`src/data/clean_flights.py`):

* Filter na lety, ktoré štartujú alebo pristávajú na jednom zo 6 letísk. Pre B (šírenie meškaní) sa
  uložia všetky do `data/interim/flights_clean.parquet`.
* Plánovaný čas odletu `FlightDate + CRSDepTime` (HHMM, US/Pacific) prepočítame na UTC s ohľadom na
  letný čas. Prílety sa prepočítavajú podľa `CRSArrTime` v čase cieľa. Dátum príletu (rovnaký alebo
  nasledujúci deň) sa určí ako kandidát najbližší k `odlet + CRSElapsedTime`, čo sa týka 27 875 letov
  cez polnoc.
* Zrušené lety (0,90 % odletov) sa do priemeru meškania nepočítajú, ich podiel v hodine je samostatný
  príznak. Odklonené lety (0,25 %) odleteli, takže ich `DepDelay` sa počíta.
* Extrémy: `DepDelay` sa pred priemerovaním oreže na interval [−60, 360] min. Týka sa to 934 letov
  (0,13 %) s maximom 1 527 min. Jeden takýto let by inak posunul priemer celej hodiny o desiatky minút.

**Počasie** (`src/data/weather_hourly.py`):

* Udalosť je aktívna v hodine [h, h+1), ak sa s ňou interval `[StartTime, EndTime)` prekrýva.
  Zrážky udalosti sa do hodín rozdelia úmerne prekryvu.
* Na hodinu a letisko vzniknú tieto príznaky: one-hot typy, maximálna závažnosť (Light/UNK = 1,
  Moderate = 2, Heavy/Severe = 3), zrážky, aktívny podiel hodiny a počet hodín od poslednej udalosti.
  Počasie sa načítava od 30. 12. 2017 kvôli vstupnému oknu.
* **Kategórie na hodnotenie:** 0 Jasno, 1 Dážď slabý, 2 Dážď stredný/silný, 3 Hmla, 4 Sneh. Pri
  prekryve platí priorita Sneh > Hmla > Dážď silný > Dážď slabý. Rozhodnutia o nejednoznačných
  prípadoch:
  * `Precipitation` (závažnosť UNK, 12 hodín v roku 2018) patrí k slabému dažďu.
  * `Cold` (14 hodín) patrí k Jasnu, pretože nejde o zrážky ani zníženú viditeľnosť.
  * `Storm` a `Hail` by patrili k silnému dažďu, v roku 2018 sa však na našich letiskách nevyskytujú.
  * Kategória „Dážď stredný/silný“ zahŕňa `Moderate` aj `Heavy`, preto má 301 platných hodín a nie
    194 ako v predbežnom výpočte, ktorý bral iba `Heavy`.
  * Kategórie sa prekrývajú iba v 32 z 5 898 hodín s udalosťou, takže na poradí priority prakticky
    nezáleží.
* Pre mriežku osoby D sa ukladá aj riedka hodinová tabuľka všetkých **227 staníc** v obdĺžniku
  32–49° N × 125–116° W (`weather_region_hourly.parquet`). Pôvodný odhad bol 222 staníc. Rozdiel nie je
  spôsobený hranicami obdĺžnika: aj pri polootvorených intervaloch vychádza 227. Všetkých 6 letísk
  padne do rôznych buniek (`grid_index.json`).

**Cieľ a maska** (`src/data/build_processed.py`):

* Cieľ *y* je priemerný orezaný `DepDelay` odletov z letiska, ktorých plánovaný odlet padne do danej
  hodiny UTC. Časová os je 1. 1. 2018 00:00 až 31. 12. 2018 23:00 UTC, spolu 8 760 hodín.
* Hodina je platná (mask = 1), ak má aspoň 3 uskutočnené odlety. Platných je 77,9 % hodín, od 68,5 %
  v SJC po 83,3 % v SEA. Neplatné sú hlavne nočné hodiny: medzi 1:00 a 4:59 miestneho času odletí za
  celý rok 966 lietadiel.
* Lety z večera 31. 12. miestneho času patria v UTC do roku 2019 (538 odletov), preto vypadnú.
  Začiatok 1. 1. UTC zase zodpovedá večeru 31. 12. 2017, z ktorého lety nemáme.

**Súbory v `data/processed/`:**

| Súbor | Tvar | Obsah |
|---|---|---|
| `X_nodes.npy` | [8760, 6, 22] | príznaky uzlov (nenormalizované, zoznam vo `feature_names.json`): 5 × one-hot kategórie, závažnosť, zrážky, aktívny podiel hodiny, hodiny od poslednej udalosti, plánované odlety, *y* z minulosti, príznak platnosti, podiel > 15 min, podiel zrušených, počet a priemerné meškanie príletov, hodina, deň v týždni a deň v roku (sin/cos) |
| `Y.npy`, `mask.npy` | [8760, 6] | cieľ v minútach (0, kde mask = 0) a maska |
| `weather_cat.npy` | [8760, 6] | kategória 0–4 |
| `timestamps.csv` | 8 760 riadkov | čas UTC aj miestny a príslušnosť každej hodiny k časti každého rozdelenia |
| `split.json`, `scaler.json` | – | rozdelenia (1.7) a μ, σ iba z tréningovej časti každého rozdelenia |
| `grid_index.json` | 6 × (riadok, stĺpec) | bunky letísk v mriežke 17 × 9 |

Trieda `WindowDataset(mode="graph"|"grid", L=24, H=(1, 3, 6), split, part)` z
`src/data/window_dataset.py` vytvára okná a normalizuje. Platné vzorky (t0 = posledná vstupná hodina)
sú iba tie, ktorých všetky hodiny od t0 − L + 1 po t0 + 6 ležia v jednej časti rozdelenia. Trieda
funguje priamo ako PyTorch `Dataset` a pre Keras má metódu `as_arrays()`.

**Kontroly spojenia** (`A4_kontroly_spojenia.csv`, `A4_letny_cas_okolie.csv`):

* Hodnota `CRSDepTime = "2400"` sa nevyskytuje ani raz. Všetky časy majú 4 znaky, preto sa musia
  načítať ako text.
* **Letný čas:** 11. 3. medzi 02:00 a 02:59 neodlieta nič. Dňa 4. 11. padne do dvojznačnej hodiny
  01:00–01:59 iba 5 odletov, ktoré priraďujeme k letnému času. Počty odletov po hodinách UTC okolo
  oboch zmien nemajú dieru ani zdvojenie, lebo zmena času padne do nočnej pauzy. Spojenie s počasím
  v UTC preto letný čas prakticky neovplyvní.
* Stĺpce s príčinami meškania sú vyplnené presne pri letoch s `ArrDelay ≥ 15` (zhoda 100 %). Chýbajúce
  hodnoty sú teda vlastnosť datasetu, nie chyba (`A1_chybajuce_hodnoty.csv`). Chýbajúce `DepDelay`
  (6 484) a `Tail_Number` (664) sú výlučne pri zrušených letoch.

## 1.3 Obsah a charakteristiky

**Rozdelenie meškania je silne šikmé** (`A1_depdelay_rozdelenie.png`, `A1_depdelay_statistiky.csv`).

| Štatistika | Hodnota |
|---|---|
| medián / priemer `DepDelay` | −2 / 8,1 min |
| 99. percentil | 157 min |
| šikmosť / excess špicatosť | 9,8 / 192 |
| lety s odletom včas alebo skôr | 60,3 % |
| lety meškajúce > 15 min | 16,9 % |
| lety meškajúce > 60 min | 5,2 % |

Lety meškajúce viac ako 60 min (5,2 %) tvoria **57 % všetkých minút meškania**. Priemer je preto
ťahaný chvostom, a preto *y* orezávame a vedľa MAE uvádzame aj RMSE, ktoré chvost penalizuje viac.

**Letiská sa líšia úrovňou aj šumom** (`A1_depdelay_po_letiskach.csv`, `A1_hodinovy_ciel.csv`).
SFO má najvyššie priemerné meškanie (10,9 min, 20,6 % letov > 15 min), PDX najnižšie (4,4 min). Medián
počtu odletov v platnej hodine je 31 v LAX, ale iba 9 v PDX a SJC. Hodinový cieľ malých letísk je
preto priemerom z menšieho počtu letov a je zašumenejší. *Pre modely:* chyba sa bude medzi letiskami
líšiť aj bez vplyvu počasia, preto ju hodnotíme po letiskách.

**Hodinový cieľ** má priemer 7,3 min, medián 4,2 min, smerodajnú odchýlku 12,5 min a šikmosť 2,7
(`A1_hodinovy_ciel.png`). Priemerovaním v rámci hodiny sa šikmosť znížila z 9,8 na 2,7, no cieľ je
stále výrazne nesymetrický.

**Zrušené lety** (`A1_zrusene_lety.png`): 0,90 % odletov, najviac v SFO (1,37 %). Tretina zrušení
(33,7 %) má ako príčinu počasie. V SFO je to 907 z 2 409 zrušení, čo je jasný signál, že dážď a hmla
v SFO zasahujú do prevádzky.

**Počasie spôsobuje meškanie hlavne nepriamo** (`A1_priciny_meskania.png`). Z minút meškania
pripadá na príčiny:

| Príčina | Podiel minút |
|---|---|
| meškajúce lietadlo z predchádzajúceho letu (`LateAircraftDelay`) | 41,3 % |
| dopravca (`CarrierDelay`) | 30,8 % |
| NAS, teda riadenie letovej prevádzky (`NASDelay`) | 26,0 % |
| počasie priamo (`WeatherDelay`) | 1,7 % |

Priamy podiel počasia je od 0,8 % (SJC) po 3,8 % (SEA). Zlé počasie sa v štatistike BTS zapisuje
prevažne ako NAS (obmedzená kapacita dráh) a následne ako meškajúce lietadlo. *Pre modely:* priamy
`WeatherDelay` by vplyv počasia výrazne podcenil. Užitočné sú príznaky prenosu, napríklad meškanie
príletov, ktoré je v `X_nodes` zahrnuté, a graf letov medzi letiskami, ktorý pripravuje B.

**Udalosti počasia** (`A1_udalosti_pocasia.png`, `A1_trvanie_udalosti.csv`): medián trvania je
1,0 h a 90. percentil 4 h. Krok 1 hodina teda zodpovedá dĺžke typickej udalosti. V roku 2018 je
v platných hodinách 36 205 hodín Jasna, 2 660 slabého dažďa, 301 stredného/silného dažďa, 1 698 hmly a
61 snehu (`A1_hodiny_kategorii_2018.csv`). Sneh sa vyskytuje iba v SEA a PDX a takmer výlučne vo
februári. Hmla dominuje v LAX (581 platných hodín, viac ako dážď), dážď v SEA a PDX.

## 1.4 Časové vlastnosti

**Denný profil** (`A2_profily.png`, vľavo hore): meškanie počas dňa narastá z −1,3 min o 5:00
miestneho času na maximum 11,9 min o 13:00 a vysoké ostáva až do večera. Ráno lietadlá štartujú
„čisté“, počas dňa sa meškanie prenáša z letu na let, čo zodpovedá 41 % podielu meškajúceho
lietadla. *Pre modely:* hodina dňa je najsilnejší deterministický prediktor, preto je v príznakoch
ako sin/cos a hodnotiť by sa mala aj anomália voči dennému profilu.

**Týždenný profil:** najhoršie sú piatok (10,5 min) a štvrtok (9,6 min), najlepšia sobota (5,8 min).
Rozdiel cez týždeň (4,7 min) je porovnateľný s efektom slabého dažďa.

**Ročný profil 2018** (`A2_mesacny_profil.csv`): priemer klesá od 12,9 min v auguste po 4,8 min
v septembri. Najhorší mesiac (august) je zároveň takmer bez počasia na našich letiskách
(`A2_profily.png`, vpravo dole). Letné meškanie má teda iný pôvod než miestne počasie: vyťaženosť
siete a prenos z iných častí USA. Mesiace s najviac počasím (január, marec, november) nemajú
najvyššie meškanie. *Pre modely:* sezónny priemer meškania sa nedá vysvetliť miestnym počasím, takže
modely sa ho nemôžu naučiť iba z príznakov počasia. Pri porovnaní po kategóriách je preto dôležitá aj
anomália voči sezónnemu profilu.

**Denné rady** (`A2_denne_rady.png`, `A2_najhorsie_dni.csv`): najhoršie dni sú epizodické a väčšinou
sprevádzané počasím. SEA 20. 11. 2018 mala priemer 126,7 min pri 13 hodinách počasia a SFO 6. 4.
56,9 min pri 23 hodinách počasia. SFO 14. 8. (43,1 min) však bola bez počasia, čo potvrdzuje, že
nie všetky extrémy vysvetlí počasie.

**Autokorelácia** (`A2_acf_pacf.png`, `A2_acf_stacionarita.csv`):

* ACF hodinového *y* pri posune 1 h je v priemere 0,45, od 0,18 v PDX po 0,71 v SFO. Pri malých
  letiskách prevláda šum z malého počtu letov.
* Výrazné vrcholy pri 24 h a 48 h (0,17–0,32) predstavujú dennú periodicitu.
* Po odčítaní profilu hodina × deň v týždni zostáva ACF anomálie 0,42 pri 1 h a 0,23 pri 6 h. Meškanie
  má teda aj krátkodobú pamäť nad rámec sezónnosti, ktorá odznie približne za pol dňa.
* PACF má významné posuny 1 (0,45) a 2 (0,23), potom je takmer nulová, s malým záporným vrcholom pri
  25–26 h. Hodinová rada sa teda správa približne ako AR(2) so sezónnou zložkou.
* ACF denného priemeru má lokálne maximum pri 7 dňoch (0,17–0,32), čo je týždenná periodicita.

*Pre modely:* okno L = 24 h pokryje krátku pamäť aj jeden denný cyklus. Horizonty 1, 3 a 6 h ležia
v oblasti, kde krátka pamäť ešte existuje (ACF anomálie 0,42 → 0,23), takže prínos histórie by mal
s horizontom klesať.

**Stacionarita** dennej rady: ADF zamieta jednotkový koreň na všetkých letiskách (p ≤ 0,021). KPSS
nezamieta stacionaritu okrem SEA (p = 0,01) a hraničného PDX (p = 0,07). Rada je stacionárna okolo
sezónneho profilu, bez trendu. Výnimkou je SEA, ktorú ovplyvňujú novembrové extrémy. Normalizácia
μ, σ z tréningu je preto oprávnená, ale test (november–december) má o niečo inú úroveň než tréning
(priemer *y* 7,8 vs. 7,6 min).

## 1.5 Vplyv počasia v čase

**Meškanie podľa kategórie počasia** (`A3_pocasie_vs_meskanie.png`, `A3_meskanie_podla_kategorie.csv`;
anomália = *y* mínus priemer letisko × hodina × deň v týždni):

| Kategória | Platných hodín | Priemer *y* (95 % CI) | Medián | Anomália |
|---|---|---|---|---|
| Jasno | 36 205 | 7,0 (6,9–7,1) | 4,1 | −0,4 |
| Dážď slabý | 2 660 | 8,0 (7,5–8,5) | 3,9 | +1,8 |
| Hmla | 1 698 | 10,2 (9,4–11,1) | 5,6 | +3,5 |
| Sneh | 61 | 12,2 (9,2–15,3) | 10,3 | +6,0 |
| Dážď stredný/silný | 301 | 15,9 (13,6–18,4) | 10,3 | +9,1 |

Poradie od Jasna po silný dážď platí aj pre anomáliu, takže nejde iba o to, že počasie prichádza
v rušnejších hodinách. Interval pre sneh je pri 61 hodinách široký (±3 min), a preto závery o snehu
treba formulovať opatrne.

**Účinok počasia je silne lokálny** (stredný panel obrázka A3):

* SFO: dážď stredný/silný +20,4 min (68 h), dážď slabý +13,7 min. Priemer pri akomkoľvek daždi je
  26,3 min oproti 9,1 min za jasna. Dôvodom sú paralelné dráhy SFO, ktoré pri zlom počasí znižujú
  kapacitu.
* SEA: hmla +12,9 min (244 h), priemer 17,6 vs. 5,2 min za jasna.
* LAX: hmla +0,3 min pri 581 hodinách (7,5 vs. 7,4 min), teda bez účinku. Hmla v LAX je bežná
  a prevádzka je na ňu pripravená. Rovnako SJC: hmla +0,3 min.
* PDX: počasie takmer nemá vplyv (slabý dážď −0,1 min, hmla +2,6 min).

*Pre modely:* rovnaký typ počasia má na rôznych letiskách úplne iný účinok. Model preto potrebuje
príznaky počasia na úrovni uzla a parametre, ktoré sa môžu líšiť medzi letiskami. Najťažšie
predikovateľné kombinácie sú dážď v SFO a hmla v SEA (hypotézy pre C a D).

**Sezóna a denná doba** (pravý panel, `A3_kategoria_x_doba_x_sezona.csv`):

* Silný dážď má na jar a na jeseň dopoludnia a večer anomáliu +14 až +20 min. V zime je jeho účinok
  oveľa menší (+2 až +11 min), hoci zimného dažďa je najviac. Neočakávaný dážď mimo zimy teda
  narušuje prevádzku viac.
* Hmla je najčastejšia na jeseň ráno (277 h), no tam má iba +4,6 min. Najväčší účinok má večer
  (v lete +10,8 min pri 63 h), keď narazí na už nahromadené meškanie.

*Pre modely:* účinok počasia závisí od sezóny a dennej doby, preto sú príznaky času a počasia vo
vstupe spolu a nelineárny model ich môže kombinovať.

**Nábeh a doznievanie** (`A3_event_study.png`, `A3_event_study.csv`). Epizóda počasia je súvislý úsek
hodín s udalosťou; medzery do 1 h sa spájajú. Anomália sa počíta voči začiatku a koncu epizódy:

* Pri silnom daždi anomália po začiatku narastá postupne: +2,8 min po 1 h, +5,8 po 2 h, +8,3 po 4 h.
  Meškanie sa teda kumuluje s oneskorením.
* Pri konci epizódy silného dažďa je anomália ešte +4,5 min a interval spoľahlivosti obsahuje nulu až
  približne 4 h po konci. Pri hmle (+3,3 min pri konci) a slabom daždi (+1,6 min) meškanie odznie
  približne za 3 h.
* Snehových epizód s platnými hodinami je iba 13, takže pre sneh tento odhad nie je spoľahlivý.

**Korelácia s posunom** (`A3_korelacia_s_posunom.png`): korelácia počasia v čase *t* s anomáliou
v čase *t + k* je na SEA a SFO vyššia pri *k* = 3 h ako pri *k* = 0 (SEA 0,131 vs. 0,104; SFO 0,229
vs. 0,214). Na PDX a SJC je pri všetkých posunoch pod 0,06.

*Pre modely:* **počasie pozorované do času T obsahuje informáciu o meškaní v T+1 až T+3 h**, pretože
účinok sa kumuluje a doznieva 3–4 h. Predikcia iba z minulého počasia (bez „oracle“ počasia v T+h)
má preto zmysel. Pre h = 6 h však bude signál počasia slabší.

## 1.6 Je rok 2018 typický?

(`A4_typickost_2018.png`, `A4_udalosti_po_rokoch.csv`, `A4_typickost_2018_mesiace.csv`)

* Rok 2018 mal na 6 letiskách **2 143 udalostí počasia, o 12 % menej** ako priemer ostatných rokov
  (2 436).
* Pre SEA a PDX bol 2018 najsuchší zo všetkých 7 rokov (poradie 7/7). SJC bola mierne nad priemerom
  (1,13 × priemer).
* Sezónny tvar je v 2018 rovnaký ako v ostatných rokoch: január má za 7 rokov 2 530 udalostí, júl
  318. Z 72 mesačných hodnôt (6 letísk × 12 mesiacov) má iba 6 odchýlku |z| > 2:
  * menej udalostí: SEA február (42 vs. 81) a máj (11 vs. 41),
  * viac udalostí: SJC november (57 vs. 25, z = 4,7), SFO november (49 vs. 24), LAX október (40 vs. 20).

*Pre modely:* rok 2018 je typický tvarom sezóny, no na severe je chudobnejší na udalosti.
**Testovací november bol v Bay Area mimoriadne daždivý**, takže hlavný test obsahuje veľa dažďa
(618 h slabého a 113 h silného), čo je pre porovnanie modelov výhodné. Zistenia o daždi v SFO a SJC
sú však založené na nadpriemerne daždivom testovacom období.

## 1.7 Rozdelenie dát a počasie v ňom

(`A4_kategorie_v_rozdeleniach.csv`, `A4_rozdelenia_datumy.csv`)

Medzi časťami je **medzera 30 h** (L + max H), aby sa okná neprekrývali. Medzera sa odreže z tréningu
alebo validácie, nikdy z testu.

**Hlavné rozdelenie:**

| Časť | Obdobie (UTC) | Platných letisko-hodín |
|---|---|---|
| tréning | 1. 1.–30. 8. 17:00 | 27 230 |
| validácia | 1. 9.–30. 10. 17:00 | 6 666 |
| test | 1. 11.–31. 12. | 6 731 |

Počasie v hlavnom teste: 5 451 h Jasna, 618 h slabého dažďa, 113 h silného dažďa, 549 h hmly a
**0 h snehu**. Validácia (september–október) má iba 30 h silného dažďa, takže ladenie hyperparametrov
na nej počasie takmer nezohľadní.

**Štvrťročné foldy:** test je postupne Q1 až Q4, tréning tvorí zvyšok roka a validácia je posledný
mesiac tréningu. Spojené testy všetkých 4 foldov pokryjú každú platnú hodinu roka práve raz,
**vrátane všetkých 61 hodín snehu (všetky vo folde Q1)**. Fold Q3 (júl–september) má v teste iba 8 h
silného dažďa, takže jeho výsledok pre silný dážď je takmer bez informácie. Hodnotenie po počasí
preto robíme nad spojenými predikciami 4 foldov. Platí pri tom upozornenie, že tréning foldov Q1–Q3
obsahuje aj neskoršie mesiace.

## 1.8 Baseline modely

(`src/baselines/`; predikcie `results/preds_<model>_<split>.npz` vo formáte dátového kontraktu,
metriky `baseline_*.csv`, obrázok `A5_baseline.png`)

| Model | Opis |
|---|---|
| persistence | posledné platné *y* na letisku (do 24 h späť) |
| sezónny priemer | priemer z tréningu podľa letisko × miestna hodina × deň v týždni |
| sezónny + AR(1) | sezónny priemer + posledná odchýlka × φ_h; φ = 0,41 / 0,33 / 0,24 pre h = 1 / 3 / 6, odhadnuté na tréningu |
| lineárny VARX | Ridge regresia z celého normalizovaného okna 24 × 6 × 22 na 3 × 6 výstupov; α vybrané na validácii, prefit na tréning + validácia |

Lineárny VARX je klasický lineárny model priestorovo-časovej autoregresie z prednášok s exogénnymi
vstupmi počasia. Slúži ako referencia, ktorú by mali neurónové modely prekonať.

**Hlavný test (november–december), MAE v minútach (95 % blokový bootstrap po dňoch):**

| Model | h = 1 | h = 3 | h = 6 | RMSE h = 1 |
|---|---|---|---|---|
| persistence | 8,29 (7,7–8,9) | 9,40 | 10,51 | 12,27 |
| sezónny priemer | 8,62 (7,9–9,5) | 8,62 | 8,62 | 13,75 |
| sezónny + AR(1) | 7,53 (7,0–8,2) | 7,88 | **8,19** | 11,36 |
| lineárny VARX | **6,76** (6,3–7,3) | **7,52** | 8,31 | **10,15** |

**MAE podľa počasia, h = 1, spojené 4 foldy:**

| Kategória (n hodín) | persistence | sezónny | sez. + AR(1) | VARX |
|---|---|---|---|---|
| Jasno (35 752) | 8,33 | 7,91 | 7,25 | **6,47** |
| Dážď slabý (2 634) | 9,06 | 8,84 | 8,01 | **7,24** |
| Hmla (1 657) | 8,49 | 9,16 | 7,82 | **7,45** |
| Sneh (61) | 11,34 | 10,01 | 9,10 | **8,85** |
| Dážď stredný/silný (301) | 12,16 | 13,55 | 10,80 | **9,02** |

Závery z baseline modelov:

* Pri h = 1 je VARX o 18 % lepší ako persistence. Pri h = 6 ho dobehne jednoduchý sezónny model
  s utlmenou odchýlkou (8,19 vs. 8,31). Pri dlhšom horizonte sa krátka pamäť stráca a zostáva najmä
  sezónny profil (zhoduje sa s ACF v 1.4).
* Pri silnom daždi sú chyby všetkých baseline modelov o 40–70 % vyššie ako za jasna. Je to kategória,
  na ktorej majú neurónové modely najviac priestoru ukázať prínos.
* **Referencia pre modely C a D:** MAE 6,8 / 7,5 / 8,2 min pre h = 1 / 3 / 6 na hlavnom teste. Model,
  ktorý tieto hodnoty neprekoná, neospravedlní svoju zložitosť.

## 1.9 Zhrnutie zistení a odporúčania pre modely

1. **Cieľ je šikmý a zašumený:** 5,2 % letov nesie 57 % minút meškania a malé letiská (PDX, SJC) majú
   v hodine medián iba 9 odletov. Preto *y* orezávame na 360 min, maskujeme hodiny s menej ako
   3 odletmi a hodnotíme po letiskách.
2. **Silná denná a týždenná periodicita** (ACF pri 24 h 0,17–0,32, pri 7 dňoch 0,17–0,32) a krátka
   pamäť anomálie (0,42 pri 1 h, 0,23 pri 6 h). Okno L = 24 h a cyklické príznaky času sú oprávnené.
3. **Ročný priebeh meškania nekopíruje počasie:** najhorší je august (12,9 min) takmer bez počasia,
   najlepší september (4,8 min). Sezónnu zložku modely nemôžu vysvetliť počasím.
4. **Počasie zvyšuje meškanie postupne a lokálne:** silný dážď +9,1 min, hmla +3,5 min, sneh +6,0 min
   (len 61 h). Dážď v SFO (+20 min) a hmla v SEA (+13 min) sú silné, hmla v LAX ani PDX nemá účinok.
5. **Počasie pôsobí nepriamo a s oneskorením:** priamy `WeatherDelay` tvorí iba 1,7 % minút, meškanie
   sa po začiatku dažďa kumuluje 2–4 h a doznieva 3–4 h. Preto má zmysel minulé počasie aj prenos
   meškania po grafe letov.
6. **Rok 2018 je typický tvarom, ale suchší** (−12 % udalostí). Testovací november bol v Bay Area
   mimoriadne daždivý. Sneh je hodnotiteľný iba cez štvrťročné foldy.
7. **Referencia je lineárny VARX** s MAE 6,8 min pri h = 1. Najväčšie chyby majú baseline modely pri
   silnom daždi (9–14 min).

**Obmedzenia:**

* Hodinový cieľ *y* je priemer z plánovaných odletov v danej hodine. Jeho hodnota sa definitívne
  ustáli až po skutočnom odlete meškajúcich lietadiel, takže posledná hodina vstupného okna je
  v skutočnej prevádzke známa iba čiastočne. Pri všetkých modeloch aj baseline modeloch je efekt
  rovnaký, porovnanie je preto férové.
* Kategórie počasia vychádzajú z udalostí jednej stanice na letisku. Prípadné chýbajúce záznamy
  stanice sa javia ako Jasno.

---

### Reprodukcia

```bash
py -3.11 -m src.data.clean_flights          # ~50 s, lety -> data/interim/
py -3.11 -m src.data.weather_hourly         # ~20 s, počasie -> data/interim/
py -3.11 -m src.data.build_processed        # data/processed/ (kontrakt)
py -3.11 -m src.baselines.baselines         # baseline predikcie + metriky
py -3.11 -m src.analysis.temporal.run_all   # tabuľky a obrázky tejto kapitoly
```
