# SkyGuard FUSION — Phase 42.3: image-space multi-object identity stress evaluation

## Qëllimi

Ky modul **nuk krijon detektime**. Ai merr JSONL me **ground truth të pavarur** dhe parashikimet reale të një tracker-i (nëse janë prodhuar realisht), nga formati i Fazës **42.1 / 42.2**, pastaj jep diagnostikë specifike për skenarët me disa dronë.

Rasti i ndërrimit të ID-së gjatë një kryqëzimi në imazh analizohet duke krahasuar **ID-të e objekteve të etiketuara në video** me ID-të e tracker-it. Një ndryshim i renditjes sipas koordinatës X dhe afërsia e kutive tregon vetëm një **kandidat kryqëzimi në projeksionin 2D**, jo manovër fizike të verifikuar. Nuk krijohen koordinata GPS, distanca apo alarme shkeljeje perimetri.

**Dallim kritik:** Nëse input-i vjen nga 42.2, vlerësohet tracker-i `Ultralytics YOLO.track` (ByteTrack/BoT-SORT), **jo** `SkyGuard LiveSessionManager`. Për të provuar tracker-in live të SkyGuard duhet eksporti i *vërtetë* i parashikimeve të tij për të njëjtat frame-e me anotime të pavarura. Kjo integrim end-to-end mbetet i hapur.

## Matjet

- ID switches të përputhura me evaluator-in origjinal 42.1 dhe ID switches afër objekteve të tjera.
- Numri i frame-eve me 2, 3 ose 4+ dronë të etiketuar, objektet e gjetura, të humbura, të ngatërruara si zogj.
- Episode të afrimit të objekteve në **pikselë**, përmbysje të renditjes X në intervale të shkurtra dhe listë e kufizuar ngjarjesh diagnostikuese.
- ID të njëjta të përdorura për disa identitete GT, ID të ndryshme që i atribuohen të njëjtit GT, frame-e me mungesë ID-je.
- Humbje/rikthim detektimi mes frame-eve të anotuar, dhe boshllëqe të anotimit të GT (këto të fundit **nuk** provojnë okluzion fizik).
- Mbështjellë rezultatet e plota të evaluator-it **42.1**, për të mos ndryshuar formulat e metrikave historike.

`close_factor` mat afërsinë e dy qendrave në pikselë kundrejt diagonalës së kutisë më të madhe; kjo nuk është njësi distance në metra. `crossing_max_gap` është interval maksimal frame-esh për ndryshimin e renditjes X pas një afrimi të afërt. `id_switches` numërohen sipas politikës 42.1 për maksimumin e diferencës së frame-eve.

## Ekzekutimi i sigurt, vetëm offline

Shembulli i parë është **SINTETIK**, i krijuar enkas që dy dronë artificialë të ndërrojnë `track_id` gjatë një kryqëzimi. Raporti është test i algoritmit, **nuk provon saktësi në video reale**:

```powershell
cd C:\Users\Amigos\Projects\skyguard-fusion
$env:PYTHONPATH = (Resolve-Path 'backend').Path
$out = Join-Path $env:USERPROFILE ('Downloads\SkyGuard_42_3_Synthetic_' + (Get-Date -Format yyyyMMdd_HHmmss) + '.json')
& 'backend\.venv\Scripts\python.exe' -m app.evaluation.crossing_stress `
  --input 'backend\examples\phase42_3_SYNTHETIC_identity_swap.jsonl' `
  --output $out --synthetic-self-test
if ($LASTEXITCODE -ne 0) { throw '42.3 synthetic diagnostic failed' }
```

Për video reale të anotuar, pas ekzekutimit të **42.2** me licencë të kontrolluar dhe checkpoint të besuar, përdorni:

```powershell
$base = 'C:\PATH\TO\PHASE42_2_OUTPUT_DIRECTORY'
$out = Join-Path $env:USERPROFILE ('Downloads\SkyGuard_42_3_Real_Analysis_' + (Get-Date -Format yyyyMMdd_HHmmss) + '.json')
& 'backend\.venv\Scripts\python.exe' -m app.evaluation.crossing_stress `
  --input (Join-Path $base 'annotated_predictions.jsonl') `
  --manifest (Join-Path $base 'manifest.json') --output $out
if ($LASTEXITCODE -ne 0) { throw '42.3 real-video evaluation failed' }
```

Kërkohet që `--output` të mos ekzistojë. JSONL duhet të përmbajë anotime të pavarura dhe parashikime; skripti nuk ekzekuton modelin ose kamerën. Raporti shënon SHA-256 të input-it dhe manifestit. Manifesti i 42.2 është **deklarim i burimit**, jo vërtetim i jashtëm i licencës, autenticitetit apo cilësisë së anotimeve.

## Kriteret e pranimit të Fazës 42.3

Testet automatike provojnë **matjet dhe logjikën** me skenarë sintetikë: dy dronë që kalojnë, ID të shkëmbyera, frame-e me humbje gjurmësh, klasifikime dron/zog, sekuenca të ndara, kufij input-esh dhe origjinë të deklaruar. Ato nuk provojnë që sistemi zbulon dronë realë ose se mund të ndjekë një maksimum të caktuar në mjedis real.

Rezultate nga video reale të licencuara dhe identitete të etiketuara nga burime të pavarura **nuk përfshihen** në skriptin e instalimit. Në hapat e radhës duhet matje me të dhëna reale, krahasim i drejtpërdrejtë me tracker-in live SkyGuard, testim i FPS dhe latencës në kamerën konkrete dhe validim i pasigurisë së pozicionit/distancës.
