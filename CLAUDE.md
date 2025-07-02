# Linear Least Squares Methods - Claude Code Notes

## ✅ FIXED: Regularizační parametry byly opraveny

### Problém (VYŘEŠEN)
Při testování různých regresních metod (Linear, Ridge, Lasso, Elastic Net) na polynomiálních datech (zejména quintic funkce) se projevuje neočekávané chování:

- **Ridge** a **Elastic Net** vykazují příliš silnou regularizaci (přehnaně hladké křivky)
- **Lasso** má správnou úroveň regularizace
- **Linear** nemá žádnou regularizaci (overfitting)

### Příčina
Všechny 3 implementace (pure, numpy, numba) používají stejnou hodnotu `alpha = 1.0` pro Ridge, Lasso i Elastic Net. Toto je problematické, protože:

1. Pro polynomiální regrese (zejména vyšších stupňů) je `alpha = 1.0` příliš vysoká hodnota pro Ridge a Elastic Net
2. Pro Lasso je `alpha = 1.0` relativně přiměřená hodnota
3. Různé regularizační metody typicky vyžadují různé hodnoty alpha parametru

### Soubory k úpravě
- `/approaches/least_squares_pure.py` (řádek 249)
- `/approaches/least_squares_numpy.py` (řádek 53)
- `/approaches/least_squares_numba.py` (řádek 227)

### Implementované řešení ✅
Upraveny výchozí hodnoty alpha pro různé metody ve všech 3 implementacích:
- **Ridge**: `alpha = 0.01` (sníženo z 1.0)
- **Lasso**: `alpha = 1.0` (ponecháno)
- **Elastic Net**: `alpha = 0.1` (sníženo z 1.0)

Změny provedeny v souborech:
- ✅ `/approaches/least_squares_pure.py` (řádky 249-257)
- ✅ `/approaches/least_squares_numpy.py` (řádky 40-48)
- ✅ `/approaches/least_squares_numba.py` (řádky 635-643)

### Další poznámky
- Problém se nejvíce projevuje na složitějších funkcích (polynomy vyšších stupňů, sinusové funkce)
- Je třeba otestovat navrhované hodnoty na různých datasetech
- Zvážit přidání možnosti nastavit alpha jako parametr při vytváření instance regresních tříd