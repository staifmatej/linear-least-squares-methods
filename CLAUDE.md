## Project: Linear Least Squares Methods

**Objective:**  
Implement and compare four variants of the Least Squares Method (LSM) for curve fitting:

1. Python using NumPy  
2. Python with `for` loops (manual calculation)  
3. Python with `for` loops and JIT compilation using Numba  
4. C++ version for performance comparison

All implementations will use **multivariable regression**, meaning the model can handle more than one input variable (feature) at once.

All implementations will be tested with the same datasets, and visualized using Python (e.g., Matplotlib).  

**Core Features:**
- Input: user-defined 2D points
- Output: fitted curve, residual error, visual graph
- Polynomial support: fitting up to 5th-degree polynomials
- Future support for:
  - Lasso Regression
  - Ridge Regression
  - Elastic Net Regression

**Tools & Libraries:**
- Python (NumPy, Numba, Matplotlib, Seaborn)
- for graphs i want to use seaborn for better design of charts and so on...
- C++
- (Planned) scikit-learn for additional regressions

here are documents written in Tex from my class Linear Algebra 2, i want to implement linear least squares method like this:
\section{Motivace a ukázka použití: odhad ceny nemovitostí}

Představte si, že chcete odhadnout tržní cenu nemovitosti, řekněme bytu. Konkrétněji takovou úlohu můžeme popsat tak, že chceme získat nějaký jasný návod, jak z věcí, které o bytu umíme zjistit (např. rozloha, adresa, rok rekonstrukce, počet místností atp.) odvodit něco, co o něm zjistit snadno neumíme, v našem případě jeho tržní hodnotu. Je dobré si na začátku nastavit realistická očekávání, takže budeme raději říkat, že chceme získat \emph{odhad tržní ceny}, protože nevěříme, že je možné získat vždy skutečnou cenu. Takovýto druh problémů se řeší ve statistice a strojovém učení\footnote{248}. Hledaný návod má formu modelu v podobě matematické funkce, která po dosazení zmíněných známých údajů, kterým budeme říkat \emph{příznaky} (angl. \emph{features}), vrátí odhad údaje neznámého, kterému budeme říkat \emph{vysvětlovaná proměnná} (angl. \emph{target variable}).

My použijeme jeden z populárních přístupů, zvaných \emph{lineární regrese}, kdy se pokusíme najít funkci ve tvaru \emph{lineární kombinace příznaků}. Konkrétně budeme předpokládat, že tržní cena bytu jde rozumně odhadnout jako lineární kombinace číselných příznaků jako např. rozlohy v metrech čtverečních, vzdálenosti od centra, letech od poslední rekonstrukce atp.

\subsection{Jednorozměrný příklad}

Nejdříve pro jednoduchost předpokládejme, že cena závisí pouze na velikosti bytu, tedy obyčejné ploše v metrech čtverečních. Označme cenu jako $Y$ a velikost jako $X$ a předpokládejme, že závislost mezi nimi je lineární\footnote{249}:
\begin{equation}
Y \approx wX, \quad w \in \mathbb{R}.
\end{equation}

Značka $\approx$ značí, že nečekáme přímo rovnost, ale něco jako „přibližnou" rovnost. Přesnější a formálnější definice takového vztahu by vyžadovala znalost pojmu \emph{náhodná veličina}, se kterým se seznámíte až v teorii pravděpodobnosti a statistice.

Když jsme si takto předepsali, jak hledaný vztah mezi známým příznakem a neznámou vysvětlovanou proměnnou vypadá, zbývá nám určit hodnotu parametru $w \in \mathbb{R}$ a tím lineární závislost přesně specifikovat. Abychom to mohli udělat, budeme potřebovat data, tedy skutečnou hodnotu $X$ a $Y$ pro nějakou sadu bytů. Z těchto dat se hodnotu parametru $w$ „naučíme" (jak by se řeklo ve strojovém učení) resp. „odhadneme" (jak by se řeklo ve statistice).

Zjistíme tržní cenu několika bytů (řekněme, že je jich $N$), které byly prodány v posledních měsících, a u nich si zjistíme i velikost. Data můžeme uložit do vektorů $\mathbf{Y}, \mathbf{X} \in \mathbb{R}^N$, kde $Y_i$ označuje cenu $i$-tého bytu a $x_i$ jeho velikost.

Předpokládejme, že máme nyní parametr našeho modelu $w$ zafixovaný, potom je odhad ceny $i$-tého bytu daný modelem roven číslu, které označíme\footnote{250} jako
\begin{equation}
\hat{Y}_i = wx_i.
\end{equation}

Máme-li odhad ceny i skutečnou cenu, můžeme spočítat \emph{chybu modelu}, která bude nějakým způsobem měřit odchylku mezi skutečností $Y_i$ a výstupem modelu (zvaným odhad nebo predikce) $\hat{Y}_i$. Tradiční volbou\footnote{251} je \emph{kvadrát rezidua}
\begin{equation}
(Y_i - \hat{Y}_i)^2.
\end{equation}

Cílem je nastavit parametr $w$ tak, aby chyba pro všechny byty z našich dat byla co nejmenší. Minimalizujeme tedy \emph{součet kvadrátů residuí} (anglicky \emph{residual sum of squares})
\begin{align}
RSS(w) &= \sum_{i=1}^{N}(Y_i - \hat{Y}_i)^2 = \sum_{i=1}^{N}(Y_i - wx_i)^2 = \sum_{i=1}^{N} \left(Y_i^2 - 2wx_iY_i + w^2x_i^2\right).
\end{align}

Jedná se o sumu parabol, což je opět parabola (v proměnné $w$):
\begin{equation}
RSS(w) = \sum_{i=1}^{N} Y_i^2 - 2w \sum_{i=1}^{N} x_iY_i + w^2 \sum_{i=1}^{N} x_i^2.
\end{equation}

Najít minimum je tedy triviální středoškolská úloha\footnote{252}. Hodnota parametru $w$ minimalizující $RSS(w)$ je rovna
\begin{equation}
\hat{w}^{(OLS)} = \frac{\sum_{i=1}^{N} x_iY_i}{\sum_{i=1}^{N} x_i^2} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{Y}.
\end{equation}

Tomuto odhadu\footnote{253} parametru $w$ se říká \emph{odhad metodou nejmenších čtverců} (angl. \emph{Ordinary Least Squares} -- slovo „ordinary" (tj. obyčejný) se do názvu přidává, aby se odlišranilo, že se nejedná o některé z mnoha „neobyčejných" variant této metody, které mají název \emph{weighted least squares}, \emph{non-linear least squares}, \emph{ridge regression}, atp.).

\subsection{Vícerozměrný příklad}

Model, který předpokládá, že mezi cenou a velikostí bytu je přímá úměra, je samozřejmě příliš jednoduchý a nebude odpovídat realitě ani přibližně. Jak jej rozumně „zesložitit", aby měl naději na to, být dobrým modelem skutečnosti? Můžeme předpokládat, že závislost není lineární, ale třeba kvadratická, tedy že má tvar
\begin{equation}
Y \approx w_0 + w_1X + w_2X^2, \quad w_0, w_1, w_2 \in \mathbb{R}.
\end{equation}

Nebo můžeme předpokládat, že cena nezávisí jen na obyčejné ploše (označme $X_1$), ale na více parametrech, např. době od poslední rekonstrukce $X_2$, vzdálenosti od centra $X_3$ a kvadrátu velikosti bytu $X_4 = X_1^2$. Model pak vypadá takto:
\begin{equation}
Y \approx w_0 + w_1X_1 + w_2X_2 + w_3X_3 + w_4X_4, \quad w_0, w_1, w_2, w_3, w_4 \in \mathbb{R}.
\end{equation}

Jistě by se dalo přijít ještě s mnoha nápady, ale z matematického pohledu můžeme vše zobecnit\footnote{254} následovně jako problém \emph{lineární regrese}: Předpokládáme, že existuje závislost vysvětlované proměnné $Y$ na $p$ příznacích $X_1, X_2, \ldots, X_p$ ve tvaru
\begin{equation}
Y \approx w_0 + w_1X_1 + \cdots + w_pX_p, \quad w_0, w_1, \ldots, w_p \in \mathbb{R}.
\end{equation}

Absolutní člen $w_0$ se obvykle nazývá \emph{intercept} a do modelu se přidává, aby se mu umožnilo „neprochá­zet nulou". Technicky jej do modelu zahrneme tak, že vytvoříme umělý příznak $X_0$ vždy rovný jedné. Označme-li (sloupcový) vektor parametrů $\mathbf{w} = (w_0, w_1, \ldots, w_p)$ a též sloupcový vektor příznaků $\mathbf{x} = (1, X_1, \ldots, X_p)$, potom lze model zapsat jako
\begin{equation}
Y \approx \mathbf{w}^T\mathbf{x}.
\end{equation}

Snažíme se tedy opět vyjádřit $Y$ jako lineární kombinaci příznaků.

Pro hledání „vhodných" hodnot \emph{parametrů modelu lineární regrese} $\mathbf{w}$ (zvaných též koeficienty či váhy) budeme opět potřebovat data, tedy známé hodnoty vysvětlované proměnné
\begin{equation}
\mathbf{Y} = (Y_1, \ldots, Y_N) \in \mathbb{R}^N
\end{equation}

a příznaků, které lze uložit do matice, kde řádek odpovídá datovému bodu (např. parametrům jednoho bytu) a sloupec odpovídá příznaku (např. velikosti bytu). Matice bude mít rozměr $N \times (p + 1)$, tedy $N$ řádků a $p + 1$ sloupců, přičemž první sloupec, reprezentující intercept a umělý příznak $X_0$, bude obsahovat samé jedničky:
\begin{equation}
\mathbf{X} = \begin{pmatrix}
1 & x_{11} & x_{12} & \cdots & x_{1p} \\
1 & x_{21} & x_{22} & \cdots & x_{2p} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_{N1} & x_{N2} & \cdots & x_{Np}
\end{pmatrix} \in \mathbb{R}^{N,p+1}.
\end{equation}

Označme $i$-tý řádek matice $\mathbf{X}$ jako (sloupcový) vektor\footnote{255} $\mathbf{x}_i \in \mathbb{R}^{p+1}$, potom je odhad $Y$ pro $i$-tý datový bod (v našem příkladu odhad ceny $i$-tého bytu) daný výrazem
\begin{equation}
\hat{Y}_i = w_0 + w_1x_{i1} + \cdots + w_px_{ip} = \mathbf{w}^T\mathbf{x}_i.
\end{equation}

S využitím tohoto značení lze opět vyjádřit chybu modelu měřenou jako sumu kvadrátů reziduí
\begin{equation}
RSS(\mathbf{w}) = \sum_{i=1}^{N}(Y_i - \hat{Y}_i)^2 = \sum_{i=1}^{N}(Y_i - \mathbf{w}^T\mathbf{x}_i)^2.
\end{equation}

Zamysleme se nad tím, co by muselo platit, aby byla tato chyba nulová? Museli bychom pro všechny $i = 1, 2, \ldots, N$ mít
\begin{equation}
Y_i = w_0 + w_1x_{i1} + \cdots + w_px_{ip} = \mathbf{w}^T\mathbf{x}_i,
\end{equation}

což můžeme přepsat „maticově" do tvaru
\begin{equation}
\mathbf{X}\mathbf{w} = \mathbf{Y}.
\end{equation}

Tato rovnice je vlastně soustava $N$ lineárních rovnic o $p + 1$ neznámých. Jaká je šance na to, že má řešení? Jak víme, řešení existuje, právě když je vektor $\mathbf{Y} \in \mathbb{R}^N$ prvkem lineárního obalu souborů sloupců matice $\mathbf{X}$, tedy právě když
\begin{equation}
\mathbf{Y} \in \langle\mathbf{X}_0, \mathbf{X}_1, \ldots, \mathbf{X}_p\rangle \subset \mathbb{R}^N.
\end{equation}

Šance, že je toto splněno, závisí na vzájemném poměru $N$ a $p + 1$. Obecně lze říci, že čím je $N$ větší než $p + 1$, tím menší je šance, že řešení existuje. V reálném použití lineárních modelů obvykle soustava
\begin{equation}
\mathbf{X}\mathbf{w} = \mathbf{Y}.
\end{equation}

nemá řešení, protože $N$ je výrazně vyšší než $p + 1$. Navíc statistická pravda říká, že čím více dat, tím lépe. Tedy máme vlastně zájem na tom, aby soustava skoro jistě řešení neměla. Proto rezignujeme na snahu soustavu vyřešit a hledáme $\mathbf{w}$ tak, aby minimalizovalo (nikoli tedy nutně vynulovalo) chybu $RSS(\mathbf{w})$. Tuto chybu lze přepsat maticově s využitím euklidovské normy následovně:
\begin{equation}
RSS(\mathbf{w}) = \sum_{i=1}^{N}(Y_i - \mathbf{w}^T\mathbf{x}_i)^2 = ||\mathbf{Y} - \mathbf{X}\mathbf{w}||^2.
\end{equation}

Než se pustíme do řešení této úlohy, zavedeme si ji formálněji v duchu lineární algebry.

\section{Řešení SLR ve smyslu nejmenších čtverců}

V předchozí části jsme si ukázali, jak se můžeme dostat do situace, kdy potřebujeme řešit soustavu lineárních rovnic (SLR), která nemá řešení ve smyslu Frobeniovy věty. Ukázali jsme si, že v takovém případě dává smysl hledat vektor, který sice není přímo řešením, ale v rámci možností je řešení nejblíže, toto si nyní definujeme řádně\footnote{256}

\begin{tcolorbox}[colback=green!20,colframe=green!60!black,title={\textbf{Definice 11.1} (Řešení ve smyslu nejmenších čtverců / Least squares solution)}]
Mějme $\mathbf{A} \in \mathbb{R}^{m,n}$ a $\mathbf{b} \in \mathbb{R}^m$, potom $\mathbf{x} \in \mathbb{R}^n$ nazveme řešením soustavy $\mathbf{A}\mathbf{x} = \mathbf{b}$ ve smyslu nejmenších čtverců, pokud pro každé $\mathbf{y} \in \mathbb{R}^n$ platí:
\begin{equation}
||\mathbf{b} - \mathbf{A}\mathbf{x}|| \leq ||\mathbf{b} - \mathbf{A}\mathbf{y}|| . \tag{11.1}
\end{equation}
\end{tcolorbox}

Definice nám říká, že $\mathbf{A}\mathbf{x}$ je nejblíže $\mathbf{b}$, jak jen to jde.

Podmínku lze přepsat i do kompaktnější formy: $\mathbf{x}$ je řešení ve smyslu nejmenších čtverců, právě když
\begin{equation}
||\mathbf{b} - \mathbf{A}\mathbf{x}|| = \inf\{||\mathbf{b} - \mathbf{A}\mathbf{y}|| \mid \mathbf{y} \in \mathbb{R}^n\} . \tag{11.2}
\end{equation}

Pokud je $\mathbf{x}$ (klasickým) řešením soustavy, potom je $||\mathbf{b} - \mathbf{A}\mathbf{x}|| = ||\mathbf{b} - \mathbf{b}|| = 0$. Tedy v nerovnosti \eqref{11.1} obdržíme na levé straně 0 a nerovnost je automaticky splněna, nebot norma nemůže být nikdy záporná.

\begin{tcolorbox}[colback=orange!20,colframe=orange!60!black,title={\textbf{Pozorování 11.1}}]
Má-li soustava lineárních rovnic řešení (tj. $\mathcal{S} \neq \emptyset$), je množina řešení shodná s množinou řešení dané soustavy ve smyslu nejmenších čtverců.
\end{tcolorbox}

Už máme „nejlepší možné" řešení definováno. Jak jej spočítat? Jak dobře víme a jak jsme si vysvětlili výše, tak výraz $\mathbf{A}\mathbf{x}$ je roven lineární kombinaci sloupců matice $\mathbf{A}$:
\begin{equation}
\mathbf{A}\mathbf{x} = x_1\mathbf{A}_{:1} + x_2\mathbf{A}_{:2} + \cdots + x_n\mathbf{A}_{:n}.
\end{equation}

Pokud minimalizujeme $||\mathbf{b} - \mathbf{A}\mathbf{x}||$, tak vlastně minimalizujeme $d(\mathbf{b}, \langle\mathbf{A}_{:1}, \ldots, \mathbf{A}_{:n}\rangle)$.

Hledáme tedy v daném podprostoru nejbližší vektor k nějakému zadanému vektoru. To je ale úloha, kterou už máme vyřešenou, viz sekci 6.3: řešením je ortogonální projekce zadaného vektoru do tohoto podprostoru.

\begin{tcolorbox}[colback=cyan!20,colframe=cyan!60!black,title={\textbf{Tvrzení 11.1}}]
Mějme $\mathbf{A} \in \mathbb{R}^{m,n}$ a $\mathbf{b} \in \mathbb{R}^m$, potom $\mathbf{x} \in \mathbb{R}^n$ je řešením soustavy $\mathbf{A}\mathbf{x} = \mathbf{b}$ ve smyslu nejmenších čtverců, právě když je řešením soustavy
\begin{equation}
\mathbf{A}\mathbf{x} = \text{proj}_{\langle\mathbf{A}_{:1}, \ldots, \mathbf{A}_{:n}\rangle} \mathbf{b}. \tag{11.3}
\end{equation}
\end{tcolorbox}

Soustava \eqref{11.3} je vždy řešitelná, Frobeniova věta nám říká, kdy má jediné řešení:

\begin{tcolorbox}[colback=cyan!20,colframe=cyan!60!black,title={\textbf{Důsledek 11.1}}]
Mějme $\mathbf{A} \in \mathbb{R}^{m,n}$ a $\mathbf{b} \in \mathbb{R}^m$. Potom existuje právě jedno řešení soustavy $\mathbf{A}\mathbf{x} = \mathbf{b}$ ve smyslu nejmenších čtverců, právě když soubor sloupců matice $\mathbf{A}$ je LN.
\end{tcolorbox}

\vspace{0.5cm}
\textcolor{blue}{\underline{Zobrazit důkaz} \textvisiblespace}

V našem případě, kdy uvažujeme $\mathbb{R}^n$ s euklidovskou normou indukovanou standardním skalárním součinem, tak umíme projekci na podprostor spočítat pomocí jeho ortogonální báze (viz Definice \eqref{6.3}).

Ukážeme si, jak řešení ve smyslu nejmenších čtverců spočítat bez toho, abychom tuto projekci počítali.

\begin{tcolorbox}[colback=cyan!20,colframe=cyan!60!black,title={\textbf{Věta 11.1} (Řešení ve smyslu nejmenších čtverců)}]
Mějme $\mathbf{A} \in \mathbb{R}^{m,n}$ a $\mathbf{b} \in \mathbb{R}^m$. Potom $\mathbf{x} \in \mathbb{R}^n$ je řešením soustavy $\mathbf{A}\mathbf{x} = \mathbf{b}$ ve smyslu nejmenších čtverců, právě když jsou splněny tzv. normální rovnice
\begin{equation}
\mathbf{A}^T\mathbf{A}\mathbf{x} = \mathbf{A}^T\mathbf{b} . \tag{11.4}
\end{equation}

Pokud navíc $h(\mathbf{A}) = n$, potom lze toto řešení vyjádřit:
\begin{equation}
\mathbf{x} = (\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T\mathbf{b} . \tag{11.5}
\end{equation}
\end{tcolorbox}

\vspace{0.5cm}
\textcolor{blue}{\underline{Zobrazit důkaz} \textvisiblespace}

\subsection{Vícerozměrný příklad -- řešení}

Nyní můžeme aplikovat získané výsledky na náš problém vícerozměrné lineární regrese a jeho řešení metodou nejmenších čtverců, kdy hledáme minimum výrazu
\begin{equation}
RSS(\mathbf{w}) = ||\mathbf{Y} - \mathbf{X}\mathbf{w}||^2
\end{equation}

pro matici $\mathbf{X} \in \mathbb{R}^{N,p+1}$ a vektor $\mathbf{Y} \in \mathbb{R}^N$.Podle věty 11.1 minimalizuje $\mathbf{w}$ funkci $RSS(\mathbf{w})$, jestliže splňuje normální rovnice
\begin{equation}
\mathbf{X}^T\mathbf{X}\mathbf{w} = \mathbf{X}^T\mathbf{Y}.
\end{equation}

Navíc víme, že tato soustava má jednoznačné řešení, právě když jsou sloupce matice $\mathbf{X}$ lineárně nezávislé\footnote{257}. To je zároveň ekvivalentní tomu, že matice $\mathbf{X}^T\mathbf{X}$ je regulární a jednoznačné řešení je rovno
\begin{equation}
\hat{\mathbf{w}}^{(OLS)} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{Y}.
\end{equation}

Vektoru $\hat{\mathbf{w}}^{(OLS)}$ se říká \emph{odhad metodou nejmenších čtverců}.

\begin{tcolorbox}[colback=cyan!20,colframe=cyan!60!black,title={\textbf{Věta 11.2}}]
Buď $\mathbf{X}$ matice příznaků a $\mathbf{Y}$ vektor vysvětlované proměnné jako výše. Pro odhad $\hat{\mathbf{w}}^{(OLS)}$ parametrů vícerozměrné lineární regrese metodou nejmenších čtverců platí:
\begin{equation}
\mathbf{X}^T\mathbf{X}\hat{\mathbf{w}}^{(OLS)} = \mathbf{X}^T\mathbf{Y}.
\end{equation}

Je-li navíc soubor sloupců matice $\mathbf{X}$ LN, pak je řešení této rovnice jednoznačné a
\begin{equation}
\hat{\mathbf{w}}^{(OLS)} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{Y}.
\end{equation}
\end{tcolorbox}

\section{Výpočet OLS pomocí QR rozkladu}

Výpočet inverze $(\mathbf{A}^T\mathbf{A})^{-1}$ v \eqref{11.5} je numericky náročný. Ukážeme si, jak se mu vyhnout a nahradit jej výpočtem QR rozkladu matice $\mathbf{A} \in \mathbb{R}^{N,p+1}$, který umíme spočítat pomocí numericky stabilních algoritmů. Ukážeme si dokonce, že stačí pouze redukovaný QR rozklad, a když úplný QR rozklad nám poskytne nějaké informace navíc.

Hlavní trik spočívá v aplikaci poznámky 6.2, která říká, že obsahuje-li $\mathbf{Q}$ ve svých sloupcích ON bázi lineárního obalu sloupců matice $\mathbf{A}$, potom je
\begin{equation}
\text{proj}_{\langle\mathbf{A}_{:1}, \ldots, \mathbf{A}_{:n}\rangle} \mathbf{b} = \mathbf{Q}\mathbf{Q}^T\mathbf{b}. \tag{11.6}
\end{equation}

Ortonormální bázi lineárního obalu sloupců matice jsme již počítali a to právě při hledání QR rozkladu (více v podkapitole 8.8.1).Označme si redukovaný QR rozklad matice $\mathbf{A}$ jako
\begin{equation}
\mathbf{A} = \hat{\mathbf{Q}}\hat{\mathbf{R}},
\end{equation}

a úplný rozklad jako
\begin{equation}
\mathbf{A} = \mathbf{Q}\mathbf{R}.
\end{equation}

Můžeme tedy psát, při blokovém maticovém zápisu, že
\begin{equation}
\mathbf{Q} = \begin{pmatrix} \hat{\mathbf{Q}} & \mathbf{Q}' \end{pmatrix} \quad \text{a} \quad \mathbf{R} = \begin{pmatrix} \hat{\mathbf{R}} \\ \boldsymbol{\Theta} \end{pmatrix},
\end{equation}

kde $\boldsymbol{\Theta}$ značí nulovou matici. Rozměry všech matic si rozmyslete!

Dáme-li dohromady rovnice \eqref{11.3} a \eqref{11.6}, dostaneme
\begin{equation}
\hat{\mathbf{Q}}\hat{\mathbf{R}}\mathbf{x} = \mathbf{A}\mathbf{x} = \text{proj}_{\langle\mathbf{A}_{:1}, \ldots, \mathbf{A}_{:n}\rangle} \mathbf{b} = \hat{\mathbf{Q}}\hat{\mathbf{Q}}^T\mathbf{b}.
\end{equation}

Víme, že tato rovnice má řešení. Snadno si rozmyslíme, že díky ortogonalitě sloupců matice $\hat{\mathbf{Q}}$ můžeme obě strany rovnice vynásobit $\hat{\mathbf{Q}}^T$ zleva a dostaneme soustavu
\begin{equation}
\hat{\mathbf{R}}\mathbf{x} = \hat{\mathbf{Q}}^T\mathbf{b},
\end{equation}

jejíž množina je stejná jako množina řešení soustavy původní. To už nám v podstatě dává důkaz následující věty.

\begin{tcolorbox}[colback=cyan!20,colframe=cyan!60!black,title={\textbf{Věta 11.3} (Řešení ve smyslu nejmenších čtverců pomocí QR rozkladu)}]
Mějme $\mathbf{A} \in \mathbb{R}^{m,n}$ a $\mathbf{b} \in \mathbb{R}^m$, $h(\mathbf{A}) = n$ a buď $\mathbf{A} = \mathbf{Q}\mathbf{R} = \hat{\mathbf{Q}}\hat{\mathbf{R}}$ úplný resp. částečný QR rozklad matice $\mathbf{A}$. Potom $\mathbf{x} \in \mathbb{R}^n$ je řešením soustavy $\mathbf{A}\mathbf{x} = \mathbf{b}$ ve smyslu nejmenších čtverců, právě když je řešením soustavy
\begin{equation}
\hat{\mathbf{R}}\mathbf{x} = \hat{\mathbf{Q}}^T\mathbf{b}. \tag{11.7}
\end{equation}

Navíc pro taková $\mathbf{x}$ platí
\begin{equation}
||\mathbf{b} - \mathbf{A}\mathbf{x}|| = ||\mathbf{Q}'^T\mathbf{b}||,
\end{equation}

kde
\begin{equation}
\mathbf{Q} = \begin{pmatrix} \hat{\mathbf{Q}} & \mathbf{Q}' \end{pmatrix}.
\end{equation}
\end{tcolorbox}

\vspace{0.5cm}
\textcolor{blue}{\underline{Zobrazit důkaz} \textvisiblespace}

\subsection{Vícerozměrný příklad -- řešení pomocí QR}

Naposledy se vrátíme k vícerozměrnému případu lineární regrese a řešení metodou nejmenších čtverců. Věta 11.3 říká, že odhad $\hat{\mathbf{w}}^{(OLS)}$ můžeme spočítat pomocí QR rozkladu matice $\mathbf{X}$.

\begin{tcolorbox}[colback=cyan!20,colframe=cyan!60!black,title={\textbf{Věta 11.4}}]
Buď $\mathbf{X}$ matice příznaků a $\mathbf{Y}$ vektor vysvětlované proměnné jako výše. Označme $\mathbf{X} = \mathbf{Q}\mathbf{R} = \hat{\mathbf{Q}}\hat{\mathbf{R}}$ úplný resp. částečný QR rozklad matice $\mathbf{X}$. Potom $\hat{\mathbf{w}}^{(OLS)}$ splňuje
\begin{equation}
\hat{\mathbf{R}}\mathbf{w} = \hat{\mathbf{Q}}^T\mathbf{Y}
\end{equation}

a navíc
\begin{equation}
\min_{\mathbf{w} \in \mathbb{R}^{p+1}} RSS(\mathbf{w}) = ||\mathbf{Q}'^T\mathbf{Y}||^2,
\end{equation}

kde
\begin{equation}
\mathbf{Q} = \begin{pmatrix} \hat{\mathbf{Q}} & \mathbf{Q}' \end{pmatrix}.
\end{equation}
\end{tcolorbox}

Můžeme tedy psát, při blokovém maticovém zápisu, že
\begin{equation}
\mathbf{Q} = \begin{pmatrix} \hat{\mathbf{Q}} & \mathbf{Q}' \end{pmatrix} \quad \text{a} \quad \mathbf{R} = \begin{pmatrix} \hat{\mathbf{R}} \\ \boldsymbol{\Theta} \end{pmatrix},
\end{equation}

kde $\boldsymbol{\Theta}$ značí nulovou matici. Rozměry všech matic si rozmyslete!

Dáme-li dohromady rovnice \eqref{11.3} a \eqref{11.6}, dostaneme
\begin{equation}
\hat{\mathbf{Q}}\hat{\mathbf{R}}\mathbf{x} = \mathbf{A}\mathbf{x} = \text{proj}_{\langle\mathbf{A}_{:1}, \ldots, \mathbf{A}_{:n}\rangle} \mathbf{b} = \hat{\mathbf{Q}}\hat{\mathbf{Q}}^T\mathbf{b}.
\end{equation}

Víme, že tato rovnice má řešení. Snadno si rozmyslíme, že díky ortogonalitě sloupců matice $\hat{\mathbf{Q}}$ můžeme obě strany rovnice vynásobit $\hat{\mathbf{Q}}^T$ zleva a dostaneme soustavu
\begin{equation}
\hat{\mathbf{R}}\mathbf{x} = \hat{\mathbf{Q}}^T\mathbf{b},
\end{equation}

jejíž množina je stejná jako množina řešení soustavy původní. To už nám v podstatě dává důkaz následující věty.
