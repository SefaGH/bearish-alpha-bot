Hedef “işlem açmayan bot” değil; hedef **pozitif beklenen değer (EV) gördüğü yerde mantıklı risk alan**, EV zayıfsa **ya risk küçülten ya da alternatif stratejiye geçen** bot.

Aşağıdaki çerçeve tam bunu yapar: **tek bir sinyali veto etmek yerine, bağlamı sınıflandırır → hangi stratejinin edge’i var onu seçer → risk/size ve R:R’ı formülle belirler.**

---

## 1) 3 sınıf = 3 farklı “edge” (veto değil, strateji seçimi)

### Sınıf A — **Breakout–Acceptance (BA)**

Özellik: impuls + üstte kabul (ret yok, tutunma var).
**Edge MR’da değil; continuation’da.**
→ MR karşı-yöne *ya hiç açma ya da mikro risk*, asıl risk **breakout continuation**.

### Sınıf B — **Genuine Overextension (GO)**

Özellik: band dışına uzama + gerçek rejection (üst fitil/kırmızı kapanış + band içine dönüş).
**Edge MR fade’de.**
→ Normal riskle MR.

### Sınıf C — **Fakeout–Rejection (FR)**

Özellik: impuls var ama ardından güçlü ret + geri dönüş (fake breakout).
**Edge reversal’da güçlü.**
→ MR daha “sert” ama **onay kalıcılığı** şart.

> Ana fikir: “kötü MR sinyalini veto edip boş kalmak” yerine, BA çıkınca **MR yerine continuation** seçiyorsun. Bot işlem açmaya devam ediyor; sadece “doğru edge”e geçiyor.

---

## 2) Sınıflandırma skorları (heuristic ama formül)

Aşağıdaki 4 skoru 0–1 aralığında hesapla:

### (i) ImpulseScore

Son 1–3 dakikadaki tek yönlü şok:
[
I=\text{clip}\left(\max\left(\frac{|r_{1m}|}{0.25%},\frac{\text{range}*{1m}}{2.5\cdot ATR*{1m}},\frac{VR_{1m}}{2.0}\right),0,1\right)
]

### (ii) TrendScore (coordinator)

[
T=\text{clip}\left(\frac{ADX_{coord}-25}{45-25},0,1\right)\cdot \mathbf{1}[\text{slope(VWAP/EMA)}>0 \text{ (yön uyumu)}]
]
(Yön uyumu yoksa T’yi düşür.)

### (iii) AcceptanceScore

İmpuls sonrası fiyatın “üst bölgede tutunması”:
[
A=\text{clip}\left(\frac{\text{time_above_VWAP}}{120s},0,1\right)\cdot (1-R)
]
Burada (R) rejection skoruna bağlı olarak azalır.

### (iv) RejectionScore (kalıcılıkla)

Son N kontrol penceresinde “ret” geçiş oranı:
[
R = 0.7\cdot \text{pass_rate}_{rej}(N) + 0.3\cdot \text{clip}\left(\frac{upperwick}{0.8},0,1\right)
]
Ve **persistency**: ardışık 2 onay yoksa (R:=0.6R) gibi ceza uygula.

---

## 3) Sınıf olasılıkları (basit softmax)

[
S_{BA}=1.2I+1.0T+0.8A-1.0R
]
[
S_{GO}=1.0R+0.8z-1.0I-0.8T
]
[
S_{FR}=1.0I+1.2R-1.0A
]

Softmax:
[
p_k=\frac{e^{S_k}}{\sum_j e^{S_j}}
]

---

## 4) Strateji seçimi (işlem açmayı “0’a” indirmeden)

**Seçim:**

* Eğer (p_{BA}) en büyükse → **Continuation moduna geç** (pullback-long / breakout-follow)
* Eğer (p_{GO}) en büyükse → **MR fade**
* Eğer (p_{FR}) en büyükse → **MR reversal (fakeout)**

> Bu noktada “veto” sadece şurada devreye girer: **seçilen stratejinin minimum edge eşiği sağlanmıyorsa** (ör. tüm p’ler düşük). Bu, işlem açmamak değil; “EV yoksa risk almamak”tır.

---

## 5) Risk (position size) formülü: “EV’ye göre kademeli”

Önce “TradeEdge” skorunu tanımla:
[
E = p_{selected}\cdot Q \cdot M
]

* (Q): quality bileşeni (0–1)
  [
  Q= \text{clip}\left(\frac{\text{quality}-0.50}{0.20},0,1\right)
  ]
* (M): execution/market penalty (0–1)
  [
  M = (1-\text{clip}(\frac{fill_delay}{60s},0,1))\cdot (1-\text{clip}(\frac{|RR-RR_{min}|}{0.2},0,1))
  ]

Sonra risk çarpanı:
[
\text{risk_mult}=\text{clip}\left(\sigma(8(E-0.55)),0.1,1.0\right)
]
[
R_{$}=R_{base}\cdot \text{risk_mult}
]

* (E<0.55) ise risk otomatik küçülür (**micro risk**), tamamen sıfıra inmek zorunda değil.
* Ama (E<0.45) gibi çok düşükte **trade açmak yerine alternatif strateji** dene; o da düşükse “flat”.

---

## 6) R:R’ı bağlama göre dinamik yap (sadece yükseltmek değil)

### MR için:

[
RR_{MR}=1.2 + 1.0\cdot (1-p_{BA}) + 0.6\cdot p_{FR}
]

* BA riski artıyorsa (breakout-acceptance), MR’ın RR talebi yükselir → trade zaten elenir ya da mikro risk olur.
* FR’de reversal edge yüksekse RR artar.

### Continuation için:

[
RR_{CONT}=1.3 + 0.7\cdot p_{BA}
]

---

## 7) Bu vakaya uygularsan ne olurdu?

Senin trade’de tipik tablo:

* (I) yüksek, (T) yüksek (coord ADX 45+), (R) düşük (8 fail), (A) yüksek (kabul)
  → (p_{BA}) büyük çıkar.
  Sonuç:
* **MR SHORT yerine continuation** seçilirdi (ya da MR short mikro risk + çok yüksek RR isterdi).
* Dolayısıyla bot “işlem açmıyor” değil; **doğru tarafa işlem açıyor**.

---

## 8) Uygulamada tek kritik şart: “fill sonrası yeniden fiyatlama”

Bu çerçevenin çalışması için:

* stop/TP **fill_price** üzerinden finalize edilmeli,
* fill_delay büyükse (M) penalty’si devreye girmeli,
* gerekirse “max_wait” ile sinyalin geçerliliği korunmalı.

---