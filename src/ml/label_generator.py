import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def generate_regime_labels(price_data: pd.DataFrame, window: int = 20, threshold: float = 0.01) -> pd.Series:
    """
    Basit kurallara dayalı olarak geçmiş fiyat verileri için rejim etiketleri oluşturur.
    Bu fonksiyon, rejim tahmin modelini eğitmek için gereklidir.

    Etiketler:
    - 0: Bullish (Boğa)
    - 1: Neutral (Nötr)
    - 2: Bearish (Ayı)

    Args:
        price_data: OHLCV içeren bir Pandas DataFrame.
        window: Gelecekteki getiriyi hesaplamak için kullanılacak periyot.
        threshold: 'Bullish' veya 'Bearish' olarak kabul edilecek getiri eşiği.

    Returns:
        Her bir zaman damgası için rejim etiketlerini içeren bir Pandas Serisi.
    """
    logger.info(f"Rejim etiketleri oluşturuluyor: pencere={window}, eşik={threshold}...")
    
    # Gelecekteki fiyat değişimini hesapla (etiketleme için ileriye bakıyoruz)
    future_returns = price_data['close'].pct_change(periods=window).shift(-window)
    
    # Etiketleri ata (Varsayılan: Neutral)
    labels = pd.Series(1, index=price_data.index, name="regime_labels")
    
    # .loc kullanarak güvenli atama yap
    labels.loc[future_returns > threshold] = 0  # Bullish
    labels.loc[future_returns < -threshold] = 2 # Bearish
    
    # Son 'window' kadar veri NaN (boş) olacağından,
    # bu boşlukları bir önceki geçerli etiketle doldur.
    # --- DÜZELTME (FutureWarning) ---
    # labels.fillna(method='ffill', inplace=True) -> labels.ffill(inplace=True)
    labels.ffill(inplace=True)
    
    # Oluşturulan etiketlerin sayısını logla (hata ayıklama için yararlı)
    label_counts = labels.value_counts()
    logger.info(f"Etiket oluşturma tamamlandı. Sayımlar: "
                f"Bullish (0): {label_counts.get(0, 0)}, "
                f"Neutral (1): {label_counts.get(1, 0)}, "
                f"Bearish (2): {label_counts.get(2, 0)}")

    return labels
