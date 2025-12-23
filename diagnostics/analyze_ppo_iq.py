import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# --- AYARLAR ---
LOOK_AHEAD_CANDLES = 2       # Karardan sonraki kaçıncı muma bakılsın? (Log sıklığına göre)
CONFIDENCE_THRESHOLD = 0.60  # Güçlü sinyal sayılması için gereken güven oranı

class PPOIQAnalyzer:
    def __init__(self, log_file_path):
        self.log_file = log_file_path
        self.ppo_data = []
        self.price_data = []
        self.health_issues = []

    def parse_logs(self):
        """Log dosyasını okur ve Regex ile verileri ayıklar."""
        if not os.path.exists(self.log_file):
            print(f"❌ HATA: Dosya bulunamadı: {self.log_file}")
            sys.exit(1)

        print(f"📂 Analiz Ediliyor: {os.path.basename(self.log_file)}...")
        
        # --- REGEX TANIMLARI ---
        
        # 1. PPO Kararı (Debug Satırı)
        # Örnek: ... [PPO-DEBUG] ... p_long=0.79 ...
        ppo_pattern = re.compile(
            r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?\[PPO-DEBUG\].*?p_long=(?P<p_long>[\d\.]+).*?conf_raw=(?P<conf>[\d\.]+).*?action_int=(?P<action>\d)"
        )
        
        # 2. Fiyat Verisi (Çoklu Kaynak)
        # Örnek 1: Price $89,747.70
        # Örnek 2: [OB R/R] Entry=$89711.60
        # Örnek 3: BTC/USDT price via 'bingx': $89719.90
        price_pattern = re.compile(
            r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?(?:Price|Entry|via 'bingx')[:\s]+\$(?P<price>[\d,\.]+)"
        )

        # 3. Sağlık Sorunu (Health Guard)
        # Örnek: 🚨 HEALTH GUARD TRIGGERED! Reasons: ['obs_clip_high']
        health_pattern = re.compile(r"HEALTH GUARD TRIGGERED! Reasons: \['(?P<reason>.*?)'\]")

        # Dosyayı Satır Satır Oku
        with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # PPO Yakala
                ppo_match = ppo_pattern.search(line)
                if ppo_match:
                    d = ppo_match.groupdict()
                    self.ppo_data.append({
                        'timestamp': pd.to_datetime(d['ts']),
                        'p_long': float(d['p_long']),
                        'confidence': float(d['conf']),
                        'action': int(d['action'])
                    })
                    continue

                # Fiyat Yakala
                price_match = price_pattern.search(line)
                if price_match:
                    d = price_match.groupdict()
                    clean_price = float(d['price'].replace(',', ''))
                    self.price_data.append({
                        'timestamp': pd.to_datetime(d['ts']),
                        'price': clean_price
                    })
                    continue

                # Hata Yakala
                health_match = health_pattern.search(line)
                if health_match:
                    self.health_issues.append(health_match.group('reason'))

        print(f"✅ Veri Çıkarıldı: {len(self.ppo_data)} PPO Kararı | {len(self.price_data)} Fiyat Noktası")

    def run_analysis(self):
        """Verileri birleştirir, metrikleri hesaplar ve raporlar."""
        if not self.ppo_data or not self.price_data:
            print("❌ Yetersiz veri. Log dosyasında PPO-DEBUG veya fiyat bilgisi olduğundan emin olun.")
            return

        # DataFrame Oluştur
        df_ppo = pd.DataFrame(self.ppo_data).sort_values('timestamp')
        df_price = pd.DataFrame(self.price_data).sort_values('timestamp')

        # Fiyat ve Kararları Eşleştir (En yakın zamanlı fiyatı bul)
        df_merged = pd.merge_asof(
            df_ppo, 
            df_price, 
            on='timestamp', 
            direction='nearest', 
            tolerance=pd.Timedelta('1min')
        )
        
        # Gelecek Performansı (Basit Lookahead)
        # Bot "Al" dediğinde fiyat sonraki adımlarda ne yapmış?
        df_merged['future_price'] = df_merged['price'].shift(-LOOK_AHEAD_CANDLES)
        df_merged['price_change_pct'] = (df_merged['future_price'] - df_merged['price']) / df_merged['price'] * 100
        
        # NaN değerleri temizle (son satırlar)
        self.df = df_merged.dropna(subset=['price_change_pct'])

        self._print_report()
        self._plot_charts()

    def _print_report(self):
        """Metin tabanlı karne çıkarır."""
        print("\n" + "="*50)
        print(f"🤖 PPO TRADING IQ KARNESİ")
        print("="*50)

        # 1. SAĞLIK DURUMU
        if self.health_issues:
            count = len(self.health_issues)
            unique = set(self.health_issues)
            print(f"🚨 SAĞLIK UYARISI: Toplam {count} kez 'Health Guard' devreye girdi.")
            print(f"   Sebepler: {unique}")
            if 'obs_clip_high' in unique:
                print("   👉 YORUM: 'obs_clip_high', piyasa verisinin (volatilite vb.) eğitilen veriden çok saptığını gösterir.")
        else:
            print("✅ SAĞLIK: Mükemmel. Hiçbir guard tetiklenmedi.")
        
        print("-" * 30)

        # 2. KORELASYON ANALİZİ
        corr = self.df['p_long'].corr(self.df['price_change_pct'])
        print(f"📈 Zeka Puanı (Korelasyon): {corr:.3f}")
        
        if corr > 0.15:
            print("   🌟 SONUÇ: ÇOK İYİ. Model piyasa yönünü tahmin ediyor.")
        elif corr > 0.05:
            print("   ✅ SONUÇ: İYİ. Pozitif bir ilişki var, rastgele değil.")
        elif corr > -0.05:
            print("   😐 SONUÇ: NÖTR. Model henüz yönü tam kestiremiyor (Yazı/Tura).")
        else:
            print("   ❌ SONUÇ: KÖTÜ. Model ters indikatör gibi çalışıyor.")

        # 3. GÜÇLÜ SİNYAL PERFORMANSI
        high_conf = self.df[self.df['p_long'] > CONFIDENCE_THRESHOLD]
        if not high_conf.empty:
            win_rate = (high_conf['price_change_pct'] > 0).mean() * 100
            avg_return = high_conf['price_change_pct'].mean()
            print(f"💪 Güçlü Sinyaller (>{int(CONFIDENCE_THRESHOLD*100)}%):")
            print(f"   - Sinyal Sayısı: {len(high_conf)}")
            print(f"   - Başarı Oranı (Win Rate): %{win_rate:.1f}")
            print(f"   - Ortalama Getiri (Kısa Vade): %{avg_return:.4f}")
        else:
            print("ℹ️ Henüz çok güçlü bir sinyal üretilmemiş.")

    def _plot_charts(self):
        """Görsel analiz grafikleri."""
        plt.figure(figsize=(12, 8))
        
        # Scatter Plot: Karar Kalitesi
        plt.subplot(2, 1, 1)
        sns.scatterplot(data=self.df, x='p_long', y='price_change_pct', hue='confidence', palette='coolwarm')
        plt.axhline(0, color='black', linestyle='-', linewidth=1)
        plt.axvline(0.5, color='gray', linestyle='--')
        plt.title(f'IQ Testi: P_Long vs Sonraki {LOOK_AHEAD_CANDLES} Adım Değişimi')
        plt.xlabel('Modelin Long İsteği (0=Hayır, 1=Evet)')
        plt.ylabel('Gerçekleşen Fiyat Değişimi (%)')
        plt.legend(title='Güven')

        # Zaman Serisi
        plt.subplot(2, 1, 2)
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax1.plot(self.df['timestamp'], self.df['price'], 'k-', alpha=0.5, label='Fiyat')
        ax2.plot(self.df['timestamp'], self.df['p_long'], 'b-', alpha=0.8, label='P_Long Skoru')
        ax2.axhline(0.5, color='red', linestyle='--', alpha=0.3)
        plt.title('Zaman Tüneli: Fiyat ve Model Kararları')
        ax1.set_ylabel('Fiyat ($)')
        ax2.set_ylabel('P_Long (0-1)')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanım: python analyze_ppo_iq.py <log_dosyasi_yolu>")
        sys.exit(1)
    
    log_path = sys.argv[1]
    analyzer = PPOIQAnalyzer(log_path)
    analyzer.parse_logs()
    analyzer.run_analysis()