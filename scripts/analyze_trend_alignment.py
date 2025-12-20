import argparse
import re
import sys
from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None

# --- Görselleştirme Ayarları ---
if sns:
    sns.set(style="darkgrid", context="talk")
plt.rcParams['figure.figsize'] = [16, 10]
plt.rcParams['lines.linewidth'] = 1.5

# Eğer grafik arayüzü hatası alırsan alttaki satırın yorumunu kaldır:
# matplotlib.use('Agg') 

def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(errors="replace")
        except Exception:
            pass

class TrendAnalyzer:
    def __init__(self, log_file, symbol):
        self.log_file = log_file
        self.symbol = symbol
        self.df = pd.DataFrame()
        
        # Regex Tanımları
        # 1. Zaman Damgası
        self.time_pattern = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
        
        # 2. Metin Bazlı Trend (Senin Logların)
        # "EMA fast $86,747.14 below EMA mid $86,953.99."
        self.ema_text_pattern = re.compile(r"EMA fast \$([\d,.]+) (below|above) EMA mid \$([\d,.]+)")
        
        # 3. Sayısal Bazlı Trend (Varsa Yedek)
        self.explicit_score_pattern = re.compile(r"trend_score=([-]?\d+\.\d+)")

    def parse_logs(self):
        """Log dosyasını satır satır okur ve yapılandırılmış veriye çevirir."""
        print(f"📖 Reading and parsing log file: {self.log_file}...")
        data = []
        
        with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # Zaman damgası yoksa atla
                time_match = self.time_pattern.search(line)
                if not time_match:
                    continue
                
                timestamp = datetime.strptime(time_match.group(1), "%Y-%m-%d %H:%M:%S")
                
                # --- Yöntem A: EMA Metin Analizi (Senin Log Formatın) ---
                ema_match = self.ema_text_pattern.search(line)
                if ema_match:
                    try:
                        # HATA DÜZELTMESİ BURADA YAPILDI:
                        # .replace(',', '') -> Virgülleri siler
                        # .rstrip('.')      -> Sondaki noktayı (varsa) siler
                        fast_val_str = ema_match.group(1).replace(',', '').rstrip('.')
                        mid_val_str = ema_match.group(3).replace(',', '').rstrip('.')

                        fast_val = float(fast_val_str)
                        mid_val = float(mid_val_str)
                        direction = ema_match.group(2)
                        
                        # Trend Gücü Hesaplama (EMA Farkı %)
                        diff_pct = ((fast_val - mid_val) / mid_val) * 100
                        
                        # Yön Kontrolü
                        if direction == 'below':
                            regime = 'BEAR'
                            score = -abs(diff_pct) if diff_pct != 0 else -0.01
                        else:
                            regime = 'BULL'
                            score = abs(diff_pct) if diff_pct != 0 else 0.01

                        data.append({
                            'timestamp': timestamp,
                            'price': fast_val,  
                            'trend_score': score,
                            'regime': regime,
                            'source': 'text_ema'
                        })
                    except ValueError as e:
                        print(f"⚠️ Skipping line due to parse error: {line.strip()} | Error: {e}")
                        continue
                    continue

                # --- Yöntem B: Açık Skor (Explicit Score) ---
                score_match = self.explicit_score_pattern.search(line)
                if score_match:
                    try:
                        score = float(score_match.group(1))
                        data.append({
                            'timestamp': timestamp,
                            'price': np.nan, 
                            'trend_score': score,
                            'regime': 'BULL' if score > 0 else 'BEAR',
                            'source': 'explicit'
                        })
                    except ValueError:
                        continue

        self.df = pd.DataFrame(data)
        
        if not self.df.empty:
            self.df = self.df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
            
            # Fiyat eksiklerini tamamla
            self.df['price'] = self.df['price'].ffill().bfill()
            
            # Veriyi dakikalık bazda normalize et (Resample)
            try:
                self.df.set_index('timestamp', inplace=True)
                # Sadece sayısal kolonları resample et
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                df_resampled = self.df[numeric_cols].resample('1min').ffill()
                self.df = df_resampled.dropna().reset_index()
                
                print(f"✅ Successfully parsed {len(self.df)} data points (Resampled to 1-min intervals).")
            except Exception as e:
                print(f"⚠️ Resampling warning: {e}. Proceeding with raw data.")
                self.df.reset_index(inplace=True)
        else:
            print("❌ No trend data found matching known patterns.")

        return not self.df.empty

    def analyze_statistics(self):
        """Trend kararlılığı ve dağılımı hakkında detaylı rapor sunar."""
        df = self.df
        print("\n" + "="*40)
        print("   📊 ADVANCED TREND STATISTICS")
        print("="*40)
        
        if len(df) == 0:
            return

        # 1. Rejim Dağılımı
        bull_points = len(df[df['trend_score'] > 0])
        bear_points = len(df[df['trend_score'] < 0])
        total = len(df)
        
        if total > 0:
            print(f"Total Duration Analyzed: {df['timestamp'].max() - df['timestamp'].min()}")
            print(f"Bullish Bias: {bull_points/total*100:.1f}%")
            print(f"Bearish Bias: {bear_points/total*100:.1f}%")
        
        # 2. Trend Değişim Sıklığı (Flip-Flop)
        df['regime_change'] = (df['trend_score'] * df['trend_score'].shift(1) < 0).astype(int)
        flips = df['regime_change'].sum()
        
        print(f"Trend Flips (Regime Changes): {flips}")
        if total > 0 and flips > (total * 0.1):
            print("⚠️  WARNING: High volatility detected! Bot is changing mind too often.")
        else:
            print("✅ Stability: Trend signals are stable.")

        # 3. Ortalama Trend Gücü
        if bull_points > 0:
            avg_bull_str = df[df['trend_score'] > 0]['trend_score'].mean()
            print(f"Avg Bullish Strength: {avg_bull_str:.4f}% (EMA divergence)")
        
        if bear_points > 0:
            avg_bear_str = df[df['trend_score'] < 0]['trend_score'].mean()
            print(f"Avg Bearish Strength: {avg_bear_str:.4f}% (EMA divergence)")
        else:
            print("Avg Bearish Strength: N/A (No Bearish Data)")

    def analyze_lag_correlation(self):
        """Fiyat hareketleri ile botun reaksiyonu arasındaki gecikmeyi ölçer."""
        df = self.df.copy()
        if len(df) < 5:
            return

        # Fiyat değişimi (%)
        df['price_pct'] = df['price'].pct_change()
        
        # Korelasyon Hesabı
        corr = df['trend_score'].corr(df['price_pct'])
        
        print("\n" + "-"*30)
        print("   ⏱️ LAG & CORRELATION")
        print("-" * 30)
        print(f"Direct Correlation: {corr:.4f}")
        
        if corr < 0:
            print("⚠️  NEGATIVE CORRELATION: Bot appears to be counter-trading (Mean Reversion).")
        elif corr > 0.3:
            print("✅ POSITIVE CORRELATION: Bot is aligned with price direction.")
        else:
            print("ℹ️  WEAK CORRELATION: No clear linear relationship detected.")

    def plot_comprehensive_chart(self):
        """Çok katmanlı, profesyonel analiz grafiği çizer."""
        df = self.df
        if len(df) == 0:
            return

        # İki alt grafik oluştur
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        
        # --- ÜST GRAFİK: FİYAT VE REJİM ---
        ax1.plot(df['timestamp'], df['price'], color='black', linewidth=1.5, label='Price (Fast EMA)')
        
        y_min, y_max = df['price'].min(), df['price'].max()
        padding = (y_max - y_min) * 0.1 if (y_max != y_min) else 100
        ax1.set_ylim(y_min - padding, y_max + padding)
        
        ax1.fill_between(df['timestamp'], y_min-padding, y_max+padding, 
                         where=df['trend_score'] > 0, 
                         color='green', alpha=0.15, label='Bull Regime')
        
        ax1.fill_between(df['timestamp'], y_min-padding, y_max+padding, 
                         where=df['trend_score'] < 0, 
                         color='red', alpha=0.15, label='Bear Regime')
        
        ax1.set_ylabel('Price (USDT)', fontsize=12, fontweight='bold')
        ax1.set_title(f'Market Regime Analysis: {self.symbol}', fontsize=16)
        ax1.legend(loc='upper left')

        # --- ALT GRAFİK: TREND GÜCÜ (OSİLATÖR) ---
        ax2.axhline(0, color='black', linewidth=1, linestyle='--')
        ax2.plot(df['timestamp'], df['trend_score'], color='blue', linewidth=1, label='Trend Strength (%)')
        
        ax2.fill_between(df['timestamp'], df['trend_score'], 0, where=df['trend_score']>0, color='green', alpha=0.3)
        ax2.fill_between(df['timestamp'], df['trend_score'], 0, where=df['trend_score']<0, color='red', alpha=0.3)
        
        ax2.set_ylabel('Trend Strength (%)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Time', fontsize=12)
        ax2.legend(loc='upper left')
        
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.xticks(rotation=0)
        
        output_file = "diagnostics/trend_alignment_chart_v2.png"
        plt.tight_layout()
        plt.savefig(output_file)
        print(f"\n📈 Chart saved successfully to: {output_file}")

def main():
    _configure_stdout()
    if sns is None:
        print("seaborn not installed; continuing without seaborn styling.")
    parser = argparse.ArgumentParser(description="Comprehensive Trend Alignment Analysis Tool")
    parser.add_argument("--log-file", required=True, help="Path to the log file")
    parser.add_argument("--symbol", default="BTC", help="Symbol to filter")
    
    args = parser.parse_args()
    
    analyzer = TrendAnalyzer(args.log_file, args.symbol)
    
    if analyzer.parse_logs():
        analyzer.analyze_statistics()
        analyzer.analyze_lag_correlation()
        try:
            analyzer.plot_comprehensive_chart()
        except Exception as e:
            print(f"❌ Error plotting chart: {e}")
            print("Tip: If running on a headless server, uncomment 'matplotlib.use(\"Agg\")' at top of script.")

if __name__ == "__main__":
    main()
