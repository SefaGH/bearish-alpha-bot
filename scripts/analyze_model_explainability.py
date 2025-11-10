"""
Model Yorumlanabilirlik Analiz Betiği (Explainability Script)
SÜRÜM 5 - Hata Düzeltmesi (StandardScaler eklendi)

Bu betik, eğitilmiş bir modeli alır ve neden belirli kararları verdiğini
analiz etmek için Permutation Importance ve SHAP yöntemlerini kullanır.

Tuning'de kullanılan scaler'ı yükleyip veriyi analizden önce ölçekler.
"""

import torch
import numpy as np
import pandas as pd
import shap
import json
import matplotlib.pyplot as plt
import argparse
import sys
import os
import joblib  # <<< YENİ İMPORT (Scaler'ı yüklemek için) >>>
from pathlib import Path # <<< YENİ İMPORT >>>
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, make_scorer, confusion_matrix
from sklearn.preprocessing import StandardScaler # <<< YENİ İMPORT >>>

# <<< YENİ: Scaler dosyasının yolu (Tuning'de kaydedilen) >>>
SCALER_PATH = Path('data/cache/scaler_production.joblib')

# --- PyTorch Modelini Sklearn Uyumlu Hale Getirme ---

class PyTorchWrapper:
    """PyTorch modelini sklearn uyumlu hale getiren sarmalayıcı (wrapper)."""
    def __init__(self, model_path):
        try:
            self.model = torch.jit.load(model_path)
            self.model.eval()
        except Exception as e:
            print(f"HATA: Model yüklenemedi: {model_path}. Hata: {e}")
            sys.exit(1)

    def fit(self, X, y, **kwargs):
        """
        Sahte (dummy) .fit() metodu.
        permutation_importance fonksiyonu için gereklidir.
        """
        return self

    def predict_proba(self, X):
        """Olasılıkları (probabilities) döndürür."""
        try:
            X_tensor = torch.tensor(X, dtype=torch.float32)
            with torch.no_grad():
                logits = self.model(X_tensor)
            probabilities = torch.softmax(logits, dim=1)
            return probabilities.numpy()
        except Exception as e:
            print(f"HATA: Model tahmini sırasında sorun oluştu: {e}")
            return np.zeros((X.shape[0], 3)) 

    def predict(self, X):
        """En yüksek olasılıklı sınıfı (0, 1, veya 2) döndürür."""
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

# --- Analiz Fonksiyonları ---

def load_data_and_features(data_path, metadata_path):
    """Veri ve özellik isimlerini yükler."""
    print(f"Veri yükleniyor: {data_path}")
    try:
        data = np.load(data_path)
        X_full = data['X']
        y_full = data['y']
    except FileNotFoundError:
        print(f"HATA: Veri dosyası bulunamadı: {data_path}")
        sys.exit(1)
    except KeyError as e:
        print(f"HATA: .npz dosyası içinde 'X' veya 'y' anahtarı bulunamadı: {e}")
        sys.exit(1)

    print(f"Özellik isimleri yükleniyor: {metadata_path}")
    feature_names_list = []
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # JSON dosyasında doğrulanan anahtar 'selected_features'
        if 'selected_features' in metadata:
            feature_names_list = metadata['selected_features']
            print("   ... 'selected_features' anahtarı başarıyla bulundu.")
        else:
            raise KeyError(f"JSON içinde 'selected_features' anahtarı bulunamadı.")

    except FileNotFoundError:
        print(f"HATA: Metadata bulunamadı: {metadata_path}. 'feature_names' olmadan devam edilecek.")
    except KeyError as e:
        print(f"HATA: Metadata JSON ({metadata_path}) içinde özellik listesi bulunamadı: {e}.")
        print("Jenerik isimler (feature_0, feature_1...) kullanılacak.")
    
    # Güvenlik kontrolü: Eğer isim listesi boşsa veya uzunluğu uyuşmuyorsa, jenerik isimler kullan
    if not feature_names_list or len(feature_names_list) != X_full.shape[1]:
        if feature_names_list: # Uyuşmazlık varsa uyar
             print(f"UYARI: Veri {X_full.shape[1]} özelliğe sahip ama metadata {len(feature_names_list)} isim listeliyor.")
        print("Jenerik isimlere (feature_0, feature_1...) dönülüyor.")
        feature_names_list = [f"feature_{i}" for i in range(X_full.shape[1])]
        
    return X_full, y_full, feature_names_list

def run_permutation_importance(model, X_test, y_test, feature_names, output_dir):
    """Genel özellik önemliliği analizini çalıştırır."""
    print("\n" + "="*50)
    print("YÖNTEM 1: PERMUTATION IMPORTANCE (Genel Özellik Önemi)")
    print("Bu işlem 1-2 dakika sürebilir...")

    balanced_scorer = make_scorer(balanced_accuracy_score)
    
    r = permutation_importance(
        model, 
        X_test, # <<< NOT: Bu veri zaten ölçeklenmiş olarak geliyor >>>
        y_test,
        n_repeats=10,
        random_state=42,
        scoring=balanced_scorer,
        n_jobs=1  # Paralel çalışmayı kapat (PicklingError fix)
    )
    
    print("En Önemli 20 Özellik (Modelin Kararlarını En Çok Etkileyenler):")
    sorted_idx = r.importances_mean.argsort()[::-1]
    
    top_features = []
    for i in sorted_idx[:20]:
        mean = r.importances_mean[i]
        std = r.importances_std[i]
        # Özellik isminin (string) 30 karakterden uzun olmamasını sağla
        feature_name_str = str(feature_names[i])[:30] 
        print(f"  {feature_name_str:<30}: {mean:.4f} +/- {std:.4f}")
        top_features.append(feature_names[i])

    # Grafik oluştur
    plt.figure(figsize=(10, 8))
    plt.barh(
        [str(feature_names[i])[:30] for i in sorted_idx[:20]][::-1], 
        r.importances_mean[sorted_idx[:20]][::-1]
    )
    plt.xlabel("Önemlilik (Dengeli Doğruluk Kaybı)")
    plt.title("En Önemli 20 Özellik (Permutation Importance)")
    plt.tight_layout()
    output_path = os.path.join(output_dir, "permutation_importance.png")
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Permutation Importance grafiği kaydedildi: {output_path}")

def find_biggest_error(y_test, y_pred):
    """Karışıklık matrisindeki en büyük hatayı (en çok yanlış sınıflandırılan) bulur."""
    cm = confusion_matrix(y_test, y_pred)
    # Diagonal (doğru tahminler) dışındaki en büyük sayıyı bul
    np.fill_diagonal(cm, 0) # Doğruları sıfırla
    
    max_error_count = np.max(cm)
    indices = np.unravel_index(np.argmax(cm), cm.shape)
    
    actual_class = indices[0] # Gerçek sınıf (örn: 2)
    predicted_class = indices[1] # Tahmin edilen sınıf (örn: 1)
    
    class_map = {0: "Bullish", 1: "Neutral", 2: "Bearish"}
    
    print("\n" + "="*50)
    print("HATA ANALİZİ: Karışıklık Matrisindeki En Büyük Hata:")
    print(f"  Gerçek Sınıf: {class_map.get(actual_class, actual_class)}")
    print(f"  Tahmin Edilen Sınıf: {class_map.get(predicted_class, predicted_class)}")
    print(f"  Örnek Sayısı: {max_error_count} adet")
    print("="*50)
    
    return actual_class, predicted_class, max_error_count, class_map

def run_shap_analysis(model, X_train, X_test, y_test, y_pred, feature_names, output_dir):
    """Modelin en büyük hatasına odaklanan SHAP analizini çalıştırır."""
    
    actual_class, predicted_class, error_count, class_map = find_biggest_error(y_test, y_pred)
    
    if error_count == 0:
        print("Modelde hiç hata bulunamadı. SHAP analizi atlanıyor.")
        return

    # Hatalı örneklerin indekslerini bul
    error_indices = np.where((y_test == actual_class) & (y_pred == predicted_class))[0]
    
    # Analiz için en fazla 10 hatalı örnek al (GitHub Actions'ta hızlı olması için)
    analysis_samples_bad = X_test[error_indices[:10]] # <<< NOT: Bu veri zaten ölçeklenmiş >>>
    
    if len(analysis_samples_bad) == 0:
        print("Analiz edilecek hatalı örnek bulunamadı. (Test setinde bu hata olmayabilir)")
        return

    print("SHAP için arka plan (background) veri seti oluşturuluyor (100 örnek)...")
    background_data = shap.sample(X_train, 100) # <<< NOT: Bu veri zaten ölçeklenmiş >>>
    
    print("SHAP Explainer oluşturuluyor...")
    explainer = shap.KernelExplainer(model.predict_proba, background_data)
    
    print(f"SHAP değerleri hesaplanıyor ({len(analysis_samples_bad)} hatalı örnek için)...")
    print("DİKKAT: Bu işlem ÇOK UZUN sürebilir (5-15 dakika)...")
    shap_values = explainer.shap_values(analysis_samples_bad)
    
    print("Hesaplama tamamlandı. Grafikler oluşturuluyor...")
    
    class_names = [class_map.get(i, f"Class {i}") for i in range(len(class_map))]
    
    # Özellik isimlerini DataFrame'e çevir (SHAP'ın bazen ihtiyaç duyduğu format)
    analysis_samples_df = pd.DataFrame(analysis_samples_bad, columns=feature_names)
    
    # Özet Grafik: Hatalı örnekler için genel özellik etkisi
    plt.figure()
    shap.summary_plot(
        shap_values, 
        analysis_samples_df, 
        class_names=class_names,
        show=False,
        plot_type="bar" # Hangi özelliğin ORTALAMA etkiye sahip olduğunu göster
    )
    plt.title(f"Hata Analizi: En Etkili Özellikler (Ortalama Etki)")
    output_path = os.path.join(output_dir, "shap_summary_bar.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"✅ SHAP Özet (Bar) grafiği kaydedildi: {output_path}")

    # Detaylı Özet Grafik
    plt.figure()
    shap.summary_plot(
        shap_values, 
        analysis_samples_df, 
        class_names=class_names,
        show=False
    )
    plt.title(f"Hata Analizi: {class_map.get(actual_class)} -> {class_map.get(predicted_class)}")
    output_path = os.path.join(output_dir, "shap_summary_detailed.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"✅ SHAP Özet (Detaylı) grafik kaydedildi: {output_path}")

def main():
    """Ana analiz fonksiyonu."""
    
    # Argümanları al (Workflow'dan gelecek)
    parser = argparse.ArgumentParser(description="Model Yorumlanabilirlik Analizi")
    parser.add_argument("--model-path", type=str, required=True, help="Eğitilmiş .ptc modelinin yolu.")
    parser.add_argument("--data-path", type=str, required=True, help="Eğitim verisi .npz dosyasının yolu.")
    parser.add_argument("--metadata-path", type=str, required=True, help="Özellik isimlerini içeren .json dosyasının yolu.")
    parser.add_argument("--output-dir", type=str, default=".", help="Analiz grafiklerinin kaydedileceği dizin.")
    args = parser.parse_args()

    # Çıktı dizinini oluştur
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Modeli yükle
    model = PyTorchWrapper(args.model_path)
    
    # 2. Veriyi ve özellikleri yükle (HENÜZ ÖLÇEKLENMEMİŞ)
    X_full, y_full, feature_names = load_data_and_features(args.data_path, args.metadata_path)

    # 3. Veriyi loglardaki gibi 80/20 böl (HENÜZ ÖLÇEKLENMEMİŞ)
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, 
        y_full, 
        test_size=0.20, #
        random_state=42, 
        stratify=y_full #
    )
    
    # <<< BAŞLANGIÇ: YENİ ÖLÇEKLEME (SCALING) ADIMI >>>
    # ==============================================================================
    print("\n" + "="*50)
    print("⚖️ LOADING STANDARD SCALER (ÖLÇEKLEYİCİ)")
    print("="*50)

    if not SCALER_PATH.exists():
        print(f"❌ HATA: Kayıtlı scaler (ölçekleyici) bulunamadı: {SCALER_PATH}")
        print("   Bu betik, 'full-lstm-tuning.yml' tarafından oluşturulan scaler'a bağımlıdır.")
        sys.exit(1)
    
    try:
        scaler = joblib.load(SCALER_PATH)
        print(f"✅ Scaler (Ölçekleyici) başarıyla yüklendi: {SCALER_PATH}")
        
        # Hem Train hem de Test verisini 'transform' et
        print("Transforming Train and Test data...")
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"   Train data shape: {X_train_scaled.shape}")
        print(f"   Test data shape: {X_test_scaled.shape}")
        
    except Exception as e:
        print(f"❌ HATA: Scaler yüklenirken veya veri dönüştürülürken hata oluştu: {e}")
        sys.exit(1)

    print("="*50)
    # ============================================================================== #
    # <<< SON: YENİ ÖLÇEKLEME (SCALING) ADIMI >>>
    
    # 4. Genel Özellik Önemliliğini Çalıştır (ÖLÇEKLENMİŞ VERİ İLE)
    run_permutation_importance(model, X_test_scaled, y_test, feature_names, args.output_dir)

    # 5. SHAP ile Hata Analizini Çalıştır (ÖLÇEKLENMİŞ VERİ İLE)
    print("Modelin hatalarını analiz etmek için test seti üzerinde tahmin yapılıyor...")
    y_pred = model.predict(X_test_scaled)
    run_shap_analysis(model, X_train_scaled, X_test_scaled, y_test, y_pred, feature_names, args.output_dir)

    print("\nAnaliz tamamlandı.")

if __name__ == "__main__":
    main()
