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
import joblib
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, make_scorer, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Scaler dosyasının yolu (Tuning'de kaydedilen) >>>
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
        with open(metadata_path, 'r', encoding='utf-8') as f:
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

def load_and_scale_data(X_train: np.ndarray, X_test: np.ndarray) -> Tuple:
    """
    Tuning'den (Ar-Ge) gelen 'scaler_production.joblib' dosyasını yükler
    ve hem eğitim hem de test verisini ölçekler.
    """
    print("\n" + "="*50)
    print("⚖️ LOADING STANDARD SCALER (ÖLÇEKLEYİCİ)")
    print("="*50)

    if not SCALER_PATH.exists():
        print(f"❌ HATA: Kayıtlı scaler (ölçekleyici) bulunamadı: {SCALER_PATH}")
        print("   Bu betik, 'full-gemma-tuning.yml' tarafından oluşturulan scaler'a bağımlıdır.")
        return None, None, None
    
    try:
        scaler = joblib.load(SCALER_PATH)
        print(f"✅ Scaler (Ölçekleyici) başarıyla yüklendi: {SCALER_PATH}")
        
        # Hem Train hem de Test verisini 'transform' et
        print("Transforming Train and Test data...")
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"   Train data shape: {X_train_scaled.shape}")
        print(f"   Test data shape: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, scaler
        
    except Exception as e:
        print(f"❌ HATA: Scaler yüklenirken veya veri dönüştürülürken hata oluştu: {e}")
        return None, None, None

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

def run_shap_analysis(model_wrapper: PyTorchWrapper, X_train_scaled: np.ndarray, 
                      X_test_scaled: np.ndarray, y_test: np.ndarray, y_pred: np.ndarray, 
                      feature_names: list, output_dir: Path):
    """
    Modelin en büyük hatalarını analiz etmek için SHAP kullanır.
    (GÜNCELLENDİ: KernelExplainer -> GradientExplainer'a geçildi)
    """
    import torch # Gerekli import
    print("\n" + "="*50)
    print("🔬 ADIM 8.2: SHAP Hata Analizi")
    print("="*50)

    try:
        # Karışıklık Matrisi (Confusion Matrix)
        cm = confusion_matrix(y_test, y_pred)
        print("Test Seti Karışıklık Matrisi:\n", cm)

        # En büyük hatayı bul (örn: Gerçekte Bullish (0) iken Neutral (1) tahmin edilmesi)
        # (Diyagonal dışındaki en yüksek sayı)
        np.fill_diagonal(cm, 0) # Doğru tahminleri sıfırla
        error_indices = np.unravel_index(np.argmax(cm, axis=None), cm.shape)
        true_class = error_indices[0]
        pred_class = error_indices[1]
        error_count = cm[true_class, pred_class]
        
        class_names = {0: 'Bullish', 1: 'Neutral', 2: 'Bearish'}
        print("="*50)
        print(f"HATA ANALİZİ: Karışıklık Matrisindeki En Büyük Hata:")
        print(f"  Gerçek Sınıf: {class_names.get(true_class, true_class)}")
        print(f"  Tahmin Edilen Sınıf: {class_names.get(pred_class, pred_class)}")
        print(f"  Örnek Sayısı: {error_count} adet")
        print("="*50)

        # Analiz için bu hatalı örnekleri seç
        error_mask = (y_test == true_class) & (y_pred == pred_class)
        X_test_errors = X_test_scaled[error_mask]
        
        if len(X_test_errors) == 0:
            print("ℹ️ SHAP analizi için yeterli sayıda hatalı örnek bulunamadı.")
            return

        # Analizi 10 örnekle sınırla (CI'da hızlı çalışması için)
        if len(X_test_errors) > 10:
            sample_indices = np.random.choice(X_test_errors.shape[0], 10, replace=False)
            X_test_errors = X_test_errors[sample_indices]

        # --- SHAP ANALİZİ (GradientExplainer ile) ---
        print("SHAP için arka plan (background) veri seti oluşturuluyor (100 örnek)...")
        
        # 1. Veriyi PyTorch tensörüne çevir
        # GradientExplainer, PyTorch tensörleri bekler.
        background_tensor = torch.from_numpy(X_train_scaled).float()
        test_samples_tensor = torch.from_numpy(X_test_errors).float()

        # 100 rastgele örnekle arka planı özetle
        indices = np.random.choice(background_tensor.shape[0], 100, replace=False)
        background_sample_tensor = background_tensor[indices]
        
        print("SHAP GradientExplainer oluşturuluyor (PyTorch için optimize)...")
        # Wrapper'ın içindeki JIT script modele (.model) erişiyoruz
        explainer = shap.GradientExplainer(model_wrapper.model, background_sample_tensor)
        
        print(f"SHAP değerleri {len(test_samples_tensor)} hatalı örnek için hesaplanıyor...")
        # GradientExplainer, JIT modelleri için shap_values döndürür
        shap_values = explainer.shap_values(test_samples_tensor)
        
        # shap_values, (sınıf sayısı, örnek sayısı, özellik sayısı) şeklinde bir listedir.
        # Hata "Gerçek Sınıf -> Tahmin Edilen Sınıf" idi.
        # "Modeli neden 'Tahmin Edilen Sınıf'a iten özellikler nelerdi?" diye soruyoruz.
        shap_values_for_error_class = shap_values[pred_class]
        
        plot_title = f"SHAP (Hata: {class_names.get(true_class)}->{class_names.get(pred_class)}) - Sınıf {pred_class} İçin İtici Güçler"
        shap_path = output_dir / "shap_error_summary_plot.png"
        
        print("SHAP özet grafiği oluşturuluyor...")
        shap.summary_plot(
            shap_values_for_error_class, 
            X_test_errors, 
            feature_names=feature_names,
            show=False,
            max_display=20
        )
        plt.title(plot_title)
        plt.tight_layout()
        plt.savefig(shap_path)
        plt.clf()
        print(f"✅ SHAP özet grafiği kaydedildi: {shap_path}")

    except Exception as e:
        print(f"❌ SHAP analizi sırasında hata oluştu: {e}")
        print("   Grafikler oluşturulamamış olabilir.")

def main():
    """Ana analiz fonksiyonu."""
    print("==================================================")
    print("🔬 MODEL YORUMLANABİLİRLİK VE HATA ANALİZİ BAŞLIYOR")
    print("==================================================")
    
    parser = argparse.ArgumentParser(description="GEMMA Model Explainability Script")
    parser.add_argument('--model-path', required=True, help="Eğitilmiş modelin (.pt) yolu.")
    parser.add_argument('--data-path', required=True, help="Eğitim verisinin (.npz) yolu.")
    parser.add_argument('--metadata-path', required=True, help="Özellik metadata JSON dosyasının yolu.")
    parser.add_argument('--output-dir', default="./analysis_artifacts", help="Çıktı grafiklerinin kaydedileceği dizin.")
    
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ==========================================================
    # ADIM 1: Modeli Yükle
    # ==========================================================
    print(f"\nModel yükleniyor: {args.model_path}")
    model = PyTorchWrapper(args.model_path)
    print("✅ Model başarıyla yüklendi.")

    # ==========================================================
    # ADIM 2: Veriyi, İsimleri ve Maskeyi Yükle
    # ==========================================================
    print(f"Veri yükleniyor: {args.data_path}")
    X_full, y_full, feature_names = load_data_and_features(
        args.data_path, 
        args.metadata_path
    )
    
    mask_path = Path('data/cache/gemma/feature_selection_mask.npy')
    if not mask_path.exists():
        print(f"❌ HATA: Özellik seçim maskesi bulunamadı: {mask_path}")
        print("   Bu betik, 'full-gemma-tuning.yml' tarafından oluşturulan maskeye bağımlıdır.")
        sys.exit(1)
    
    print(f"Özellik seçim maskesi yükleniyor: {mask_path}")
    feature_mask = np.load(mask_path)

    # ==========================================================
    # ADIM 3: Maskeyi Uygula
    # ==========================================================
    # Veri ile maskenin uyumlu olduğunu doğrula
    if X_full.shape[1] != len(feature_mask):
        raise ValueError(f"Ham veri ({X_full.shape[1]}) ve maske ({len(feature_mask)}) boyutu uyuşmuyor!")
    
    # Veriyi maskele
    X_selected = X_full[:, feature_mask]
    print(f"✅ Özellik maskesi veriye uygulandı. {X_full.shape[1]} -> {X_selected.shape[1]} özellik.")

    # Özellik isim listesini maskele
    if len(feature_names) == len(feature_mask):
        feature_names = [name for name, selected in zip(feature_names, feature_mask) if selected]
        print(f"✅ Özellik isimleri maskelendi. Yeni isim sayısı: {len(feature_names)}")
    else:
         print(f"UYARI: Özellik ismi sayısı ({len(feature_names)}) maske ({len(feature_mask)}) ile eşleşmiyor.")
         feature_names = [f"feature_{i}" for i in range(X_selected.shape[1])]
    
    # ==========================================================
    # ADIM 4: Veriyi Böl (Maskelenmiş Veriyi)       
    # ==========================================================
    print(f"Maskelenmiş {X_selected.shape[1]} özellikli veri, train/test olarak bölünüyor (shuffle=False)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected,  # <-- DOĞRU VERİ
        y_full, 
        test_size=0.20, 
        random_state=42, 
        shuffle=False # Zaman serisi için 'False' olmalı
    )
    print(f"   Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # ==========================================================
    # ADIM 5: Veriyi Ölçekle (Scaler ile)
    # ==========================================================
    X_train_scaled, X_test_scaled, scaler = load_and_scale_data(X_train, X_test)
    
    if X_train_scaled is None or X_test_scaled is None:
        print("❌ Ölçekleme hatası nedeniyle analiz durduruluyor.")
        sys.exit(1)

    print("="*50)
    
    # ==========================================================
    # ADIM 6: Analizleri Çalıştır
    # ==========================================================
    
    # Genel Özellik Önemliliğini Çalıştır (ÖLÇEKLENMİŞ VERİ İLE)
    run_permutation_importance(model, X_test_scaled, y_test, feature_names, output_path)

    # Test kümesi üzerinden model tahminlerini al (hata analizi için)
    print("\n🔎 Model test tahminleri hesaplanıyor (hata analizi için)...")
    y_pred = model.predict(X_test_scaled)
    if len(y_pred) != len(y_test):
        print(f"UYARI: y_pred uzunluğu ({len(y_pred)}) != y_test uzunluğu ({len(y_test)})")

    # SHAP ile Hata Analizini Çalıştır (ÖLÇEKLENMİŞ VERİ İLE)
    print("SHAP analizi başlatılıyor...")
    run_shap_analysis(model, X_train_scaled, X_test_scaled, y_test, y_pred, feature_names, output_path)

    print("\n" + "="*50)
    print("✅ Hata analizi tamamlandı.")
    print(f"Raporlar şuraya kaydedildi: {output_path.resolve()}")
    print("="*50)

if __name__ == "__main__":
    main()
