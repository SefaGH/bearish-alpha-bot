#!/usr/bin/env python3
"""
Bu script, bir PyTorch checkpoint (dict) dosyasını ve bir sınıf adını alır.
Checkpoint'ten 'state_size' ve 'action_size' gibi kurucu (constructor) 
argümanlarını tahmin etmeye çalışır.
Başarılı olursa, sınıfı bu argümanlarla "canlandırır", ağırlıkları yükler
ve çalıştırılabilir modeli 'diagnostics/inst_model.pth' olarak kaydeder.
Ayrıca tahmin ettiği 'state_size'ı 'diagnostics/inferred_state_size.txt'ye yazar.
"""
import torch
import json
import os
import importlib
import traceback
import inspect
import sys # sys.exit için eklendi

def main():
    CHECKPOINT_PATH = os.environ.get("MODEL_PATH", "data/models/rl_agent_final.pth")
    MODEL_CLASS_IMPORT = os.environ.get("MODEL_CLASS_IMPORT", "")
    
    # Tüm denemeleri raporlamak için bir çıktı sözlüğü
    report = {
        "path": CHECKPOINT_PATH, 
        "model_class_import": MODEL_CLASS_IMPORT,
        "instantiation_kwargs": None,
        "instantiate_error": None,
        "load_error": None,
        "save_error": None,
        "loaded": False,
        "inst_saved": None
    }

    try:
        # 1. Checkpoint'i (sözlüğü) yükle
        ck = torch.load(CHECKPOINT_PATH, map_location="cpu")
    except Exception as e:
        report["error"] = f"torch.load failed: {e}"
        report["traceback"] = traceback.format_exc()
        save_report(report)
        sys.exit(1)

    if not MODEL_CLASS_IMPORT:
        report["error"] = "No MODEL_CLASS_IMPORT provided"
        save_report(report)
        sys.exit(1)

    try:
        # 2. Sınıfı (mimarîyi) import et
        module_path, cls_name = MODEL_CLASS_IMPORT.rsplit(".", 1)
        mod = importlib.import_module(module_path)
        Klass = getattr(mod, cls_name)
    except Exception as e:
        report["import_error"] = str(e)
        report["traceback"] = traceback.format_exc()
        save_report(report)
        sys.exit(1)

    # 3. 'q_network' state_dict'ini bul (model_load.json'a göre)
    q_sd = None
    for key in ("q_network", "q_network_state_dict", "qnet", "state_dict"):
        if isinstance(ck, dict) and key in ck and isinstance(ck[key], dict):
            q_sd = ck[key]
            break
    
    if q_sd is None:
        report["error"] = "Could not find a 'q_network' or 'state_dict' key in the checkpoint."
        save_report(report)
        sys.exit(1)

    # 4. Argümanları (kwargs) tahmin et
    inferred = {"state_size": None, "action_size": None}
    try:
        weight_names = [n for n in q_sd.keys() if n.endswith(".weight")]
        if weight_names:
            first = q_sd[weight_names[0]]
            last = q_sd[weight_names[-1]]
            if hasattr(first, "size") and first.dim() >= 2:
                inferred["state_size"] = int(first.size()[1]) # Giriş boyutu
            if hasattr(last, "size") and last.dim() >= 2:
                inferred["action_size"] = int(last.size()[0]) # Çıkış boyutu
    except Exception:
        pass # Tahmin başarısız olursa None kalır

    # Fallback (varsayılan) değerler
    if inferred["state_size"] is None:
        inferred["state_size"] = int(ck.get("state_size", 50)) if isinstance(ck, dict) else 50
    if inferred["action_size"] is None:
        inferred["action_size"] = int(ck.get("action_size", 3)) if isinstance(ck, dict) else 3

    # Config'i al
    cfg = ck.get("config") if isinstance(ck, dict) else None
    if not isinstance(cfg, dict):
        cfg = ck.get("training_history") if isinstance(ck, dict) else {}

    # 5. Modeli "canlandırmayı" dene
    inst = None
    kwargs = {"state_size": inferred["state_size"], "action_size": inferred["action_size"], "config": cfg}
    report["instantiation_kwargs"] = kwargs
    
    try:
        inst = Klass(**kwargs)
        
        # --- YENİ EKLENEN ADIM ---
        # Tahmin ettiğimiz state_size'ı diğer script'ler için kaydet
        state_size_to_save = inferred["state_size"]
        os.makedirs("diagnostics", exist_ok=True) # Dizin yoksa oluştur
        with open("diagnostics/inferred_state_size.txt", 'w', encoding='utf-8') as f:
            f.write(str(state_size_to_save))
        print(f"Wrote inferred state_size ({state_size_to_save}) to diagnostics/inferred_state_size.txt")
        # --- YENİ EKLENEN ADIM SONU ---

    except Exception as e:
        report["instantiate_error"] = str(e)
        report["instantiate_traceback"] = traceback.format_exc()
        save_report(report)
        sys.exit(1) # Canlandırma başarısızsa devam etme

    # 6. Ağırlıkları "canlı" modele yükle
    loaded = False
    if inst is not None and q_sd is not None:
        try:
            if hasattr(inst, "q_network"):
                inst.q_network.load_state_dict(q_sd)
                loaded = True
            elif hasattr(inst, "load_state_dict"):
                inst.load_state_dict(q_sd)
                loaded = True
        except Exception as e:
            report["load_error"] = str(e)
            report["load_traceback"] = traceback.format_exc()

    # 7. "Canlı" modeli kaydet
    if loaded and inst is not None:
        try:
            torch.save(inst, "diagnostics/inst_model.pth")
            report["inst_saved"] = "diagnostics/inst_model.pth"
        except Exception as e:
            report["save_error"] = str(e)
            report["save_traceback"] = traceback.format_exc()

    report["loaded"] = loaded
    save_report(report)
    print(f"Instantiation attempt report saved to diagnostics/model_inst_attempt.json")

def save_report(report_data: dict):
    os.makedirs("diagnostics", exist_ok=True)
    with open("diagnostics/model_inst_attempt.json", 'w', encoding='utf-8') as f:
        # Tensorları veya diğer serileştirilemeyen objeleri string'e çevir
        try:
            json.dump(report_data, f, indent=2, default=str)
        except Exception:
            # En kötü durum fallback'i
            fallback_report = {}
            for k, v in report_data.items():
                try:
                    json.dumps(v)
                    fallback_report[k] = v
                except Exception:
                    fallback_report[k] = f"Error: Non-serializable object of type {type(v).__name__}"
            json.dump(fallback_report, f, indent=2)

if __name__ == "__main__":
    main()
