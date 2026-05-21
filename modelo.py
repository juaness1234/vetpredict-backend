"""
modelo.py — Árbol de decisión con guardado/carga de .pkl
"""
import os, csv
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

SINTOMA_ID_MAP = {
    1: "Fever", 2: "Diarrhea", 3: "Lameness",
    4: "Labored Breathing", 5: "Lethargy", 6: "Coughing",
    7: "Vomiting", 8: "Sneezing", 9: "Appetite Loss",
    10: "Skin Lesions", 11: "Nasal Discharge", 12: "Eye Discharge",
}
ALL_SYMPTOM_TEXTS = list(SINTOMA_ID_MAP.values())

NIVEL_RIESGO = {
    "Parvovirus": "critico", "Canine Parvovirus": "critico",
    "Canine Distemper": "critico", "Distemper": "critico",
    "Leptospirosis": "critico", "Canine Leptospirosis": "critico",
    "Lyme Disease": "alto", "Tick-Borne Disease": "alto",
    "Heartworm Disease": "alto", "Canine Heartworm Disease": "alto",
    "Pancreatitis": "alto", "Canine Hepatitis": "alto",
    "Canine Infectious Hepatitis": "alto",
    "Gastroenteritis": "medio", "Kennel Cough": "medio",
    "Canine Cough": "medio", "Chronic Bronchitis": "medio",
    "Bordetella Infection": "medio", "Canine Flu": "medio",
    "Canine Influenza": "medio",
    "Allergic Rhinitis": "bajo", "Arthritis": "bajo",
}

PKL_PATH = os.path.join(os.path.dirname(__file__), "vetpredict_model.pkl")

def _parse_temp(val):
    try:
        return float(str(val).replace("°C","").replace("°F","").strip())
    except:
        return 38.5

class DiseasePredictor:
    def __init__(self, csv_path):
        self.model    = None
        self.classes  = []
        self.features = []
        self.accuracy = 0.0

        if os.path.exists(PKL_PATH):
            self._load_pkl()
        else:
            self._train(csv_path)
            self._save_pkl()

    # ── Serialización ──────────────────────────────────────────
    def _save_pkl(self):
        data = {
            "model":    self.model,
            "classes":  self.classes,
            "features": self.features,
            "accuracy": self.accuracy,
        }
        joblib.dump(data, PKL_PATH, compress=3)
        print(f"[ML] Modelo guardado en {PKL_PATH}")

    def _load_pkl(self):
        data = joblib.load(PKL_PATH)
        self.model    = data["model"]
        self.classes  = data["classes"]
        self.features = data["features"]
        self.accuracy = data["accuracy"]
        print(f"[ML] Modelo cargado desde {PKL_PATH} "
              f"| Accuracy: {self.accuracy:.1%} "
              f"| Clases: {len(self.classes)}")

    # ── Feature builder ────────────────────────────────────────
    def _build_feat(self, record):
        feat = {}
        present = set()
        for col in ["Symptom_1","Symptom_2","Symptom_3","Symptom_4"]:
            v = record.get(col,"").strip()
            if v:
                present.add(v)
        for sym in ALL_SYMPTOM_TEXTS:
            feat[f"sym_{sym.replace(' ','_')}"] = 1.0 if sym in present else 0.0
        try:    feat["Age"]    = float(record.get("Age", 3) or 3)
        except: feat["Age"]   = 3.0
        try:    feat["Weight"] = float(str(record.get("Weight",15)).replace("kg","").strip() or 15)
        except: feat["Weight"] = 15.0
        try:    feat["Heart_Rate"] = float(record.get("Heart_Rate", 90) or 90)
        except: feat["Heart_Rate"] = 90.0
        feat["Body_Temperature"] = _parse_temp(
            record.get("Body_Temperature","38.5") or "38.5")
        return feat

    # ── Entrenamiento ──────────────────────────────────────────
    def _train(self, csv_path):
        rows = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(row)
        if not rows:
            raise ValueError("CSV vacío")

        self.features = list(self._build_feat(rows[0]).keys())
        diseases = [r.get("Disease_Prediction","Unknown").strip() for r in rows]
        self.classes = sorted(set(diseases))
        class_to_idx = {c: i for i, c in enumerate(self.classes)}

        X = np.array([[self._build_feat(r)[f] for f in self.features]
                       for r in rows], dtype=np.float32)
        y = np.array([class_to_idx[d] for d in diseases], dtype=np.int32)

        self.model = DecisionTreeClassifier(
            max_depth=None, min_samples_leaf=1,
            criterion="gini", random_state=42)
        self.model.fit(X, y)
        self.accuracy = float((self.model.predict(X) == y).mean())
        print(f"[ML] Árbol entrenado. Train accuracy: {self.accuracy:.1%} "
              f"| Clases: {len(self.classes)} | Features: {len(self.features)}")

    # ── Predicción ─────────────────────────────────────────────
    def predict(self, sintoma_ids, temperatura=38.5,
                frecuencia_cardiaca=90, peso=15.0, edad=3):
        feat = {f: 0.0 for f in self.features}
        for sid in sintoma_ids:
            nombre = SINTOMA_ID_MAP.get(sid)
            if nombre:
                key = f"sym_{nombre.replace(' ','_')}"
                if key in feat:
                    feat[key] = 1.0
        feat["Age"]              = float(edad)
        feat["Weight"]           = float(peso)
        feat["Heart_Rate"]       = float(frecuencia_cardiaca)
        feat["Body_Temperature"] = float(temperatura)

        X = np.array([[feat[f] for f in self.features]], dtype=np.float32)
        pred_idx   = int(self.model.predict(X)[0])
        proba      = self.model.predict_proba(X)[0]
        disease    = self.classes[pred_idx]
        confidence = float(proba[pred_idx]) * 100.0

        top5 = np.argsort(proba)[::-1][:5]
        probabilidades = {
            self.classes[int(i)]: round(float(proba[i]), 4)
            for i in top5 if proba[i] > 0.01
        }
        return {
            "enfermedad":      disease,
            "nivel_riesgo":    NIVEL_RIESGO.get(disease, "medio"),
            "confianza":       round(confidence, 2),
            "probabilidades":  probabilidades,
            "sintomas_usados": [SINTOMA_ID_MAP[s] for s in sintoma_ids
                                 if s in SINTOMA_ID_MAP],
            "accuracy_modelo": round(self.accuracy * 100, 1),
        }

# ── Instancia global ───────────────────────────────────────────
_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        csv_path = os.path.join(os.path.dirname(__file__), "dogs_only2.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Copia dogs_only2.csv a la carpeta backend/. No encontrado: {csv_path}")
        _predictor = DiseasePredictor(csv_path)
    return _predictor
