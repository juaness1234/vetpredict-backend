"""
modelo_v5.py — Árbol de decisión REAL con preguntas secuenciales
Como el ejemplo de la imagen: pregunta → sí/no → siguiente pregunta → diagnóstico
+ Recomendaciones médicas y caseras por enfermedad
Compatible con Python 3.13, sin pandas
"""
import os, csv, json
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder
import joblib

# ── Mapa de síntomas ID → nombre ──────────────────────────────
SINTOMA_ID_MAP = {
    1:  "Fever",
    2:  "Coughing",
    3:  "Lethargy",
    4:  "Vomiting",
    5:  "Diarrhea",
    6:  "Lameness",
    7:  "Labored Breathing",
    8:  "Sneezing",
    9:  "Appetite Loss",
    10: "Skin Lesions",
    11: "Nasal Discharge",
    12: "Eye Discharge",
}

# Todos los síntomas posibles en el dataset
ALL_SYMPTOMS = list(SINTOMA_ID_MAP.values())

# ── Niveles de riesgo ──────────────────────────────────────────
NIVEL_RIESGO = {
    "Canine Parvovirus":           "critico",
    "Parvovirus":                  "critico",
    "Canine Distemper":            "critico",
    "Distemper":                   "critico",
    "Canine Leptospirosis":        "critico",
    "Leptospirosis":               "critico",
    "Lyme Disease":                "alto",
    "Tick-Borne Disease":          "alto",
    "Canine Heartworm Disease":    "alto",
    "Heartworm Disease":           "alto",
    "Pancreatitis":                "alto",
    "Canine Hepatitis":            "alto",
    "Canine Infectious Hepatitis": "alto",
    "Gastroenteritis":             "medio",
    "Kennel Cough":                "medio",
    "Canine Cough":                "medio",
    "Chronic Bronchitis":          "medio",
    "Bordetella Infection":        "medio",
    "Canine Flu":                  "medio",
    "Canine Influenza":            "medio",
    "Allergic Rhinitis":           "bajo",
    "Arthritis":                   "bajo",
}

# ── Recomendaciones médicas por enfermedad ────────────────────
RECOMENDACIONES = {
    "Canine Parvovirus": {
        "urgencia": "🚨 URGENTE — Lleva a tu perro al veterinario INMEDIATAMENTE",
        "descripcion": "El Parvovirus canino es una enfermedad viral altamente contagiosa y mortal si no se trata a tiempo.",
        "que_hacer": [
            "Acude a urgencias veterinarias de inmediato",
            "Aísla al perro de otros animales",
            "No le des comida ni agua hasta que el veterinario lo indique",
            "El tratamiento incluye sueros intravenosos y antibióticos"
        ],
        "casero": [
            "Mantén al perro hidratado con pequeños sorbos de agua si no vomita",
            "Ambiente tranquilo, cálido y sin estrés",
            "NO administres medicamentos sin indicación veterinaria"
        ],
        "medicamentos_comunes": "Suero IV, antieméticos (metoclopramida), antibióticos (ampicilina). Solo bajo prescripción veterinaria.",
        "prevencion": "Vacuna DHPP al día. Desinfectar con lejía diluida (1:30)."
    },
    "Parvovirus": {
        "urgencia": "🚨 URGENTE — Lleva a tu perro al veterinario INMEDIATAMENTE",
        "descripcion": "El Parvovirus canino es una enfermedad viral altamente contagiosa y mortal si no se trata a tiempo.",
        "que_hacer": [
            "Acude a urgencias veterinarias de inmediato",
            "Aísla al perro de otros animales",
            "No le des comida ni agua hasta que el veterinario lo indique",
            "El tratamiento incluye sueros intravenosos y antibióticos"
        ],
        "casero": [
            "Mantén al perro hidratado con pequeños sorbos de agua si no vomita",
            "Ambiente tranquilo, cálido y sin estrés",
            "NO administres medicamentos sin indicación veterinaria"
        ],
        "medicamentos_comunes": "Suero IV, antieméticos, antibióticos. Solo bajo prescripción veterinaria.",
        "prevencion": "Vacuna DHPP al día. Desinfectar superficies con lejía diluida."
    },
    "Canine Distemper": {
        "urgencia": "🚨 URGENTE — Consulta veterinaria inmediata",
        "descripcion": "El moquillo canino es una enfermedad viral que afecta el sistema nervioso, respiratorio y digestivo.",
        "que_hacer": [
            "Consulta veterinaria urgente — no hay cura, solo tratamiento de soporte",
            "Aísla al perro de otros animales",
            "Hospitalización probablemente necesaria",
            "Tratamiento de convulsiones si las hay"
        ],
        "casero": [
            "Mantén los ojos y nariz limpios con gasa húmeda",
            "Alimentación blanda si come",
            "Ambiente cálido y libre de corrientes de aire"
        ],
        "medicamentos_comunes": "Anticonvulsivos, antibióticos secundarios, vitamina C. Solo bajo prescripción.",
        "prevencion": "Vacuna DHPP obligatoria y refuerzos anuales."
    },
    "Distemper": {
        "urgencia": "🚨 URGENTE — Consulta veterinaria inmediata",
        "descripcion": "El moquillo canino afecta múltiples sistemas del organismo del perro.",
        "que_hacer": [
            "Consulta veterinaria urgente",
            "Aísla al perro",
            "Tratamiento de soporte hospitalario"
        ],
        "casero": [
            "Limpieza de secreciones nasales y oculares",
            "Dieta blanda y agua fresca disponible",
            "Reposo absoluto"
        ],
        "medicamentos_comunes": "Tratamiento de soporte únicamente. Bajo prescripción veterinaria.",
        "prevencion": "Vacunación DHPP anual."
    },
    "Canine Leptospirosis": {
        "urgencia": "🚨 URGENTE — Zoonosis: peligroso también para humanos",
        "descripcion": "Infección bacteriana que afecta riñones e hígado. TRANSMISIBLE A HUMANOS. Usa guantes al manipular al perro.",
        "que_hacer": [
            "Veterinario urgente con antibióticos intravenosos",
            "Usa guantes y mascarilla al manipular al perro",
            "Lava tus manos frecuentemente",
            "Limpia el área de descanso con desinfectante"
        ],
        "casero": [
            "Hidratación constante",
            "Evita el contacto directo con orina del perro",
            "Aísla al perro de niños y personas mayores"
        ],
        "medicamentos_comunes": "Penicilina o doxiciclina IV. Solo bajo prescripción veterinaria.",
        "prevencion": "Vacuna Lepto 4. Evita charcos y agua estancada."
    },
    "Leptospirosis": {
        "urgencia": "🚨 URGENTE — Zoonosis: peligroso también para humanos",
        "descripcion": "Infección bacteriana transmisible a humanos. Afecta riñones e hígado.",
        "que_hacer": [
            "Consulta veterinaria inmediata",
            "Usa guantes al manipular al perro",
            "Tratamiento con antibióticos"
        ],
        "casero": [
            "Hidratación constante",
            "Higiene estricta del entorno",
            "Aislamiento del animal"
        ],
        "medicamentos_comunes": "Doxiciclina o amoxicilina. Solo bajo prescripción.",
        "prevencion": "Vacunación anual contra Leptospira."
    },
    "Kennel Cough": {
        "urgencia": "⚠️ Consulta veterinaria en los próximos 1-2 días",
        "descripcion": "Enfermedad respiratoria contagiosa causada por Bordetella y virus. Tos seca y persistente característica.",
        "que_hacer": [
            "Consulta veterinaria para antibióticos y/o antitusivos",
            "Aísla al perro de otros perros",
            "Evita paseos en zonas con mucho contacto animal"
        ],
        "casero": [
            "Miel de abeja pura: 1 cucharadita 3 veces al día (calmante natural de la tos)",
            "Vapor de eucalipto en el ambiente (no directo al perro)",
            "Humidificador en la habitación",
            "Agua tibia disponible siempre",
            "Evita collares ajustados — usa arnés"
        ],
        "medicamentos_comunes": "Doxiciclina, amoxicilina, butorfanol (antitusivo). Bajo prescripción.",
        "prevencion": "Vacuna Bordetella intranasal antes de estancias en perreras."
    },
    "Canine Cough": {
        "urgencia": "⚠️ Consulta veterinaria recomendada",
        "descripcion": "Tos canina infecciosa con síntomas respiratorios.",
        "que_hacer": [
            "Consulta veterinaria para diagnóstico preciso",
            "Reposo y evitar ejercicio intenso"
        ],
        "casero": [
            "Miel de abeja: 1 cucharadita 2-3 veces al día",
            "Agua tibia y ambiente húmedo",
            "Usa arnés en lugar de collar"
        ],
        "medicamentos_comunes": "Antitusivos y antibióticos si hay infección bacteriana. Bajo prescripción.",
        "prevencion": "Vacunación Bordetella y evitar contacto con perros enfermos."
    },
    "Gastroenteritis": {
        "urgencia": "⚠️ Si dura más de 24h o hay sangre, ve al veterinario",
        "descripcion": "Inflamación del tracto digestivo con vómito y diarrea. Generalmente se resuelve en 2-3 días.",
        "que_hacer": [
            "Ayuno de 12-24 horas para el estómago descanse",
            "Luego dieta blanda: arroz blanco + pollo hervido sin sal",
            "Si hay sangre en heces o vómito, ve al veterinario urgente",
            "Si el perro está muy débil o deshidratado, veterinario inmediato"
        ],
        "casero": [
            "Agua con un poco de sal y azúcar (suero oral casero)",
            "Arroz blanco hervido sin sal como primera comida",
            "Pollo hervido sin piel ni huesos",
            "Calabaza cocida sin condimentos (protege la mucosa)",
            "Probióticos naturales: yogur natural sin azúcar (1 cucharada)"
        ],
        "medicamentos_comunes": "Metronidazol, sucralfato, probióticos veterinarios. Bajo prescripción.",
        "prevencion": "No cambios bruscos de alimento. Evitar basura y alimentos del piso."
    },
    "Canine Hepatitis": {
        "urgencia": "🚨 URGENTE — Consulta veterinaria inmediata",
        "descripcion": "Hepatitis infecciosa canina causada por adenovirus. Afecta hígado y riñones.",
        "que_hacer": [
            "Veterinario inmediato para soporte hepático",
            "Hospitalización puede ser necesaria",
            "Análisis de sangre para evaluar función hepática"
        ],
        "casero": [
            "Dieta baja en grasas y proteínas de fácil digestión",
            "Agua fresca siempre disponible",
            "Reposo absoluto"
        ],
        "medicamentos_comunes": "Hepatoprotectores (silimarina), suero IV, antibióticos. Bajo prescripción.",
        "prevencion": "Vacuna DHPP protege contra adenovirus canino."
    },
    "Canine Infectious Hepatitis": {
        "urgencia": "🚨 URGENTE — Consulta veterinaria inmediata",
        "descripcion": "Hepatitis infecciosa canina aguda. Puede ser fatal sin tratamiento.",
        "que_hacer": [
            "Veterinario urgente",
            "Aísla al perro",
            "Soporte hospitalario necesario"
        ],
        "casero": [
            "Dieta blanda y baja en grasas",
            "Hidratación constante",
            "Reposo total"
        ],
        "medicamentos_comunes": "Tratamiento de soporte. Solo bajo prescripción veterinaria.",
        "prevencion": "Vacunación DHPP anual."
    },
    "Pancreatitis": {
        "urgencia": "🚨 Consulta veterinaria urgente — puede ser grave",
        "descripcion": "Inflamación del páncreas, frecuente tras consumo de alimentos grasos.",
        "que_hacer": [
            "Ayuno completo de 24-48h (agua únicamente)",
            "Veterinario para confirmar diagnóstico con ecografía",
            "Hospitalización si hay vómitos incontrolables"
        ],
        "casero": [
            "Ayuno estricto las primeras 24 horas",
            "Solo agua en pequeñas cantidades",
            "Luego arroz blanco y pollo hervido sin grasa",
            "Nunca alimentos grasos, fritos o condimentados"
        ],
        "medicamentos_comunes": "Analgésicos, antieméticos, fluidoterapia IV. Solo bajo prescripción.",
        "prevencion": "Dieta equilibrada. Evitar alimentos grasos y sobras de comida humana."
    },
    "Lyme Disease": {
        "urgencia": "⚠️ Consulta veterinaria en 24-48 horas",
        "descripcion": "Enfermedad bacteriana transmitida por garrapatas. Causa cojera y fiebre.",
        "que_hacer": [
            "Revisar el cuerpo del perro en busca de garrapatas",
            "Remover garrapatas con pinzas especiales (no retorcer)",
            "Consulta veterinaria para antibióticos",
            "Análisis de sangre para confirmar"
        ],
        "casero": [
            "Extracción cuidadosa de garrapatas con pinzas",
            "Desinfectar la zona de extracción con yodo o alcohol",
            "Compresas frías en las articulaciones inflamadas",
            "Reposo y limitar el ejercicio"
        ],
        "medicamentos_comunes": "Doxiciclina durante 4 semanas. Solo bajo prescripción.",
        "prevencion": "Antiparasitarios externos (pipetas, collares). Revisar el pelo tras paseos."
    },
    "Tick-Borne Disease": {
        "urgencia": "⚠️ Consulta veterinaria en 24-48 horas",
        "descripcion": "Enfermedad transmitida por garrapatas. Puede incluir varias patologías (ehrlichiosis, anaplasmosis).",
        "que_hacer": [
            "Revisión y extracción de garrapatas",
            "Consulta veterinaria para análisis de sangre",
            "Antibióticos según diagnóstico"
        ],
        "casero": [
            "Revisión diaria del pelaje",
            "Baño con champú antiparasitario",
            "Compresas frías para el dolor articular"
        ],
        "medicamentos_comunes": "Doxiciclina. Solo bajo prescripción veterinaria.",
        "prevencion": "Antiparasitarios mensuales y revisión del pelaje tras salidas."
    },
    "Canine Heartworm Disease": {
        "urgencia": "🚨 Consulta veterinaria urgente",
        "descripcion": "Infestación por gusanos del corazón (Dirofilaria). Grave y potencialmente fatal.",
        "que_hacer": [
            "Veterinario urgente para análisis de sangre",
            "Tratamiento largo y costoso — detección temprana es clave",
            "Reposo absoluto durante el tratamiento"
        ],
        "casero": [
            "Reposo estricto — evitar todo ejercicio",
            "Dieta nutritiva de alta calidad",
            "Monitoreo constante de respiración"
        ],
        "medicamentos_comunes": "Melarsomin (adulticida), ivermectina preventiva. Solo bajo prescripción.",
        "prevencion": "Preventivo mensual contra heartworm (ivermectina oral). Prueba anual de sangre."
    },
    "Heartworm Disease": {
        "urgencia": "🚨 Consulta veterinaria urgente",
        "descripcion": "Gusanos del corazón. Requiere tratamiento especializado.",
        "que_hacer": [
            "Diagnóstico veterinario con antígenos en sangre",
            "Tratamiento con adulticidas",
            "Reposo total durante el tratamiento"
        ],
        "casero": [
            "Reposo absoluto",
            "Alimentación de calidad",
            "Evitar estrés y ejercicio"
        ],
        "medicamentos_comunes": "Melarsomin. Solo bajo prescripción veterinaria.",
        "prevencion": "Preventivo mensual de heartworm."
    },
    "Bordetella Infection": {
        "urgencia": "⚠️ Consulta veterinaria en 1-2 días",
        "descripcion": "Infección respiratoria por Bordetella bronchiseptica. Causa tos seca intensa.",
        "que_hacer": [
            "Consulta veterinaria para antibióticos",
            "Aísla del resto de perros",
            "Evita humedad y frío"
        ],
        "casero": [
            "Miel pura: 1 cucharadita 3 veces al día",
            "Vapor de menta o eucalipto en el ambiente",
            "Agua tibia siempre disponible",
            "Reposo y ambiente cálido"
        ],
        "medicamentos_comunes": "Doxiciclina, amoxicilina. Bajo prescripción veterinaria.",
        "prevencion": "Vacuna intranasal Bordetella antes de contacto con otros perros."
    },
    "Canine Flu": {
        "urgencia": "⚠️ Consulta veterinaria recomendada",
        "descripcion": "Gripe canina causada por influenza H3N2 o H3N8. Altamente contagiosa entre perros.",
        "que_hacer": [
            "Aísla al perro inmediatamente",
            "Consulta veterinaria para tratamiento de soporte",
            "Reposo y buena hidratación"
        ],
        "casero": [
            "Caldo de pollo sin sal (hidratante y nutritivo)",
            "Miel de abeja para la tos",
            "Vapor de eucalipto en el ambiente",
            "Pañitos húmedos tibios para bajar fiebre leve"
        ],
        "medicamentos_comunes": "Antivirales específicos, antibióticos si hay complicaciones. Bajo prescripción.",
        "prevencion": "Vacuna bivalente contra gripe canina. Evitar perreras en épocas de brote."
    },
    "Canine Influenza": {
        "urgencia": "⚠️ Consulta veterinaria recomendada",
        "descripcion": "Influenza canina con síntomas respiratorios.",
        "que_hacer": [
            "Aislamiento del perro",
            "Consulta veterinaria",
            "Reposo y hidratación"
        ],
        "casero": [
            "Caldo de pollo sin sal",
            "Miel para la tos",
            "Ambiente cálido y sin corrientes de aire"
        ],
        "medicamentos_comunes": "Tratamiento de soporte. Antibióticos si hay infección secundaria.",
        "prevencion": "Vacunación anual contra influenza canina."
    },
    "Chronic Bronchitis": {
        "urgencia": "⚠️ Consulta veterinaria para manejo a largo plazo",
        "descripcion": "Inflamación crónica de los bronquios. Requiere manejo continuo.",
        "que_hacer": [
            "Diagnóstico veterinario con radiografías",
            "Tratamiento con broncodilatadores y corticoides",
            "Evitar humo, polvo y contaminantes"
        ],
        "casero": [
            "Humidificador en el ambiente",
            "Evitar paseos en días de mucho frío o contaminación",
            "Dieta antiinflamatoria: aceite de salmón (omega-3)",
            "Peso ideal — el sobrepeso empeora la condición"
        ],
        "medicamentos_comunes": "Broncodilatadores (teofilina), corticoides inhalados. Bajo prescripción.",
        "prevencion": "Evitar exposición a humo y alérgenos. Control de peso."
    },
    "Allergic Rhinitis": {
        "urgencia": "ℹ️ Consulta veterinaria de rutina",
        "descripcion": "Rinitis alérgica — reacción del sistema inmune a alérgenos ambientales.",
        "que_hacer": [
            "Identificar y eliminar el alérgeno (polvo, polen, ciertos alimentos)",
            "Consulta veterinaria para antihistamínicos",
            "Pruebas de alergia si los síntomas persisten"
        ],
        "casero": [
            "Limpieza nasal con suero fisiológico (0.9% NaCl)",
            "Baños frecuentes para eliminar alérgenos del pelaje",
            "Aceite de coco aplicado en hocico para aliviar irritación",
            "Miel local (reduce sensibilización a pólenes locales)",
            "Evitar salidas en épocas de alta polinización"
        ],
        "medicamentos_comunes": "Cetirizina, loratadina, corticoides tópicos. Bajo indicación veterinaria.",
        "prevencion": "Control del ambiente, HEPA en casa, dieta hipoalergénica si se sospecha alergia alimentaria."
    },
    "Arthritis": {
        "urgencia": "ℹ️ Consulta veterinaria para manejo del dolor",
        "descripcion": "Inflamación crónica de las articulaciones. Más frecuente en perros mayores o de razas grandes.",
        "que_hacer": [
            "Consulta veterinaria para analgésicos y antiinflamatorios",
            "Radiografías para evaluar el grado de afectación",
            "Fisioterapia veterinaria si está disponible"
        ],
        "casero": [
            "Cama ortopédica cálida y cómoda",
            "Compresas tibias en articulaciones afectadas (15 min, 2 veces/día)",
            "Aceite de salmón: 1 cucharadita/día (antiinflamatorio natural)",
            "Cúrcuma con pimienta negra en la comida (dosis pequeña)",
            "Ejercicio moderado y suave — evitar saltos",
            "Rampas en lugar de escaleras",
            "Control de peso — el sobrepeso aumenta el dolor"
        ],
        "medicamentos_comunes": "Meloxicam, carprofeno, glucosamina + condroitina. Bajo prescripción.",
        "prevencion": "Control de peso, ejercicio regular moderado, suplementos articulares preventivos."
    },
}

# Recomendación genérica para enfermedades sin recomendación específica
RECOMENDACION_GENERICA = {
    "urgencia": "⚠️ Consulta veterinaria recomendada",
    "descripcion": "Se ha detectado una posible enfermedad. Consulta con un veterinario para confirmar el diagnóstico.",
    "que_hacer": [
        "Consulta a un veterinario lo antes posible",
        "Anota todos los síntomas observados y hace cuánto aparecieron",
        "Lleva el historial de vacunación del perro",
        "No automedicar sin indicación veterinaria"
    ],
    "casero": [
        "Mantén al perro hidratado y en reposo",
        "Dieta blanda si hay problemas digestivos",
        "Ambiente tranquilo y cálido"
    ],
    "medicamentos_comunes": "Consulta veterinaria necesaria para prescripción adecuada.",
    "prevencion": "Vacunación al día, desparasitación regular y revisiones anuales."
}


# ── PKL path ──────────────────────────────────────────────────
PKL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "vetpredict_model_v5.pkl")


def _parse_temp(val):
    try:
        return float(str(val).replace("°C", "").replace("°F", "").strip())
    except Exception:
        return 38.5


class DiseasePredictor:
    """
    Árbol de decisión que funciona con preguntas secuenciales.
    El usuario responde SÍ/NO a cada síntoma — exactamente
    como el ejemplo del diagrama.
    """

    def __init__(self, csv_path: str):
        self.model    = None
        self.classes  = []
        self.features = []
        self.accuracy = 0.0

        if os.path.exists(PKL_PATH):
            self._load()
        else:
            self._train(csv_path)
            self._save()

    # ── Persistencia ──────────────────────────────────────────
    def _save(self):
        joblib.dump({
            "model":    self.model,
            "classes":  self.classes,
            "features": self.features,
            "accuracy": self.accuracy,
        }, PKL_PATH, compress=3)
        print(f"[ML v5] Modelo guardado en {PKL_PATH}")

    def _load(self):
        data = joblib.load(PKL_PATH)
        self.model    = data["model"]
        self.classes  = data["classes"]
        self.features = data["features"]
        self.accuracy = data["accuracy"]
        print(f"[ML v5] Modelo cargado | Accuracy: {self.accuracy:.1%}"
              f" | Clases: {len(self.classes)}")

    # ── Feature builder ───────────────────────────────────────
    def _build_feat(self, record: dict) -> dict:
        """
        Convierte un registro del CSV en vector de features.
        Cada síntoma es una columna binaria (1=presente, 0=ausente).
        """
        feat = {}

        # Síntomas de Symptom_1..4 → one-hot
        present = set()
        for col in ["Symptom_1", "Symptom_2", "Symptom_3", "Symptom_4"]:
            v = record.get(col, "").strip()
            # Normalizar variantes ("Loss of Appetite" → "Appetite Loss")
            if v in ("Loss of Appetite",): v = "Appetite Loss"
            if v: present.add(v)

        for sym in ALL_SYMPTOMS:
            feat[sym] = 1.0 if sym in present else 0.0

        # Vitales
        try:    feat["Age"]    = float(record.get("Age", 3) or 3)
        except: feat["Age"]   = 3.0
        try:    feat["Weight"] = float(str(record.get("Weight", 15))
                                       .replace("kg", "").strip() or 15)
        except: feat["Weight"] = 15.0
        try:    feat["Heart_Rate"] = float(record.get("Heart_Rate", 90) or 90)
        except: feat["Heart_Rate"] = 90.0
        feat["Body_Temperature"] = _parse_temp(
            record.get("Body_Temperature", "38.5") or "38.5")

        return feat

    # ── Entrenamiento ─────────────────────────────────────────
    def _train(self, csv_path: str):
        rows = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(row)
        if not rows:
            raise ValueError("CSV vacío")

        self.features = list(self._build_feat(rows[0]).keys())
        diseases      = [r.get("Disease_Prediction", "Unknown").strip()
                         for r in rows]
        self.classes  = sorted(set(diseases))
        c2i           = {c: i for i, c in enumerate(self.classes)}

        X = np.array(
            [[self._build_feat(r)[f] for f in self.features] for r in rows],
            dtype=np.float32)
        y = np.array([c2i[d] for d in diseases], dtype=np.int32)

        # Árbol sin límite de profundidad para máxima precisión
        self.model = DecisionTreeClassifier(
            max_depth=None,
            min_samples_leaf=1,
            criterion="gini",
            random_state=42,
        )
        self.model.fit(X, y)
        self.accuracy = float((self.model.predict(X) == y).mean())
        print(f"[ML v5] Árbol entrenado | Accuracy: {self.accuracy:.1%}"
              f" | Clases: {len(self.classes)}"
              f" | Features: {len(self.features)}")

    # ── Predicción desde IDs de síntomas ─────────────────────
    def predict(self,
                sintoma_ids: list,
                temperatura: float = 38.5,
                frecuencia_cardiaca: int = 90,
                peso: float = 15.0,
                edad: int = 3) -> dict:
        """
        Recibe solo los síntomas que el usuario marcó como SÍ.
        El árbol evalúa presencia/ausencia de cada síntoma.
        """
        feat = {f: 0.0 for f in self.features}

        # Mapear IDs → nombre de síntoma
        for sid in sintoma_ids:
            nombre = SINTOMA_ID_MAP.get(sid)
            if nombre and nombre in feat:
                feat[nombre] = 1.0

        feat["Age"]              = float(edad)
        feat["Weight"]           = float(peso)
        feat["Heart_Rate"]       = float(frecuencia_cardiaca)
        feat["Body_Temperature"] = float(temperatura)

        X        = np.array([[feat[f] for f in self.features]],
                            dtype=np.float32)
        pred_idx = int(self.model.predict(X)[0])
        proba    = self.model.predict_proba(X)[0]

        disease    = self.classes[pred_idx]
        confidence = float(proba[pred_idx]) * 100.0

        # Top 5 probabilidades
        top5 = np.argsort(proba)[::-1][:5]
        probabilidades = {
            self.classes[int(i)]: round(float(proba[i]), 4)
            for i in top5 if proba[i] > 0.01
        }

        # Recomendación
        rec = RECOMENDACIONES.get(disease, RECOMENDACION_GENERICA)

        return {
            "enfermedad":       disease,
            "nivel_riesgo":     NIVEL_RIESGO.get(disease, "medio"),
            "confianza":        round(confidence, 2),
            "probabilidades":   probabilidades,
            "sintomas_usados":  [SINTOMA_ID_MAP[s] for s in sintoma_ids
                                 if s in SINTOMA_ID_MAP],
            "accuracy_modelo":  round(self.accuracy * 100, 1),
            "recomendacion":    rec,
        }

    # ── Flujo secuencial (árbol real pregunta por pregunta) ───
    def get_next_question(self, respuestas: dict) -> dict:
        """
        Simula el flujo del árbol de decisión con preguntas
        secuenciales como en el diagrama del ejemplo.

        respuestas: {"Fever": True, "Coughing": False, ...}

        Retorna:
          - Si hay más preguntas: {"tipo": "pregunta", "sintoma": "...", "pregunta": "..."}
          - Si el árbol ya puede decidir: {"tipo": "diagnostico", ...}
        """
        # Orden de preguntas según importancia clínica
        ORDEN_PREGUNTAS = [
            ("Fever",            "¿Tu perro tiene fiebre (temperatura > 39.2°C)?"),
            ("Vomiting",         "¿Tu perro presenta vómitos?"),
            ("Diarrhea",         "¿Tu perro tiene diarrea?"),
            ("Lethargy",         "¿Tu perro está letárgico o sin energía?"),
            ("Coughing",         "¿Tu perro tose frecuentemente?"),
            ("Labored Breathing","¿Tu perro tiene dificultad para respirar?"),
            ("Lameness",         "¿Tu perro presenta cojera o dificultad para caminar?"),
            ("Appetite Loss",    "¿Tu perro ha perdido el apetito?"),
            ("Sneezing",         "¿Tu perro estornuda frecuentemente?"),
            ("Nasal Discharge",  "¿Tu perro tiene secreción nasal?"),
            ("Eye Discharge",    "¿Tu perro tiene secreción en los ojos?"),
            ("Skin Lesions",     "¿Tu perro tiene lesiones, ronchas o costras en la piel?"),
        ]

        # Encontrar el siguiente síntoma no respondido
        for sintoma, pregunta in ORDEN_PREGUNTAS:
            if sintoma not in respuestas:
                return {
                    "tipo":     "pregunta",
                    "sintoma":  sintoma,
                    "pregunta": pregunta,
                    "respondidas": len(respuestas),
                    "total":    len(ORDEN_PREGUNTAS),
                }

   # Todas respondidas — hacer predicción
        sintoma_ids = [
            sid for sid, nombre in SINTOMA_ID_MAP.items()
            if respuestas.get(nombre, False)
        ]

        if len(sintoma_ids) == 0:
            return {
                "tipo": "diagnostico",
                "enfermedad": "Sin síntomas detectados",
                "nivel_riesgo": "bajo",
                "confianza": 0.0,
                "probabilidades": {},
                "sintomas_usados": [],
                "recomendacion": {
                    "urgencia": "✅ Tu perro parece estar sano",
                    "descripcion": "No se detectaron síntomas relevantes. Consulta a un veterinario de rutina si tienes dudas.",
                    "que_hacer": [
                        "Continúa con las vacunas y desparasitación al día",
                        "Revisión veterinaria anual de rutina",
                        "Mantén una dieta equilibrada y ejercicio regular"
                    ],
                    "casero": [
                        "Agua fresca siempre disponible",
                        "Ejercicio diario según la raza",
                        "Revisión de pelaje y orejas semanalmente"
                    ],
                    "medicamentos_comunes": "No se requiere medicación.",
                    "prevencion": "Vacunación y desparasitación regular."
                }
            }

        resultado = self.predict(sintoma_ids)
        resultado["tipo"] = "diagnostico"
        return resultado

# ── Instancia global ──────────────────────────────────────────
_predictor = None


def get_predictor(csv_path: str = None) -> DiseasePredictor:
    global _predictor
    if _predictor is None:
        if csv_path is None:
            csv_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "dogs_only2.csv"
            )
        if not os.path.exists(PKL_PATH) and not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Copia dogs_only2.csv a la carpeta backend/. "
                f"No encontrado: {csv_path}"
            )
        _predictor = DiseasePredictor(csv_path)
    return _predictor
