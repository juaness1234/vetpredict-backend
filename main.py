"""
main.py — VetPredict API v5
Árbol de decisión secuencial + recomendaciones médicas
"""
import os, json, logging, hashlib, datetime
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, field_validator, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv

from database import query_one, query_all, execute
from auth import hash_password, verify_password, create_token, get_current_user
from modelo_v5 import get_predictor

load_dotenv()

# ── Logging ───────────────────────────────────────────────────
log_file = os.getenv("LOG_FILE", "logs/vetpredict_security.log")
Path(log_file).parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("vetpredict")

limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])

app = FastAPI(
    title="VetPredict API",
    version="5.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)
    
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

_origins = [o.strip() for o in os.getenv(
    "ALLOWED_ORIGINS",
    "http://10.0.2.2:8000,http://localhost:8000,http://127.0.0.1:8000"
).split(",")]
app.add_middleware(CORSMiddleware, allow_origins=_origins,
    allow_credentials=True, allow_methods=["GET","POST","PUT","DELETE"],
    allow_headers=["Authorization","Content-Type"])

@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"]        = "DENY"
    response.headers["X-XSS-Protection"]       = "1; mode=block"
    response.headers["Cache-Control"]          = "no-store"
    return response

@app.middleware("http")
async def log_requests(request: Request, call_next):
    response = await call_next(request)
    ip = get_remote_address(request)
    if response.status_code >= 400:
        logger.warning(f"[{response.status_code}] {request.method} {request.url.path} IP:{ip}")
    return response

_bearer = HTTPBearer()

def _token_hash(token): 
    import hashlib
    return hashlib.sha256(token.encode()).hexdigest()

def _revoke_token(token, user_id, motivo="logout"):
    expira = datetime.datetime.utcnow() + datetime.timedelta(hours=2)
    try:
        execute("INSERT IGNORE INTO token_blacklist (token_hash,usuario_id,motivo,expira_en) VALUES (%s,%s,%s,%s)",
            (_token_hash(token), user_id, motivo, expira))
    except: pass

def _is_revoked(token):
    try:
        row = query_one("SELECT id FROM token_blacklist WHERE token_hash=%s AND expira_en > NOW()",
            (_token_hash(token),))
        return row is not None
    except: return False

def get_secure_user(creds: HTTPAuthorizationCredentials = Depends(_bearer)):
    if _is_revoked(creds.credentials):
        raise HTTPException(401, "Token revocado. Inicia sesión nuevamente.")
    return get_current_user(creds)

def serial(row):
    for k, v in row.items():
        if isinstance(v, (datetime.datetime, datetime.date)):
            row[k] = v.isoformat()
    return row

def get_ip(r):
    fwd = r.headers.get("X-Forwarded-For")
    return fwd.split(",")[0].strip() if fwd else (r.client.host if r.client else "unknown")

# ── Schemas ───────────────────────────────────────────────────
class RegisterIn(BaseModel):
    nombre:   str = Field(..., min_length=2, max_length=100)
    apellido: str = Field(..., min_length=2, max_length=100)
    email:    EmailStr
    password: str = Field(..., min_length=8, max_length=100)
    @field_validator("password")
    @classmethod
    def pwd_strength(cls, v):
        if not any(c.isupper() for c in v): raise ValueError("Necesita una mayúscula")
        if not any(c.isdigit() for c in v): raise ValueError("Necesita un número")
        return v

class LoginIn(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1, max_length=100)

class MascotaIn(BaseModel):
    nombre: str   = Field(..., min_length=1, max_length=100)
    raza:   Optional[str]   = Field(None, max_length=100)
    edad:   Optional[int]   = Field(None, ge=0, le=30)
    genero: Optional[str]   = Field(None)
    peso:   Optional[float] = Field(None, ge=0.1, le=200)

class DiagnosticoIn(BaseModel):
    mascota_id:          int       = Field(..., ge=1)
    sintoma_ids:         list[int] = Field(..., min_length=1, max_length=12)
    temperatura:         Optional[float] = Field(None, ge=35.0, le=43.0)
    frecuencia_cardiaca: Optional[int]   = Field(None, ge=20, le=300)
    peso_momento:        Optional[float] = Field(None, ge=0.1, le=200)
    duracion_sintomas:   Optional[str]   = Field(None, max_length=100)

class RespuestasIn(BaseModel):
    """Flujo secuencial: envía respuestas acumuladas y recibe siguiente pregunta o diagnóstico."""
    mascota_id:          int  = Field(..., ge=1)
    respuestas:          dict = Field(default={})
    temperatura:         Optional[float] = Field(None, ge=35.0, le=43.0)
    frecuencia_cardiaca: Optional[int]   = Field(None, ge=20, le=300)
    peso_momento:        Optional[float] = Field(None, ge=0.1, le=200)

class SeguimientoIn(BaseModel):
    consulta_id:         int       = Field(..., ge=1)
    estado_general:      str       = Field(...)
    sintoma_ids:         list[int] = Field(default=[], max_length=12)
    temperatura:         Optional[float] = Field(None, ge=35.0, le=43.0)
    frecuencia_cardiaca: Optional[int]   = Field(None, ge=20, le=300)
    notas:               Optional[str]   = Field(None, max_length=500)

class DeleteAccountIn(BaseModel):
    password: str = Field(..., min_length=1)
    motivo:   Optional[str] = Field(None, max_length=255)

class ChangePasswordIn(BaseModel):
    password_actual: str = Field(..., min_length=1)
    password_nuevo:  str = Field(..., min_length=8, max_length=100)
    @field_validator("password_nuevo")
    @classmethod
    def pwd_strength(cls, v):
        if not any(c.isupper() for c in v): raise ValueError("Necesita una mayúscula")
        if not any(c.isdigit() for c in v): raise ValueError("Necesita un número")
        return v

# ── Startup ───────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    get_predictor()
    logger.info("VetPredict API v5 iniciada — Árbol secuencial + Recomendaciones activas")

@app.get("/")
def root(): return {"status":"ok","api":"VetPredict","version":"5.0.0"}

@app.get("/health")
def health():
    db = ml = True
    try: query_one("SELECT 1")
    except: db = False
    try: get_predictor()
    except: ml = False
    return {"database":"ok" if db else "error","modelo_ml":"ok" if ml else "error"}

# ── Auth ──────────────────────────────────────────────────────
@app.post("/api/auth/register", status_code=201)
@limiter.limit("3/minute")
async def register(request: Request, body: RegisterIn):
    ip = get_ip(request)
    if query_one("SELECT id FROM usuarios WHERE email=%s", (body.email,)):
        raise HTTPException(400, "Este correo ya está registrado.")
    uid = execute("INSERT INTO usuarios (nombre,apellido,email,password_hash) VALUES (%s,%s,%s,%s)",
        (body.nombre.strip(), body.apellido.strip(), body.email.lower().strip(), hash_password(body.password)))
    logger.info(f"[REGISTER] ID={uid} email={body.email} IP={ip}")
    return {"token": create_token(uid, body.email),
            "usuario": {"id":uid,"nombre":body.nombre,"apellido":body.apellido,"email":body.email}}

@app.post("/api/auth/login")
@limiter.limit("5/minute")
async def login(request: Request, body: LoginIn):
    ip = get_ip(request)
    ua = request.headers.get("User-Agent","")[:255]
    email = body.email.lower().strip()
    user = query_one("SELECT * FROM usuarios WHERE email=%s AND activo=1", (email,))
    if user and user.get("bloqueado_hasta"):
        b = user["bloqueado_hasta"]
        if isinstance(b, datetime.datetime) and b > datetime.datetime.utcnow():
            raise HTTPException(429, f"Cuenta bloqueada hasta las {b.strftime('%H:%M')}.")
    if not user or not verify_password(body.password, user["password_hash"]):
        try: execute("INSERT INTO login_intentos (email,ip,exitoso,user_agent) VALUES (%s,%s,0,%s)", (email,ip,ua))
        except: pass
        if user:
            i = user["intentos_fallidos"] + 1
            b = datetime.datetime.utcnow() + datetime.timedelta(minutes=15) if i >= 5 else None
            execute("UPDATE usuarios SET intentos_fallidos=%s,bloqueado_hasta=%s WHERE id=%s", (i,b,user["id"]))
        logger.warning(f"[LOGIN] Fallo: {email} IP={ip}")
        raise HTTPException(401, "Correo o contraseña incorrectos.")
    execute("UPDATE usuarios SET intentos_fallidos=0,bloqueado_hasta=NULL WHERE id=%s", (user["id"],))
    try: execute("INSERT INTO login_intentos (email,ip,exitoso,user_agent) VALUES (%s,%s,1,%s)", (email,ip,ua))
    except: pass
    logger.info(f"[LOGIN] OK: {email} ID={user['id']} IP={ip}")
    return {"token": create_token(user["id"], user["email"]),
            "usuario": {"id":user["id"],"nombre":user["nombre"],"apellido":user["apellido"],"email":user["email"]}}

@app.get("/api/auth/me")
async def me(current=Depends(get_secure_user)): return current

@app.post("/api/auth/logout")
async def logout(request: Request, creds=Depends(_bearer), current=Depends(get_secure_user)):
    _revoke_token(creds.credentials, current["id"], "logout")
    return {"message": "Sesión cerrada correctamente."}

@app.put("/api/auth/password")
@limiter.limit("3/minute")
async def change_password(request: Request, body: ChangePasswordIn, creds=Depends(_bearer), current=Depends(get_secure_user)):
    user = query_one("SELECT * FROM usuarios WHERE id=%s", (current["id"],))
    if not verify_password(body.password_actual, user["password_hash"]):
        raise HTTPException(401, "Contraseña actual incorrecta.")
    execute("UPDATE usuarios SET password_hash=%s WHERE id=%s", (hash_password(body.password_nuevo), current["id"]))
    _revoke_token(creds.credentials, current["id"], "password_change")
    return {"message": "Contraseña actualizada. Inicia sesión nuevamente."}

@app.delete("/api/auth/cuenta")
@limiter.limit("2/minute")
async def delete_account(request: Request, body: DeleteAccountIn, creds=Depends(_bearer), current=Depends(get_secure_user)):
    user = query_one("SELECT * FROM usuarios WHERE id=%s", (current["id"],))
    if not verify_password(body.password, user["password_hash"]):
        raise HTTPException(401, "Contraseña incorrecta.")
    _revoke_token(creds.credentials, current["id"], "account_deleted")
    execute("DELETE FROM usuarios WHERE id=%s", (current["id"],))
    logger.warning(f"[DELETE_ACCOUNT] ID={current['id']} email={current['email']}")
    return {"message": "Tu cuenta ha sido eliminada. 🐾"}

@app.get("/api/perfil")
async def get_perfil(current=Depends(get_secure_user)):
    uid = current["id"]
    tm  = query_one("SELECT COUNT(*) AS t FROM mascotas WHERE usuario_id=%s AND activa=1", (uid,))
    tc  = query_one("SELECT COUNT(*) AS t FROM consultas WHERE usuario_id=%s", (uid,))
    ult = query_one("SELECT creado_en FROM consultas WHERE usuario_id=%s ORDER BY creado_en DESC LIMIT 1", (uid,))
    return {"id":uid,"nombre":current["nombre"],"apellido":current["apellido"],"email":current["email"],
            "total_mascotas":tm["t"] if tm else 0,"total_consultas":tc["t"] if tc else 0,
            "ultima_consulta":serial({"f":ult["creado_en"]})["f"] if ult else None}

# ── Mascotas ──────────────────────────────────────────────────
@app.get("/api/mascotas")
async def listar(current=Depends(get_secure_user)):
    return [serial(r) for r in query_all("SELECT * FROM mascotas WHERE usuario_id=%s AND activa=1 ORDER BY creado_en DESC", (current["id"],))]

@app.get("/api/mascotas/{mid}")
async def obtener(mid: int, current=Depends(get_secure_user)):
    row = query_one("SELECT * FROM mascotas WHERE id=%s AND usuario_id=%s AND activa=1", (mid, current["id"]))
    if not row: raise HTTPException(404, "Mascota no encontrada.")
    return serial(row)

@app.post("/api/mascotas", status_code=201)
async def crear(body: MascotaIn, request: Request, current=Depends(get_secure_user)):
    mid = execute("INSERT INTO mascotas (usuario_id,nombre,raza,edad,genero,peso) VALUES (%s,%s,%s,%s,%s,%s)",
        (current["id"],body.nombre.strip(),body.raza,body.edad,body.genero,body.peso))
    logger.info(f"[MASCOTA] Creada ID={mid} usuario={current['id']}")
    return serial(query_one("SELECT * FROM mascotas WHERE id=%s", (mid,)))

@app.put("/api/mascotas/{mid}")
async def actualizar(mid: int, body: MascotaIn, current=Depends(get_secure_user)):
    if not query_one("SELECT id FROM mascotas WHERE id=%s AND usuario_id=%s", (mid, current["id"])):
        raise HTTPException(404, "Mascota no encontrada.")
    execute("UPDATE mascotas SET nombre=%s,raza=%s,edad=%s,genero=%s,peso=%s WHERE id=%s",
        (body.nombre.strip(),body.raza,body.edad,body.genero,body.peso,mid))
    return serial(query_one("SELECT * FROM mascotas WHERE id=%s", (mid,)))

@app.delete("/api/mascotas/{mid}")
async def eliminar(mid: int, request: Request, current=Depends(get_secure_user)):
    ex = query_one("SELECT id,nombre FROM mascotas WHERE id=%s AND usuario_id=%s", (mid, current["id"]))
    if not ex: raise HTTPException(404, "Mascota no encontrada.")
    execute("DELETE FROM mascotas WHERE id=%s", (mid,))
    logger.info(f"[MASCOTA] Eliminada ID={mid} nombre='{ex['nombre']}' usuario={current['id']}")
    return {"message":f"{ex['nombre']} eliminado.","eliminado":True}

# ── Diagnóstico v5 — DOS modos ────────────────────────────────

@app.post("/api/diagnostico/siguiente-pregunta")
async def siguiente_pregunta(body: RespuestasIn, current=Depends(get_secure_user)):
    """
    MODO SECUENCIAL — Como el árbol del ejemplo.
    El frontend envía las respuestas acumuladas (sí/no por síntoma).
    El backend responde con la siguiente pregunta O con el diagnóstico final.
    """
    mascota = query_one(
        "SELECT * FROM mascotas WHERE id=%s AND usuario_id=%s AND activa=1",
        (body.mascota_id, current["id"])
    )
    if not mascota: raise HTTPException(404, "Mascota no encontrada.")

    predictor = get_predictor()
    result = predictor.get_next_question(body.respuestas)

    if result["tipo"] == "diagnostico":
        # Guardar en BD
        sintoma_ids = [
            sid for sid, nombre in predictor.SINTOMA_ID_MAP.items()
            if body.respuestas.get(nombre, False)
        ] if hasattr(predictor, 'SINTOMA_ID_MAP') else []

        from modelo_v5 import SINTOMA_ID_MAP as SIM
        sintoma_ids = [sid for sid, nombre in SIM.items() if body.respuestas.get(nombre, False)]

        cid = execute(
            "INSERT INTO consultas (mascota_id,usuario_id,temperatura,frecuencia_cardiaca,"
            "peso_momento,enfermedad_predicha,nivel_riesgo,confianza,sintomas_json,resultado_json) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (body.mascota_id, current["id"],
             body.temperatura, body.frecuencia_cardiaca, body.peso_momento,
             result["enfermedad"], result["nivel_riesgo"], result["confianza"],
             json.dumps(sintoma_ids), json.dumps(result))
        )
        result["consulta_id"] = cid
        result["mascota"]     = serial(mascota)
        logger.info(f"[DIAG] Secuencial ID={cid} resultado='{result['enfermedad']}' usuario={current['id']}")

    return result


@app.post("/api/diagnostico")
@limiter.limit("30/minute")
async def diagnosticar(request: Request, body: DiagnosticoIn, current=Depends(get_secure_user)):
    """
    MODO DIRECTO — El frontend envía todos los síntomas marcados de una vez.
    Útil para compatibilidad con versión anterior.
    """
    mascota = query_one(
        "SELECT * FROM mascotas WHERE id=%s AND usuario_id=%s AND activa=1",
        (body.mascota_id, current["id"])
    )
    if not mascota: raise HTTPException(404, "Mascota no encontrada.")

    sintoma_ids = list(set(body.sintoma_ids))
    if not all(1 <= s <= 12 for s in sintoma_ids):
        raise HTTPException(400, "IDs de síntomas inválidos (1-12)")

    r = get_predictor().predict(
        sintoma_ids, body.temperatura or 38.5,
        body.frecuencia_cardiaca or 90,
        body.peso_momento or float(mascota.get("peso") or 15),
        int(mascota.get("edad") or 3)
    )

    cid = execute(
        "INSERT INTO consultas (mascota_id,usuario_id,temperatura,frecuencia_cardiaca,"
        "peso_momento,duracion_sintomas,enfermedad_predicha,nivel_riesgo,confianza,"
        "sintomas_json,resultado_json) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
        (body.mascota_id, current["id"], body.temperatura, body.frecuencia_cardiaca,
         body.peso_momento, body.duracion_sintomas, r["enfermedad"], r["nivel_riesgo"],
         r["confianza"], json.dumps(sintoma_ids), json.dumps(r))
    )
    logger.info(f"[DIAG] Directo ID={cid} resultado='{r['enfermedad']}' usuario={current['id']}")
    return {"consulta_id":cid,"enfermedad":r["enfermedad"],"nivel_riesgo":r["nivel_riesgo"],
            "confianza":r["confianza"],"probabilidades":r["probabilidades"],
            "sintomas_usados":r["sintomas_usados"],"recomendacion":r["recomendacion"],
            "mascota":serial(mascota)}

# ── Historial ─────────────────────────────────────────────────
@app.get("/api/historial")
async def historial(mascota_id: Optional[int] = None, current=Depends(get_secure_user)):
    if mascota_id:
        rows = query_all("SELECT c.*,m.nombre AS mascota_nombre,m.raza FROM consultas c JOIN mascotas m ON m.id=c.mascota_id WHERE c.usuario_id=%s AND c.mascota_id=%s ORDER BY c.creado_en DESC", (current["id"],mascota_id))
    else:
        rows = query_all("SELECT c.*,m.nombre AS mascota_nombre,m.raza FROM consultas c JOIN mascotas m ON m.id=c.mascota_id WHERE c.usuario_id=%s ORDER BY c.creado_en DESC", (current["id"],))
    return [serial(r) for r in rows]

@app.get("/api/historial/{cid}")
async def detalle(cid: int, current=Depends(get_secure_user)):
    row = query_one("SELECT c.*,m.nombre AS mascota_nombre,m.raza FROM consultas c JOIN mascotas m ON m.id=c.mascota_id WHERE c.id=%s AND c.usuario_id=%s", (cid,current["id"]))
    if not row: raise HTTPException(404, "Consulta no encontrada.")
    row = serial(row)
    try: row["resultado"] = json.loads(row.get("resultado_json","{}") or "{}")
    except: row["resultado"] = {}
    try: row["sintoma_ids"] = json.loads(row.get("sintomas_json","[]") or "[]")
    except: row["sintoma_ids"] = []
    # Incluir recomendación en el detalle
    from modelo_v5 import RECOMENDACIONES, RECOMENDACION_GENERICA
    enf = row.get("enfermedad_predicha","")
    row["recomendacion"] = RECOMENDACIONES.get(enf, RECOMENDACION_GENERICA)
    row["seguimientos"] = [serial(s) for s in query_all("SELECT * FROM seguimientos WHERE consulta_id=%s ORDER BY creado_en ASC", (cid,))]
    return row

# ── Seguimientos ──────────────────────────────────────────────
@app.post("/api/seguimientos", status_code=201)
async def crear_seg(body: SeguimientoIn, current=Depends(get_secure_user)):
    if not query_one("SELECT id FROM consultas WHERE id=%s AND usuario_id=%s", (body.consulta_id,current["id"])):
        raise HTTPException(404, "Consulta no encontrada.")
    ne = nc = None
    if body.sintoma_ids:
        res = get_predictor().predict(list(set(body.sintoma_ids)))
        ne, nc = res["enfermedad"], res["confianza"]
    sid = execute("INSERT INTO seguimientos (consulta_id,usuario_id,estado_general,temperatura,frecuencia_cardiaca,sintomas_json,enfermedad_predicha,confianza,notas) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
        (body.consulta_id,current["id"],body.estado_general,body.temperatura,body.frecuencia_cardiaca,json.dumps(body.sintoma_ids),ne,nc,body.notas))
    execute("UPDATE consultas SET estado='en_seguimiento' WHERE id=%s", (body.consulta_id,))
    return serial(query_one("SELECT * FROM seguimientos WHERE id=%s", (sid,)))

@app.get("/api/seguimientos/{cid}")
async def listar_seg(cid: int, current=Depends(get_secure_user)):
    return [serial(r) for r in query_all("SELECT * FROM seguimientos WHERE consulta_id=%s ORDER BY creado_en ASC", (cid,))]
