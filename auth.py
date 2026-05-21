"""
auth.py — JWT + bcrypt para Python 3.13
"""
import os, datetime
from dotenv import load_dotenv

import jwt
import bcrypt
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from database import query_one

load_dotenv(override=False)

SECRET     = os.getenv("JWT_SECRET", "cambiar-en-produccion")
ALGORITHM  = "HS256"
ACCESS_EXP = datetime.timedelta(hours=2)

bearer_scheme = HTTPBearer()


# ── Contraseñas ───────────────────────────────────────────────
def hash_password(plain: str) -> str:
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(plain.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


# ── JWT ───────────────────────────────────────────────────────
def create_token(user_id: int, email: str) -> str:
    payload = {
        "sub":   str(user_id),
        "email": email,
        "exp":   datetime.datetime.utcnow() + ACCESS_EXP,
        "iat":   datetime.datetime.utcnow(),
    }
    return jwt.encode(payload, SECRET, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expirado. Inicia sesión nuevamente.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Token inválido.")


# ── Dependencia FastAPI ───────────────────────────────────────
def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(bearer_scheme)
) -> dict:
    payload = decode_token(creds.credentials)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Token sin usuario.")

    user = query_one(
        "SELECT id, nombre, apellido, email FROM usuarios WHERE id=%s AND activo=1",
        (int(user_id),)
    )
    if not user:
        raise HTTPException(status_code=401, detail="Usuario no encontrado o inactivo.")
    return user
