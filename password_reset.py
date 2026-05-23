"""
password_reset.py — Recuperación de contraseña con SendGrid
"""
import os, random, string, datetime
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

def _generar_codigo():
    return ''.join(random.choices(string.digits, k=6))

def enviar_codigo_reset(email: str, nombre: str) -> str:
    codigo = _generar_codigo()
    mensaje = Mail(
        from_email=('juanestebancardozoricardo95@gmail.com', 'VetPredict'),
        to_emails=email,
        subject='🐾 Código para restablecer tu contraseña — VetPredict',
        html_content=f"""
        <div style="font-family: Arial, sans-serif; max-width: 500px; margin: auto;">
            <div style="background: #2E7D6B; padding: 24px; border-radius: 12px 12px 0 0; text-align: center;">
                <h1 style="color: white; margin: 0;">🐾 VetPredict</h1>
            </div>
            <div style="background: #f4f7f5; padding: 32px; border-radius: 0 0 12px 12px;">
                <p style="font-size: 16px;">Hola <strong>{nombre}</strong>,</p>
                <p>Recibimos una solicitud para restablecer tu contraseña.</p>
                <p>Tu código de verificación es:</p>
                <div style="background: white; border: 2px solid #2E7D6B; border-radius: 12px;
                            padding: 24px; text-align: center; margin: 24px 0;">
                    <span style="font-size: 42px; font-weight: bold; color: #2E7D6B;
                                 letter-spacing: 8px;">{codigo}</span>
                </div>
                <p style="color: #6B8F85; font-size: 13px;">
                    ⏱ Este código expira en <strong>15 minutos</strong>.<br>
                    Si no solicitaste esto, ignora este correo.
                </p>
            </div>
        </div>
        """
    )
    sg = SendGridAPIClient(os.getenv('SENDGRID_API_KEY'))
    sg.send(mensaje)
    return codigo