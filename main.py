from fastapi import FastAPI, status, Depends, HTTPException
import models
from database import engine, SessionLocal
from typing import Annotated
from sqlalchemy.orm import Session
import auth

from fastapi.security import OAuth2PasswordBearer
from fastapi import Security
from jose import JWTError, jwt
from auth import SECRET_KEY, ALGORITHM

from models import Users

app = FastAPI()

app.include_router(auth.router)

models.Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends (get_db)]

oauth2_scheme = OAuth2PasswordBearer(tokenUrl='/auth/token')

async def get_current_user(token: str = Security(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(Users).filter(Users.username == username).first()
    if user is None:
        raise credentials_exception
    return user

@app.get("/", status_code=status.HTTP_200_OK)
async def read_authenticated_user(current_user: Users = Depends(get_current_user)):
    return {"user": current_user.username}