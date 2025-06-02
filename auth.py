from datetime import timedelta, datetime
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from starlette import status
from database import SessionLocal
from models import Users
from passlib.context import CryptContext
#from fastapi.security import OAuth2Password, RequestForm, OAuth2PasswordBearer
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import jwt, JWTError

router = APIRouter(prefix='/auth',
tags=['auth'])

SECRET_KEY = '29ab7c4d15f5ca6e8387cb8e3988fdcb8f932a965c5b94295ec4f9724bc5073b'
ALGORITHM = 'HS256'

bcrypt_context = CryptContext(schemes=['bcrypt'], deprecated="auto")
oauth2_bearer = OAuth2PasswordBearer(tokenUrl='/auth/token')

class CreateUserRequest(BaseModel):
    username: str
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

class TokenData(BaseModel):
    access_token: str
    token_type: str

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_db)]

@router.post("/signup", status_code=status.HTTP_201_CREATED)
async def create_user(db: db_dependency, create_user_request: CreateUserRequest):
    create_user_model = Users(
        username=create_user_request.username,
        email=create_user_request.email,  
        hashed_password=bcrypt_context.hash(create_user_request.password),
    )
    db.add(create_user_model)
    db.commit()
    return {"message": "User created successfully"}

@router.post("/login", response_model=TokenData)
async def login_for_access_token(login_request: LoginRequest, db: db_dependency):
    print(f"Login attempt for user: {login_request.email}")
    user = authenticate_user(login_request.email, login_request.password, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=30)
    access_token = create_access_token(
        data={"email": user.email}, expires_delta=access_token_expires
    )
    print("Access token generated successfully")
    return {"access_token": access_token, "token_type": "bearer"}

def create_access_token(data: dict, expires_delta: timedelta):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def authenticate_user(email: str, password: str, db):
    user = db.query(Users).filter(Users.email == email).first()
    if not user:
        print("User not found during authentication")
        return False
    if not bcrypt_context.verify(password, user.hashed_password):
        print("Password verification failed")
        return False
    print("User authenticated successfully")
    return user
