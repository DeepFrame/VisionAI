
# deepframe-backend

This API provides user signup, login (with JWT token generation), and retrieval of the authenticated user's username.

## Install Dependencies
In a virtual environment, install all the dependencies at once, by running:

```bash
pip install -r requirements.txt
```

## JWT auth endpoints

#### POST `/auth/signup`

Create a new user.

- **Postman Request Body:**
  Body Type: raw
  
  Content-Type: application/json
  
    ```json
    {
      "username": "string",
      "email": "string",
      "password": "string"
    }

- **Postman Response**
    ```json
    {"message": "User created successfully"}


#### POST `/auth/login`

Authenticates a user and returns a JWT token if successful.

- **Postman Request Body:**
  Body Type: x-www-form-urlencoded

  Content-Type: application/x-www-form-urlencoded
  ```python
  username=string
  password=string

- **Postman Response:**
  ```json
  {
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6InRlc3RQU0BnbWFpbC5jb20iLCJwYXNzd29yZCI6InRlc3RpbmdQUzEyMyIsImV4cCI6MTc0ODYxNTkyNX0.gEyl3UA3yFQNybJ5ysAiX9ZjtjAnOqiU01r6GMxo2Yc",
    "token_type": "bearer"
  }

## Running the Application

#### Start the FastAPI server

``uvicorn main:app --reload``
