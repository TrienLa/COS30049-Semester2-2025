# Email Classification Web Application
## Overview
Using the previously trained models, utilising FastAPI and React to create a responsive web application. Able to identify and classify spam emails from text input.

## Setting up
Installing the required packages for both the frontend and backend using the following commands.

### Frontend
```
cd [path to repo]/frontend
npm install
```

### Backend
For the backend, you can install the packages using `pip` and `requirements.txt`.

```
cd [path to repo]/backend
pip install -r requirements.txt
```

## Running the app
### Frontend
The frontend can be started using the following command. You will need to open another command line window to do this.
```
cd [path to repo]/frontend
npm run dev
```

The previous command will only host the frontend on your local machine. To create a distributable build, use this instead. It will create a compiled build of the frontend ready to be hosted on a dedicated server.
```
npm run build
```

### Backend
The backend can be started using the following command. It will open the backend to port `8000`.
```
cd [path to repo]/backend
python main.py
```
