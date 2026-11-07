from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Support Ticket Analyzer API!"}


# validations


@app.post("/analyze")
async def analyze_ticket():
    return {"status":200}