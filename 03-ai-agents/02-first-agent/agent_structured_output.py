from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain.messages import HumanMessage
from pydantic import BaseModel
from pprint import pprint
from typing import List

load_dotenv()

class CapitalInfo(BaseModel):
    nome: str
    populacao: int
    pontos_turisticos: List[str]
    economia: str

llm = ChatGroq(model="llama-3.3-70b-versatile")
agent_structured = create_agent(
    model=llm,
    system_prompt="Você é um expert em turismo. Responda com as informações sobre a capital solicitada.",
    response_format=CapitalInfo
)

questions = [
    "Qual é a capital do Brasil?",
    "Qual é a capital da França"
]

for q in questions:
    response_structured = agent_structured.invoke(
        {"messages": [HumanMessage(content=q)]}
    )
    print("Output do Agente:")
    capital_info = response_structured["messages"][-1].content
    pprint(capital_info)
    pprint("-*" * 20)
