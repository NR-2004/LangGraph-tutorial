import os
import uuid
from typing import TypedDict, Optional
 
from dotenv import load_dotenv
from langchain_community.utilities import SerpAPIWrapper
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
 
load_dotenv()
 

# 1. STATE
class TravelState(TypedDict):
    user_input: str
    destination: Optional[str]
    days: Optional[int]
    budget: Optional[int]
    daily_cost: Optional[int]
    searched_destination: Optional[str]  # tracks which destination we last searched, to avoid re-searching
    estimated_cost: Optional[int]
    final_message: Optional[str]
 
 
# 2. MODEL + SEARCH TOOL
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
)
 
search_tool = SerpAPIWrapper(serpapi_api_key=os.getenv("SERPAPI_API_KEY"))
 
 
class ExtractedDetails(BaseModel):
    destination: Optional[str] = Field(
        None, description="Travel destination city, ONLY if mentioned in this message."
    )
    days: Optional[int] = Field(
        None, description="Number of days for the trip, ONLY if mentioned in this message."
    )
    budget: Optional[int] = Field(
        None, description="Total budget in INR, ONLY if mentioned in this message. "
        "Convert phrases like '3000 rs' or '3k' to a plain integer."
    )
 
 
class DailyCostEstimate(BaseModel):
    daily_cost_inr: int = Field(
        description="Average daily travel cost in INR for a mid-range "
        "budget traveler at this destination, based on the search results."
    )
 
 
extractor = llm.with_structured_output(ExtractedDetails)
cost_extractor = llm.with_structured_output(DailyCostEstimate)
 
 
# 3. NODES
def get_user_details(state: TravelState) -> TravelState:
    """LLM extraction instead of regex - handles any phrasing. Only
    returns keys the message actually mentioned, so a follow-up like
    'increase to 3000 rs' updates budget WITHOUT wiping out the
    destination/days already stored from earlier turns."""
    extracted = extractor.invoke(
        f"Extract only what THIS message states, leave the rest null:\n"
        f"\"{state['user_input']}\""
    )
 
    updates: TravelState = {}
    if extracted.destination:
        updates["destination"] = extracted.destination
    if extracted.days:
        updates["days"] = extracted.days
    if extracted.budget:
        updates["budget"] = extracted.budget
    return updates
 
 
def route_after_details(state: TravelState) -> str:
    """If we still don't have all three pieces (even after merging with
    prior turns), stop and ask - don't let a crash be the way we find out."""
    if not state.get("destination") or not state.get("days") or not state.get("budget"):
        return "need_more_info"
    return "ok"
 
 
def need_more_info(state: TravelState) -> TravelState:
    missing = []
    if not state.get("destination"):
        missing.append("destination")
    if not state.get("days"):
        missing.append("number of days")
    if not state.get("budget"):
        missing.append("budget")
    return {"final_message": f"I still need: {', '.join(missing)}."}
 
 
def check_destination(state: TravelState) -> TravelState:
    """Real web search (SerpAPI/Google) for daily cost - but only when
    the destination has actually changed since the last search. This
    is what stops 'increase to 3000 rs' from re-triggering a search for
    a destination we already looked up two turns ago."""
    destination = state["destination"]
    if state.get("daily_cost") is not None and state.get("searched_destination") == destination:
        return {}  # cached - nothing to do
 
    search_results = search_tool.results(
        f"average daily travel budget cost {destination} India per day INR"
    )
    snippets = "\n".join(
        r.get("snippet", "") for r in search_results.get("organic_results", [])
    )[:1500]
 
    estimate = cost_extractor.invoke(
        f"Destination: {destination}\nGoogle search results:\n{snippets}\n\n"
        f"Estimate the average daily travel cost in INR."
    )
    return {"daily_cost": estimate.daily_cost_inr, "searched_destination": destination}
 
 
def check_budget(state: TravelState) -> TravelState:
    estimated_cost = state["daily_cost"] * state["days"]
    return {"estimated_cost": estimated_cost}
 
 
def route_on_budget(state: TravelState) -> str:
    return "yes" if state["budget"] >= state["estimated_cost"] else "no"
 
 
def create_travel_plan(state: TravelState) -> TravelState:
    reply = llm.invoke(
        f"In under 40 words, give a {state['days']}-day plan outline for "
        f"{state['destination']} within ₹{state['budget']}."
    )
    leftover = state["budget"] - state["estimated_cost"]
    message = (
        f"{reply.content}\n\n"
        f"Cost: ₹{state['estimated_cost']} (₹{state['daily_cost']}/day x {state['days']} days). "
        f"₹{leftover} left of your ₹{state['budget']} budget."
    )
    return {"final_message": message}
 
 
def recalculate_budget(state: TravelState) -> TravelState:
    affordable_days = state["budget"] // state["daily_cost"]
    if affordable_days == 0:
        message = (
            f"₹{state['budget']} doesn't cover even 1 day in "
            f"{state['destination']} (₹{state['daily_cost']}/day)."
        )
    else:
        message = (
            f"₹{state['budget']} isn't enough for {state['days']} days in "
            f"{state['destination']} (needs ₹{state['estimated_cost']} at "
            f"₹{state['daily_cost']}/day).\n"
            f"With this budget you can afford {affordable_days} day(s) instead."
        )
    return {"final_message": message}
 
# 4. BUILD THE GRAPH
def build_graph():
    builder = StateGraph(TravelState)
 
    builder.add_node("get_user_details", get_user_details)
    builder.add_node("need_more_info", need_more_info)
    builder.add_node("check_destination", check_destination)
    builder.add_node("check_budget", check_budget)
    builder.add_node("create_travel_plan", create_travel_plan)
    builder.add_node("recalculate_budget", recalculate_budget)
 
    builder.add_edge(START, "get_user_details")
    builder.add_conditional_edges(
        "get_user_details",
        route_after_details,
        {"need_more_info": "need_more_info", "ok": "check_destination"},
    )
    builder.add_edge("need_more_info", END)
 
    builder.add_edge("check_destination", "check_budget")
    builder.add_conditional_edges(
        "check_budget",
        route_on_budget,
        {"yes": "create_travel_plan", "no": "recalculate_budget"},
    )
 
    builder.add_edge("create_travel_plan", END)
    builder.add_edge("recalculate_budget", END)
 
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)
 
 
def run():
    graph = build_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
 
    print("AI Travel Planner.")
    print('"Type your destination and budget:"\n')
 
    while True:
        user_input = input("> ").strip()
        if not user_input:
            continue
        result = graph.invoke({"user_input": user_input}, config)
        print("\n" + result["final_message"] + "\n")
 
 
if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\nbye")
 