import os
import json
from kdra.core.reasoning.engine import OpenAIEngine
from pydantic import BaseModel, Field
from typing import List

class QueryPlan(BaseModel):
    strategy: str = Field(description="The execution strategy: 'graph_first', 'vector_first', or 'hybrid'")
    search_terms: List[str] = Field(description="Keywords to use in vector or graph search")
    sub_tasks: List[str] = Field(description="The step-by-step logic the agent should follow")

engine = OpenAIEngine()

if engine.instructor_client:
    try:
        plan = engine.generate_structured(
            prompt="Which paper performed better on the GLUE dataset, paper A or paper B?",
            schema_class=QueryPlan,
            system_prompt="You are an expert Query routing agent.",
            max_retries=2
        )
        print("Plan:")
        print(plan.model_dump_json(indent=2))
    except Exception as e:
        print("Error:", e)
else:
    print("Instructor not ready")
