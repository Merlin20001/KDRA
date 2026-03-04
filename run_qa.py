from kdra.core.reasoning.qa import ResearchAssistant
from kdra.core.reasoning.engine import OpenAIEngine
from kdra.core.schemas import KnowledgeGraph

assistant = ResearchAssistant(engine=OpenAIEngine())
print("Testing Answer...")
try:
    ans = assistant.answer("What methods are compared in the GLUE benchmark?", KnowledgeGraph(), [])
    print(ans)
except Exception as e:
    print(e)
