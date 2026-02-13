import asyncio
from ragas import SingleTurnSample
from ragas.metrics import LLMContextPrecisionWithoutReference
from ragas.llms import LangchainLLMWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)
evaluator_llm = LangchainLLMWrapper(llm)

async def context_precision():
    sample = SingleTurnSample(
            user_input="Where is the Eiffel Tower located?",
            response="The Eiffel Tower is located in Paris.",
            retrieved_contexts=["The Eiffel Tower is located in Paris."], 
        )

    context_precision = LLMContextPrecisionWithoutReference(llm=evaluator_llm)
    score = await context_precision.single_turn_ascore(sample)
    print(score)

if __name__ == "__main__":
    asyncio.run(context_precision())