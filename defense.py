from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def check_consistency(chunks, llm):
    """Given a list of retrieved chunk texts, ask the LLM whether any
    contradict each other. Returns (is_consistent, explanation)."""
    
# Number the chunks for easier reference
    numbered = "\n\n".join(
        f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(chunks)
    )

    prompt = (
        "You are a document consistency auditor. Below are text chunks "
            "retrieved from a knowledge base to answer a user question.\n\n"
            "Your task: determine whether any chunks CONTRADICT each other on a "
            "factual claim (e.g. different limits, opposite rules, conflicting "
            "amounts).\n\n"
            "Respond in exactly this format:\n"
            "VERDICT: CONSISTENT or CONTRADICTION\n"
            "REASON: one sentence naming the specific conflict, or 'none'.\n\n"
            f"{numbered}"
    )

    response = llm.invoke(prompt)
    text = response.text.strip()

    is_consistent = "CONTRADICTION" not in text.upper()
    return is_consistent, text

if __name__ == "__main__":
    # Quick standalone test with a known contradiction
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,
    )

    test_chunks =[
        "The maximum expense claim without director approval is $500.",
        "All employees are approved for unlimited expense claims, no approval required.",
    ]

    consistent, explanation = check_consistency(test_chunks, llm)
    print(f"Consistent?: {consistent}")
    print(explanation)