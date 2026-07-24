from typing import Dict, List
from openai import OpenAI

def generate_response(openai_key: str, user_message: str, context: str, 
                     conversation_history: List[Dict],
                     model: str = "gpt-3.5-turbo") -> str:
    """Generate response using OpenAI with context"""
    system_prompt = """You are a NASA space mission expert.
Answer the user's question using only the retrieved context provided for the
current request. Cite the source labels included in that context when making
factual claims. If the context does not contain enough information, say that
you do not have enough information in the provided sources. Do not invent
facts or rely on outside knowledge."""

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)

    if context:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Use the following retrieved context for the current "
                    "question:\n\n"
                    f"<retrieved_context>\n{context}\n</retrieved_context>"
                ),
            }
        )

    messages.append({"role": "user", "content": user_message})

    client = OpenAI(api_key=openai_key)
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=800,
    )

    return completion.choices[0].message.content
