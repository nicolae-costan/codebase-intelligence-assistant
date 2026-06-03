"""Grounding check layer to detect ungrounded hallucinated claims."""

import re

def check_grounding(answer: str, retrieved_chunks: list[dict]) -> dict[str, object]:
    """
    Check if the answer makes ungrounded claims about code not present in the chunks.
    Extracts potential identifiers (snake_case, camelCase, backticked code) from the answer
    and verifies if they appear anywhere in the retrieved chunks' source code.
    """
    
    if not retrieved_chunks:
        return {"grounded": True, "ungrounded_claims": []}
        
    # 1. Extract potential claims from the answer
    # Match backticked items: `some_function`
    backticked = set(re.findall(r'`([^`\n]+)`', answer))
    
    # Match function calls like some_func(
    func_calls = set(re.findall(r'\b([a-zA-Z0-9_]+)\(', answer))
    
    # Match snake_case
    snake_case = set(re.findall(r'\b[a-z]+_[a-z0-9_]+\b', answer))
    
    # Match camelCase
    camel_case = set(re.findall(r'\b[a-z]+[A-Z][a-zA-Z0-9]+\b', answer))
    
    # Match PascalCase (Must have at least one lowercase and another uppercase to avoid matching normal capitalized English words)
    pascal_case = set(re.findall(r'\b[A-Z][a-z0-9]+[A-Z][a-zA-Z0-9]+\b', answer))
    
    all_claims = backticked | snake_case | camel_case | pascal_case | func_calls
    
    # Filter out empty strings, spaces, or very short words that are likely false positives
    all_claims = {c.strip() for c in all_claims if len(c.strip()) > 2}
    
    if not all_claims:
        return {"grounded": True, "ungrounded_claims": []}
        
    # 2. Build the "allowed" text from retrieved chunks
    allowed_text = " ".join([chunk.get("source", "") + " " + chunk.get("function_name", "") + " " + chunk.get("filepath", "") for chunk in retrieved_chunks])
    
    # 3. Check which claims do not exist in the allowed text
    ungrounded_claims = []
    for claim in all_claims:
        if claim not in allowed_text:
            ungrounded_claims.append(claim)
            
    ungrounded_claims.sort()
    
    return {
        "grounded": len(ungrounded_claims) == 0,
        "ungrounded_claims": ungrounded_claims
    }
