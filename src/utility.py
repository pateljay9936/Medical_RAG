"""
Utility functions for query classification and response generation
"""
import re
from typing import Tuple



class QueryClassifier:
    """Classify queries to determine if retrieval is needed"""

    # Simple greetings/acknowledgments (no retrieval needed)
    SIMPLE_PATTERNS = [
        r"\b(hi|hello|hey|greetings|good morning|good evening|good afternoon)\b",
        r"\b(thank you|thanks|thx|appreciate it)\b",
        r"\b(bye|goodbye|see you|take care)\b",
        r"\b(ok|okay|got it|understood|alright|sure)\b",
        r"\b(yes|yeah|yep|no|nope)\b",
    ]

    # Medical keywords (definitely needs retrieval)
    MEDICAL_KEYWORDS = [
        "symptom",
        "treatment",
        "disease",
        "diagnosis",
        "medicine",
        "medication",
        "cure",
        "pain",
        "fever",
        "infection",
        "doctor",
        "hospital",
        "prescription",
        "side effect",
        "dosage",
        "therapy",
        "vaccine",
        "surgery",
        "condition",
        "blood",
        "pressure",
        "diabetes",
        "cancer",
        "heart",
        "lung",
        "kidney",
        "test",
        "scan",
        "mri",
        "x-ray",
        "injury",
        "allergy",
        "chronic",
        "acute",
        "disorder",
        "illness",
        "sick",
        "health",
    ]

    @classmethod
    def needs_retrieval(cls, query: str) -> Tuple[bool, str]:
        """
        Determine if query needs document retrieval

        Args:
            query: User's input message

        Returns:
            Tuple[bool, str]: (needs_retrieval, reason)
        """
        query_lower = query.lower().strip()
        word_count = len(query_lower.split())

        # Rule 1: Very short queries with simple patterns (no retrieval)
        if word_count <= 3:
            for pattern in cls.SIMPLE_PATTERNS:
                if re.search(pattern, query_lower):
                    return False, "simple_greeting"

        # Rule 2: Contains medical keywords (needs retrieval)
        for keyword in cls.MEDICAL_KEYWORDS:
            if keyword in query_lower:
                return True, "medical_keyword_detected"

        # Rule 3: Question words in longer queries (likely needs retrieval)
        question_words = [
            "what",
            "how",
            "why",
            "when",
            "where",
            "which",
            "who",
            "can",
            "should",
            "is",
            "are",
            "does",
            "do",
            "could",
            "would",
            "will",
        ]
        if word_count >= 3 and any(q in query_lower.split()[:3] for q in question_words):
            return True, "question_detected"

        # Rule 4: Single word queries (context-dependent, default to no retrieval)
        if word_count == 1:
            return False, "single_word"

        # Default: If uncertain and query is substantial, use retrieval
        if word_count >= 4:
            return True, "substantial_query"

        return False, "default_no_retrieval"

    @classmethod
    def get_simple_response(cls, query: str) -> str:
        """
        Generate appropriate response for non-retrieval queries

        Args:
            query: User's input message

        Returns:
            str: Appropriate response without retrieval
        """
        query_lower = query.lower().strip()

        # Greetings
        if re.search(cls.SIMPLE_PATTERNS[0], query_lower):
            return (
                "Hello! I'm your medical assistant. I can help answer questions about "
                "symptoms, treatments, medications, and general health information. "
                "How can I assist you today?"
            )

        # Thanks
        if re.search(cls.SIMPLE_PATTERNS[1], query_lower):
            return (
                "You're very welcome! If you have any other health-related questions, "
                "feel free to ask. I'm here to help!"
            )

        # Goodbye
        if re.search(cls.SIMPLE_PATTERNS[2], query_lower):
            return (
                "Goodbye! Take care of your health. Feel free to return anytime you "
                "have questions. Stay well!"
            )

        # Acknowledgments
        if re.search(cls.SIMPLE_PATTERNS[3], query_lower):
            return (
                "Is there anything else you'd like to know about your health or medical concerns?"
            )

        # Yes/No
        if re.search(cls.SIMPLE_PATTERNS[4], query_lower):
            return (
                "Could you please provide more details about your question? "
                "I'm here to help with any health-related information you need."
            )

        # Default
        return (
            "I'm here to help with medical and health-related questions. "
            "Could you please elaborate on what you'd like to know?"
        )


class StreamingHandler:
    """Handle streaming responses from LangChain"""

    @staticmethod
    async def stream_rag_response(rag_chain, input_data: dict):
        """
        Stream tokens from RAG chain
        
        Args:
            rag_chain: The retrieval chain to stream from
            input_data: Dict with 'input' and 'chat_history' keys
            
        Yields:
            str: JSON formatted chunks with token data
        """
        import json
        
        try:
            # Stream the response
            full_answer = ""
            async for chunk in rag_chain.astream(input_data):
                # Extract answer tokens from the chunk
                if "answer" in chunk:
                    token = chunk["answer"]
                    full_answer += token
                    # Send token as JSON
                    yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"
            
            # Send completion signal
            yield f"data: {json.dumps({'token': '', 'done': True, 'full_answer': full_answer})}\n\n"
            
        except Exception as e:
            error_msg = f"Streaming error: {str(e)}"
            yield f"data: {json.dumps({'error': error_msg, 'done': True})}\n\n"

    @staticmethod
    async def stream_simple_response(response: str):
        """
        Stream a simple non-retrieval response character by character
        
        Args:
            response: The complete response text
            
        Yields:
            str: JSON formatted chunks with token data
        """
        import json
        import asyncio
        
        # Stream character by character with slight delay for smooth effect
        for char in response:
            yield f"data: {json.dumps({'token': char, 'done': False})}\n\n"
            await asyncio.sleep(0.01)  # Small delay for smooth streaming
        
        # Send completion signal
        yield f"data: {json.dumps({'token': '', 'done': True, 'full_answer': response})}\n\n"
