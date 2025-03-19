import os
import litellm
from litellm import acompletion
import logging
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMManager:
    def __init__(self):
        self.model_config = {
            "gpt-3.5-turbo": os.getenv("OPENAI_API_KEY"),
            # "claude-3": os.getenv("ANTHROPIC_API_KEY"),
            # "command": os.getenv("COHERE_API_KEY")
        }
        # Validate environment variables
        for model, key in self.model_config.items():
            if not key:
                logger.error(f"Missing API key for {model} in environment variables")
                raise ValueError(f"API key for {model} not configured")

    async def get_llm_response(self, prompt: str, model_name: str = "gpt-4") -> str:
        """
        Get LLM response with usage tracking and error handling
        """
        try:
            logger.info(f"LLMManager: Sending request to {model_name}...")
            response = await acompletion(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                api_key=self.model_config[model_name]
            )

            # Track usage
            input_tokens = response['usage']['prompt_tokens']
            output_tokens = response['usage']['completion_tokens']
            cost = self._calculate_cost(model_name, input_tokens, output_tokens)
            
            logger.info(f"LLM Usage - Model: {model_name}, Input Tokens: {input_tokens}, "
                        f"Output Tokens: {output_tokens}, Estimated Cost: ${cost:.4f}")

            content = response.choices[0].message.content
            if not isinstance(content, str):
                logger.error(f"Invalid response content type: {type(content)}")
                raise Exception("Invalid LLM response format")
            return content

        except litellm.exceptions.APIError as e:
            logger.error(f"API Error: {str(e)}")
            raise Exception("LLM service unavailable")
        except litellm.exceptions.Timeout as e:
            logger.error(f"Request Timeout: {str(e)}")
            raise Exception("LLM request timed out")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise Exception("Failed to process LLM request")

    def _calculate_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate estimated cost based on model pricing
        """
        pricing = {
            "gpt-3.5-turbo": {"input": 0.03, "output": 0.06},
            # "claude-3": {"input": 0.015, "output": 0.075},
            # "command": {"input": 0.015, "output": 0.045}
        }
        
        if model_name not in pricing:
            logger.warning(f"No pricing data for model: {model_name}")
            return 0.0

        logger.info(f"Calculating cost for model: {model_name}")
        cost = (input_tokens * pricing[model_name]["input"] / 1000) + \
               (output_tokens * pricing[model_name]["output"] / 1000)
        return cost

    async def get_summary(self, text: str, max_tokens: int) -> str:
        """Generate summary from markdown-formatted text"""
        prompt = (
            "Analyze this markdown document and create a comprehensive summary. "
            "Focus on the main content while ignoring markdown syntax. "
            "Keep the summary under {max_tokens} tokens.\n\n"
            "Document content:\n{content}"
        ).format(max_tokens=max_tokens, content=text[:15000])  # Truncate to 15k chars

        try:
            logger.info("LLMManager: Generating summary...")
            response = await self.get_llm_response(prompt, model_name="gpt-3.5-turbo")
            return response.strip()
        except Exception as e:
            logger.error(f"Summary generation failed: {str(e)}")
            return "Summary unavailable: processing error occurred"

    async def ask_question(self, context: str, question: str, max_tokens: int) -> str:
        """Answer question based on provided context"""
        prompt = (
        "Context:\n{context}\n\nQuestion: {question}\n\n"
        "Requirements:"
        "- Answer must be factual based on context"
        "- Maximum {max_tokens} tokens"
        "- If unsure, state \"I cannot determine from the provided content"
        ).format(context=context, question=question, max_tokens=max_tokens)
        
        try:
            logger.info("LLMManager: Asking question...")
            response = await self.get_llm_response(prompt, model_name="gpt-3.5-turbo")
            return response.strip()
        except Exception as e:
            logger.error(f"Question answering failed: {str(e)}")
            return "Answer unavailable: processing error occurred"
        