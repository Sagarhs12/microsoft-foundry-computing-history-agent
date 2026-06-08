"""
Agent Client - Handles interaction with the Microsoft Foundry agent.

This module contains the core logic for connecting to and communicating with
the agent published in Microsoft Foundry.
"""

import os
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import OpenAI

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class AgentClient:
    """Client for interacting with a Microsoft Foundry agent."""

    def __init__(self):
        """Initialize the agent client."""

        endpoint = os.getenv("AGENT_ENDPOINT")

        if not endpoint:
            raise ValueError("AGENT_ENDPOINT not found in environment variables")

        # Remove /responses if present
        endpoint = endpoint.replace("/responses", "")

        self.agent_endpoint = endpoint

        # Create authenticated OpenAI client
        self.client = OpenAI(
            api_key=get_bearer_token_provider(
                DefaultAzureCredential(),
                "https://ai.azure.com/.default"
            ),
            base_url=self.agent_endpoint,
            default_query={
                "api-version": "2025-11-15-preview"
            }
        )

        # Conversation history
        self.conversation_history: List[Dict[str, Any]] = []
        self.max_history = 3

    def send_message(self, user_message: str) -> str:
        """Send message to the agent."""

        self.conversation_history.append(
            {
                "role": "user",
                "content": user_message
            }
        )

        try:
            response = self.client.responses.create(
                input=self.conversation_history
            )

            assistant_message = response.output_text

            self.conversation_history.append(
                {
                    "role": "assistant",
                    "content": assistant_message
                }
            )

            # Keep only last 3 conversations
            user_count = sum(
                1 for msg in self.conversation_history
                if msg["role"] == "user"
            )

            while user_count > self.max_history:
                for i, msg in enumerate(self.conversation_history):
                    if msg["role"] == "user":
                        self.conversation_history.pop(i)

                        if (
                            i < len(self.conversation_history)
                            and self.conversation_history[i]["role"] == "assistant"
                        ):
                            self.conversation_history.pop(i)

                        user_count -= 1
                        break

            return assistant_message

        except Exception as e:
            logger.exception("Error communicating with agent")
            return f"Error: {str(e)}"

    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []