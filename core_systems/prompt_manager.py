import json
import os
import logging

logger = logging.getLogger(__name__)

class PromptManager:
    """
    A centralized class for creating and managing prompts sent to LLMs.
    This helps in keeping prompts consistent and easy to modify without
    touching the core logic of the system components.
    """
    def __init__(self, prompts_filepath: str):
        """
        Initializes the PromptManager.

        Args:
            prompts_filepath (str): The path to the JSON file containing the prompts.
        """
        if not os.path.exists(prompts_filepath):
            raise FileNotFoundError(f"Prompts file not found at: {prompts_filepath}")
        self.prompts_filepath = prompts_filepath
        self.prompts = {}
        self.load_prompts()

    def load_prompts(self):
        """
        Loads the prompts from the JSON file into the prompts dictionary.
        """
        try:
            with open(self.prompts_filepath, 'r', encoding='utf-8') as f:
                self.prompts = json.load(f)
            logger.info(f"Successfully loaded {len(self.prompts)} prompts from {self.prompts_filepath}.")
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load or parse prompts file: {e}", exc_info=True)
            self.prompts = {} # Ensure prompts is a dict even on failure

    def get(self, prompt_name: str, default: str = "") -> str:
        """
        Retrieves a prompt by its name.

        Args:
            prompt_name (str): The name of the prompt to retrieve.
            default (str, optional): A default value to return if the prompt name
                                     is not found. Defaults to "".

        Returns:
            str: The prompt template string.
        """
        return self.prompts.get(prompt_name, default)
