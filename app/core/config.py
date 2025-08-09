import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # pydantic-settings will automatically load this from the .env file
    # or environment variables. The type hint ensures it's a string.
    GEMINI_API_KEY: str
    HACKRX_API_KEY: str = "default_hackrx_key"

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

# This creates the settings instance, loading and validating the variables.
settings = Settings()

# Debug: Print the current working directory and environment
print("\n=== Debug: Config Loading ===")
print(f"Current working directory: {os.getcwd()}")
print(f"HACKRX_API_KEY: {'*' * len(settings.HACKRX_API_KEY) if settings.HACKRX_API_KEY else 'Not set'}")
print("==========================\n")
