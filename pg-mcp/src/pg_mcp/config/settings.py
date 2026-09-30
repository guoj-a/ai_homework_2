"""Configuration management for PostgreSQL MCP Server.

This module defines all configuration settings using Pydantic for validation
and type safety. Configuration is loaded from environment variables with
sensible defaults.
"""

import json
from typing import Any, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    Field,
    SecretStr,
    field_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfig(BaseSettings):
    """PostgreSQL database connection configuration."""

    model_config = SettingsConfigDict(
        env_prefix="DATABASE_",
        extra="ignore",
        env_file=".env",
        enable_decoding=False,
    )

    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, ge=1, le=65535, description="Database port")
    name: str = Field(default="postgres", description="Database name")
    user: str = Field(default="postgres", description="Database user")
    password: str = Field(default="", description="Database password")

    # Per-database access policy (applies on top of global SecurityConfig)
    blocked_tables: list[str] = Field(
        default_factory=list,
        description="Tables queries against this database may not reference",
    )
    blocked_columns: list[str] = Field(
        default_factory=list,
        description="Columns queries against this database may not reference",
    )

    # Connection pool settings
    min_pool_size: int = Field(default=5, ge=1, le=100, description="Minimum pool size")
    max_pool_size: int = Field(default=20, ge=1, le=100, description="Maximum pool size")
    pool_timeout: float = Field(
        default=30.0, ge=1.0, le=300.0, description="Pool acquire timeout in seconds"
    )
    command_timeout: float = Field(
        default=30.0, ge=1.0, le=300.0, description="Command execution timeout in seconds"
    )

    @property
    def dsn(self) -> str:
        """Build PostgreSQL DSN connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

    @property
    def safe_dsn(self) -> str:
        """Build DSN with masked password for logging."""
        return f"postgresql://{self.user}:***@{self.host}:{self.port}/{self.name}"

    @field_validator("blocked_tables", "blocked_columns", mode="before")
    @classmethod
    def parse_blocked_objects(cls, v: str | list[str]) -> list[str]:
        """Parse comma-separated string or list of table/column names."""
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v


class DatabaseAccessPolicy(BaseModel):
    """Per-database object access restrictions.

    Applied on top of the global SecurityConfig: a query against a specific
    database may not reference the tables/columns listed here.
    """

    blocked_tables: list[str] = Field(default_factory=list)
    blocked_columns: list[str] = Field(default_factory=list)

    @property
    def has_restrictions(self) -> bool:
        """Whether this policy restricts anything."""
        return bool(self.blocked_tables or self.blocked_columns)


class OpenAIConfig(BaseSettings):
    """OpenAI API configuration."""

    model_config = SettingsConfigDict(env_prefix="OPENAI_", extra="ignore", env_file=".env")

    api_key: SecretStr = Field(default=SecretStr(""), description="OpenAI API key")
    base_url: str | None = Field(
        default=None,
        description=(
            "Optional OpenAI-compatible API base URL (e.g. 'http://host:3030/v1' "
            "for a local gateway). None uses the official OpenAI endpoint."
        ),
    )
    model: str = Field(default="gpt-4o-mini", description="Model to use for SQL generation")
    max_tokens: int = Field(
        default=2000, ge=100, le=128000, description="Maximum tokens in response"
    )
    temperature: float = Field(
        default=0.0, ge=0.0, le=2.0, description="Temperature for response randomness"
    )
    timeout: float = Field(
        default=30.0, ge=5.0, le=120.0, description="API request timeout in seconds"
    )

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: SecretStr) -> SecretStr:
        """Validate API key is not empty and has correct format."""
        api_key_str = v.get_secret_value()
        if not api_key_str or not api_key_str.strip():
            raise ValueError("OpenAI API key must not be empty")
        if not api_key_str.startswith("sk-"):
            raise ValueError("OpenAI API key must start with 'sk-'")
        return v


class SecurityConfig(BaseSettings):
    """Security and access control configuration."""

    model_config = SettingsConfigDict(
        env_prefix="SECURITY_", extra="ignore", enable_decoding=False, env_file=".env"
    )

    allow_write_operations: bool = Field(
        default=False, description="Allow write operations (INSERT, UPDATE, DELETE)"
    )
    blocked_functions: list[str] = Field(
        default_factory=lambda: [
            "pg_sleep",
            "pg_read_file",
            "pg_write_file",
            "lo_import",
            "lo_export",
        ],
        description="List of blocked PostgreSQL functions",
    )
    max_rows: int = Field(default=10000, ge=1, le=100000, description="Maximum rows to return")
    max_execution_time: float = Field(
        default=30.0, ge=1.0, le=300.0, description="Maximum query execution time in seconds"
    )
    readonly_role: str | None = Field(
        default=None, description="PostgreSQL role to switch to for read-only access"
    )
    safe_search_path: str = Field(
        default="public", description="Safe search_path to set during query execution"
    )
    blocked_tables: list[str] = Field(
        default_factory=list,
        description="Tables that queries may not reference (comma-separated env)",
    )
    blocked_columns: list[str] = Field(
        default_factory=list,
        description="Columns that queries may not reference (comma-separated env)",
    )
    allow_explain: bool = Field(default=False, description="Allow EXPLAIN statements in queries")

    @field_validator("blocked_functions", mode="before")
    @classmethod
    def parse_blocked_functions(cls, v: str | list[str]) -> list[str]:
        """Parse comma-separated string or list."""
        if isinstance(v, str):
            return [f.strip() for f in v.split(",") if f.strip()]
        return v

    @field_validator("blocked_tables", "blocked_columns", mode="before")
    @classmethod
    def parse_blocked_objects(cls, v: str | list[str]) -> list[str]:
        """Parse comma-separated string or list of table/column names."""
        if isinstance(v, str):
            return [f.strip() for f in v.split(",") if f.strip()]
        return v


class ValidationConfig(BaseSettings):
    """Query validation configuration."""

    model_config = SettingsConfigDict(env_prefix="VALIDATION_", extra="ignore", env_file=".env")

    max_question_length: int = Field(
        default=10000, ge=1, le=50000, description="Maximum question length in characters"
    )
    min_confidence_score: int = Field(
        default=70, ge=0, le=100, description="Minimum confidence score (0-100)"
    )

    # Result validation settings
    enabled: bool = Field(default=True, description="Enable result validation using LLM")
    sample_rows: int = Field(
        default=5, ge=1, le=100, description="Number of sample rows to include in validation"
    )
    timeout_seconds: float = Field(
        default=10.0, ge=1.0, le=60.0, description="Result validation timeout in seconds"
    )
    confidence_threshold: int = Field(
        default=70, ge=0, le=100, description="Minimum confidence for acceptable results"
    )


class CacheConfig(BaseSettings):
    """Schema cache configuration."""

    model_config = SettingsConfigDict(env_prefix="CACHE_", extra="ignore", env_file=".env")

    schema_ttl: int = Field(
        default=3600, ge=60, le=86400, description="Schema cache TTL in seconds"
    )
    max_size: int = Field(default=100, ge=1, le=1000, description="Maximum cache entries")
    enabled: bool = Field(default=True, description="Enable schema caching")


class ResilienceConfig(BaseSettings):
    """Resilience and fault tolerance configuration."""

    model_config = SettingsConfigDict(env_prefix="RESILIENCE_", extra="ignore", env_file=".env")

    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")
    retry_delay: float = Field(
        default=1.0, ge=0.1, le=10.0, description="Initial retry delay in seconds"
    )
    backoff_factor: float = Field(
        default=2.0, ge=1.0, le=10.0, description="Exponential backoff factor"
    )
    circuit_breaker_threshold: int = Field(
        default=5, ge=1, le=100, description="Failures before circuit opens"
    )
    circuit_breaker_timeout: float = Field(
        default=60.0, ge=10.0, le=300.0, description="Circuit breaker timeout in seconds"
    )
    query_rate_limit: int = Field(
        default=10, ge=1, le=1000, description="Max concurrent query executions"
    )
    llm_rate_limit: int = Field(default=5, ge=1, le=1000, description="Max concurrent LLM calls")
    rate_limit_timeout: float = Field(
        default=5.0,
        ge=0.1,
        le=300.0,
        description="Seconds to wait for a rate limiter slot before failing",
    )


class ObservabilityConfig(BaseSettings):
    """Observability and monitoring configuration."""

    model_config = SettingsConfigDict(env_prefix="OBSERVABILITY_", extra="ignore", env_file=".env")

    metrics_enabled: bool = Field(default=True, description="Enable Prometheus metrics")
    metrics_port: int = Field(
        default=9090, ge=1024, le=65535, description="Metrics HTTP server port"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )
    log_format: Literal["json", "text"] = Field(default="text", description="Log format")


class Settings(BaseSettings):
    """Main application settings aggregating all config sections."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        enable_decoding=False,
    )

    environment: Literal["development", "staging", "production"] = Field(
        default="development", description="Application environment"
    )

    # Nested configurations read .env themselves (see their model_config) —
    # without env_file declared there they would only see OS environment
    # variables and .env values like DATABASE_PASSWORD would be ignored.
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    database_extra: list[DatabaseConfig] = Field(
        default_factory=list,
        validation_alias=AliasChoices("database_extra", "DATABASE_EXTRA_JSON"),
        description=(
            "Additional databases via DATABASE_EXTRA_JSON env var (JSON array of "
            "DatabaseConfig objects). Fields omitted from a JSON entry fall back to "
            "the DATABASE_* environment variables / global defaults."
        ),
    )
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    resilience: ResilienceConfig = Field(default_factory=ResilienceConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    @field_validator("database_extra", mode="before")
    @classmethod
    def parse_database_extra(
        cls, v: str | list[dict[str, Any]] | None
    ) -> list[dict[str, Any]] | None:
        """Parse DATABASE_EXTRA_JSON env var (JSON string) or pass through a list."""
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
            except json.JSONDecodeError as e:
                raise ValueError(f"DATABASE_EXTRA_JSON is not valid JSON: {e}") from e
            if not isinstance(parsed, list):
                raise ValueError("DATABASE_EXTRA_JSON must be a JSON array of objects")
            return parsed
        return v

    @property
    def all_databases(self) -> list[DatabaseConfig]:
        """All configured databases, primary first.

        Extra databases whose name collides with the primary database (or with
        an earlier extra) are skipped to keep pool keys unique.
        """
        seen = {self.database.name}
        result = [self.database]
        for extra in self.database_extra:
            if extra.name not in seen:
                seen.add(extra.name)
                result.append(extra)
        return result

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create global settings instance.

    Returns:
        Settings: The global settings instance.
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset global settings instance. Useful for testing."""
    global _settings
    _settings = None
