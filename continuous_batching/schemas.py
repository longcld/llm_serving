from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum, IntEnum
from abc import ABC, abstractmethod
import uuid


@dataclass
class GenerationParams:
    """Parameters for text generation."""
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.1
    stop_sequences: List[str] = field(default_factory=list)
    stream: bool = False
    seed: Optional[int] = None


class Priority(IntEnum):
    LOW = 1
    NORMAL = 2
    MEDIUM = 3
    HIGH = 4
    URGENT = 5


class RequestStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class RequestSchema:
    request_id: str = field(
        metadata={
            "description": "Unique identifier for the request"
        }
    )

    prompt: str = field(
        metadata={
            "description": "The input prompt for the model"
        }
    )

    params: GenerationParams = field(
        default_factory=GenerationParams,
        metadata={
            "description": "Generation parameters for the request"
        }
    )

    priority: Priority = field(
        default=Priority.NORMAL,
        metadata={
            "description": "Priority level for the request"
        }
    )

    status: RequestStatus = field(
        default=RequestStatus.PENDING,
        metadata={
            "description": "Current status of the request"
        }
    )

    created_at: Optional[float] = field(
        default=None,
        metadata={
            "description": "Timestamp when the request was created"
        }
    )

    completed_at: Optional[float] = field(
        default=None,
        metadata={
            "description": "Timestamp when the request was completed"
        }
    )

    def __post_init__(self):
        if not self.request_id:
            self.request_id = str(uuid.uuid4())


@dataclass
class BatchParamsCompatibility:
    """Parameters to ensure compatibility within a batch."""
    temperature: float = field(
        default=0.05,
        metadata={
            "description": "max temperature difference allowed within the batch"
        }
    )
    top_p: float = field(
        default=0.05,
        metadata={
            "description": "max top_p difference allowed within the batch"
        }
    )
    # top_k: int = field(
    #     default=0,
    #     metadata={
    #         "description": "max top_k difference allowed within the batch"
    #     }
    # )
    # repetition_penalty: float = field(
    #     default=0.05,
    #     metadata={
    #         "description": "max repetition_penalty difference allowed within the batch"
    #     }
    # )


@dataclass
class BaseBatchRequestSchema(ABC):
    batch_id: str = field(
        metadata={
            "description": "Unique identifier for the batch request"
        }
    )
    requests: List[RequestSchema] = field(
        metadata={
            "description": "List of individual request schemas"
        }
    )
    priority: Priority = field(
        default=Priority.NORMAL,
        metadata={
            "description": "Priority level for the batch request"
        }
    )
    max_batch_size: int = field(
        default=8,
        metadata={
            "description": "Maximum number of requests allowed in the batch"
        }
    )
    compatibility_params: BatchParamsCompatibility = field(
        default_factory=BatchParamsCompatibility,
        metadata={
            "description": "Parameters to ensure compatibility within the batch"
        }
    )
    max_tokens: int = field(
        default=0,
        metadata={
            "description": "Maximum token length among all requests in the batch"
        }
    )
    created_at: Optional[float] = field(
        default=None,
        metadata={
            "description": "Timestamp when the batch request was created"
        }
    )

    def __post_init__(self):
        if not self.batch_id:
            self.batch_id = str(uuid.uuid4())

        if self.max_tokens == 0 and self.requests:
            self.max_tokens = max(
                request.params.max_tokens for request in self.requests)

    @abstractmethod
    def validate_request(self, request: RequestSchema) -> Tuple[bool, str]:
        """Validate if a request can be added to the batch.

        Args:
            request: The request to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        pass

    @abstractmethod
    def add_request(self, request: RequestSchema) -> None:
        """Add a request to the batch.

        Args:
            request: The request to add to the batch

        Raises:
            ValueError: If the request cannot be added to the batch
        """
        pass

    @abstractmethod
    def remove_request(self, request_id: str) -> bool:
        """Remove a request from the batch.

        Args:
            request_id: The ID of the request to remove

        Returns:
            True if the request was found and removed, False otherwise
        """
        pass
