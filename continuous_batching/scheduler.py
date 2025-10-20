from .schemas import RequestSchema, BaseBatchRequestSchema, BatchParamsCompatibility, Priority
from typing import Tuple, Optional, List, Dict
from loguru import logger
import asyncio
from collections import deque


class BatchRequest(BaseBatchRequestSchema):
    def validate_request(self, request: RequestSchema) -> Tuple[bool, str]:
        """Validate if a request can be added to the batch."""

        # Check if adding the request would exceed max batch size
        if len(self.requests) >= self.max_batch_size:
            message = f"Cannot add more requests to batch {self.batch_id}: max batch size: {self.max_batch_size} reached."
            return False, message

        # Check if there are no existing requests to compare with
        if not self.requests:
            return True, ""

        # Check if the generation parameters of the request match to other requests in the batch
        sample_existing_params = self.requests[0].params
        # Check temperature
        if abs(sample_existing_params.temperature - request.params.temperature) >= self.compatibility_params.temperature:
            message = f"Request {request.request_id} temperature {request.params.temperature} is not compatible with batch {self.batch_id} (existing temperature: {sample_existing_params.temperature})"
            return False, message
        # Check top_p
        if abs(sample_existing_params.top_p - request.params.top_p) >= self.compatibility_params.top_p:
            message = f"Request {request.request_id} top_p {request.params.top_p} is not compatible with batch {self.batch_id} (existing top_p: {sample_existing_params.top_p})"
            return False, message

        return True, ""

    def add_request(self, request: RequestSchema) -> bool:
        """Add a request to the batch."""
        # Validate the request before adding it to the batch
        is_valid, message = self.validate_request(request)
        if not is_valid:
            raise ValueError(message)

        # Update batch-level parameters
        self.max_tokens = max(self.max_tokens, request.params.max_tokens)
        self.priority = max(self.priority, request.priority)

        self.requests.append(request)

        return True

    def remove_request(self, request_id: str) -> bool:
        """Remove a request from the batch."""
        for i, req in enumerate(self.requests):
            if req.request_id == request_id:
                self.requests.pop(i)
                return True
        return False


class RequestScheduler:
    """Continuous batching request scheduler. The main workflow are:

    1. Initialize: active_batches = []
    2. While system is running:
        1. For each request in active_batches:
            - Generate next token
            - If request completed, remove from active_batches
        2. While active_batches has capacity AND waiting_queue is not empty:
            - Pop request from waiting_queue
            - Add request to active_batches
        3. Proceed to next iteration

    Args:
        model: The model to use for processing requests.
        max_processing_batches (Optional[int]): Maximum number of batches to process concurrently. Defaults to 3.
        max_queue_size (Optional[int]): Maximum number of requests allowed in the queue. Defaults to 1000.
        max_batch_size (Optional[int]): Maximum number of requests allowed in a batch. Defaults to 8.
        max_tokens_per_batch (Optional[int]): Maximum token length allowed in a batch. Defaults to None (no limit).
        temperature_compatibility (Optional[float]): Maximum allowed temperature difference within a batch. Defaults to 0.1.
        top_p_compatibility (Optional[float]): Maximum allowed top_p difference within a batch. Defaults to 0.1.
    """

    def __init__(
        self,
        model,
        max_processing_batches: Optional[int] = 3,
        max_queue_size: Optional[int] = 1000,
        max_batch_size: Optional[int] = 8,
        max_tokens_per_batch: Optional[int] = None,
        # Batch compatibility parameters
        temperature_compatibility: Optional[float] = None,
        top_p_compatibility: Optional[float] = None,
    ):
        self.model = model
        self.max_processing_batches = max_processing_batches
        self.max_queue_size = max_queue_size
        self.max_batch_size = max_batch_size
        self.max_tokens_per_batch = max_tokens_per_batch
        self.compatibility_params = BatchParamsCompatibility(
            temperature=temperature_compatibility if temperature_compatibility is not None else 0.1,
            top_p=top_p_compatibility if top_p_compatibility is not None else 0.1,
        )

        # Scheduler state
        self.running = False
        self.scheduler_task = None

        # Locks
        self.waiting_queue_lock = asyncio.Lock()
        self.batching_queue_lock = asyncio.Lock()

        self.waiting_queue_by_priority: Dict[Priority, deque] = {
            priority: deque() for priority in Priority
        }
        self.active_batches: List[BatchRequest] = []

        logger.info("RequestScheduler Initialized")

    async def schedule(self, request: RequestSchema) -> None:
        """Schedule a new request.

        Args:
            request: The request to schedule.
        """
        pass

    async def batching(self, queue, priority: Priority) -> None | Optional[BatchRequest]:
        """Process continuous batching queue.

        1. Acquire the batching queue lock.
        2. Get active batch for the priority.
        """

        # Skip empty queues
        if not queue:
            return None

        # TODO: Implement batching logic here

    async def process_waiting_queue(self) -> None:
        """Process the waiting queue:

        1. Acquire the waiting queue lock.
        2. Process requests in priority order.
        """

        with self.waiting_queue_lock:
            for priority in sorted(self.waiting_queue_by_priority.keys(), reverse=True):
                waiting_queue = self.waiting_queue_by_priority[priority]

                # Skip empty queues
                if not waiting_queue:
                    continue

                # TODO: Process requests in the current priority queue

    async def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.running:
            try:
                await self._process_queues()
                # Small delay to prevent busy waiting
                # await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                # await asyncio.sleep(0.1)

    async def start(self):
        """Start the continuous batching scheduler."""
        logger.info("Starting RequestScheduler...")
        self.running = True

        self.scheduler_task = asyncio.create_task(self._scheduler_loop())

        logger.info("RequestScheduler Started.")
