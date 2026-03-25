"""execution.models — Canonical execution domain types (Domain 3: core plumbing).

Venue-agnostic data models for orders, fills, acks, rejections, balances,
positions, instruments, intents, transfers, and errors. Every venue adapter
maps its raw API types into these canonicals before they enter the pipeline.
"""
from execution.models.orders import CanonicalOrder, ingress_order_dedup_identity
from execution.models.fills import CanonicalFill
from execution.models.acks import CanonicalAck, normalize_ack
from execution.models.rejections import (
    CanonicalRejection,
    ack_to_rejection,
    classify_rejection_reason,
    rejection_routing_key,
)
from execution.models.rejection_events import CanonicalRejectionEvent, rejection_to_event
from execution.models.fill_events import (
    CanonicalFillIngressEvent,
    build_ingress_fill_event,
    build_synthetic_ingress_fill_event,
    canonical_fill_to_ingress_event,
    canonical_fill_to_public_event,
    ingress_fill_dedup_identity,
)
from execution.models.commands import (
    BaseCommand, SubmitOrderCommand, CancelOrderCommand,
    make_submit_order_command, make_cancel_order_command,
)
from execution.models.balances import CanonicalBalance, BalanceSnapshot
from execution.models.errors import (
    ExecutionError, ExecutionErrorKind,
    ValidationError, MappingError, VenueError,
    InsufficientBalanceError, DuplicateError,
)
from execution.models.instruments import InstrumentInfo, InstrumentRegistry
from execution.models.intents import ExecutionIntent, IntentStatus
from execution.models.positions import VenuePosition, PositionSnapshot
from execution.models.transfers import (
    TransferRequest, TransferResult, TransferType, TransferStatus,
    FundingRecord,
)
from execution.models.venue import VenueInfo, VenueType, VenueFeature

__all__ = [
    # Orders
    "CanonicalOrder",
    "ingress_order_dedup_identity",
    # Fills
    "CanonicalFill",
    # Acks
    "CanonicalAck",
    "normalize_ack",
    # Rejections
    "CanonicalRejection",
    "ack_to_rejection",
    "classify_rejection_reason",
    "rejection_routing_key",
    "CanonicalRejectionEvent",
    "rejection_to_event",
    # Fill events
    "CanonicalFillIngressEvent",
    "build_ingress_fill_event",
    "build_synthetic_ingress_fill_event",
    "canonical_fill_to_ingress_event",
    "canonical_fill_to_public_event",
    "ingress_fill_dedup_identity",
    # Commands
    "BaseCommand",
    "SubmitOrderCommand",
    "CancelOrderCommand",
    "make_submit_order_command",
    "make_cancel_order_command",
    # Balances
    "CanonicalBalance",
    "BalanceSnapshot",
    # Errors
    "ExecutionError",
    "ExecutionErrorKind",
    "ValidationError",
    "MappingError",
    "VenueError",
    "InsufficientBalanceError",
    "DuplicateError",
    # Instruments
    "InstrumentInfo",
    "InstrumentRegistry",
    # Intents
    "ExecutionIntent",
    "IntentStatus",
    # Positions
    "VenuePosition",
    "PositionSnapshot",
    # Transfers
    "TransferRequest",
    "TransferResult",
    "TransferType",
    "TransferStatus",
    "FundingRecord",
    # Venue metadata
    "VenueInfo",
    "VenueType",
    "VenueFeature",
]
