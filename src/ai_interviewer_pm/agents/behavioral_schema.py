"""Enhanced state schema for behavioral interview system using 2025 LangGraph best practices."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field


class BehavioralQuestion(BaseModel):
    """Structured behavioral question with metadata."""

    id: str = Field(description="Unique identifier for the question")
    text: str = Field(description="The actual question text")
    category: Literal[
        "leadership",
        "conflict_resolution",
        "prioritization",
        "stakeholder_management",
        "product_decisions",
        "failure_recovery",
    ] = Field(description="Question category")
    difficulty: Literal["junior", "mid", "senior"] = Field(description="Target seniority level")
    expected_duration: int = Field(description="Expected response time in minutes", default=5)
    follow_up_strategy: Literal["deep_dive", "metrics", "stakeholders", "alternatives"] = Field(
        description="Strategy for follow-up questions"
    )


class ResponseEvaluation(BaseModel):
    """Structured evaluation of candidate response."""

    completeness_score: float = Field(ge=0, le=10, description="How complete is the STAR structure")
    clarity_score: float = Field(ge=0, le=10, description="Clarity and communication quality")
    depth_score: float = Field(ge=0, le=10, description="Depth of insight and analysis")
    impact_score: float = Field(ge=0, le=10, description="Business impact demonstrated")
    leadership_score: float = Field(ge=0, le=10, description="Leadership qualities shown")
    overall_score: float = Field(ge=0, le=10, description="Overall response quality")
    key_strengths: list[str] = Field(description="Identified strengths in the response")
    improvement_areas: list[str] = Field(description="Areas needing improvement")
    follow_up_needed: bool = Field(description="Whether follow-up questions are warranted")
    contexts: list[dict[str, Any]] = Field(
        default_factory=list, description="Retrieved contexts used for evaluation"
    )


class InterviewSession(BaseModel):
    """Interview session metadata and progress tracking."""

    session_id: str = Field(description="Unique session identifier")
    candidate_name: str | None = Field(description="Candidate name if provided", default=None)
    target_level: Literal["junior", "mid", "senior"] = Field(description="Target seniority level")
    start_time: datetime = Field(default_factory=datetime.now)
    current_question_index: int = Field(description="Index of current question", default=0)
    questions_completed: int = Field(description="Number of questions completed", default=0)
    total_planned_questions: int = Field(description="Total planned questions", default=5)
    interview_stage: Literal[
        "introduction", "questioning", "follow_up", "evaluation", "wrap_up"
    ] = Field(description="Current interview stage", default="introduction")


class BehavioralInterviewState(TypedDict):
    """Enhanced state for behavioral interview system following 2025 LangGraph patterns."""

    # Conversation management using LangGraph's message annotation pattern
    messages: Annotated[list[BaseMessage], add_messages]

    # Session and flow management
    session: InterviewSession

    # Question management
    question_pool: list[BehavioralQuestion]
    current_question: BehavioralQuestion | None

    # Response handling
    current_answer: str | None
    template_answer: str | None  # Template/example answer for current question
    evaluation: ResponseEvaluation | None
    improvement_tips: list[str]  # Specific tips for improving the answer
    refinement_count: int  # Track how many times answer has been refined

    # Follow-up management
    follow_up_questions: list[str]
    follow_up_count: int
    max_follow_ups: int  # Configurable limit
    display_followups: list[str]  # Follow-up questions for display purposes

    # Context and retrieval
    retrieved_context: list[dict[str, Any]]
    web_search_results: list[dict[str, Any]]

    # NEW: Enhanced evaluation fields
    coaching_patterns: dict[str, Any] | None  # Dr. Nancy's coaching patterns
    coaching_feedback: dict[str, Any] | None  # Dr. Nancy's coaching feedback
    grail_evaluation: dict[str, Any] | None  # GRAIL rubric evaluation
    agent_evaluations: list[dict[str, Any]]  # Multi-agent evaluations
    consensus_evaluation: dict[str, Any] | None  # Consensus from agents

    # NEW: Adaptive questioning fields
    performance_metrics: dict[str, Any] | None  # Performance tracking
    adaptive_decision: dict[str, Any] | None  # Adaptive system decision
    evaluation_history: list[ResponseEvaluation]  # History for adaptation
    next_question_override: BehavioralQuestion | None  # Override from adaptive

    # NEW: Iteration and recursion control
    node_iterations: dict[str, int]  # Track iterations per node
    max_node_iterations: dict[str, int]  # Max iterations per node
    graph_recursion_depth: int  # Current recursion depth
    max_recursion_depth: int  # Maximum allowed recursion

    # Flow control
    next_action: Literal[
        "generate_questions",
        "ask_question",
        "wait_for_response",
        "evaluate_response",
        "ask_follow_up",
        "move_to_next",
        "conclude",
        "evaluation_complete",
        "multi_agent_eval",  # NEW
        "grail_eval",  # NEW
        "adaptive_selection",  # NEW
    ]

    # Configuration
    config: dict[str, Any]  # Runtime configuration

    # Error handling
    error_state: dict[str, Any] | None
    retry_count: int

    # NEW: Backup data for persistence
    evaluation_backup: dict[str, Any] | None  # Backup evaluation data


def validate_interview_state(state: dict[str, Any]) -> BehavioralInterviewState:
    """Validate and normalize interview state with robust error handling."""
    # Implementation follows your existing pattern but enhanced for behavioral interviews
    try:
        # Validate session data
        session_data = state.get("session", {})
        if not isinstance(session_data, InterviewSession):
            session = (
                InterviewSession(**session_data)
                if session_data
                else InterviewSession(
                    session_id=f"session_{datetime.now().isoformat()}", target_level="mid"
                )
            )
        else:
            session = session_data

        # Ensure required fields are present with defaults
        normalized_state: BehavioralInterviewState = {
            "messages": state.get("messages", []),
            "session": session,
            "question_pool": state.get("question_pool", []),
            "current_question": state.get("current_question"),
            "current_answer": state.get("current_answer"),
            "evaluation": state.get("evaluation"),
            "follow_up_questions": state.get("follow_up_questions", []),
            "follow_up_count": state.get("follow_up_count", 0),
            "max_follow_ups": state.get("max_follow_ups", 2),
            "display_followups": state.get("display_followups", []),
            "retrieved_context": state.get("retrieved_context", []),
            "web_search_results": state.get("web_search_results", []),
            "next_action": state.get("next_action", "generate_questions"),
            "config": state.get("config", {}),
            "error_state": state.get("error_state"),
            "retry_count": state.get("retry_count", 0),
            "improvement_tips": state.get("improvement_tips", []),
            "refinement_count": state.get("refinement_count", 0),
            "template_answer": state.get("template_answer"),
            # NEW: Enhanced evaluation fields
            "coaching_patterns": state.get("coaching_patterns"),
            "grail_evaluation": state.get("grail_evaluation"),
            "agent_evaluations": state.get("agent_evaluations", []),
            "consensus_evaluation": state.get("consensus_evaluation"),
            # NEW: Adaptive questioning fields
            "performance_metrics": state.get("performance_metrics"),
            "adaptive_decision": state.get("adaptive_decision"),
            "evaluation_history": state.get("evaluation_history", []),
            "next_question_override": state.get("next_question_override"),
            # NEW: Iteration and recursion control
            "node_iterations": state.get("node_iterations", {}),
            "max_node_iterations": state.get(
                "max_node_iterations",
                {
                    "response_evaluator": 3,
                    "follow_up_generator": 3,
                    "context_retrieval": 2,
                    "multi_agent_evaluator": 2,
                },
            ),
            "graph_recursion_depth": state.get("graph_recursion_depth", 0),
            "max_recursion_depth": state.get("max_recursion_depth", 50),
            # Backup data
            "evaluation_backup": state.get("evaluation_backup"),
        }

        return normalized_state

    except Exception as e:
        # Fail fast with clear error messaging
        raise ValueError(f"Invalid interview state: {e}") from e
