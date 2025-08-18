from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class EvaluateOptions(BaseModel):
    """Optional switches for evaluation behavior.

    Attributes:
        k: Top-k to retrieve.
        prefer_coach: Whether to prefer coach-style sources.
        do_web_search: Whether to perform Tavily search using search_query.
        do_ragas: Whether to compute RAGAS metrics (requires packages installed).
        do_judge: Whether to compute LLM-as-judge score.
    """

    k: int = Field(
        default=5, ge=1, le=20, description="Top-k documents to retrieve before reranking."
    )
    prefer_coach: bool = Field(
        default=True, description="Prefer coach-style sources when available."
    )
    do_web_search: bool = Field(
        default=False, description="Enable Tavily web search using search_query."
    )
    do_ragas: bool = Field(
        default=True, description="Compute RAGAS metrics if dependencies are installed."
    )
    do_judge: bool = Field(default=True, description="Compute LLM-as-judge score and rationale.")
    use_rrf: bool = Field(
        default=False, description="Fuse dense and sparse rankings using Reciprocal Rank Fusion."
    )


# EvaluateRequest removed - use behavioral interview endpoints instead


class JudgeResult(BaseModel):
    """LLM-as-judge evaluation result."""
    
    score: float
    rationale: str


class GRAILScoreDetail(BaseModel):
    """Individual GRAIL component score."""
    
    score: float = Field(ge=0, le=10)
    evidence: list[str]
    missing_elements: list[str]
    strength_level: Literal["weak", "developing", "proficient", "strong", "exceptional"]


class GRAILEvaluationResult(BaseModel):
    """Complete GRAIL evaluation result."""
    
    goal_score: GRAILScoreDetail
    resources_score: GRAILScoreDetail
    actions_score: GRAILScoreDetail
    impact_score: GRAILScoreDetail
    learning_score: GRAILScoreDetail
    overall_score: float = Field(ge=0, le=10)
    overall_assessment: str
    pm_competency_mapping: dict[str, str]


class AgentEvaluationResult(BaseModel):
    """Individual agent evaluation."""
    
    agent_name: str
    score: float = Field(ge=0, le=10)
    confidence: float = Field(ge=0, le=1)
    key_observations: list[str]
    strengths: list[str]
    improvements: list[str]
    rationale: str


class ConsensusEvaluationResult(BaseModel):
    """Multi-agent consensus evaluation."""
    
    final_score: float = Field(ge=0, le=10)
    confidence: float = Field(ge=0, le=1)
    agent_evaluations: list[AgentEvaluationResult]
    consensus_strengths: list[str]
    consensus_improvements: list[str]
    divergent_opinions: dict[str, list[str]]
    recommendation: str


class AdaptiveDecisionResult(BaseModel):
    """Adaptive questioning decision."""
    
    action: Literal["continue", "adjust_difficulty", "switch_category", "deep_dive", "conclude"]
    next_question_id: str | None = None
    reasoning: str
    difficulty_adjustment: Literal["easier", "same", "harder"]
    focus_area: str | None


class CoachingFeedback(BaseModel):
    """Dr. Nancy's coaching style feedback."""
    
    feedback_text: str
    coaching_patterns: dict[str, list[str]]
    encouragement_message: str
    adapted_followups: list[str]


class EvaluateResponse(BaseModel):
    """Response including feedback, rubric, followups, and evaluation metrics."""

    feedback: str
    rubric_score: dict[str, Any]
    followups: list[str]
    template_answer: str | None = None  # Template/example answer for the question
    contexts: list[dict[str, Any]] | None = None
    ragas: dict[str, Any] | None = None
    judge: JudgeResult | None = None
    # NEW: Enhanced evaluation fields
    grail_evaluation: GRAILEvaluationResult | None = None
    consensus_evaluation: ConsensusEvaluationResult | None = None
    adaptive_decision: AdaptiveDecisionResult | None = None
    coaching_feedback: CoachingFeedback | None = None


# Behavioral Interview Models
class BehavioralStartRequest(BaseModel):
    """Request to start a new behavioral interview session."""

    total_questions: int = Field(
        default=5, ge=1, le=10, description="Total number of questions for the interview"
    )
    difficulty: Literal["entry", "mid", "senior"] = Field(
        default="mid", description="Interview difficulty level"
    )


class BehavioralQuestionResponse(BaseModel):
    """Single behavioral question with metadata."""

    question: str
    category: str
    difficulty: str
    question_index: int
    total_questions: int


class BehavioralStartResponse(BaseModel):
    """Response when starting behavioral interview with first question."""

    session_id: str
    question: BehavioralQuestionResponse
    message: str = "Behavioral interview started successfully"


class BehavioralSubmitRequest(BaseModel):
    """Request to submit answer for current behavioral question."""

    session_id: str
    answer: str = Field(
        min_length=10, description="Candidate's answer using STAR method recommended"
    )
    is_refinement: bool = Field(
        default=False, description="Whether this is a refined answer after feedback"
    )
    refinement_count: int = Field(
        default=0, ge=0, le=2, description="Number of times answer has been refined"
    )
    options: EvaluateOptions = Field(default_factory=EvaluateOptions)


class BehavioralSubmitResponse(BaseModel):
    """Response after evaluating behavioral answer."""

    session_id: str
    evaluation: EvaluateResponse
    next_question: BehavioralQuestionResponse | None = None
    interview_completed: bool = False
    progress: dict[str, Any] = Field(
        default_factory=lambda: {"questions_completed": 0, "questions_remaining": 0}
    )
    refinement_allowed: bool = Field(
        default=True, description="Whether user can refine their answer"
    )
    refinement_count: int = Field(
        default=0, description="Current refinement iteration"
    )
    improvement_tips: list[str] = Field(
        default_factory=list, description="Specific tips for improving the answer"
    )
    # NEW: Performance tracking
    performance_metrics: dict[str, Any] | None = Field(
        default=None, description="Current performance metrics for adaptive questioning"
    )


class BehavioralContinueRequest(BaseModel):
    """Request to continue to next question or get interview summary."""

    session_id: str


class BehavioralContinueResponse(BaseModel):
    """Response when continuing behavioral interview."""

    session_id: str
    question: BehavioralQuestionResponse | None = None
    interview_summary: str | None = None
    interview_completed: bool
    conversation_history: list[dict[str, Any]] = Field(default_factory=list)
    overall_score: float | None = None
