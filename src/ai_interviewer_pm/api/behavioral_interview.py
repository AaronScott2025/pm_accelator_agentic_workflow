"""API endpoints for behavioral interview system with proper session management."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langgraph.pregel import Pregel

from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from ai_interviewer_pm.agents.behavioral_graph import compile_behavioral_interview_graph
from ai_interviewer_pm.agents.behavioral_schema import (
    BehavioralInterviewState,
    InterviewSession,
)

router = APIRouter(prefix="/api/v1/behavioral", tags=["behavioral-interview"])

# Global app instance - in production, use dependency injection
_interview_app = None


def get_interview_app() -> Pregel:
    """Get compiled interview application singleton."""
    global _interview_app
    if _interview_app is None:
        _interview_app = compile_behavioral_interview_graph()
    return _interview_app


class StartInterviewRequest(BaseModel):
    """Request to start a new behavioral interview session."""

    candidate_name: str | None = Field(None, description="Optional candidate name")
    target_level: str = Field("mid", description="Target seniority: junior, mid, senior")
    total_questions: int = Field(5, ge=1, le=10, description="Total planned questions")


class StartInterviewResponse(BaseModel):
    """Response when starting a new interview."""

    session_id: str
    message: str
    next_action: str


class SubmitResponseRequest(BaseModel):
    """Request to submit a candidate response."""

    session_id: str
    response: str = Field(
        ..., min_length=10, description="Candidate's response to current question"
    )


class SubmitResponseResponse(BaseModel):
    """Response after submitting candidate response."""

    session_id: str
    message: str | None
    evaluation: dict[str, Any] | None
    next_action: str
    interview_completed: bool


class InterviewStatusResponse(BaseModel):
    """Current interview session status."""

    session_id: str
    current_question: str | None
    progress: str
    stage: str
    messages_count: int


@router.post("/start", response_model=StartInterviewResponse)
async def start_interview(request: StartInterviewRequest) -> StartInterviewResponse:
    """Start a new behavioral interview session."""
    try:
        # Create new session
        session_id = f"session_{uuid.uuid4().hex[:12]}"

        session = InterviewSession(
            session_id=session_id,
            candidate_name=request.candidate_name,
            target_level=request.target_level,  # type: ignore
            total_planned_questions=request.total_questions,
            interview_stage="introduction",
        )

        # Initialize state
        initial_state: BehavioralInterviewState = {
            "messages": [],
            "session": session,
            "question_pool": [],
            "current_question": None,
            "current_answer": None,
            "evaluation": None,
            "follow_up_questions": [],
            "follow_up_count": 0,
            "max_follow_ups": 2,
            "display_followups": [],
            "retrieved_context": [],
            "web_search_results": [],
            "next_action": "generate_questions",
            "config": {"model_temperature": 0.3},
            "error_state": None,
            "retry_count": 0,
        }

        # Run the graph to generate questions and start interview
        app = get_interview_app()
        config = {"configurable": {"thread_id": session_id}}

        result = await app.ainvoke(initial_state, config=config)

        # Extract the introduction message
        messages = result.get("messages", [])
        intro_message = messages[-1].content if messages else "Interview started successfully!"

        return StartInterviewResponse(
            session_id=session_id,
            message=intro_message,
            next_action=result.get("next_action", "wait_for_response"),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start interview: {str(e)}") from e


@router.post("/respond", response_model=SubmitResponseResponse)
async def submit_response(request: SubmitResponseRequest) -> SubmitResponseResponse:
    """Submit candidate response and get next question or feedback."""
    try:
        app = get_interview_app()
        config = {"configurable": {"thread_id": request.session_id}}

        # Get current state
        current_state = await app.aget_state(config)
        if not current_state.values:
            raise HTTPException(status_code=404, detail="Interview session not found")

        # Add the candidate's response
        state = dict(current_state.values)
        state["current_answer"] = request.response

        # Add human message to conversation
        human_msg = HumanMessage(content=request.response)
        if "messages" not in state:
            state["messages"] = []
        state["messages"].append(human_msg)

        # Process the response through the graph
        result = await app.ainvoke(state, config=config)

        # Extract response information
        messages = result.get("messages", [])
        latest_ai_message = None

        # Find the latest AI message
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.__class__.__name__ == "AIMessage":
                latest_ai_message = msg.content
                break

        evaluation_data = result.get("evaluation")
        evaluation_dict = None
        if evaluation_data:
            if hasattr(evaluation_data, "model_dump"):
                evaluation_dict = evaluation_data.model_dump()
            else:
                evaluation_dict = (
                    dict(evaluation_data) if isinstance(evaluation_data, dict) else None
                )

        # Check if interview is completed
        next_action = result.get("next_action", "wait_for_response")
        is_completed = (
            next_action == "conclude"
            or result.get("session", {}).get("interview_stage") == "wrap_up"
        )

        return SubmitResponseResponse(
            session_id=request.session_id,
            message=latest_ai_message,
            evaluation=evaluation_dict,
            next_action=next_action,
            interview_completed=is_completed,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process response: {str(e)}") from e


@router.get("/status/{session_id}", response_model=InterviewStatusResponse)
async def get_interview_status(session_id: str) -> InterviewStatusResponse:
    """Get current status of an interview session."""
    try:
        app = get_interview_app()
        config = {"configurable": {"thread_id": session_id}}

        current_state = await app.aget_state(config)
        if not current_state.values:
            raise HTTPException(status_code=404, detail="Interview session not found")

        state = current_state.values
        session_data = state.get("session", {})
        current_question = state.get("current_question")

        # Calculate progress
        current_index = session_data.get("current_question_index", 0)
        total_questions = len(state.get("question_pool", []))
        progress = f"{current_index}/{total_questions}" if total_questions > 0 else "0/0"

        # Get current question text
        question_text = None
        if current_question and hasattr(current_question, "text"):
            question_text = current_question.text
        elif current_question and isinstance(current_question, dict):
            question_text = current_question.get("text")

        return InterviewStatusResponse(
            session_id=session_id,
            current_question=question_text,
            progress=progress,
            stage=session_data.get("interview_stage", "unknown"),
            messages_count=len(state.get("messages", [])),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}") from e


@router.delete("/session/{session_id}")
async def end_interview_session(session_id: str) -> dict[str, str]:
    """End and clean up an interview session."""
    try:
        # In production, you'd clean up the checkpoint data here
        # For now, just return success
        return {"message": f"Interview session {session_id} ended successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to end session: {str(e)}") from e


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}
