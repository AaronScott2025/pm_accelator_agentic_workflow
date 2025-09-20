from __future__ import annotations

import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ai_interviewer_pm.agents.behavioral_graph import compile_behavioral_interview_graph
from ai_interviewer_pm.agents.behavioral_schema import (
    BehavioralInterviewState,
    BehavioralQuestion,
    InterviewSession,
    validate_interview_state,
)
from ai_interviewer_pm.api.evaluation import llm_as_judge, try_ragas_single
from ai_interviewer_pm.api.models import (
    AdaptiveDecisionResult,
    AgentEvaluationResult,
    BehavioralContinueRequest,
    BehavioralContinueResponse,
    BehavioralQuestionResponse,
    BehavioralStartRequest,
    BehavioralStartResponse,
    BehavioralSubmitRequest,
    BehavioralSubmitResponse,
    CoachingFeedback,
    ConsensusEvaluationResult,
    EvaluateResponse,
    GRAILEvaluationResult,
    GRAILScoreDetail,
    JudgeResult,
)

app = FastAPI(title="AI Interviewer PM API", version="0.1.0")

@app.get("/")
def root():
    return {"message": "FastAPI is working!"}

# CORS: allow local Next.js dev by default
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_behavioral_graph_app = compile_behavioral_interview_graph()  # For behavioral interviews


def create_initial_behavioral_state(
    session_id: str, total_questions: int, difficulty: str = "mid"
) -> BehavioralInterviewState:
    """Create initial state for behavioral interview graph."""
    from datetime import datetime

    session = InterviewSession(
        session_id=session_id,
        target_level=difficulty,
        total_planned_questions=total_questions,
        start_time=datetime.now(),
        interview_stage="introduction",
    )

    state: BehavioralInterviewState = {
        "messages": [],
        "session": session,
        "question_pool": [],
        "current_question": None,
        "current_answer": None,
        "template_answer": None,
        "evaluation": None,
        "improvement_tips": [],
        "refinement_count": 0,
        "follow_up_questions": [],
        "follow_up_count": 0,
        "max_follow_ups": 2,
        "display_followups": [],
        "retrieved_context": [],
        "web_search_results": [],
        # Enhanced evaluation fields
        "coaching_patterns": None,
        "coaching_feedback": None,
        "grail_evaluation": None,
        "agent_evaluations": [],
        "consensus_evaluation": None,
        # Adaptive questioning fields
        "performance_metrics": None,
        "adaptive_decision": None,
        "evaluation_history": [],
        "next_question_override": None,
        # Iteration control
        "node_iterations": {},
        "max_node_iterations": {},
        "graph_recursion_depth": 0,
        "max_recursion_depth": 50,
        # Flow control
        "next_action": "generate_questions",
        "config": {},
        "error_state": None,
        "retry_count": 0,
        "evaluation_backup": None,
    }

    return state


def behavioral_question_to_api_model(question: BehavioralQuestion, index: int, total: int) -> BehavioralQuestionResponse:
    """Convert behavioral graph question to API response model."""
    return BehavioralQuestionResponse(
        question=question.text,
        category=question.category,
        difficulty=question.difficulty,
        question_index=index,
        total_questions=total,
    )


def extract_evaluation_from_state(state: BehavioralInterviewState) -> EvaluateResponse:
    """Extract evaluation data from behavioral graph state including new enhanced evaluations."""
    evaluation = state.get("evaluation")
    evaluation_backup = state.get("evaluation_backup")
    print(f"DEBUG extract_evaluation_from_state: evaluation = {evaluation}")
    print(f"DEBUG extract_evaluation_from_state: evaluation_backup = {evaluation_backup}")
    print(f"DEBUG extract_evaluation_from_state: state keys = {list(state.keys())}")
    print(f"DEBUG extract_evaluation_from_state: improvement_tips = {state.get('improvement_tips', [])}")
    print(f"DEBUG extract_evaluation_from_state: display_followups = {state.get('display_followups', [])}")

    # Use backup evaluation data if primary evaluation is missing
    feedback = ""
    rubric_score = {}

    if evaluation_backup:
        # Use the backup evaluation data
        feedback = evaluation_backup.get("feedback", "")
        rubric_score = evaluation_backup.get("rubric_score", {})
    elif evaluation:
        # Use primary evaluation data
        rubric_score = {
            "clarity": evaluation.clarity_score,
            "structure": evaluation.completeness_score,
            "depth": evaluation.depth_score,
            "impact": evaluation.impact_score,
            "leadership": evaluation.leadership_score,
            "overall": evaluation.overall_score,
            "summary": f"Overall score: {evaluation.overall_score:.1f}/10. Key strengths: {', '.join(evaluation.key_strengths[:2])}",
        }

        feedback = f"""**Evaluation Summary**

**Strengths:**
{chr(10).join(f"• {strength}" for strength in evaluation.key_strengths)}

**Areas for Improvement:**
{chr(10).join(f"• {area}" for area in evaluation.improvement_areas)}

**Detailed Scores:**
• Clarity & Communication: {evaluation.clarity_score:.1f}/10
• STAR Structure: {evaluation.completeness_score:.1f}/10
• Depth of Analysis: {evaluation.depth_score:.1f}/10
• Business Impact: {evaluation.impact_score:.1f}/10
• Leadership Qualities: {evaluation.leadership_score:.1f}/10

**Overall Assessment:** {evaluation.overall_score:.1f}/10"""
    else:
        # Fallback: Generate basic feedback from improvement tips if evaluation is missing
        improvement_tips = state.get("improvement_tips", [])
        if improvement_tips:
            feedback = f"""**Evaluation Summary**

**Areas for Improvement:**
{chr(10).join(f"• {tip}" for tip in improvement_tips[:3])}

**Note:** This is a simplified evaluation. For detailed scoring, please try again."""

    # Extract GRAIL evaluation if available
    grail_evaluation = None
    grail_data = state.get("grail_evaluation")
    if grail_data and isinstance(grail_data, dict):
        try:
            grail_evaluation = GRAILEvaluationResult(
                goal_score=GRAILScoreDetail(**grail_data["goal_score"]),
                resources_score=GRAILScoreDetail(**grail_data["resources_score"]),
                actions_score=GRAILScoreDetail(**grail_data["actions_score"]),
                impact_score=GRAILScoreDetail(**grail_data["impact_score"]),
                learning_score=GRAILScoreDetail(**grail_data["learning_score"]),
                overall_score=grail_data["overall_score"],
                overall_assessment=grail_data["overall_assessment"],
                pm_competency_mapping=grail_data.get("pm_competency_mapping", {})
            )
        except Exception as e:
            print(f"DEBUG: Failed to parse GRAIL evaluation: {e}")

    # Extract consensus evaluation if available
    consensus_evaluation = None
    consensus_data = state.get("consensus_evaluation")
    if consensus_data and isinstance(consensus_data, dict):
        try:
            agent_evals = [
                AgentEvaluationResult(**agent_data) 
                for agent_data in consensus_data.get("agent_evaluations", [])
            ]
            consensus_evaluation = ConsensusEvaluationResult(
                final_score=consensus_data["final_score"],
                confidence=consensus_data["confidence"],
                agent_evaluations=agent_evals,
                consensus_strengths=consensus_data.get("consensus_strengths", []),
                consensus_improvements=consensus_data.get("consensus_improvements", []),
                divergent_opinions=consensus_data.get("divergent_opinions", {}),
                recommendation=consensus_data.get("recommendation", "")
            )
        except Exception as e:
            print(f"DEBUG: Failed to parse consensus evaluation: {e}")

    # Extract adaptive decision if available
    adaptive_decision = None
    adaptive_data = state.get("adaptive_decision")
    if adaptive_data and isinstance(adaptive_data, dict):
        try:
            adaptive_decision = AdaptiveDecisionResult(
                action=adaptive_data["action"],
                next_question_id=adaptive_data.get("next_question_id"),
                reasoning=adaptive_data["reasoning"],
                difficulty_adjustment=adaptive_data["difficulty_adjustment"],
                focus_area=adaptive_data.get("focus_area")
            )
        except Exception as e:
            print(f"DEBUG: Failed to parse adaptive decision: {e}")

    # Extract coaching feedback if available
    coaching_feedback = None
    coaching_data = state.get("coaching_feedback")
    if coaching_data and isinstance(coaching_data, dict):
        try:
            coaching_feedback = CoachingFeedback(
                feedback_text=coaching_data.get("feedback_text", ""),
                coaching_patterns=coaching_data.get("coaching_patterns", {}),
                encouragement_message=coaching_data.get("encouragement_message", ""),
                adapted_followups=coaching_data.get("adapted_followups", [])
            )
        except Exception as e:
            print(f"DEBUG: Failed to parse coaching feedback: {e}")

    return EvaluateResponse(
        feedback=feedback,
        rubric_score=rubric_score,
        followups=state.get("display_followups", [])[:3],  # Display follow-ups for frontend
        template_answer=state.get("template_answer"),
        contexts=state.get("retrieved_context"),
        ragas=None,  # Set later if requested
        judge=None,  # Set later if requested
        grail_evaluation=grail_evaluation,
        consensus_evaluation=consensus_evaluation,
        adaptive_decision=adaptive_decision,
        coaching_feedback=coaching_feedback
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# Note: The old /evaluate endpoint has been removed in favor of the behavioral interview system.
# Use /behavioral/start and /behavioral/submit for comprehensive interview sessions.


@app.post("/behavioral/start", response_model=BehavioralStartResponse)
def start_behavioral_interview(req: BehavioralStartRequest) -> BehavioralStartResponse:
    """Start a new behavioral interview session and return the first question."""
    session_id = str(uuid.uuid4())

    # Create initial state for behavioral graph
    state = create_initial_behavioral_state(
        session_id=session_id,
        total_questions=req.total_questions,
        difficulty=req.difficulty,
    )

    # Run behavioral graph to generate questions with recursion limit
    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": 50,  # Increase limit for complex flows
    }
    try:
        final_state = None
        for event in _behavioral_graph_app.stream(state, config=config, stream_mode="values"):
            final_state = event

        if not final_state or not final_state.get("question_pool"):
            raise HTTPException(status_code=500, detail="Failed to generate behavioral questions")

        question_pool = final_state.get("question_pool", [])
        if not question_pool:
            raise HTTPException(status_code=500, detail="No questions generated")

        # Get first question
        first_question = question_pool[0]
        question_response = behavioral_question_to_api_model(first_question, 0, len(question_pool))

        return BehavioralStartResponse(session_id=session_id, question=question_response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start behavioral interview: {str(e)}") from e


@app.post("/behavioral/submit", response_model=BehavioralSubmitResponse)
def submit_behavioral_answer(req: BehavioralSubmitRequest) -> BehavioralSubmitResponse:
    """Submit answer for current behavioral question and get evaluation."""
    session_id = req.session_id

    # Get current state from behavioral graph
    config = {"configurable": {"thread_id": session_id}}
    try:
        current_state_info = _behavioral_graph_app.get_state(config)
        if not current_state_info or not current_state_info.values:
            raise HTTPException(status_code=404, detail="Session not found")

        current_state = current_state_info.values

        # Update state with the submitted answer
        updated_state = dict(current_state)
        updated_state["current_answer"] = req.answer
        updated_state["refinement_count"] = req.refinement_count if req.is_refinement else 0
        updated_state["next_action"] = "wait_for_response"

        # Add retrieval configuration
        updated_state["config"] = {
            "k": req.options.k,
            "prefer_coach": req.options.prefer_coach,
            "use_rrf": req.options.use_rrf,
        }

        # Process the answer through the behavioral graph
        final_state = None
        for event in _behavioral_graph_app.stream(updated_state, config=config, stream_mode="values"):
            final_state = event

        if not final_state:
            raise HTTPException(status_code=500, detail="Failed to process behavioral answer")

        # Extract evaluation from the behavioral graph state
        evaluation_response = extract_evaluation_from_state(final_state)

        # Get session info for progress tracking
        session = final_state.get("session")
        question_pool = final_state.get("question_pool", [])
        current_question_index = session.current_question_index if session else 0

        # Determine next question and interview completion
        next_question = None
        interview_completed = current_question_index + 1 >= len(question_pool)

        if not interview_completed and not req.is_refinement:
            next_question_obj = question_pool[current_question_index + 1]
            next_question = behavioral_question_to_api_model(
                next_question_obj, current_question_index + 1, len(question_pool)
            )

        # Calculate refinement allowance
        evaluation = final_state.get("evaluation")
        avg_score = evaluation.overall_score if evaluation else 0
        refinement_allowed = avg_score < 7.0 and req.refinement_count < 2

        progress = {
            "questions_completed": current_question_index + (0 if req.is_refinement else 1),
            "questions_remaining": max(0, len(question_pool) - current_question_index - 1),
        }

        # Optional RAGAS and LLM-as-judge evaluation
        ragas = None
        judge = None
        if req.options.do_ragas:
            flat_contexts = [c.get("text", "") for c in evaluation_response.contexts or [] if isinstance(c, dict)]
            current_question = final_state.get("current_question")
            question_text = current_question.text if current_question else ""
            print(f"DEBUG: Computing RAGAS for question: {question_text[:100]}...")
            print(f"DEBUG: Answer length: {len(req.answer)}")
            print(f"DEBUG: Number of contexts: {len(flat_contexts)}")
            ragas = try_ragas_single(
                question_text,
                req.answer,
                flat_contexts
            )
            print(f"DEBUG: RAGAS result: {ragas}")
        if req.options.do_judge:
            current_question = final_state.get("current_question")
            question_text = current_question.text if current_question else ""
            out = llm_as_judge(
                question_text,
                req.answer,
                evaluation_response.feedback
            )
            if out is not None:
                judge = JudgeResult(score=float(out["score"]), rationale=str(out["rationale"]))

        # Update evaluation response with optional metrics
        evaluation_response.ragas = ragas
        evaluation_response.judge = judge

        # Extract performance metrics if available
        performance_metrics = None
        perf_data = final_state.get("performance_metrics")
        if perf_data and isinstance(perf_data, dict):
            performance_metrics = {
                "avg_score": perf_data.get("avg_score", 0),
                "trend": perf_data.get("trend", "stable"),
                "strengths": perf_data.get("strengths", []),
                "weaknesses": perf_data.get("weaknesses", []),
                "confidence_level": perf_data.get("confidence_level", 0),
                "questions_answered": perf_data.get("questions_answered", 0),
                "time_spent_minutes": perf_data.get("time_spent_minutes", 0)
            }

        return BehavioralSubmitResponse(
            session_id=session_id,
            evaluation=evaluation_response,
            next_question=next_question,
            interview_completed=interview_completed,
            progress=progress,
            refinement_allowed=refinement_allowed,
            refinement_count=req.refinement_count if req.is_refinement else 0,
            improvement_tips=final_state.get("improvement_tips", []),
            performance_metrics=performance_metrics
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit behavioral answer: {str(e)}") from e


@app.post("/behavioral/continue", response_model=BehavioralContinueResponse)
def continue_behavioral_interview(req: BehavioralContinueRequest) -> BehavioralContinueResponse:
    """Advance to next question or get final interview summary."""
    session_id = req.session_id

    # Get current state from behavioral graph
    config = {"configurable": {"thread_id": session_id}}

    try:
        # Get the latest state from the behavioral graph's checkpointer
        latest_state = _behavioral_graph_app.get_state(config)
        if not latest_state or not latest_state.values:
            raise HTTPException(status_code=404, detail="Session not found")

        current_state = latest_state.values
        session = current_state.get("session")
        question_pool = current_state.get("question_pool", [])

        if not session:
            raise HTTPException(status_code=500, detail="Invalid session state")

        # Move to next question
        next_index = session.current_question_index + 1

        # Check if interview is completed
        if next_index >= len(question_pool):
            # Calculate overall score from messages/evaluations
            # For now, use a simple completion message
            interview_summary = (
                f"Behavioral interview completed! You answered {session.questions_completed} questions. "
                f"Thank you for participating in this {session.target_level}-level PM behavioral interview."
            )

            return BehavioralContinueResponse(
                session_id=session_id,
                question=None,
                interview_summary=interview_summary,
                interview_completed=True,
                conversation_history=[],  # Could extract from messages if needed
                overall_score=None,  # Could calculate from stored evaluations
            )

        # Update session to move to next question
        updated_session = session.model_copy()
        updated_session.current_question_index = next_index
        updated_session.questions_completed += 1

        # Update state with next question
        updated_state = dict(current_state)
        updated_state["session"] = updated_session
        updated_state["current_question"] = question_pool[next_index]
        updated_state["current_answer"] = None  # Reset for new question
        updated_state["next_action"] = "ask_question"

        # Save updated state to behavioral graph
        _behavioral_graph_app.update_state(config, updated_state, as_node="session_manager")

        # Return next question
        next_question = question_pool[next_index]
        question_response = behavioral_question_to_api_model(next_question, next_index, len(question_pool))

        return BehavioralContinueResponse(
            session_id=session_id,
            question=question_response,
            interview_summary=None,
            interview_completed=False,
            conversation_history=[],  # Could extract from messages if needed
            overall_score=None,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to continue interview: {str(e)}") from e
