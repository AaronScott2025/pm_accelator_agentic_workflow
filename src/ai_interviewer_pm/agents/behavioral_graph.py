"""Behavioral interview graph implementation using 2025 LangGraph best practices."""

from __future__ import annotations

import json
import random
from functools import wraps
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from langgraph.pregel import Pregel

from ai_interviewer_pm.agents.adaptive_questioning import create_adaptive_selector
from ai_interviewer_pm.agents.behavioral_schema import (
    BehavioralInterviewState,
    BehavioralQuestion,
    ResponseEvaluation,
)
from ai_interviewer_pm.agents.coaching_style import create_coaching_style_handler
from ai_interviewer_pm.agents.grail_rubric import create_grail_evaluator
from ai_interviewer_pm.agents.multi_agent_evaluator import create_multi_agent_evaluator
from ai_interviewer_pm.settings import settings
from ai_interviewer_pm.tools.internet import internet_search
from ai_interviewer_pm.tools.vector_db import vector_search
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph


def with_iteration_limit(node_name: str, max_iterations: int = 5) -> Callable:
    """Decorator to add iteration limiting to nodes."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(state: BehavioralInterviewState) -> BehavioralInterviewState:
            if "node_iterations" not in state:
                state["node_iterations"] = {}

            current = state["node_iterations"].get(node_name, 0)
            state["node_iterations"][node_name] = current + 1

            max_iter = state.get("max_node_iterations", {}).get(node_name, max_iterations)
            if current >= max_iter:
                state["error_state"] = {
                    "error": f"Node {node_name} exceeded max iterations ({max_iter})",
                    "type": "iteration_limit",
                }
                state["next_action"] = "conclude"
                return state

            return func(state)

        return wrapper

    return decorator


def with_recursion_check(func: Callable) -> Callable:
    """Decorator to check recursion depth."""

    @wraps(func)
    def wrapper(state: BehavioralInterviewState) -> BehavioralInterviewState:
        current_depth = state.get("graph_recursion_depth", 0)
        max_depth = state.get("max_recursion_depth", 50)

        if current_depth >= max_depth:
            state["error_state"] = {
                "error": f"Maximum recursion depth ({max_depth}) exceeded",
                "type": "recursion_limit",
            }
            state["next_action"] = "conclude"
            return state

        state["graph_recursion_depth"] = current_depth + 1
        result = func(state)
        state["graph_recursion_depth"] = current_depth
        return result

    return wrapper


def _get_llm() -> ChatOpenAI:
    """Get configured LLM instance."""
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.1, api_key=settings.openai_api_key)


# Initialize LLM with tools
llm = _get_llm()
llm_with_tools = llm.bind_tools([vector_search, internet_search])

# Pre-defined behavioral questions pool for PM interviews
BEHAVIORAL_QUESTIONS_POOL = [
    BehavioralQuestion(
        id="leadership_conflict",
        text="Tell me about a time when you had to lead a project where team members had conflicting opinions about the product direction. How did you handle it?",
        category="leadership",
        difficulty="senior",
        follow_up_strategy="stakeholders",
    ),
    BehavioralQuestion(
        id="prioritization_pressure",
        text="Describe a situation where you had to make a difficult prioritization decision with limited resources and multiple stakeholders pushing for their features.",
        category="prioritization",
        difficulty="mid",
        follow_up_strategy="metrics",
    ),
    BehavioralQuestion(
        id="product_failure",
        text="Walk me through a product decision you made that didn't work out as expected. What happened and how did you recover?",
        category="failure_recovery",
        difficulty="mid",
        follow_up_strategy="deep_dive",
    ),
    BehavioralQuestion(
        id="stakeholder_disagreement",
        text="Tell me about a time when engineering and design disagreed on a technical approach, and you had to facilitate a resolution.",
        category="stakeholder_management",
        difficulty="senior",
        follow_up_strategy="alternatives",
    ),
    BehavioralQuestion(
        id="junior_leadership",
        text="Describe a time when you had to influence a decision without having direct authority over the people involved.",
        category="leadership",
        difficulty="junior",
        follow_up_strategy="deep_dive",
    ),
]


def _get_llm() -> ChatOpenAI:
    """Get configured LLM instance with optimal settings for interviews."""
    return ChatOpenAI(
        model="gpt-4o-mini",  # Using latest model for better conversation quality
        temperature=0.3,  # Slightly higher for more natural conversation
        api_key=settings.openai_api_key,
        max_tokens=1000,  # Reasonable limit for interview responses
    )


def question_generator_node(state: BehavioralInterviewState) -> BehavioralInterviewState:
    """Generate or select appropriate behavioral questions based on target level and session progress.

    Uses intelligent selection considering:
    - Target seniority level
    - Question variety and coverage
    - Session progress and time constraints
    """
    session = state["session"]
    existing_pool = state.get("question_pool", [])

    if existing_pool:
        # Questions already generated, no need to regenerate
        state["next_action"] = "ask_question"
        return state

    # Check for custom question from config
    custom_question = state.get("config", {}).get("custom_question")

    # Filter questions by target level and ensure variety
    level_appropriate = [
        q
        for q in BEHAVIORAL_QUESTIONS_POOL
        if q.difficulty == session.target_level
        or q.difficulty == "mid"  # Mid-level questions work for all
    ]

    # Ensure category diversity
    selected_questions: list[BehavioralQuestion] = []
    used_categories: set[str] = set()

    # If custom question provided, create a custom BehavioralQuestion and use it as first question
    if custom_question:
        # Determine category based on question content or default to prioritization
        category = "prioritization"  # Default category for follow-up questions
        if "conflict" in custom_question.lower() or "disagreement" in custom_question.lower():
            category = "conflict_resolution"
        elif "leadership" in custom_question.lower() or "team" in custom_question.lower():
            category = "leadership"
        elif "stakeholder" in custom_question.lower():
            category = "stakeholder_management"
        elif "product" in custom_question.lower() or "feature" in custom_question.lower():
            category = "product_decisions"
        elif "failure" in custom_question.lower() or "mistake" in custom_question.lower():
            category = "failure_recovery"

        custom_q = BehavioralQuestion(
            id=f"custom_{session.session_id}",
            text=custom_question,
            category=category,
            difficulty=session.target_level,
            follow_up_strategy="deep_dive",
        )
        selected_questions.append(custom_q)
        used_categories.add(category)

    # First, prioritize one question from each category (excluding custom if already added)
    for question in level_appropriate:
        if (
            question.category not in used_categories
            and len(selected_questions) < session.total_planned_questions
        ):
            selected_questions.append(question)
            used_categories.add(question.category)

    # Fill remaining slots randomly from appropriate level
    remaining_count = session.total_planned_questions - len(selected_questions)
    remaining_questions = [q for q in level_appropriate if q not in selected_questions]
    selected_questions.extend(
        random.sample(remaining_questions, min(remaining_count, len(remaining_questions)))
    )

    # Randomize order to avoid predictable patterns
    random.shuffle(selected_questions)

    state["question_pool"] = selected_questions
    state["next_action"] = "ask_question"

    # Add system message about the interview structure
    if not state["messages"]:
        intro_msg = AIMessage(
            content=(
                f"Welcome to your PM behavioral interview! I'll be asking you {len(selected_questions)} "
                f"behavioral questions focused on {session.target_level}-level PM scenarios. "
                "Please structure your responses using the STAR method (Situation, Task, Action, Result) "
                "when applicable. Let's begin!"
            )
        )
        state["messages"] = [intro_msg]

    return state


def question_asker_node(state: BehavioralInterviewState) -> BehavioralInterviewState:
    """Present the current question to the candidate and wait for response."""
    session = state["session"]
    question_pool = state["question_pool"]

    # Check if we have more questions to ask
    if session.current_question_index >= len(question_pool):
        state["next_action"] = "conclude"
        return state

    # Get current question
    current_q = question_pool[session.current_question_index]
    state["current_question"] = current_q

    # Create the question message with context
    question_msg = AIMessage(
        content=(
            f"**Question {session.current_question_index + 1} of {len(question_pool)}**\n\n"
            f"{current_q.text}\n\n"
            f"*This is a {current_q.category} question. Please take your time to provide a detailed response "
            f"using specific examples from your experience.*"
        )
    )

    state["messages"].append(question_msg)
    state["next_action"] = "wait_for_response"

    return state


def response_processor_node(state: BehavioralInterviewState) -> BehavioralInterviewState:
    """Process the candidate's response and prepare for evaluation."""
    # In a real implementation, this would handle the incoming response
    # For now, we assume the response is in current_answer
    current_answer = state.get("current_answer")

    if not current_answer or not current_answer.strip():
        # No response yet, continue waiting
        state["next_action"] = "wait_for_response"
        return state

    # Add the candidate's response to message history
    response_msg = HumanMessage(content=current_answer)
    state["messages"].append(response_msg)

    state["next_action"] = "retrieve_context"
    return state


@with_iteration_limit("context_retrieval", max_iterations=2)
def context_retrieval_node(state: BehavioralInterviewState) -> BehavioralInterviewState:
    """Retrieve relevant context for evaluating the behavioral response with Dr. Nancy's prioritization."""
    current_question = state.get("current_question")
    current_answer = state.get("current_answer")

    if not current_question or not current_answer:
        state["retrieved_context"] = []
        state["next_action"] = "evaluate_response"
        return state

    # Create search query combining question category and answer content
    search_query = f"{current_question.category} {current_question.text} {current_answer[:200]}"

    try:
        # Vector search for relevant PM knowledge
        vector_results = vector_search(search_query, k=10)

        # Web search for current PM best practices
        web_results = internet_search(
            f"product manager {current_question.category} best practices examples"
        )

        # Combine and format results
        all_contexts = []

        # Add vector search results
        for result in vector_results:
            all_contexts.append(
                {
                    "text": result.get("content", ""),
                    "source": result.get("source", "knowledge_base"),
                    "score": result.get("score", 0.0),
                }
            )

        # Add web search results
        for result in web_results:
            all_contexts.append(
                {
                    "text": result.get("content", ""),
                    "source": result.get("url", "web_search"),
                    "score": 0.5,  # Default score for web results
                }
            )

        # NEW: Apply Dr. Nancy's coaching filter
        coaching_handler = create_coaching_style_handler()
        dr_nancy_contexts = coaching_handler.filter_dr_nancy_content(all_contexts)
        coaching_patterns = coaching_handler.extract_coaching_patterns(dr_nancy_contexts)

        state["retrieved_context"] = dr_nancy_contexts
        state["coaching_patterns"] = coaching_patterns

    except Exception as e:
        print(f"Context retrieval failed: {e}")
        state["retrieved_context"] = []
        state["coaching_patterns"] = None

    state["next_action"] = "evaluate_response"
    return state


def improvement_tips_generator_node(state: BehavioralInterviewState) -> BehavioralInterviewState:
    """Generate specific improvement tips based on evaluation."""
    current_question = state.get("current_question")
    current_answer = state.get("current_answer")
    evaluation = state.get("evaluation")

    if not all([current_question, current_answer, evaluation]):
        state["improvement_tips"] = []
        return state

    tips_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    "You are a PM interview coach providing actionable improvement tips. "
                    "Based on the evaluation, provide 3-5 specific, actionable tips "
                    "for improving the answer. Focus on: "
                    "1) Adding missing STAR elements 2) Including specific metrics "
                    "3) Demonstrating clearer leadership 4) Improving clarity"
                )
            ),
            (
                "human",
                (
                    "Question: {question}\n"
                    "Answer: {answer}\n"
                    "Evaluation Scores: {scores}\n"
                    "Areas for Improvement: {areas}\n\n"
                    "Generate 3-5 specific improvement tips as a JSON array."
                ),
            ),
        ]
    )

    llm = _get_llm()
    tips_chain = tips_prompt | llm

    try:
        response = tips_chain.invoke(
            {
                "question": current_question.text,
                "answer": current_answer,
                "scores": (
                    str(evaluation.model_dump())
                    if hasattr(evaluation, "model_dump")
                    else str(evaluation)
                ),
                "areas": (
                    evaluation.improvement_areas if hasattr(evaluation, "improvement_areas") else []
                ),
            }
        )

        import json

        content = getattr(response, "content", "")
        # Clean the content to extract JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        if content.strip().startswith("["):
            parsed_tips = json.loads(content)
            # Handle both array of strings and array of objects with "tip" field
            if parsed_tips and isinstance(parsed_tips[0], dict):
                tips = [tip.get("tip", str(tip)) for tip in parsed_tips]
            else:
                tips = parsed_tips
        else:
            tips = []

        state["improvement_tips"] = tips[:5]  # Limit to 5 tips
    except Exception:
        # Fallback tips
        state["improvement_tips"] = [
            "Add specific metrics and numbers to quantify your impact",
            "Include more detail about the initial situation and context",
            "Clarify your specific role and actions taken",
            "Emphasize the measurable results and outcomes",
            "Consider mentioning lessons learned or follow-up actions",
        ]

    return state


def template_answer_generator_node(state: BehavioralInterviewState) -> BehavioralInterviewState:
    """Generate a brief, high-quality template answer for the current question."""
    current_question = state.get("current_question")

    if not current_question:
        return state

    template_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    "You are an expert PM coach generating brief template answers. "
                    "Create a concise STAR-format answer (2-3 sentences per section) that: "
                    "1) Shows clear structure 2) Includes specific metrics 3) Demonstrates PM skills "
                    "4) Is realistic and actionable. Keep it brief but impactful."
                )
            ),
            (
                "human",
                (
                    "Question: {question}\n"
                    "Category: {category}\n\n"
                    "Generate a brief template answer using STAR format. "
                    "Make it specific enough to be helpful but generic enough to apply broadly."
                ),
            ),
        ]
    )

    llm = _get_llm()
    template_chain = template_prompt | llm

    try:
        response = template_chain.invoke(
            {
                "question": current_question.text,
                "category": current_question.category,
            }
        )

        template_answer = getattr(response, "content", "")
        state["template_answer"] = template_answer
    except Exception:
        # Fallback template based on category
        category_templates = {
            "leadership": (
                "Situation: Led cross-functional team during critical product pivot. "
                "Task: Align 15+ stakeholders on new direction within 2 weeks. "
                "Action: Conducted 1:1s, created shared vision doc, ran design sprints. "
                "Result: 100% buy-in, launched MVP 3 weeks early, 40% adoption rate."
            ),
            "prioritization": (
                "Situation: 5 high-priority features, only resources for 2. "
                "Task: Choose features maximizing user value and business impact. "
                "Action: Analyzed usage data, ran opportunity sizing, stakeholder mapping. "
                "Result: Selected features increased NPS by 15 points, $2M additional revenue."
            ),
            "stakeholder_management": (
                "Situation: Engineering and design disagreed on implementation approach. "
                "Task: Find solution balancing technical feasibility and user experience. "
                "Action: Facilitated workshop, created decision matrix, ran A/B test. "
                "Result: Hybrid approach reduced dev time 30%, improved usability score 25%."
            ),
        }

        state["template_answer"] = category_templates.get(
            current_question.category,
            "Use STAR: Describe Situation clearly, define your Task/role, "
            "explain specific Actions taken, quantify Results achieved.",
        )

    return state


@with_iteration_limit("response_evaluator", max_iterations=3)
def response_evaluator_node(state: BehavioralInterviewState) -> BehavioralInterviewState:
    """Evaluate the candidate's response using Dr. Nancy's coaching style."""
    current_question = state["current_question"]
    current_answer = state["current_answer"]
    retrieved_context = state.get("retrieved_context", [])
    coaching_patterns = state.get("coaching_patterns")

    if not current_question or not current_answer:
        state["error_state"] = {"error": "Missing question or answer for evaluation"}
        state["next_action"] = "ask_question"  # Retry
        return state

    # NEW: Get Dr. Nancy's coaching prompt
    coaching_handler = create_coaching_style_handler()

    # Determine performance level from history
    evaluation_history = state.get("evaluation_history", [])
    if evaluation_history:
        avg_score = sum(e.overall_score for e in evaluation_history) / len(evaluation_history)
        if avg_score < 5:
            performance_level = "struggling"
        elif avg_score < 7:
            performance_level = "developing"
        elif avg_score < 8.5:
            performance_level = "strong"
        else:
            performance_level = "excellent"
    else:
        performance_level = "developing"

    coaching_prompt = coaching_handler.get_coaching_prompt(
        current_question.category, performance_level
    )

    # Format context for evaluation
    context_text = ""
    if retrieved_context:
        context_text = "\n\nRelevant PM Knowledge & Best Practices:\n"
        for i, ctx in enumerate(retrieved_context[:3], 1):  # Use top 3 contexts
            context_text += f"{i}. {ctx['text'][:300]}...\n"

    # Create evaluation prompt with Dr. Nancy's style
    evaluation_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=coaching_prompt),
            (
                "human",
                (
                    "Question: {question}\n"
                    "Category: {category}\n"
                    "Response: {response}\n"
                    "{context}\n"
                    "Coaching Patterns: {patterns}\n\n"
                    "Evaluate this response and return a JSON object with:\n"
                    "- completeness_score (0-10): STAR structure completeness\n"
                    "- clarity_score (0-10): Communication clarity\n"
                    "- depth_score (0-10): Depth of insight\n"
                    "- impact_score (0-10): Business impact demonstrated\n"
                    "- leadership_score (0-10): Leadership qualities\n"
                    "- overall_score (0-10): Overall quality\n"
                    "- key_strengths (array): Main strengths\n"
                    "- improvement_areas (array): Areas for improvement\n"
                    "- follow_up_needed (boolean): Whether follow-ups are warranted"
                ),
            ),
        ]
    )

    # Use structured output for reliable evaluation
    structured_llm = _get_llm().with_structured_output(ResponseEvaluation)
    evaluation_chain = evaluation_prompt | structured_llm

    try:
        evaluation = evaluation_chain.invoke(
            {
                "question": current_question.text,
                "category": current_question.category,
                "response": current_answer,
                "context": context_text,
                "patterns": (
                    str(coaching_patterns) if coaching_patterns else "Standard coaching approach"
                ),
            }
        )

        # Add retrieved contexts to evaluation for RAGAS
        evaluation.contexts = retrieved_context
        state["evaluation"] = evaluation

        # Add to evaluation history for adaptive questioning
        if "evaluation_history" not in state:
            state["evaluation_history"] = []
        state["evaluation_history"].append(evaluation)

        # Determine next action
        state["next_action"] = "multi_agent_eval"

    except Exception as e:
        state["error_state"] = {"error": f"Evaluation failed: {str(e)}"}
        state["next_action"] = "move_to_next"  # Continue interview even if evaluation fails

    return state


def follow_up_generator_node(state: BehavioralInterviewState) -> BehavioralInterviewState:
    """Generate targeted follow-up questions based on the response and evaluation."""
    current_question = state["current_question"]
    current_answer = state["current_answer"]
    evaluation = state["evaluation"]

    if not all([current_question, current_answer, evaluation]):
        state["next_action"] = "move_to_next"
        return state

    # Generate follow-ups based on strategy and evaluation
    follow_up_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    "Generate 1-2 specific follow-up questions to probe deeper into the candidate's response. "
                    "Base the questions on the evaluation findings and the original question's follow-up strategy. "
                    "Focus on areas that need more depth or clarity."
                )
            ),
            (
                "human",
                (
                    "Original Question: {question}\n"
                    "Strategy: {strategy}\n"
                    "Response: {response}\n"
                    "Evaluation: {evaluation}\n\n"
                    "Generate targeted follow-up questions as a JSON array of strings."
                ),
            ),
        ]
    )

    follow_up_chain = follow_up_prompt | _get_llm()

    try:
        response = follow_up_chain.invoke(
            {
                "question": current_question.text,
                "strategy": current_question.follow_up_strategy,
                "response": current_answer,
                "evaluation": (
                    evaluation.model_dump()
                    if hasattr(evaluation, "model_dump")
                    else str(evaluation)
                ),
            }
        )

        content = getattr(response, "content", "")
        follow_ups = json.loads(content) if content.strip().startswith("[") else [content]

        state["follow_up_questions"] = follow_ups[:2]  # Limit to 2 follow-ups
        state["next_action"] = "ask_follow_up"

    except Exception:
        # Fallback to generic follow-up based on strategy
        strategy_follow_ups = {
            "deep_dive": "Can you provide more specific details about your decision-making process?",
            "metrics": "What metrics did you use to measure success, and what were the results?",
            "stakeholders": "How did different stakeholders react to your approach?",
            "alternatives": "What alternative approaches did you consider and why didn't you choose them?",
        }

        fallback_question = strategy_follow_ups.get(
            current_question.follow_up_strategy,
            "Can you elaborate on the most challenging aspect of this situation?",
        )
        state["follow_up_questions"] = [fallback_question]
        state["next_action"] = "ask_follow_up"

    return state


def follow_up_asker_node(state: BehavioralInterviewState) -> BehavioralInterviewState:
    """Ask follow-up questions based on the candidate's previous response."""
    follow_ups = state.get("follow_up_questions", [])

    if not follow_ups or state["follow_up_count"] >= state["max_follow_ups"]:
        state["next_action"] = "move_to_next"
        return state

    # Ask the first follow-up question
    follow_up = follow_ups[0]
    follow_up_msg = AIMessage(content=f"**Follow-up:** {follow_up}")
    state["messages"].append(follow_up_msg)

    # Update follow-up tracking
    state["follow_up_count"] += 1
    state["follow_up_questions"] = follow_ups[1:]  # Remove asked question
    state["current_answer"] = None  # Reset for new response
    state["next_action"] = "wait_for_response"

    return state


def display_followups_generator_node(state: BehavioralInterviewState) -> BehavioralInterviewState:
    """Generate 3 follow-up questions for display purposes (not for interview flow)."""
    current_question = state["current_question"]
    current_answer = state["current_answer"]

    if not all([current_question, current_answer]):
        state["display_followups"] = []
        state["next_action"] = "move_to_next"
        return state

    # Generate display follow-ups based on the question and answer
    display_followup_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    "Generate exactly 3 thoughtful follow-up questions that would help explore "
                    "the candidate's response more deeply. These questions should:\n"
                    "1. Probe specific aspects of their decision-making process\n"
                    "2. Explore alternative approaches they considered\n"
                    "3. Understand the broader impact or lessons learned\n\n"
                    "Focus on questions that would be valuable for a product manager interview. "
                    "Make them specific to the candidate's response, not generic."
                )
            ),
            (
                "human",
                (
                    "Original Question: {question}\n"
                    "Category: {category}\n"
                    "Candidate's Response: {response}\n\n"
                    "Generate exactly 3 follow-up questions as a JSON array of strings. "
                    "Each question should be specific to this response and help understand "
                    "the candidate's PM skills and thinking process better."
                ),
            ),
        ]
    )

    display_followup_chain = display_followup_prompt | _get_llm()

    try:
        response = display_followup_chain.invoke(
            {
                "question": current_question.text,
                "category": current_question.category,
                "response": current_answer,
            }
        )

        content = getattr(response, "content", "")

        # Clean up the content and try to parse as JSON array
        try:
            import json
            import re

            # Remove markdown code blocks and clean up the content
            cleaned_content = content.strip()
            if cleaned_content.startswith("```"):
                # Remove markdown code blocks
                cleaned_content = re.sub(r"^```[a-z]*\n?", "", cleaned_content, flags=re.MULTILINE)
                cleaned_content = re.sub(r"\n?```$", "", cleaned_content)

            # Try to extract JSON array from the content
            json_match = re.search(r"\[.*?\]", cleaned_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                display_followups = json.loads(json_str)
                if isinstance(display_followups, list):
                    state["display_followups"] = display_followups[:3]  # Ensure exactly 3
                else:
                    raise ValueError("Not a list")
            else:
                raise ValueError("No JSON array found")

        except (json.JSONDecodeError, ValueError):
            # Fallback to splitting by lines and cleaning up
            lines = [
                line.strip()
                for line in content.split("\n")
                if line.strip() and not line.startswith("```")
            ]
            # Remove any JSON artifacts
            clean_lines = []
            for line in lines:
                if line.startswith('"') and line.endswith('",'):
                    # Remove quotes and trailing comma
                    clean_lines.append(line[1:-2])
                elif line.startswith('"') and line.endswith('"'):
                    # Remove quotes
                    clean_lines.append(line[1:-1])
                elif line not in ["[", "]", "{", "}"]:
                    clean_lines.append(line)
            state["display_followups"] = clean_lines[:3]

    except Exception:
        # Fallback to generic follow-ups based on category
        category_followups = {
            "leadership": [
                "How did you measure the success of your leadership approach in this situation?",
                "What would you do differently if you faced a similar leadership challenge again?",
                "How did you ensure team buy-in for your decisions throughout this process?",
            ],
            "conflict_resolution": [
                "What early warning signs helped you identify this conflict?",
                "How did you balance different stakeholder perspectives in your resolution?",
                "What processes did you put in place to prevent similar conflicts?",
            ],
            "prioritization": [
                "What framework did you use to evaluate competing priorities?",
                "How did you communicate these prioritization decisions to stakeholders?",
                "What metrics did you use to validate your prioritization choices?",
            ],
            "stakeholder_management": [
                "How did you tailor your communication style for different stakeholders?",
                "What strategies did you use to maintain stakeholder alignment over time?",
                "How did you handle pushback from key stakeholders?",
            ],
            "product_decisions": [
                "What data sources influenced your product decision-making process?",
                "How did you validate your assumptions before implementing this decision?",
                "What was the long-term impact of this product decision?",
            ],
            "failure_recovery": [
                "What early indicators helped you recognize this failure?",
                "How did you communicate the failure and recovery plan to stakeholders?",
                "What systems did you implement to prevent similar failures?",
            ],
        }

        state["display_followups"] = category_followups.get(
            current_question.category,
            [
                "Can you elaborate on the decision-making process you used?",
                "What alternative approaches did you consider?",
                "How did you measure the success of your actions?",
            ],
        )

    # Store evaluation data in a persistent field that won't get cleared
    evaluation = state.get("evaluation")
    if evaluation:
        state["evaluation_backup"] = {
            "feedback": f"""**Evaluation Summary**

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

**Overall Assessment:** {evaluation.overall_score:.1f}/10""",
            "rubric_score": {
                "clarity": evaluation.clarity_score,
                "structure": evaluation.completeness_score,
                "depth": evaluation.depth_score,
                "impact": evaluation.impact_score,
                "leadership": evaluation.leadership_score,
                "overall": evaluation.overall_score,
                "summary": f"Overall score: {evaluation.overall_score:.1f}/10. Key strengths: {', '.join(evaluation.key_strengths[:2])}",
            },
        }

    # For API evaluation, we want to end here to preserve evaluation data
    # The session manager will clear the evaluation, so we end the graph here
    state["next_action"] = "evaluation_complete"
    return state


def session_manager_node(state: BehavioralInterviewState) -> BehavioralInterviewState:
    """Manage session progression and prepare for next question or conclusion."""
    session = state["session"]

    # NEW: Check for adaptive question override
    if state.get("next_question_override"):
        # Find the override question in the pool
        override_q = state["next_question_override"]
        question_pool = state["question_pool"]

        # Find index of override question or add it
        override_index = None
        for i, q in enumerate(question_pool):
            if q.id == override_q.id:
                override_index = i
                break

        if override_index is not None:
            session.current_question_index = override_index
        else:
            # Add the new question to the pool
            question_pool.append(override_q)
            session.current_question_index = len(question_pool) - 1
            state["question_pool"] = question_pool

        # Clear the override
        state["next_question_override"] = None
    else:
        # Normal progression
        session.current_question_index += 1

    session.questions_completed += 1

    # Reset question-specific state
    state["current_question"] = None
    state["current_answer"] = None
    state["evaluation"] = None
    state["follow_up_questions"] = []
    state["follow_up_count"] = 0

    # Update session state
    state["session"] = session

    # Determine next action
    if session.current_question_index >= len(state["question_pool"]):
        state["next_action"] = "conclude"
    else:
        state["next_action"] = "ask_question"

    return state


def multi_agent_evaluator_node(state: BehavioralInterviewState) -> BehavioralInterviewState:
    """Orchestrate multiple evaluation agents for comprehensive assessment."""
    current_question = state.get("current_question")
    current_answer = state.get("current_answer")
    retrieved_context = state.get("retrieved_context", [])

    if not current_question or not current_answer:
        state["next_action"] = "grail_eval"
        return state

    try:
        # Create multi-agent evaluator
        multi_evaluator = create_multi_agent_evaluator(use_all_agents=True)

        # Run parallel evaluation
        agent_evaluations = multi_evaluator.evaluate_parallel(
            current_question.text, current_answer, retrieved_context
        )

        # Build consensus
        consensus = multi_evaluator.build_consensus(agent_evaluations)

        # Store evaluations
        state["agent_evaluations"] = [e.model_dump() for e in agent_evaluations]
        state["consensus_evaluation"] = consensus.model_dump()

        # Update overall evaluation with consensus
        if state["evaluation"]:
            state["evaluation"].overall_score = consensus.final_score

    except Exception as e:
        print(f"Multi-agent evaluation failed: {e}")
        # Continue without multi-agent evaluation

    state["next_action"] = "grail_eval"
    return state


def grail_evaluator_node(state: BehavioralInterviewState) -> BehavioralInterviewState:
    """Apply GRAIL rubric evaluation for comprehensive PM assessment."""
    current_question = state.get("current_question")
    current_answer = state.get("current_answer")
    retrieved_context = state.get("retrieved_context", [])

    if not current_question or not current_answer:
        state["next_action"] = "improvement_tips_generator"
        return state

    try:
        # Create GRAIL evaluator
        grail_evaluator = create_grail_evaluator()

        # Perform GRAIL evaluation
        grail_evaluation = grail_evaluator.evaluate(
            current_question.text, current_answer, retrieved_context, current_question.category
        )

        # Store GRAIL evaluation
        state["grail_evaluation"] = grail_evaluation.model_dump()

        # Get improvement recommendations
        grail_recommendations = grail_evaluator.get_improvement_recommendations(grail_evaluation)

        # Merge with existing improvement tips
        existing_tips = state.get("improvement_tips", [])
        state["improvement_tips"] = existing_tips[:2] + grail_recommendations[:3]

        # Update overall evaluation with GRAIL score
        if state["evaluation"]:
            # Average the scores from different evaluations
            scores = [state["evaluation"].overall_score]
            if state.get("consensus_evaluation"):
                scores.append(state["consensus_evaluation"]["final_score"])
            scores.append(grail_evaluation.overall_score)
            state["evaluation"].overall_score = sum(scores) / len(scores)

    except Exception as e:
        print(f"GRAIL evaluation failed: {e}")
        # Continue without GRAIL evaluation

    state["next_action"] = "coaching_generator"
    return state


def coaching_feedback_node(state: BehavioralInterviewState) -> BehavioralInterviewState:
    """Generate Dr. Nancy's coaching style feedback."""
    print("DEBUG: coaching_feedback_node called")
    current_question = state.get("current_question")
    current_answer = state.get("current_answer")
    evaluation = state.get("evaluation")
    grail_evaluation = state.get("grail_evaluation")
    retrieved_context = state.get("retrieved_context", [])
    
    if not current_question or not current_answer or not evaluation:
        state["next_action"] = "improvement_tips_generator"
        return state
    
    try:
        # Create coaching style handler
        coaching_handler = create_coaching_style_handler()
        
        # Filter and prioritize Dr. Nancy's content
        filtered_contexts = coaching_handler.filter_dr_nancy_content(retrieved_context)
        
        # Extract coaching patterns from contexts
        coaching_patterns = coaching_handler.extract_coaching_patterns(filtered_contexts)
        state["coaching_patterns"] = coaching_patterns
        
        # Create evaluation scores dict for coaching feedback
        evaluation_scores = {
            "clarity": evaluation.clarity_score,
            "structure": evaluation.completeness_score,
            "depth": evaluation.depth_score,
            "impact": evaluation.impact_score,
            "leadership": evaluation.leadership_score,
            "overall": evaluation.overall_score,
        }
        
        # Add GRAIL scores if available
        if grail_evaluation and isinstance(grail_evaluation, dict):
            evaluation_scores["grail_overall"] = grail_evaluation.get("overall_score", 0)
        
        # Generate coaching feedback
        coaching_feedback = coaching_handler.generate_coaching_feedback(
            current_question.text,
            current_answer,
            evaluation_scores,
            coaching_patterns
        )
        
        # Get encouragement message
        improvement_areas = evaluation.improvement_areas[:2] if evaluation.improvement_areas else []
        encouragement_message = coaching_handler.get_encouragement_message(
            evaluation.overall_score,
            improvement_areas
        )
        
        # Adapt follow-up questions based on performance
        performance_level = "struggling" if evaluation.overall_score < 5 else (
            "developing" if evaluation.overall_score < 7 else (
                "strong" if evaluation.overall_score < 8.5 else "excellent"
            )
        )
        
        original_followups = state.get("display_followups", [])
        adapted_followups = coaching_handler.adapt_follow_up_questions(
            current_question.text,
            current_answer,
            performance_level,
            original_followups
        )
        
        # Store coaching feedback in state
        state["coaching_feedback"] = {
            "feedback_text": coaching_feedback,
            "coaching_patterns": coaching_patterns,
            "encouragement_message": encouragement_message,
            "adapted_followups": adapted_followups
        }
        print(f"DEBUG: Coaching feedback stored in state: {state['coaching_feedback']}")
        
    except Exception as e:
        print(f"Coaching feedback generation failed: {e}")
        # Continue without coaching feedback
    
    state["next_action"] = "improvement_tips_generator"
    return state


def adaptive_question_selector_node(state: BehavioralInterviewState) -> BehavioralInterviewState:
    """Select next question based on performance analysis."""
    evaluation_history = state.get("evaluation_history", [])
    current_evaluation = state.get("evaluation")
    question_pool = state.get("question_pool", [])
    session = state.get("session")

    if not question_pool or not session:
        state["next_action"] = "move_to_next"
        return state

    try:
        # Create adaptive selector
        adaptive_selector = create_adaptive_selector(session.target_level)

        # Analyze performance
        performance_metrics = adaptive_selector.analyze_performance(
            evaluation_history, current_evaluation
        )

        # Store performance metrics
        state["performance_metrics"] = performance_metrics.model_dump()

        # Select next question
        current_question = state.get("current_question")
        adaptive_decision = adaptive_selector.select_next_question(
            question_pool,
            current_question,
            performance_metrics,
            completed_categories=[
                q.category for q in question_pool[: session.current_question_index]
            ],
        )

        # Store adaptive decision
        state["adaptive_decision"] = adaptive_decision.model_dump()

        # Apply adaptive decision
        if adaptive_decision.action == "conclude":
            state["next_action"] = "conclude"
        elif adaptive_decision.next_question:
            state["next_question_override"] = adaptive_decision.next_question
            state["next_action"] = "move_to_next"
        else:
            state["next_action"] = "move_to_next"

        # Check if should conclude early
        if adaptive_selector.should_conclude_early(
            performance_metrics, session.questions_completed * 5
        ):
            state["next_action"] = "conclude"

    except Exception as e:
        print(f"Adaptive selection failed: {e}")
        state["next_action"] = "move_to_next"

    return state


def conclusion_node(state: BehavioralInterviewState) -> BehavioralInterviewState:
    """Provide interview conclusion and summary feedback."""
    session = state["session"]

    # Generate summary message
    summary_msg = AIMessage(
        content=(
            f"Thank you for completing the behavioral interview! We covered {session.questions_completed} "
            f"questions across various PM scenarios. Your responses demonstrated experience in "
            f"{session.target_level}-level product management situations.\n\n"
            "The interview is now complete. You'll receive detailed feedback on your responses shortly."
        )
    )

    state["messages"].append(summary_msg)
    state["session"].interview_stage = "wrap_up"
    state["next_action"] = "conclude"

    return state


# Routing functions for conditional edges
def route_from_generator(state: BehavioralInterviewState) -> str:
    """Route after question generation based on state."""
    next_action = state.get("next_action", "ask_question")
    return next_action


def route_from_asker(state: BehavioralInterviewState) -> str:
    """Route after asking question."""
    return state.get("next_action", "wait_for_response")


def route_from_processor(state: BehavioralInterviewState) -> str:
    """Route after processing response."""
    return state.get("next_action", "evaluate_response")


def route_from_evaluator(state: BehavioralInterviewState) -> str:
    """Route after evaluation based on follow-up need."""
    return state.get("next_action", "move_to_next")


def route_from_multi_agent(state: BehavioralInterviewState) -> str:
    """Route after multi-agent evaluation."""
    return state.get("next_action", "grail_eval")


def route_from_grail(state: BehavioralInterviewState) -> str:
    """Route after GRAIL evaluation."""
    return state.get("next_action", "coaching_generator")


def route_from_adaptive(state: BehavioralInterviewState) -> str:
    """Route after adaptive selection."""
    return state.get("next_action", "move_to_next")


def route_from_coaching(state: BehavioralInterviewState) -> str:
    """Route after coaching feedback."""
    return state.get("next_action", "improvement_tips_generator")


def route_from_follow_up_generator(state: BehavioralInterviewState) -> str:
    """Route after generating follow-ups."""
    return state.get("next_action", "ask_follow_up")


def route_from_follow_up_asker(state: BehavioralInterviewState) -> str:
    """Route after asking follow-up."""
    return state.get("next_action", "wait_for_response")


def route_from_session_manager(state: BehavioralInterviewState) -> str:
    """Route after managing session progression."""
    return state.get("next_action", "ask_question")


def build_behavioral_interview_graph() -> StateGraph:
    """Build the behavioral interview graph with proper 2025 LangGraph patterns."""
    graph = StateGraph(BehavioralInterviewState)

    # Add all nodes
    graph.add_node("question_generator", question_generator_node)
    graph.add_node("question_asker", question_asker_node)
    graph.add_node("response_processor", response_processor_node)
    graph.add_node("context_retrieval", context_retrieval_node)
    graph.add_node("template_answer_generator", template_answer_generator_node)
    graph.add_node("response_evaluator", response_evaluator_node)
    # NEW: Enhanced evaluation nodes
    graph.add_node("multi_agent_evaluator", multi_agent_evaluator_node)
    graph.add_node("grail_evaluator", grail_evaluator_node)
    graph.add_node("coaching_generator", coaching_feedback_node)  # Dr. Nancy's coaching
    graph.add_node("improvement_tips_generator", improvement_tips_generator_node)
    graph.add_node("display_followups_generator", display_followups_generator_node)
    # NEW: Adaptive questioning node
    graph.add_node("adaptive_selector", adaptive_question_selector_node)
    graph.add_node("follow_up_generator", follow_up_generator_node)
    graph.add_node("follow_up_asker", follow_up_asker_node)
    graph.add_node("session_manager", session_manager_node)
    graph.add_node("conclusion", conclusion_node)

    # Set entry point
    graph.set_entry_point("question_generator")

    # Add conditional edges for dynamic flow control
    graph.add_conditional_edges(
        "question_generator",
        route_from_generator,
        {"ask_question": "question_asker", "conclude": "conclusion"},
    )

    graph.add_conditional_edges(
        "question_asker",
        route_from_asker,
        {"wait_for_response": "response_processor", "conclude": "conclusion"},
    )

    graph.add_conditional_edges(
        "response_processor",
        route_from_processor,
        {
            "retrieve_context": "context_retrieval",
            "wait_for_response": END,  # End if no response - API will handle this
        },
    )

    graph.add_conditional_edges(
        "context_retrieval",
        route_from_processor,  # Reuse same routing logic
        {
            "evaluate_response": "template_answer_generator",
        },
    )

    # Template answer generator always goes to response evaluator
    graph.add_edge("template_answer_generator", "response_evaluator")

    # NEW: Enhanced evaluation pipeline
    graph.add_conditional_edges(
        "response_evaluator",
        route_from_evaluator,
        {
            "multi_agent_eval": "multi_agent_evaluator",
            "move_to_next": "session_manager",  # Fallback
        },
    )

    graph.add_conditional_edges(
        "multi_agent_evaluator",
        route_from_multi_agent,
        {
            "grail_eval": "grail_evaluator",
            "improvement_tips_generator": "improvement_tips_generator",  # Fallback
        },
    )

    graph.add_conditional_edges(
        "grail_evaluator",
        route_from_grail,
        {
            "coaching_generator": "coaching_generator",
        },
    )
    
    # Coaching feedback leads to improvement tips
    graph.add_conditional_edges(
        "coaching_generator",
        route_from_coaching,
        {
            "improvement_tips_generator": "improvement_tips_generator",
        },
    )

    # After improvement tips, generate display follow-ups
    graph.add_edge("improvement_tips_generator", "display_followups_generator")

    # NEW: Adaptive questioning after display follow-ups
    graph.add_edge("display_followups_generator", "adaptive_selector")

    graph.add_conditional_edges(
        "adaptive_selector",
        route_from_adaptive,
        {
            "ask_follow_up": "follow_up_generator",
            "move_to_next": "session_manager",
            "conclude": "conclusion",
            "evaluation_complete": END,  # End here to preserve evaluation data
        },
    )

    graph.add_conditional_edges(
        "follow_up_generator",
        route_from_follow_up_generator,
        {"ask_follow_up": "follow_up_asker", "move_to_next": "session_manager"},
    )

    graph.add_conditional_edges(
        "follow_up_asker",
        route_from_follow_up_asker,
        {
            "wait_for_response": END,
            "move_to_next": "session_manager",
        },  # End if waiting for response
    )

    graph.add_conditional_edges(
        "session_manager",
        route_from_session_manager,
        {"ask_question": "question_asker", "conclude": "conclusion"},
    )

    # Conclusion leads to END
    graph.add_edge("conclusion", END)

    return graph


def create_checkpointer() -> MemorySaver:
    """Create a memory checkpointer for session persistence."""
    return MemorySaver()


def compile_behavioral_interview_graph(
    checkpointer: MemorySaver | None = None, max_recursion: int = 50
) -> Pregel:
    """Compile the behavioral interview graph with recursion protection."""
    graph = build_behavioral_interview_graph()

    # Use provided checkpointer or create a default one
    if checkpointer is None:
        checkpointer = create_checkpointer()

    # Compile with proper config
    app = graph.compile(
        checkpointer=checkpointer,
        debug=False,
    )

    return app
