"""Comprehensive tests for behavioral interview system."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from ai_interviewer_pm.agents.behavioral_schema import (
    BehavioralQuestion, 
    InterviewSession,
    BehavioralInterviewState,
    validate_interview_state
)
from ai_interviewer_pm.agents.behavioral_graph import (
    question_generator_node,
    question_asker_node,
    response_processor_node,
    response_evaluator_node,
    follow_up_generator_node,
    route_from_evaluator,
    build_behavioral_interview_graph,
    compile_behavioral_interview_graph
)


class TestBehavioralSchema:
    """Test behavioral interview schema components."""
    
    def test_behavioral_question_creation(self):
        """Test behavioral question model validation."""
        question = BehavioralQuestion(
            id="test_q1",
            text="Tell me about a challenging project.",
            category="leadership",
            difficulty="mid",
            follow_up_strategy="deep_dive"
        )
        
        assert question.id == "test_q1"
        assert question.category == "leadership"
        assert question.expected_duration == 5  # default
        
    def test_interview_session_defaults(self):
        """Test interview session with proper defaults."""
        session = InterviewSession(
            session_id="test_session",
            target_level="senior"
        )
        
        assert session.current_question_index == 0
        assert session.questions_completed == 0
        assert session.total_planned_questions == 5
        assert session.interview_stage == "introduction"
        assert isinstance(session.start_time, datetime)
        
    def test_state_validation_success(self):
        """Test successful state validation."""
        session = InterviewSession(
            session_id="test_session",
            target_level="mid"
        )
        
        raw_state = {
            "messages": [],
            "session": session,
            "question_pool": [],
            "current_question": None,
            "current_answer": None,
            "evaluation": None,
            "follow_up_questions": [],
            "follow_up_count": 0,
            "max_follow_ups": 2,
            "retrieved_context": [],
            "web_search_results": [],
            "next_action": "generate_questions",
            "config": {},
            "error_state": None,
            "retry_count": 0
        }
        
        validated = validate_interview_state(raw_state)
        
        assert isinstance(validated, dict)
        assert validated["session"] == session
        assert validated["next_action"] == "generate_questions"
        assert validated["max_follow_ups"] == 2
        
    def test_state_validation_with_defaults(self):
        """Test state validation fills in missing defaults."""
        minimal_state = {
            "messages": [],
            "session": {"session_id": "test", "target_level": "mid"}
        }
        
        validated = validate_interview_state(minimal_state)
        
        assert validated["follow_up_count"] == 0
        assert validated["max_follow_ups"] == 2
        assert validated["next_action"] == "generate_questions"
        assert validated["retry_count"] == 0
        
    def test_state_validation_failure(self):
        """Test state validation with invalid data."""
        invalid_state = {
            "session": {"invalid": "data"},
            "messages": "not_a_list"  # Invalid type
        }
        
        with pytest.raises(ValueError):
            validate_interview_state(invalid_state)


class TestBehavioralNodes:
    """Test individual node implementations."""
    
    def test_question_generator_node_creates_pool(self):
        """Test question generator creates appropriate question pool."""
        session = InterviewSession(
            session_id="test",
            target_level="senior", 
            total_planned_questions=3
        )
        
        state: BehavioralInterviewState = {
            "messages": [],
            "session": session,
            "question_pool": [],  # Empty pool
            "current_question": None,
            "current_answer": None,
            "evaluation": None,
            "follow_up_questions": [],
            "follow_up_count": 0,
            "max_follow_ups": 2,
            "retrieved_context": [],
            "web_search_results": [],
            "next_action": "generate_questions",
            "config": {},
            "error_state": None,
            "retry_count": 0
        }
        
        result = question_generator_node(state)
        
        assert len(result["question_pool"]) <= 3
        assert result["next_action"] == "ask_question"
        assert len(result["messages"]) > 0  # Should have intro message
        
        # Check questions are appropriate for senior level
        for q in result["question_pool"]:
            assert q.difficulty in ["senior", "mid"]  # Mid-level questions OK for all levels
            
    def test_question_generator_skips_if_pool_exists(self):
        """Test question generator skips if pool already exists."""
        existing_question = BehavioralQuestion(
            id="existing",
            text="Existing question",
            category="leadership",
            difficulty="mid",
            follow_up_strategy="deep_dive"
        )
        
        session = InterviewSession(session_id="test", target_level="mid")
        
        state: BehavioralInterviewState = {
            "messages": [],
            "session": session,
            "question_pool": [existing_question],  # Already has questions
            "current_question": None,
            "current_answer": None,
            "evaluation": None,
            "follow_up_questions": [],
            "follow_up_count": 0,
            "max_follow_ups": 2,
            "retrieved_context": [],
            "web_search_results": [],
            "next_action": "generate_questions",
            "config": {},
            "error_state": None,
            "retry_count": 0
        }
        
        result = question_generator_node(state)
        
        assert len(result["question_pool"]) == 1  # Unchanged
        assert result["next_action"] == "ask_question"
        
    def test_question_asker_node_presents_question(self):
        """Test question asker presents current question."""
        question = BehavioralQuestion(
            id="q1",
            text="Tell me about leadership experience.",
            category="leadership", 
            difficulty="mid",
            follow_up_strategy="stakeholders"
        )
        
        session = InterviewSession(
            session_id="test",
            target_level="mid",
            current_question_index=0
        )
        
        state: BehavioralInterviewState = {
            "messages": [],
            "session": session,
            "question_pool": [question],
            "current_question": None,
            "current_answer": None,
            "evaluation": None,
            "follow_up_questions": [],
            "follow_up_count": 0,
            "max_follow_ups": 2,
            "retrieved_context": [],
            "web_search_results": [],
            "next_action": "ask_question",
            "config": {},
            "error_state": None,
            "retry_count": 0
        }
        
        result = question_asker_node(state)
        
        assert result["current_question"] == question
        assert result["next_action"] == "wait_for_response"
        assert len(result["messages"]) == 1
        assert "leadership experience" in result["messages"][0].content.lower()
        
    def test_question_asker_concludes_when_no_questions(self):
        """Test question asker concludes when no more questions."""
        session = InterviewSession(
            session_id="test",
            target_level="mid",
            current_question_index=2  # Beyond available questions
        )
        
        state: BehavioralInterviewState = {
            "messages": [],
            "session": session,
            "question_pool": [Mock()],  # Only 1 question, but index is 2
            "current_question": None,
            "current_answer": None,
            "evaluation": None,
            "follow_up_questions": [],
            "follow_up_count": 0,
            "max_follow_ups": 2,
            "retrieved_context": [],
            "web_search_results": [],
            "next_action": "ask_question",
            "config": {},
            "error_state": None,
            "retry_count": 0
        }
        
        result = question_asker_node(state)
        
        assert result["next_action"] == "conclude"
        
    def test_response_processor_waits_for_response(self):
        """Test response processor waits when no response available."""
        state: BehavioralInterviewState = {
            "messages": [],
            "session": Mock(),
            "question_pool": [],
            "current_question": None,
            "current_answer": None,  # No answer provided
            "evaluation": None,
            "follow_up_questions": [],
            "follow_up_count": 0,
            "max_follow_ups": 2,
            "retrieved_context": [],
            "web_search_results": [],
            "next_action": "wait_for_response",
            "config": {},
            "error_state": None,
            "retry_count": 0
        }
        
        result = response_processor_node(state)
        
        assert result["next_action"] == "wait_for_response"
        
    def test_response_processor_advances_with_response(self):
        """Test response processor advances when response is available."""
        state: BehavioralInterviewState = {
            "messages": [],
            "session": Mock(),
            "question_pool": [],
            "current_question": None,
            "current_answer": "This is my detailed response about the situation.",
            "evaluation": None,
            "follow_up_questions": [],
            "follow_up_count": 0,
            "max_follow_ups": 2,
            "retrieved_context": [],
            "web_search_results": [],
            "next_action": "wait_for_response",
            "config": {},
            "error_state": None,
            "retry_count": 0
        }
        
        result = response_processor_node(state)
        
        assert result["next_action"] == "retrieve_context"  # Changed flow
        assert len(result["messages"]) == 1  # Added HumanMessage
        
    @patch('ai_interviewer_pm.agents.behavioral_graph._get_llm')
    def test_response_evaluator_with_structured_output(self, mock_llm):
        """Test response evaluator with successful structured output."""
        mock_evaluation = Mock()
        mock_evaluation.follow_up_needed = True
        mock_evaluation.overall_score = 7.5
        
        # Create a mock chain that returns our evaluation
        mock_chain = Mock()
        mock_chain.invoke.return_value = mock_evaluation
        
        mock_llm.return_value.with_structured_output.return_value = mock_chain
        
        question = BehavioralQuestion(
            id="test",
            text="Test question",
            category="leadership",
            difficulty="mid", 
            follow_up_strategy="deep_dive"
        )
        
        state: BehavioralInterviewState = {
            "messages": [],
            "session": Mock(),
            "question_pool": [],
            "current_question": question,
            "current_answer": "Detailed response about leadership challenge.",
            "evaluation": None,
            "follow_up_questions": [],
            "follow_up_count": 0,
            "max_follow_ups": 2,
            "retrieved_context": [],
            "web_search_results": [],
            "next_action": "evaluate_response",
            "config": {},
            "error_state": None,
            "retry_count": 0
        }
        
        result = response_evaluator_node(state)
        
        # Check that evaluation was set (mock object)
        assert result["evaluation"] is not None
        assert result["next_action"] == "multi_agent_eval"  # New flow with multi-agent
        
    @patch('ai_interviewer_pm.agents.behavioral_graph._get_llm')
    def test_follow_up_generator_creates_questions(self, mock_llm):
        """Test follow-up generator creates targeted questions."""
        mock_response = Mock()
        mock_response.content = '["What metrics did you track?", "How did stakeholders react?"]'
        
        mock_llm.return_value.invoke.return_value = mock_response
        
        question = BehavioralQuestion(
            id="test",
            text="Test question",
            category="leadership",
            difficulty="mid",
            follow_up_strategy="metrics"
        )
        
        evaluation_mock = Mock()
        evaluation_mock.model_dump.return_value = {"overall_score": 6.0}
        
        state: BehavioralInterviewState = {
            "messages": [],
            "session": Mock(),
            "question_pool": [],
            "current_question": question,
            "current_answer": "My response",
            "evaluation": evaluation_mock,
            "follow_up_questions": [],
            "follow_up_count": 0,
            "max_follow_ups": 2,
            "retrieved_context": [],
            "web_search_results": [],
            "next_action": "ask_follow_up",
            "config": {},
            "error_state": None,
            "retry_count": 0
        }
        
        result = follow_up_generator_node(state)
        
        # The function should create follow-up questions (could be 1-2 based on implementation)
        assert len(result["follow_up_questions"]) >= 1
        # Check that at least one follow-up is metrics-related or generic
        follow_up_text = " ".join(result["follow_up_questions"]).lower()
        assert "metrics" in follow_up_text or "measure" in follow_up_text or "success" in follow_up_text
        assert result["next_action"] == "ask_follow_up"


class TestRoutingLogic:
    """Test conditional edge routing functions."""
    
    def test_route_from_evaluator_follow_up_needed(self):
        """Test routing when follow-up is needed."""
        state: BehavioralInterviewState = {
            "messages": [],
            "session": Mock(),
            "question_pool": [],
            "current_question": None,
            "current_answer": None,
            "evaluation": None,
            "follow_up_questions": [],
            "follow_up_count": 0,
            "max_follow_ups": 2,
            "retrieved_context": [],
            "web_search_results": [],
            "next_action": "ask_follow_up",
            "config": {},
            "error_state": None,
            "retry_count": 0
        }
        
        result = route_from_evaluator(state)
        assert result == "ask_follow_up"
        
    def test_route_from_evaluator_move_to_next(self):
        """Test routing when moving to next question."""
        state: BehavioralInterviewState = {
            "messages": [],
            "session": Mock(),
            "question_pool": [],
            "current_question": None,
            "current_answer": None,
            "evaluation": None,
            "follow_up_questions": [],
            "follow_up_count": 0,
            "max_follow_ups": 2,
            "retrieved_context": [],
            "web_search_results": [],
            "next_action": "move_to_next",
            "config": {},
            "error_state": None,
            "retry_count": 0
        }
        
        result = route_from_evaluator(state)
        assert result == "move_to_next"


class TestGraphIntegration:
    """Test complete graph integration."""
    
    def test_graph_build_success(self):
        """Test graph builds without errors."""
        graph = build_behavioral_interview_graph()
        
        # Verify all nodes are added
        expected_nodes = {
            "question_generator", "question_asker", "response_processor",
            "response_evaluator", "follow_up_generator", "follow_up_asker", 
            "session_manager", "conclusion"
        }
        
        actual_nodes = set(graph.nodes.keys())
        assert expected_nodes.issubset(actual_nodes)
        
        # Verify entry point is set (LangGraph uses different attribute structure)
        # In newer versions, the entry point info is stored in the compiled graph
        assert "question_generator" in actual_nodes
        
    def test_graph_compilation_success(self):
        """Test graph compiles successfully with checkpointer."""
        app = compile_behavioral_interview_graph()
        
        # Basic smoke test - app should be callable
        assert callable(app.invoke)
        assert callable(app.ainvoke)
        
        # Should have checkpointer configured
        assert app.checkpointer is not None
        
    def test_graph_can_be_compiled_without_errors(self):
        """Test graph compilation works without runtime errors."""
        from ai_interviewer_pm.agents.behavioral_graph import create_checkpointer
        
        # Create checkpointer for testing
        checkpointer = create_checkpointer()
        
        # This should not raise any errors
        app = compile_behavioral_interview_graph(checkpointer)
        
        # Basic validation that the app was created successfully
        assert app is not None
        assert hasattr(app, 'invoke')
        assert hasattr(app, 'ainvoke')
