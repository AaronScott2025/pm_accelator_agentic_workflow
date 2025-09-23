"""Tests for GRAIL rubric evaluation system."""

from __future__ import annotations

import pytest

from ai_interviewer_pm.agents.grail_rubric import (
    GRAILEvaluation,
    GRAILEvaluator,
    GRAILScore,
    create_grail_evaluator,
)


@pytest.fixture
def sample_question() -> str:
    """Sample PM interview question."""
    return "Tell me about a time when you had to prioritize features with limited resources."


@pytest.fixture
def strong_answer() -> str:
    """Strong PM answer with GRAIL elements."""
    return """
    At my previous company, we faced a critical decision point when we had to choose 
    between three major features with only resources for one.
    
    GOAL: Our objective was to increase user retention by 20% within Q3 while maintaining 
    our development velocity. This aligned with our company's OKR of improving user engagement.
    
    RESOURCES: We had a team of 5 engineers for 6 weeks, $50K budget for tools/infrastructure, 
    and limited design resources. The main constraint was our upcoming compliance deadline 
    that would require 2 engineers in week 5-6.
    
    ACTIONS: I led a structured decision process:
    1. Conducted user research with 50 power users to understand pain points
    2. Created an opportunity sizing framework scoring each feature on impact vs effort
    3. Ran a 2-day design sprint with stakeholders to prototype the top option
    4. Built consensus through data-driven presentations to leadership
    
    IMPACT: The prioritized feature (smart notifications) resulted in:
    - 28% increase in user retention (exceeding our 20% goal)
    - 15% reduction in support tickets
    - $2M additional ARR from retained customers
    - Feature shipped 1 week ahead of schedule
    
    LEARNING: This experience taught me that involving stakeholders early in the prioritization
    process reduces friction later. I've since implemented a quarterly prioritization ritual
    that has improved our planning efficiency by 30%.
    """


@pytest.fixture
def weak_answer() -> str:
    """Weak PM answer lacking GRAIL elements."""
    return """
    We had to choose between different features at my last job. It was challenging because
    everyone wanted their feature built. I talked to some people and we decided to build
    the one that seemed most important. It worked out okay in the end.
    """


class TestGRAILScore:
    """Test GRAIL score model."""
    
    def test_score_creation(self) -> None:
        """Test creating a GRAIL score."""
        score = GRAILScore(
            score=8.5,
            evidence=["Clear goal stated", "Metrics provided"],
            missing_elements=["Could elaborate on alternatives"],
            strength_level="strong"
        )
        
        assert score.score == 8.5
        assert len(score.evidence) == 2
        assert len(score.missing_elements) == 1
        assert score.strength_level == "strong"
    
    def test_score_validation(self) -> None:
        """Test score validation."""
        with pytest.raises(ValueError):
            GRAILScore(
                score=11,  # Invalid: > 10
                evidence=[],
                missing_elements=[],
                strength_level="strong"
            )


class TestGRAILEvaluator:
    """Test GRAIL evaluator functionality."""
    
    def test_evaluator_creation(self) -> None:
        """Test creating evaluator."""
        evaluator = create_grail_evaluator()
        assert isinstance(evaluator, GRAILEvaluator)
    
    def test_evaluate_component(self, sample_question: str, strong_answer: str) -> None:
        """Test evaluating a single GRAIL component."""
        evaluator = GRAILEvaluator()
        
        goal_score = evaluator.evaluate_component(
            "goal",
            strong_answer,
            sample_question
        )
        
        assert isinstance(goal_score, GRAILScore)
        assert 0 <= goal_score.score <= 10
        assert goal_score.strength_level in ["weak", "developing", "proficient", "strong", "exceptional"]
    
    @pytest.mark.skip(reason="Requires API key")
    def test_full_evaluation(self, sample_question: str, strong_answer: str) -> None:
        """Test full GRAIL evaluation."""
        evaluator = GRAILEvaluator()
        
        evaluation = evaluator.evaluate(
            sample_question,
            strong_answer,
            context=None,
            question_category="prioritization"
        )
        
        assert isinstance(evaluation, GRAILEvaluation)
        assert 0 <= evaluation.overall_score <= 10
        assert evaluation.goal_score is not None
        assert evaluation.resources_score is not None
        assert evaluation.actions_score is not None
        assert evaluation.impact_score is not None
        assert evaluation.learning_score is not None
    
    def test_weighted_score_calculation(self) -> None:
        """Test weighted score calculation."""
        evaluator = GRAILEvaluator()
        
        mock_scores = {
            "goal": GRAILScore(score=8, evidence=[], missing_elements=[], strength_level="strong"),
            "resources": GRAILScore(score=7, evidence=[], missing_elements=[], strength_level="proficient"),
            "actions": GRAILScore(score=9, evidence=[], missing_elements=[], strength_level="strong"),
            "impact": GRAILScore(score=8, evidence=[], missing_elements=[], strength_level="strong"),
            "learning": GRAILScore(score=6, evidence=[], missing_elements=[], strength_level="developing"),
        }
        
        # Test default weights
        score = evaluator.calculate_weighted_score(mock_scores)
        assert 7 <= score <= 8  # Should be around 7.7
        
        # Test category-specific weights
        score_leadership = evaluator.calculate_weighted_score(mock_scores, "leadership")
        score_prioritization = evaluator.calculate_weighted_score(mock_scores, "prioritization")
        
        # Scores should differ based on weights
        assert score_leadership != score_prioritization
    
    def test_competency_mapping(self) -> None:
        """Test mapping GRAIL scores to competencies."""
        evaluator = GRAILEvaluator()
        
        mock_scores = {
            "goal": GRAILScore(score=9, evidence=[], missing_elements=[], strength_level="exceptional"),
            "resources": GRAILScore(score=5, evidence=[], missing_elements=[], strength_level="developing"),
            "actions": GRAILScore(score=7, evidence=[], missing_elements=[], strength_level="proficient"),
            "impact": GRAILScore(score=8, evidence=[], missing_elements=[], strength_level="strong"),
            "learning": GRAILScore(score=4, evidence=[], missing_elements=[], strength_level="weak"),
        }
        
        competencies = evaluator.map_competencies(mock_scores)
        
        assert "strategic_thinking" in competencies
        assert "Demonstrated strongly" in competencies["strategic_thinking"]
        assert "resource_management" in competencies
        assert "Developing" in competencies["resource_management"]
    
    def test_improvement_recommendations(self) -> None:
        """Test generating improvement recommendations."""
        evaluator = GRAILEvaluator()
        
        mock_evaluation = GRAILEvaluation(
            goal_score=GRAILScore(
                score=5,
                evidence=[],
                missing_elements=["Clearer business alignment needed"],
                strength_level="developing"
            ),
            resources_score=GRAILScore(
                score=8,
                evidence=[],
                missing_elements=[],
                strength_level="strong"
            ),
            actions_score=GRAILScore(
                score=6,
                evidence=[],
                missing_elements=["More specific actions", "Decision rationale"],
                strength_level="proficient"
            ),
            impact_score=GRAILScore(
                score=4,
                evidence=[],
                missing_elements=["Quantified metrics needed"],
                strength_level="weak"
            ),
            learning_score=GRAILScore(
                score=7,
                evidence=[],
                missing_elements=[],
                strength_level="proficient"
            ),
            overall_score=6.0,
            overall_assessment="Developing PM skills",
            pm_competency_mapping={}
        )
        
        recommendations = evaluator.get_improvement_recommendations(mock_evaluation)
        
        assert len(recommendations) > 0
        assert len(recommendations) <= 5
        assert any("[GOAL]" in r for r in recommendations)
        assert any("[IMPACT]" in r for r in recommendations)
