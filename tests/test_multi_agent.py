"""Tests for multi-agent evaluation system."""

from __future__ import annotations

import pytest

from ai_interviewer_pm.agents.multi_agent_evaluator import (
    AgentEvaluation,
    CommunicationSkillsAgent,
    ConsensusEvaluation,
    CustomerFocusAgent,
    LeadershipEvaluationAgent,
    MultiAgentEvaluator,
    StrategicThinkingAgent,
    TechnicalAssessmentAgent,
    create_multi_agent_evaluator,
)


@pytest.fixture
def sample_question() -> str:
    """Sample PM interview question."""
    return "Describe a time when you had to make a difficult product decision with incomplete information."


@pytest.fixture
def sample_answer() -> str:
    """Sample PM answer."""
    return """
    Last year, we had to decide whether to invest in a new AI feature with only 60% confidence 
    in the technical feasibility. I gathered available data from user research showing strong 
    demand, created a risk matrix, and proposed a phased approach starting with a prototype. 
    This allowed us to validate assumptions before full investment. The feature ultimately 
    increased engagement by 35%.
    """


class TestEvaluationAgents:
    """Test individual evaluation agents."""
    
    def test_technical_agent_creation(self) -> None:
        """Test creating technical assessment agent."""
        agent = TechnicalAssessmentAgent()
        assert agent.name == "Technical Assessment"
        assert agent.focus_area == "technical_skills"
    
    def test_leadership_agent_creation(self) -> None:
        """Test creating leadership evaluation agent."""
        agent = LeadershipEvaluationAgent()
        assert agent.name == "Leadership Evaluation"
        assert agent.focus_area == "leadership"
    
    def test_communication_agent_creation(self) -> None:
        """Test creating communication skills agent."""
        agent = CommunicationSkillsAgent()
        assert agent.name == "Communication Skills"
        assert agent.focus_area == "communication"
    
    def test_strategic_agent_creation(self) -> None:
        """Test creating strategic thinking agent."""
        agent = StrategicThinkingAgent()
        assert agent.name == "Strategic Thinking"
        assert agent.focus_area == "strategy"
    
    def test_customer_agent_creation(self) -> None:
        """Test creating customer focus agent."""
        agent = CustomerFocusAgent()
        assert agent.name == "Customer Focus"
        assert agent.focus_area == "customer_centricity"
    
    def test_agent_evaluation_prompt(self) -> None:
        """Test that agents have unique evaluation prompts."""
        tech_agent = TechnicalAssessmentAgent()
        lead_agent = LeadershipEvaluationAgent()
        
        tech_prompt = tech_agent.get_evaluation_prompt()
        lead_prompt = lead_agent.get_evaluation_prompt()
        
        assert tech_prompt != lead_prompt
        assert "technical" in tech_prompt.lower()
        assert "leadership" in lead_prompt.lower()
    
    @pytest.mark.skip(reason="Requires API key")
    def test_single_agent_evaluation(self, sample_question: str, sample_answer: str) -> None:
        """Test single agent evaluation."""
        agent = TechnicalAssessmentAgent()
        
        evaluation = agent.evaluate(sample_question, sample_answer)
        
        assert isinstance(evaluation, AgentEvaluation)
        assert evaluation.agent_name == "Technical Assessment"
        assert 0 <= evaluation.score <= 10
        assert 0 <= evaluation.confidence <= 1
        assert len(evaluation.key_observations) > 0
        assert len(evaluation.strengths) > 0
        assert len(evaluation.improvements) > 0


class TestMultiAgentEvaluator:
    """Test multi-agent evaluator orchestration."""
    
    def test_evaluator_creation_all_agents(self) -> None:
        """Test creating evaluator with all agents."""
        evaluator = create_multi_agent_evaluator(use_all_agents=True)
        assert len(evaluator.agents) == 5
    
    def test_evaluator_creation_partial_agents(self) -> None:
        """Test creating evaluator with partial agents."""
        evaluator = create_multi_agent_evaluator(use_all_agents=False)
        assert len(evaluator.agents) == 3
    
    def test_custom_agent_list(self) -> None:
        """Test creating evaluator with custom agent list."""
        agents = [
            TechnicalAssessmentAgent(),
            LeadershipEvaluationAgent(),
        ]
        evaluator = MultiAgentEvaluator(agents)
        assert len(evaluator.agents) == 2
    
    @pytest.mark.skip(reason="Requires API key")
    def test_parallel_evaluation(self, sample_question: str, sample_answer: str) -> None:
        """Test parallel evaluation with multiple agents."""
        evaluator = MultiAgentEvaluator([
            TechnicalAssessmentAgent(),
            LeadershipEvaluationAgent(),
            CommunicationSkillsAgent(),
        ])
        
        evaluations = evaluator.evaluate_parallel(sample_question, sample_answer)
        
        assert len(evaluations) == 3
        assert all(isinstance(e, AgentEvaluation) for e in evaluations)
        
        # Check that different agents gave evaluations
        agent_names = {e.agent_name for e in evaluations}
        assert len(agent_names) == 3
    
    def test_consensus_building(self) -> None:
        """Test building consensus from evaluations."""
        evaluations = [
            AgentEvaluation(
                agent_name="Agent1",
                score=7.5,
                confidence=0.8,
                key_observations=["Good structure"],
                strengths=["Clear communication", "Data-driven"],
                improvements=["More metrics needed"],
                rationale="Strong response"
            ),
            AgentEvaluation(
                agent_name="Agent2",
                score=8.0,
                confidence=0.9,
                key_observations=["Strong leadership"],
                strengths=["Clear communication", "Strategic thinking"],
                improvements=["More metrics needed", "Team impact"],
                rationale="Excellent approach"
            ),
            AgentEvaluation(
                agent_name="Agent3",
                score=6.5,
                confidence=0.7,
                key_observations=["Needs depth"],
                strengths=["Problem solving"],
                improvements=["More metrics needed", "Stakeholder management"],
                rationale="Adequate but could improve"
            ),
        ]
        
        evaluator = MultiAgentEvaluator([])
        consensus = evaluator.build_consensus(evaluations)
        
        assert isinstance(consensus, ConsensusEvaluation)
        assert 7 <= consensus.final_score <= 8  # Weighted average
        assert 0.7 <= consensus.confidence <= 0.9
        assert "Clear communication" in consensus.consensus_strengths
        assert "More metrics needed" in consensus.consensus_improvements
        assert len(consensus.agent_evaluations) == 3
    
    def test_divergence_identification(self) -> None:
        """Test identifying divergent opinions."""
        evaluations = [
            AgentEvaluation(
                agent_name="Optimist",
                score=9.0,
                confidence=0.9,
                key_observations=[],
                strengths=[],
                improvements=[],
                rationale="Excellent"
            ),
            AgentEvaluation(
                agent_name="Pessimist",
                score=3.0,
                confidence=0.9,
                key_observations=[],
                strengths=[],
                improvements=[],
                rationale="Poor"
            ),
            AgentEvaluation(
                agent_name="Moderate",
                score=6.0,
                confidence=0.8,
                key_observations=[],
                strengths=[],
                improvements=[],
                rationale="Average"
            ),
        ]
        
        evaluator = MultiAgentEvaluator([])
        divergence = evaluator.identify_divergence(evaluations)
        
        assert "split_opinion" in divergence
        assert "Optimist" in str(divergence["split_opinion"])
        assert "Pessimist" in str(divergence["split_opinion"])
    
    def test_recommendation_generation(self) -> None:
        """Test generating final recommendations."""
        evaluator = MultiAgentEvaluator([])
        
        # Test strong candidate
        rec_strong = evaluator.generate_recommendation(
            8.5,
            ["Leadership", "Technical skills"],
            ["Written communication"]
        )
        assert "Strong hire" in rec_strong
        
        # Test borderline candidate
        rec_borderline = evaluator.generate_recommendation(
            5.5,
            ["Problem solving"],
            ["Leadership", "Communication"]
        )
        assert "Borderline" in rec_borderline
        
        # Test weak candidate
        rec_weak = evaluator.generate_recommendation(
            3.0,
            [],
            ["Everything"]
        )
        assert "Not ready" in rec_weak
    
    def test_empty_evaluations_handling(self) -> None:
        """Test handling empty evaluations list."""
        evaluator = MultiAgentEvaluator([])
        
        with pytest.raises(ValueError, match="No evaluations provided"):
            evaluator.build_consensus([])
    
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires API key")
    async def test_async_evaluation(self, sample_question: str, sample_answer: str) -> None:
        """Test asynchronous evaluation."""
        evaluator = MultiAgentEvaluator([
            TechnicalAssessmentAgent(),
            LeadershipEvaluationAgent(),
        ])
        
        evaluations = await evaluator.evaluate_async(sample_question, sample_answer)
        
        assert len(evaluations) == 2
        assert all(isinstance(e, AgentEvaluation) for e in evaluations)