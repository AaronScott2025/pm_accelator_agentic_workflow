"""Multi-agent evaluation system for comprehensive PM interview assessment."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from src.ai_interviewer_pm.settings import settings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class AgentEvaluation(BaseModel):
    """Individual agent's evaluation result."""

    agent_name: str = Field(description="Name of the evaluating agent")
    score: float = Field(ge=0, le=10, description="Agent's score")
    confidence: float = Field(ge=0, le=1, description="Confidence in evaluation")
    key_observations: list[str] = Field(description="Key observations from this agent")
    strengths: list[str] = Field(description="Identified strengths")
    improvements: list[str] = Field(description="Suggested improvements")
    rationale: str = Field(description="Reasoning behind the evaluation")


class ConsensusEvaluation(BaseModel):
    """Consensus evaluation from multiple agents."""

    final_score: float = Field(ge=0, le=10, description="Consensus score")
    confidence: float = Field(ge=0, le=1, description="Overall confidence")
    agent_evaluations: list[AgentEvaluation] = Field(description="Individual agent evaluations")
    consensus_strengths: list[str] = Field(description="Agreed upon strengths")
    consensus_improvements: list[str] = Field(description="Agreed upon improvements")
    divergent_opinions: dict[str, list[str]] = Field(description="Areas where agents disagreed")
    recommendation: str = Field(description="Final recommendation")


class EvaluationAgent(ABC):
    """Base class for specialized evaluation agents."""

    def __init__(self, name: str, focus_area: str) -> None:
        """Initialize evaluation agent."""
        self.name = name
        self.focus_area = focus_area
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=settings.openai_api_key)

    @abstractmethod
    def get_evaluation_prompt(self) -> str:
        """Get the agent's specific evaluation prompt."""
        pass

    def evaluate(
        self, question: str, answer: str, context: Optional[list[dict[str, Any]]] = None
    ) -> AgentEvaluation:
        """Evaluate response from this agent's perspective."""
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.get_evaluation_prompt()),
                (
                    "human",
                    """Question: {question}
Answer: {answer}
Additional Context: {context}

Provide your evaluation as JSON with:
- score (0-10)
- confidence (0-1)
- key_observations (list of 3-5 observations)
- strengths (list of 2-3 strengths)
- improvements (list of 2-3 improvements)
- rationale (brief explanation of your evaluation)""",
                ),
            ]
        )

        structured_llm = self.llm.with_structured_output(AgentEvaluation)
        chain = prompt | structured_llm

        evaluation = chain.invoke(
            {
                "question": question,
                "answer": answer,
                "context": str(context[:2]) if context else "No additional context",
            }
        )

        evaluation.agent_name = self.name
        return evaluation


class TechnicalAssessmentAgent(EvaluationAgent):
    """Agent focused on technical competency and problem-solving."""

    def __init__(self) -> None:
        """Initialize technical assessment agent."""
        super().__init__("Technical Assessment", "technical_skills")

    def get_evaluation_prompt(self) -> str:
        """Get technical evaluation prompt."""
        return """You are a technical PM evaluation specialist focusing on:

1. **Problem Decomposition**: How well did they break down the problem?
2. **Technical Understanding**: Did they grasp technical constraints and possibilities?
3. **Data-Driven Approach**: Did they use data and metrics appropriately?
4. **System Thinking**: Did they consider system-wide implications?
5. **Technical Communication**: How well did they explain technical concepts?

Evaluate responses through a technical lens, looking for:
- Structured problem-solving approach
- Understanding of technical trade-offs
- Appropriate use of technical concepts
- Scalability considerations
- Performance and reliability thinking"""


class LeadershipEvaluationAgent(EvaluationAgent):
    """Agent focused on leadership and influence skills."""

    def __init__(self) -> None:
        """Initialize leadership evaluation agent."""
        super().__init__("Leadership Evaluation", "leadership")

    def get_evaluation_prompt(self) -> str:
        """Get leadership evaluation prompt."""
        return """You are a leadership assessment specialist for PM roles focusing on:

1. **Vision & Direction**: Did they set clear vision and direction?
2. **Influence Without Authority**: How did they influence stakeholders?
3. **Team Empowerment**: Did they enable and empower their team?
4. **Conflict Resolution**: How did they handle conflicts and disagreements?
5. **Decision Making**: Did they show decisive leadership when needed?

Evaluate responses for leadership qualities:
- Inspiring and motivating others
- Building consensus and alignment
- Taking ownership and accountability
- Developing others and delegation
- Leading through ambiguity"""


class CommunicationSkillsAgent(EvaluationAgent):
    """Agent focused on communication and stakeholder management."""

    def __init__(self) -> None:
        """Initialize communication skills agent."""
        super().__init__("Communication Skills", "communication")

    def get_evaluation_prompt(self) -> str:
        """Get communication evaluation prompt."""
        return """You are a communication assessment specialist for PM interviews focusing on:

1. **Clarity**: Is the response clear and well-structured?
2. **Audience Awareness**: Did they tailor communication to stakeholders?
3. **Storytelling**: How compelling is their narrative?
4. **Active Listening**: Evidence of understanding others' perspectives?
5. **Executive Communication**: Can they communicate up effectively?

Evaluate communication effectiveness:
- STAR structure completeness
- Conciseness without losing important details
- Use of specific examples and evidence
- Ability to explain complex concepts simply
- Professional tone and presentation"""


class StrategicThinkingAgent(EvaluationAgent):
    """Agent focused on strategic and business thinking."""

    def __init__(self) -> None:
        """Initialize strategic thinking agent."""
        super().__init__("Strategic Thinking", "strategy")

    def get_evaluation_prompt(self) -> str:
        """Get strategic thinking evaluation prompt."""
        return """You are a strategic thinking assessment specialist for PM roles focusing on:

1. **Business Acumen**: Understanding of business goals and metrics?
2. **Market Awareness**: Consideration of market and competitive factors?
3. **Long-term Vision**: Balance of short-term and long-term thinking?
4. **Strategic Alignment**: Connection to company strategy and OKRs?
5. **Innovation**: Creative problem-solving and innovation?

Evaluate strategic thinking:
- Big picture thinking
- Understanding of business impact
- Risk assessment and mitigation
- Opportunity identification
- Strategic prioritization"""


class CustomerFocusAgent(EvaluationAgent):
    """Agent focused on customer-centricity and user empathy."""

    def __init__(self) -> None:
        """Initialize customer focus agent."""
        super().__init__("Customer Focus", "customer_centricity")

    def get_evaluation_prompt(self) -> str:
        """Get customer focus evaluation prompt."""
        return """You are a customer-centricity assessment specialist for PM interviews focusing on:

1. **User Empathy**: Understanding of user needs and pain points?
2. **Customer Research**: Use of customer insights and data?
3. **User Experience**: Consideration of end-to-end user journey?
4. **Customer Value**: Focus on delivering customer value?
5. **Feedback Integration**: How they incorporate customer feedback?

Evaluate customer focus:
- Deep understanding of users
- Customer problem validation
- User-centric decision making
- Balancing user needs with business goals
- Customer success metrics"""


class MultiAgentEvaluator:
    """Orchestrator for multi-agent evaluation system."""

    def __init__(self, agents: Optional[list[EvaluationAgent]] = None) -> None:
        """Initialize multi-agent evaluator."""
        if agents is None:
            agents = [
                TechnicalAssessmentAgent(),
                LeadershipEvaluationAgent(),
                CommunicationSkillsAgent(),
                StrategicThinkingAgent(),
                CustomerFocusAgent(),
            ]
        self.agents = agents
        self.executor = ThreadPoolExecutor(max_workers=max(1, len(agents)))

    def evaluate_parallel(
        self, question: str, answer: str, context: Optional[list[dict[str, Any]]] = None
    ) -> list[AgentEvaluation]:
        """Run all agents in parallel for evaluation."""
        futures = []
        for agent in self.agents:
            future = self.executor.submit(agent.evaluate, question, answer, context)
            futures.append(future)

        evaluations = []
        for future in futures:
            try:
                result = future.result(timeout=30)
                evaluations.append(result)
            except Exception as e:
                print(f"Agent evaluation failed: {e}")
                continue

        return evaluations

    async def evaluate_async(
        self, question: str, answer: str, context: Optional[dict[str, Any]] = None
    ) -> list[AgentEvaluation]:
        """Asynchronously evaluate with all agents."""
        tasks = []
        for agent in self.agents:
            task = asyncio.create_task(asyncio.to_thread(agent.evaluate, question, answer, context))
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        evaluations = []
        for result in results:
            if isinstance(result, AgentEvaluation):
                evaluations.append(result)
            else:
                print(f"Agent evaluation failed: {result}")

        return evaluations

    def build_consensus(self, evaluations: list[AgentEvaluation]) -> ConsensusEvaluation:
        """Build consensus from multiple agent evaluations."""
        if not evaluations:
            raise ValueError("No evaluations provided for consensus")

        weighted_scores = []
        total_confidence = 0

        for eval in evaluations:
            weighted_scores.append(eval.score * eval.confidence)
            total_confidence += eval.confidence

        avg_confidence = total_confidence / len(evaluations)
        final_score = sum(weighted_scores) / total_confidence if total_confidence > 0 else 0

        all_strengths = {}
        all_improvements = {}

        for eval in evaluations:
            for strength in eval.strengths:
                all_strengths[strength] = all_strengths.get(strength, 0) + 1
            for improvement in eval.improvements:
                all_improvements[improvement] = all_improvements.get(improvement, 0) + 1

        consensus_threshold = len(evaluations) / 2
        consensus_strengths = [
            s for s, count in all_strengths.items() if count >= consensus_threshold
        ][:3]

        consensus_improvements = [
            i for i, count in all_improvements.items() if count >= consensus_threshold
        ][:3]

        divergent_opinions = self.identify_divergence(evaluations)

        recommendation = self.generate_recommendation(
            final_score, consensus_strengths, consensus_improvements
        )

        return ConsensusEvaluation(
            final_score=round(final_score, 1),
            confidence=round(avg_confidence, 2),
            agent_evaluations=evaluations,
            consensus_strengths=consensus_strengths,
            consensus_improvements=consensus_improvements,
            divergent_opinions=divergent_opinions,
            recommendation=recommendation,
        )

    def identify_divergence(self, evaluations: list[AgentEvaluation]) -> dict[str, list[str]]:
        """Identify areas where agents significantly disagree."""
        divergence = {}

        scores = [e.score for e in evaluations]
        avg_score = sum(scores) / len(scores)
        std_dev = (sum((s - avg_score) ** 2 for s in scores) / len(scores)) ** 0.5

        if std_dev > 2:
            outliers = []
            for eval in evaluations:
                if abs(eval.score - avg_score) > 2:
                    outliers.append(f"{eval.agent_name}: {eval.score} ({eval.rationale[:100]}...)")
            if outliers:
                divergence["score_divergence"] = outliers

        high_scorers = [e.agent_name for e in evaluations if e.score >= 7]
        low_scorers = [e.agent_name for e in evaluations if e.score < 5]

        if high_scorers and low_scorers:
            divergence["split_opinion"] = [
                f"High scores from: {', '.join(high_scorers)}",
                f"Low scores from: {', '.join(low_scorers)}",
            ]

        return divergence

    def generate_recommendation(
        self, score: float, strengths: list[str], improvements: list[str]
    ) -> str:
        """Generate final recommendation based on consensus."""
        if score >= 8:
            level = "Strong hire recommendation"
            action = "Move to next round with focus on senior-level challenges"
        elif score >= 6.5:
            level = "Positive lean"
            action = "Proceed with additional behavioral validation"
        elif score >= 5:
            level = "Borderline"
            action = "Consider for junior role or provide specific feedback for re-application"
        else:
            level = "Not ready"
            action = "Provide developmental feedback and suggest preparation resources"

        recommendation = f"{level}. "

        if strengths:
            recommendation += f"Strong in: {', '.join(strengths[:2])}. "

        if improvements:
            recommendation += f"Development areas: {', '.join(improvements[:2])}. "

        recommendation += f"Recommendation: {action}"

        return recommendation


def create_multi_agent_evaluator(use_all_agents: bool = True) -> MultiAgentEvaluator:
    """Factory function to create multi-agent evaluator."""
    if use_all_agents:
        return MultiAgentEvaluator()
    else:
        return MultiAgentEvaluator(
            [
                LeadershipEvaluationAgent(),
                CommunicationSkillsAgent(),
                StrategicThinkingAgent(),
            ]
        )
