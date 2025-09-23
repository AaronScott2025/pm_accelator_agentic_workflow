"""GRAIL rubric evaluation framework for comprehensive PM interview assessment."""

from __future__ import annotations

from typing import Any, Literal, Optional

from src.ai_interviewer_pm.settings import settings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class GRAILScore(BaseModel):
    """Individual GRAIL component score with detailed breakdown."""

    score: float = Field(ge=0, le=10, description="Component score from 0-10")
    evidence: list[str] = Field(description="Specific evidence from answer supporting score")
    missing_elements: list[str] = Field(description="Elements that would improve the score")
    strength_level: Literal["weak", "developing", "proficient", "strong", "exceptional"] = Field(
        description="Qualitative assessment level"
    )


class GRAILEvaluation(BaseModel):
    """Complete GRAIL evaluation with all components."""

    goal_score: GRAILScore = Field(description="Goal clarity and alignment assessment")
    resources_score: GRAILScore = Field(description="Resources and constraints consideration")
    actions_score: GRAILScore = Field(description="Actions taken and decision-making process")
    impact_score: GRAILScore = Field(description="Impact measurement and quantification")
    learning_score: GRAILScore = Field(description="Learning and iteration demonstrated")

    overall_score: float = Field(ge=0, le=10, description="Weighted overall GRAIL score")
    overall_assessment: str = Field(description="Holistic assessment summary")
    pm_competency_mapping: dict[str, str] = Field(
        description="Mapping to PM competencies demonstrated"
    )


class GRAILEvaluator:
    """Evaluates PM interview responses using the GRAIL framework."""

    GRAIL_RUBRIC = {
        "goal": {
            "description": "Goal clarity and alignment with business objectives",
            "criteria": {
                "weak": "Unclear or missing goals, no business alignment mentioned",
                "developing": "Basic goal stated but lacks clarity or business connection",
                "proficient": "Clear goal with some business alignment",
                "strong": "Well-defined goal clearly tied to business objectives",
                "exceptional": "Crystal clear goal with strong business case and strategic alignment",
            },
            "evaluation_points": [
                "Was the goal clearly articulated?",
                "Was it aligned with business objectives?",
                "Were success criteria defined?",
                "Was the strategic importance explained?",
                "Were stakeholder goals considered?",
            ],
        },
        "resources": {
            "description": "Understanding and management of resources and constraints",
            "criteria": {
                "weak": "No mention of resources or constraints",
                "developing": "Basic acknowledgment of some resources/constraints",
                "proficient": "Good understanding of key resources and constraints",
                "strong": "Comprehensive resource analysis with trade-offs",
                "exceptional": "Masterful resource optimization with creative solutions",
            },
            "evaluation_points": [
                "Were available resources identified?",
                "Were constraints acknowledged?",
                "Were trade-offs discussed?",
                "Was resource allocation strategic?",
                "Were creative solutions proposed?",
            ],
        },
        "actions": {
            "description": "Quality of actions taken and decision-making process",
            "criteria": {
                "weak": "Vague or unclear actions, no clear process",
                "developing": "Basic actions described but lacks structure",
                "proficient": "Clear actions with reasonable decision process",
                "strong": "Well-structured actions with strong rationale",
                "exceptional": "Systematic approach with excellent judgment and execution",
            },
            "evaluation_points": [
                "Were specific actions clearly described?",
                "Was the decision-making process logical?",
                "Were alternatives considered?",
                "Was the approach systematic?",
                "Was execution well-planned?",
            ],
        },
        "impact": {
            "description": "Measurement and quantification of impact",
            "criteria": {
                "weak": "No impact mentioned or purely qualitative",
                "developing": "Some impact described but lacks metrics",
                "proficient": "Clear impact with some quantification",
                "strong": "Well-quantified impact with multiple metrics",
                "exceptional": "Comprehensive impact analysis with short and long-term metrics",
            },
            "evaluation_points": [
                "Was impact clearly measured?",
                "Were specific metrics provided?",
                "Were both quantitative and qualitative impacts covered?",
                "Was the business value demonstrated?",
                "Were long-term effects considered?",
            ],
        },
        "learning": {
            "description": "Learning extraction and iteration mindset",
            "criteria": {
                "weak": "No learning or reflection mentioned",
                "developing": "Basic learning acknowledged",
                "proficient": "Clear learnings with some application",
                "strong": "Strong reflection with actionable insights",
                "exceptional": "Deep insights with systematic improvement approach",
            },
            "evaluation_points": [
                "Were key learnings explicitly stated?",
                "Was there reflection on what could be improved?",
                "Were learnings applied to future situations?",
                "Was there evidence of iteration?",
                "Was a growth mindset demonstrated?",
            ],
        },
    }

    COMPETENCY_MAPPING = {
        "goal": ["strategic_thinking", "business_acumen", "vision_setting"],
        "resources": ["resource_management", "prioritization", "constraint_optimization"],
        "actions": ["execution", "decision_making", "leadership"],
        "impact": ["data_driven", "results_orientation", "measurement"],
        "learning": ["growth_mindset", "adaptability", "continuous_improvement"],
    }

    def __init__(self) -> None:
        """Initialize GRAIL evaluator."""
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, api_key=settings.openai_api_key)

    def evaluate_component(
        self,
        component: str,
        answer: str,
        question: str,
        context: Optional[list[dict[str, Any]]] = None,
    ) -> GRAILScore:
        """Evaluate a single GRAIL component."""
        rubric = self.GRAIL_RUBRIC[component]

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert PM interviewer evaluating responses using the GRAIL framework.
            
Evaluate the {component} component based on:
{description}

Evaluation Criteria:
{criteria}

Key Questions to Consider:
{evaluation_points}

Provide a detailed assessment with specific evidence from the answer.""",
                ),
                (
                    "human",
                    """Question: {question}
Answer: {answer}
Context: {context}

Evaluate the {component} component and return a JSON with:
- score (0-10)
- evidence (list of specific quotes/examples from answer)
- missing_elements (list of what would improve the score)
- strength_level (weak/developing/proficient/strong/exceptional)""",
                ),
            ]
        )

        structured_llm = self.llm.with_structured_output(GRAILScore)
        chain = prompt | structured_llm

        score = chain.invoke(
            {
                "component": component.upper(),
                "description": rubric["description"],
                "criteria": str(rubric["criteria"]),
                "evaluation_points": "\n".join(
                    f"- {point}" for point in rubric["evaluation_points"]
                ),
                "question": question,
                "answer": answer,
                "context": str(context[:2]) if context else "No additional context",
            }
        )

        return score

    def evaluate(
        self,
        question: str,
        answer: str,
        context: Optional[list[dict[str, Any]]] = None,
        question_category: Optional[str] = None,
    ) -> GRAILEvaluation:
        """Perform complete GRAIL evaluation of a response."""
        component_scores = {}

        for component in ["goal", "resources", "actions", "impact", "learning"]:
            component_scores[component] = self.evaluate_component(
                component, answer, question, context
            )

        weighted_score = self.calculate_weighted_score(component_scores, question_category)

        competency_mapping = self.map_competencies(component_scores)

        overall_assessment = self.generate_overall_assessment(
            component_scores, weighted_score, question_category
        )

        return GRAILEvaluation(
            goal_score=component_scores["goal"],
            resources_score=component_scores["resources"],
            actions_score=component_scores["actions"],
            impact_score=component_scores["impact"],
            learning_score=component_scores["learning"],
            overall_score=weighted_score,
            overall_assessment=overall_assessment,
            pm_competency_mapping=competency_mapping,
        )

    def calculate_weighted_score(
        self, scores: dict[str, GRAILScore], question_category: Optional[str] = None
    ) -> float:
        """Calculate weighted overall score based on question type."""
        default_weights = {
            "goal": 0.20,
            "resources": 0.15,
            "actions": 0.25,
            "impact": 0.25,
            "learning": 0.15,
        }

        category_weights = {
            "leadership": {
                "goal": 0.25,
                "resources": 0.10,
                "actions": 0.30,
                "impact": 0.20,
                "learning": 0.15,
            },
            "prioritization": {
                "goal": 0.20,
                "resources": 0.25,
                "actions": 0.20,
                "impact": 0.25,
                "learning": 0.10,
            },
            "stakeholder_management": {
                "goal": 0.25,
                "resources": 0.15,
                "actions": 0.25,
                "impact": 0.20,
                "learning": 0.15,
            },
            "failure_recovery": {
                "goal": 0.15,
                "resources": 0.15,
                "actions": 0.20,
                "impact": 0.20,
                "learning": 0.30,
            },
            "product_decisions": {
                "goal": 0.25,
                "resources": 0.15,
                "actions": 0.20,
                "impact": 0.30,
                "learning": 0.10,
            },
        }

        weights = category_weights.get(question_category, default_weights)

        weighted_sum = sum(scores[component].score * weights[component] for component in scores)

        return round(weighted_sum, 1)

    def map_competencies(self, scores: dict[str, GRAILScore]) -> dict[str, str]:
        """Map GRAIL scores to PM competencies."""
        competency_assessment = {}

        for component, score in scores.items():
            competencies = self.COMPETENCY_MAPPING[component]
            for competency in competencies:
                if score.strength_level in ["strong", "exceptional"]:
                    competency_assessment[competency] = (
                        f"Demonstrated strongly in {component.upper()}"
                    )
                elif score.strength_level == "proficient":
                    competency_assessment[competency] = f"Shows proficiency in {component.upper()}"
                elif score.strength_level == "developing":
                    competency_assessment[competency] = f"Developing through {component.upper()}"
                else:
                    competency_assessment[competency] = f"Needs improvement in {component.upper()}"

        return competency_assessment

    def generate_overall_assessment(
        self, scores: dict[str, GRAILScore], overall_score: float, question_category: Optional[str]
    ) -> str:
        """Generate holistic assessment summary."""
        strengths = [
            component.upper()
            for component, score in scores.items()
            if score.strength_level in ["strong", "exceptional"]
        ]

        improvements = [
            component.upper()
            for component, score in scores.items()
            if score.strength_level in ["weak", "developing"]
        ]

        if overall_score >= 8.5:
            level = "exceptional"
            summary = "Outstanding PM response demonstrating mastery"
        elif overall_score >= 7:
            level = "strong"
            summary = "Strong PM response with good structure"
        elif overall_score >= 5:
            level = "proficient"
            summary = "Solid response with room for enhancement"
        elif overall_score >= 3:
            level = "developing"
            summary = "Developing PM skills, needs more structure"
        else:
            level = "needs improvement"
            summary = "Significant gaps in PM approach"

        assessment = f"{summary}. "

        if strengths:
            assessment += f"Strengths in: {', '.join(strengths)}. "

        if improvements:
            assessment += f"Focus areas: {', '.join(improvements)}. "

        assessment += f"Overall GRAIL score: {overall_score}/10 ({level})"

        return assessment

    def get_improvement_recommendations(self, evaluation: GRAILEvaluation) -> list[str]:
        """Generate specific improvement recommendations based on GRAIL evaluation."""
        recommendations = []

        components = {
            "goal": evaluation.goal_score,
            "resources": evaluation.resources_score,
            "actions": evaluation.actions_score,
            "impact": evaluation.impact_score,
            "learning": evaluation.learning_score,
        }

        for component_name, score in components.items():
            if score.score < 7:
                for missing in score.missing_elements[:2]:
                    recommendations.append(f"[{component_name.upper()}] {missing}")

        if not recommendations:
            recommendations = [
                "Add more specific metrics to quantify impact",
                "Include alternative approaches you considered",
                "Elaborate on long-term learnings and applications",
            ]

        return recommendations[:5]


def create_grail_evaluator() -> GRAILEvaluator:
    """Factory function to create GRAIL evaluator."""
    return GRAILEvaluator()
