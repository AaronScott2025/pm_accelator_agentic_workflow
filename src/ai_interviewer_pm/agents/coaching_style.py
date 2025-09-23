"""Dr. Nancy's coaching style implementation for PM interview feedback."""

from __future__ import annotations

from typing import Any, Optional

from src.ai_interviewer_pm.settings import settings
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


class DrNancyCoachingStyle:
    """Implements Dr. Nancy's specific coaching patterns and feedback style."""

    COACHING_PRINCIPLES = {
        "empathetic": "Start with understanding and acknowledgment of effort",
        "specific": "Provide concrete, actionable feedback with examples",
        "growth_mindset": "Frame improvements as opportunities for growth",
        "structured": "Use clear frameworks like STAR and GRAIL",
        "encouraging": "Balance critique with genuine recognition of strengths",
        "practical": "Connect feedback to real PM scenarios and outcomes",
    }

    COACHING_TEMPLATES = {
        "behavioral": """As a PM coach following Dr. Nancy's methodology, evaluate this response with:
        
1. **Empathetic Recognition**: Start by acknowledging what the candidate did well
2. **Structural Analysis**: Assess STAR format completeness (Situation, Task, Action, Result)
3. **PM Competency Mapping**: Connect to key PM skills (leadership, prioritization, stakeholder management)
4. **Growth Opportunities**: Frame improvements as skill development areas
5. **Practical Application**: Suggest how to apply lessons to future PM scenarios

Remember: Be supportive yet honest, specific yet encouraging.""",
        "technical": """Following Dr. Nancy's technical interview coaching approach:
        
1. **Problem Understanding**: Did they clarify requirements and constraints?
2. **Structured Thinking**: Did they use frameworks appropriately?
3. **Trade-off Analysis**: Did they consider multiple solutions?
4. **Metrics Focus**: Did they define success metrics?
5. **Implementation Reality**: Did they consider practical execution?

Provide feedback that builds confidence while addressing gaps.""",
        "product_sense": """Using Dr. Nancy's product sense evaluation framework:
        
1. **User Empathy**: How well did they understand user needs?
2. **Market Awareness**: Did they consider competitive landscape?
3. **Business Alignment**: Did they connect to business objectives?
4. **Innovation vs Feasibility**: Did they balance creativity with practicality?
5. **Iteration Mindset**: Did they show willingness to test and learn?

Guide them toward stronger product thinking patterns.""",
    }

    FEEDBACK_STARTERS = {
        "strong": [
            "I really appreciate how you...",
            "Your response effectively demonstrated...",
            "You showed strong PM thinking when you...",
            "What stood out positively was...",
        ],
        "improvement": [
            "To make your answer even stronger, consider...",
            "One area to develop further would be...",
            "Next time, you might also include...",
            "A powerful addition would be to...",
        ],
        "encouragement": [
            "You're on the right track with...",
            "Keep building on your strength in...",
            "With practice, you'll excel at...",
            "Your potential really shows when...",
        ],
    }

    def __init__(self) -> None:
        """Initialize coaching style handler."""
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            api_key=settings.openai_api_key,
        )

    def get_coaching_prompt(self, question_category: str, performance_level: str = "mid") -> str:
        """Get Dr. Nancy's coaching prompt based on question type and performance."""
        base_template = self.COACHING_TEMPLATES.get(
            "behavioral", self.COACHING_TEMPLATES["behavioral"]
        )

        performance_adjustments = {
            "struggling": "Be extra encouraging and break down concepts simply. Focus on fundamentals.",
            "developing": "Balance encouragement with specific improvement areas. Provide clear examples.",
            "strong": "Challenge them to go deeper. Push for more strategic thinking.",
            "excellent": "Focus on nuanced improvements and senior-level considerations.",
        }

        adjustment = performance_adjustments.get(
            performance_level, performance_adjustments["developing"]
        )

        return f"{base_template}\n\nPerformance Context: {adjustment}"

    def filter_dr_nancy_content(self, contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Prioritize Dr. Nancy's content in retrieved contexts."""
        dr_nancy_contexts = []
        other_contexts = []

        for context in contexts:
            source = context.get("source", "").lower()
            text = context.get("text", "").lower()

            if any(
                indicator in source + text
                for indicator in ["dr. nancy", "nancy li", "coach", "office hour"]
            ):
                context["priority"] = 1.5
                dr_nancy_contexts.append(context)
            else:
                context["priority"] = 1.0
                other_contexts.append(context)

        dr_nancy_contexts.sort(key=lambda x: x.get("score", 0), reverse=True)
        other_contexts.sort(key=lambda x: x.get("score", 0), reverse=True)

        return dr_nancy_contexts[:3] + other_contexts[:2]

    def extract_coaching_patterns(self, contexts: list[dict[str, Any]]) -> dict[str, Any]:
        """Extract Dr. Nancy's coaching patterns from contexts."""
        patterns = {
            "feedback_style": [],
            "common_mistakes": [],
            "success_patterns": [],
            "frameworks_used": [],
        }

        for context in contexts:
            text = context.get("text", "")

            if "feedback" in text.lower():
                patterns["feedback_style"].append(text[:200])

            if any(word in text.lower() for word in ["mistake", "avoid", "don't"]):
                patterns["common_mistakes"].append(text[:200])

            if any(word in text.lower() for word in ["success", "effective", "strong"]):
                patterns["success_patterns"].append(text[:200])

            if any(framework in text.upper() for framework in ["STAR", "GRAIL", "CIRCLES", "RICE"]):
                patterns["frameworks_used"].append(text[:200])

        return patterns

    def generate_coaching_feedback(
        self,
        question: str,
        answer: str,
        evaluation_scores: dict[str, float],
        coaching_patterns: Optional[dict[str, Any]] = None,
    ) -> str:
        """Generate feedback in Dr. Nancy's coaching style."""
        avg_score = (
            sum(evaluation_scores.values()) / len(evaluation_scores) if evaluation_scores else 0
        )

        if avg_score < 5:
            performance_level = "struggling"
        elif avg_score < 7:
            performance_level = "developing"
        elif avg_score < 8.5:
            performance_level = "strong"
        else:
            performance_level = "excellent"

        coaching_prompt = self.get_coaching_prompt("behavioral", performance_level)

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=coaching_prompt),
                (
                    "human",
                    """Question: {question}
Answer: {answer}
Scores: {scores}
Coaching Context: {patterns}

Generate supportive yet honest feedback in Dr. Nancy's style. Include:
1. Positive recognition (1-2 sentences)
2. Specific improvement areas (2-3 points)
3. Actionable next steps (2-3 suggestions)
4. Encouraging conclusion (1 sentence)""",
                ),
            ]
        )

        chain = prompt | self.llm

        response = chain.invoke(
            {
                "question": question,
                "answer": answer,
                "scores": str(evaluation_scores),
                "patterns": (
                    str(coaching_patterns) if coaching_patterns else "Standard coaching approach"
                ),
            }
        )

        return response.content

    def adapt_follow_up_questions(
        self, original_question: str, answer: str, performance_level: str, follow_ups: list[str]
    ) -> list[str]:
        """Adapt follow-up questions based on Dr. Nancy's coaching approach."""
        if performance_level == "struggling":
            adapted = [
                f"Let me help you structure this better. {follow_ups[0] if follow_ups else 'Can you break down your approach step by step?'}",
                "What specific challenge did you face, and how did you initially approach it?",
                "Think about the stakeholders involved - who were they and what did they need?",
            ]
        elif performance_level == "developing":
            adapted = [
                follow_ups[0] if follow_ups else "Can you quantify the impact of your actions?",
                "What alternative approaches did you consider?",
                "How did you measure success in this situation?",
            ]
        elif performance_level == "strong":
            adapted = [
                "How would you handle this differently at a larger scale?",
                "What systemic changes did you implement to prevent similar issues?",
                follow_ups[0] if follow_ups else "How did you influence without authority?",
            ]
        else:
            adapted = (
                follow_ups[:3]
                if follow_ups
                else [
                    "What was the strategic implication of your decision?",
                    "How did this experience shape your PM philosophy?",
                    "What would you advise other PMs facing similar situations?",
                ]
            )

        return adapted[:3]

    def get_encouragement_message(self, score: float, improvement_areas: list[str]) -> str:
        """Generate an encouraging message based on performance."""
        if score < 5:
            base = "You're building important foundations. "
        elif score < 7:
            base = "You're developing strong PM instincts. "
        elif score < 9:
            base = "You're demonstrating excellent PM thinking. "
        else:
            base = "You're showing exceptional PM leadership. "

        if improvement_areas:
            focus = (
                f"Focus on {improvement_areas[0].lower()} to take your answer to the next level."
            )
        else:
            focus = "Keep practicing to refine your storytelling."

        return base + focus


def create_coaching_style_handler() -> DrNancyCoachingStyle:
    """Factory function to create coaching style handler."""
    return DrNancyCoachingStyle()
