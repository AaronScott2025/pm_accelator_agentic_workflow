"""Adaptive questioning system for dynamic interview flow based on performance."""

from __future__ import annotations

from typing import Literal

from ai_interviewer_pm.agents.behavioral_schema import BehavioralQuestion, ResponseEvaluation
from pydantic import BaseModel, Field


class PerformanceMetrics(BaseModel):
    """Performance metrics for adaptive questioning decisions."""

    avg_score: float = Field(description="Average score across evaluations")
    trend: Literal["improving", "stable", "declining"] = Field(description="Performance trend")
    strengths: list[str] = Field(description="Identified strength areas")
    weaknesses: list[str] = Field(description="Identified weakness areas")
    confidence_level: float = Field(ge=0, le=1, description="Confidence in assessment")
    questions_answered: int = Field(description="Number of questions answered")
    time_spent_minutes: float = Field(description="Total time spent")


class AdaptiveDecision(BaseModel):
    """Decision made by adaptive system."""

    action: Literal["continue", "adjust_difficulty", "switch_category", "deep_dive", "conclude"]
    next_question: BehavioralQuestion | None = Field(description="Next question to ask")
    reasoning: str = Field(description="Reasoning for the decision")
    difficulty_adjustment: Literal["easier", "same", "harder"] = Field(
        description="Difficulty adjustment"
    )
    focus_area: str | None = Field(description="Area to focus on")


class AdaptiveQuestionSelector:
    """Manages adaptive question selection based on candidate performance."""

    PERFORMANCE_THRESHOLDS = {
        "struggling": {"min": 0, "max": 4.5},
        "developing": {"min": 4.5, "max": 6.5},
        "proficient": {"min": 6.5, "max": 8.0},
        "strong": {"min": 8.0, "max": 10.0},
    }

    CATEGORY_PROGRESSION = {
        "junior": ["leadership", "prioritization", "stakeholder_management"],
        "mid": ["prioritization", "stakeholder_management", "product_decisions", "leadership"],
        "senior": ["leadership", "stakeholder_management", "failure_recovery", "product_decisions"],
    }

    FOLLOW_UP_STRATEGIES = {
        "struggling": {
            "strategy": "simplify",
            "actions": ["break_down", "provide_structure", "focus_basics"],
        },
        "developing": {
            "strategy": "guide",
            "actions": ["clarify_expectations", "probe_deeper", "scaffold_thinking"],
        },
        "proficient": {
            "strategy": "challenge",
            "actions": ["increase_complexity", "add_constraints", "explore_edge_cases"],
        },
        "strong": {
            "strategy": "stretch",
            "actions": ["senior_scenarios", "strategic_thinking", "organizational_impact"],
        },
    }

    def __init__(self, initial_level: str = "mid") -> None:
        """Initialize adaptive question selector."""
        self.initial_level = initial_level
        self.performance_history: list[float] = []
        self.category_performance: dict[str, list[float]] = {}
        self.questions_asked: list[str] = []
        self.iteration_count = 0
        self.max_iterations = 10

    def analyze_performance(
        self, evaluations: list[ResponseEvaluation], current_eval: ResponseEvaluation | None = None
    ) -> PerformanceMetrics:
        """Analyze candidate's performance across evaluations."""
        all_evals = evaluations.copy()
        if current_eval:
            all_evals.append(current_eval)

        if not all_evals:
            return PerformanceMetrics(
                avg_score=0,
                trend="stable",
                strengths=[],
                weaknesses=[],
                confidence_level=0,
                questions_answered=0,
                time_spent_minutes=0,
            )

        scores = [e.overall_score for e in all_evals]
        avg_score = sum(scores) / len(scores)

        self.performance_history.extend(scores)

        trend = self.calculate_trend(scores)

        strengths = self.identify_strengths(all_evals)
        weaknesses = self.identify_weaknesses(all_evals)

        confidence_level = min(len(all_evals) / 5, 1.0)

        return PerformanceMetrics(
            avg_score=round(avg_score, 1),
            trend=trend,
            strengths=strengths[:3],
            weaknesses=weaknesses[:3],
            confidence_level=confidence_level,
            questions_answered=len(all_evals),
            time_spent_minutes=len(all_evals) * 5,
        )

    def calculate_trend(self, scores: list[float]) -> Literal["improving", "stable", "declining"]:
        """Calculate performance trend from scores."""
        if len(scores) < 2:
            return "stable"

        recent = scores[-3:] if len(scores) >= 3 else scores
        earlier = scores[:-3] if len(scores) > 3 else scores[:1]

        recent_avg = sum(recent) / len(recent)
        earlier_avg = sum(earlier) / len(earlier)

        if recent_avg > earlier_avg + 1:
            return "improving"
        elif recent_avg < earlier_avg - 1:
            return "declining"
        else:
            return "stable"

    def identify_strengths(self, evaluations: list[ResponseEvaluation]) -> list[str]:
        """Identify consistent strength areas."""
        strength_counts: dict[str, int] = {}

        for eval in evaluations:
            for strength in eval.key_strengths:
                category = self.categorize_strength(strength)
                strength_counts[category] = strength_counts.get(category, 0) + 1

        sorted_strengths = sorted(strength_counts.items(), key=lambda x: x[1], reverse=True)
        return [s[0] for s in sorted_strengths]

    def identify_weaknesses(self, evaluations: list[ResponseEvaluation]) -> list[str]:
        """Identify consistent weakness areas."""
        weakness_counts: dict[str, int] = {}

        for eval in evaluations:
            for area in eval.improvement_areas:
                category = self.categorize_weakness(area)
                weakness_counts[category] = weakness_counts.get(category, 0) + 1

        sorted_weaknesses = sorted(weakness_counts.items(), key=lambda x: x[1], reverse=True)
        return [w[0] for w in sorted_weaknesses]

    def categorize_strength(self, strength: str) -> str:
        """Categorize a strength description."""
        strength_lower = strength.lower()

        if any(word in strength_lower for word in ["leadership", "influence", "team"]):
            return "leadership"
        elif any(word in strength_lower for word in ["prioritiz", "trade-off", "decision"]):
            return "prioritization"
        elif any(word in strength_lower for word in ["stakeholder", "communication", "align"]):
            return "stakeholder_management"
        elif any(word in strength_lower for word in ["data", "metric", "measure"]):
            return "analytical_thinking"
        elif any(word in strength_lower for word in ["strategy", "vision", "long-term"]):
            return "strategic_thinking"
        else:
            return "general_pm_skills"

    def categorize_weakness(self, weakness: str) -> str:
        """Categorize a weakness description."""
        weakness_lower = weakness.lower()

        if any(word in weakness_lower for word in ["metric", "quantif", "measure"]):
            return "quantification"
        elif any(word in weakness_lower for word in ["structure", "star", "organization"]):
            return "structure"
        elif any(word in weakness_lower for word in ["specific", "detail", "example"]):
            return "specificity"
        elif any(word in weakness_lower for word in ["impact", "outcome", "result"]):
            return "impact_articulation"
        elif any(word in weakness_lower for word in ["strategic", "business", "alignment"]):
            return "business_alignment"
        else:
            return "general_improvement"

    def select_next_question(
        self,
        question_pool: list[BehavioralQuestion],
        current_question: BehavioralQuestion | None,
        performance: PerformanceMetrics,
        completed_categories: list[str] | None = None,
    ) -> AdaptiveDecision:
        """Select the next question based on performance analysis."""
        self.iteration_count += 1

        if self.iteration_count >= self.max_iterations:
            return AdaptiveDecision(
                action="conclude",
                next_question=None,
                reasoning="Maximum iteration limit reached",
                difficulty_adjustment="same",
                focus_area=None,
            )

        if completed_categories is None:
            completed_categories = []

        available_questions = [q for q in question_pool if q.id not in self.questions_asked]

        if not available_questions:
            return AdaptiveDecision(
                action="conclude",
                next_question=None,
                reasoning="No more questions available",
                difficulty_adjustment="same",
                focus_area=None,
            )

        performance_level = self.get_performance_level(performance.avg_score)

        if performance.questions_answered < 2:
            decision = self.handle_early_interview(
                available_questions, performance_level, current_question
            )
        elif performance_level == "struggling":
            decision = self.handle_struggling_candidate(
                available_questions, performance, current_question
            )
        elif performance_level == "strong":
            decision = self.handle_strong_candidate(
                available_questions, performance, current_question
            )
        else:
            decision = self.handle_normal_progression(
                available_questions, performance, current_question
            )

        if decision.next_question:
            self.questions_asked.append(decision.next_question.id)

        return decision

    def get_performance_level(self, avg_score: float) -> str:
        """Determine performance level from average score."""
        for level, thresholds in self.PERFORMANCE_THRESHOLDS.items():
            if thresholds["min"] <= avg_score < thresholds["max"]:
                return level
        return "developing"

    def handle_early_interview(
        self,
        available_questions: list[BehavioralQuestion],
        performance_level: str,
        current_question: BehavioralQuestion | None,
    ) -> AdaptiveDecision:
        """Handle early interview progression."""
        mid_level_questions = [q for q in available_questions if q.difficulty == "mid"]

        if mid_level_questions:
            next_q = mid_level_questions[0]
        else:
            next_q = available_questions[0]

        return AdaptiveDecision(
            action="continue",
            next_question=next_q,
            reasoning="Early interview stage - establishing baseline",
            difficulty_adjustment="same",
            focus_area=next_q.category,
        )

    def handle_struggling_candidate(
        self,
        available_questions: list[BehavioralQuestion],
        performance: PerformanceMetrics,
        current_question: BehavioralQuestion | None,
    ) -> AdaptiveDecision:
        """Handle candidate who is struggling."""
        easier_questions = [q for q in available_questions if q.difficulty in ["junior", "mid"]]

        if current_question and performance.weaknesses:
            same_category = [q for q in easier_questions if q.category == current_question.category]
            if same_category:
                return AdaptiveDecision(
                    action="adjust_difficulty",
                    next_question=same_category[0],
                    reasoning="Providing easier question in same category for practice",
                    difficulty_adjustment="easier",
                    focus_area=current_question.category,
                )

        if easier_questions:
            next_q = easier_questions[0]
        else:
            next_q = available_questions[0]

        return AdaptiveDecision(
            action="adjust_difficulty",
            next_question=next_q,
            reasoning="Adjusting to easier questions to build confidence",
            difficulty_adjustment="easier",
            focus_area=next_q.category,
        )

    def handle_strong_candidate(
        self,
        available_questions: list[BehavioralQuestion],
        performance: PerformanceMetrics,
        current_question: BehavioralQuestion | None,
    ) -> AdaptiveDecision:
        """Handle high-performing candidate."""
        harder_questions = [q for q in available_questions if q.difficulty in ["mid", "senior"]]

        if performance.weaknesses:
            weakness_questions = [
                q for q in harder_questions if any(w in q.category for w in performance.weaknesses)
            ]
            if weakness_questions:
                return AdaptiveDecision(
                    action="switch_category",
                    next_question=weakness_questions[0],
                    reasoning="Testing identified weakness area at higher difficulty",
                    difficulty_adjustment="harder",
                    focus_area=weakness_questions[0].category,
                )

        senior_questions = [q for q in available_questions if q.difficulty == "senior"]

        if senior_questions:
            next_q = senior_questions[0]
            return AdaptiveDecision(
                action="adjust_difficulty",
                next_question=next_q,
                reasoning="Challenging with senior-level questions",
                difficulty_adjustment="harder",
                focus_area=next_q.category,
            )

        if harder_questions:
            next_q = harder_questions[0]
        else:
            next_q = available_questions[0]

        return AdaptiveDecision(
            action="continue",
            next_question=next_q,
            reasoning="Continuing with challenging questions",
            difficulty_adjustment="same",
            focus_area=next_q.category,
        )

    def handle_normal_progression(
        self,
        available_questions: list[BehavioralQuestion],
        performance: PerformanceMetrics,
        current_question: BehavioralQuestion | None,
    ) -> AdaptiveDecision:
        """Handle normal interview progression."""
        target_categories = self.CATEGORY_PROGRESSION.get(
            self.initial_level, self.CATEGORY_PROGRESSION["mid"]
        )

        for category in target_categories:
            category_questions = [q for q in available_questions if q.category == category]
            if category_questions:
                return AdaptiveDecision(
                    action="continue",
                    next_question=category_questions[0],
                    reasoning=f"Progressing to {category} questions",
                    difficulty_adjustment="same",
                    focus_area=category,
                )

        return AdaptiveDecision(
            action="continue",
            next_question=available_questions[0],
            reasoning="Continuing with available questions",
            difficulty_adjustment="same",
            focus_area=available_questions[0].category,
        )

    def should_conclude_early(self, performance: PerformanceMetrics, time_spent: float) -> bool:
        """Determine if interview should conclude early."""
        if performance.questions_answered >= 3 and performance.avg_score < 3:
            return True

        if time_spent > 45 and performance.questions_answered >= 5:
            return True

        if performance.trend == "declining" and performance.avg_score < 5:
            return True

        return False

    def generate_adaptive_feedback(
        self, performance: PerformanceMetrics, question_category: str
    ) -> str:
        """Generate adaptive feedback based on performance."""
        level = self.get_performance_level(performance.avg_score)
        strategy = self.FOLLOW_UP_STRATEGIES[level]

        if level == "struggling":
            feedback = (
                f"Let's focus on building your fundamentals in {question_category}. "
                "Take your time to structure your thoughts using the STAR method."
            )
        elif level == "developing":
            feedback = (
                f"You're showing good understanding. Let's work on adding more "
                f"specific details and metrics to strengthen your {question_category} examples."
            )
        elif level == "proficient":
            feedback = (
                "Strong response! Now let's explore how you'd handle this "
                "at greater scale or with additional complexity."
            )
        else:
            feedback = (
                f"Excellent demonstration of {question_category} skills. "
                "Let's discuss the strategic implications and organizational impact."
            )

        return feedback

    def reset(self) -> None:
        """Reset the adaptive selector for a new session."""
        self.performance_history = []
        self.category_performance = {}
        self.questions_asked = []
        self.iteration_count = 0


def create_adaptive_selector(initial_level: str = "mid") -> AdaptiveQuestionSelector:
    """Factory function to create adaptive question selector."""
    return AdaptiveQuestionSelector(initial_level)
