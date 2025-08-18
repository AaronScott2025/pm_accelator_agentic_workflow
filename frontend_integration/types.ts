/**
 * TypeScript interfaces for AI Interviewer PM API responses
 * These types match the enhanced Pydantic models in the backend
 */

// GRAIL Evaluation Types
interface GRAILScoreDetail {
  score: number; // 0-10
  evidence: string[];
  missing_elements: string[];
  strength_level: 'weak' | 'developing' | 'proficient' | 'strong' | 'exceptional';
}

interface GRAILEvaluationResult {
  goal_score: GRAILScoreDetail;
  resources_score: GRAILScoreDetail;
  actions_score: GRAILScoreDetail;
  impact_score: GRAILScoreDetail;
  learning_score: GRAILScoreDetail;
  overall_score: number;
  overall_assessment: string;
  pm_competency_mapping: Record<string, string>;
}

// Multi-Agent Evaluation Types
interface AgentEvaluationResult {
  agent_name: string;
  score: number; // 0-10
  confidence: number; // 0-1
  key_observations: string[];
  strengths: string[];
  improvements: string[];
  rationale: string;
}

interface ConsensusEvaluationResult {
  final_score: number;
  confidence: number;
  agent_evaluations: AgentEvaluationResult[];
  consensus_strengths: string[];
  consensus_improvements: string[];
  divergent_opinions: Record<string, string[]>;
  recommendation: string;
}

// Adaptive Questioning Types
interface AdaptiveDecisionResult {
  action: 'continue' | 'adjust_difficulty' | 'switch_category' | 'deep_dive' | 'conclude';
  next_question_id: string | null;
  reasoning: string;
  difficulty_adjustment: 'easier' | 'same' | 'harder';
  focus_area: string | null;
}

// Coaching Feedback Types
interface CoachingFeedback {
  feedback_text: string;
  coaching_patterns: Record<string, string[]>;
  encouragement_message: string;
  adapted_followups: string[];
}

// Performance Metrics
interface PerformanceMetrics {
  avg_score: number;
  trend: 'improving' | 'stable' | 'declining';
  strengths: string[];
  weaknesses: string[];
  confidence_level: number;
  questions_answered: number;
  time_spent_minutes: number;
}

// Main Evaluation Response
interface EvaluateResponse {
  feedback: string;
  rubric_score: Record<string, any>;
  followups: string[];
  template_answer?: string;
  contexts?: Array<Record<string, any>>;
  ragas?: Record<string, any>;
  judge?: {
    score: number;
    rationale: string;
  };
  // NEW: Enhanced evaluation fields
  grail_evaluation?: GRAILEvaluationResult;
  consensus_evaluation?: ConsensusEvaluationResult;
  adaptive_decision?: AdaptiveDecisionResult;
  coaching_feedback?: CoachingFeedback;
}

// Behavioral Submit Response
interface BehavioralSubmitResponse {
  session_id: string;
  evaluation: EvaluateResponse;
  next_question?: {
    question: string;
    category: string;
    difficulty: string;
    question_index: number;
    total_questions: number;
  };
  interview_completed: boolean;
  progress: {
    questions_completed: number;
    questions_remaining: number;
  };
  refinement_allowed: boolean;
  refinement_count: number;
  improvement_tips: string[];
  // NEW: Performance tracking
  performance_metrics?: PerformanceMetrics;
}

export type {
  GRAILScoreDetail,
  GRAILEvaluationResult,
  AgentEvaluationResult,
  ConsensusEvaluationResult,
  AdaptiveDecisionResult,
  CoachingFeedback,
  PerformanceMetrics,
  EvaluateResponse,
  BehavioralSubmitResponse,
};